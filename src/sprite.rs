pub use image::ImageFormat;
use { CPU_POOL, FS_POOL, Drawable, ObjectId, RenderTarget, window::Window };
use atom::Atom;
use image;
use std::{ fs::File, io::prelude::*, path::Path, sync::{ Arc, Mutex, Weak } };
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, CommandBufferExecFuture, DynamicState },
	descriptor::{ DescriptorSet, descriptor_set::{ FixedSizeDescriptorSetsPool, PersistentDescriptorSet } },
	device::Device,
	format::{ Format, R8G8B8A8Srgb },
	framebuffer::{ Framebuffer, FramebufferAbstract, FramebufferCreationError, RenderPassAbstract, Subpass },
	image::{ Dimensions, ImageViewAccess, immutable::ImmutableImage },
	instance::QueueFamily,
	memory::DeviceMemoryAllocError,
	pipeline::{ GraphicsPipeline, GraphicsPipelineAbstract, viewport::Viewport },
	sync::{ FenceSignalFuture, GpuFuture, NowFuture },
};

pub struct SpriteBatch {
	shared: Arc<SpriteBatchShared>,
	meshes: Vec<Sprite>,
	framebuffers:
		Vec<Option<(Weak<ImageViewAccess + Send + Sync + 'static>, Arc<FramebufferAbstract + Send + Sync + 'static>)>>,
	target_id: ObjectId,
	target_descs: TargetDescs,
}
impl SpriteBatch {
	pub fn new(target: &mut RenderTarget, shared: Arc<SpriteBatchShared>) -> Result<Self, DeviceMemoryAllocError> {
		let dimensions = target.images()[0].dimensions();
		let target_descs =
			Self::make_target_descs(target, &shared, dimensions.width() as f32, dimensions.height() as f32)?;

		Ok(Self {
			shared: shared,
			meshes: vec![],
			framebuffers: vec![None; target.images().len()],
			target_id: target.id_root().make_id(),
			target_descs: target_descs,
		})
	}

	pub fn add_sprite(&mut self, sprite: Sprite) {
		self.meshes.push(sprite);
	}

	fn make_target_descs(
		target: &mut RenderTarget,
		shared: &SpriteBatchShared,
		width: f32,
		height: f32
	) -> Result<TargetDescs, DeviceMemoryAllocError> {
		let (target_size, future) =
			ImmutableBuffer::from_data([width, height], BufferUsage::uniform_buffer(), target.queue().clone())?;
		target.join_future(Box::new(future));

		Ok(TargetDescs {
			color:
				Arc::new(
					PersistentDescriptorSet::start(shared.color_pipeline.clone(), 0)
						.add_buffer(target_size.clone())
						.unwrap()
						.build()
						.unwrap()
				),
			image:
				Arc::new(
					PersistentDescriptorSet::start(shared.image_pipeline.clone(), 0)
						.add_buffer(target_size.clone())
						.unwrap()
						.build()
						.unwrap()
				),
		})
	}
}
impl Drawable for SpriteBatch {
	fn commands(
		&mut self,
		target: &mut RenderTarget,
		image_num: usize,
	) -> Result<AutoCommandBuffer, DeviceMemoryAllocError> {
		assert!(self.target_id.is_child_of(target.id_root()));

		let framebuffer = self.framebuffers[image_num].as_ref()
			.and_then(|(old_image, fb)| {
				old_image.upgrade()
					.iter()
					.filter(|old_image| Arc::ptr_eq(&target.images()[image_num], &old_image))
					.next()
					.map(|_| fb.clone())
			});
		let framebuffer =
			if let Some(framebuffer) = framebuffer {
				framebuffer
			} else {
				let framebuffer = Framebuffer::start(self.shared.subpass.render_pass().clone())
					.add(target.images()[image_num].clone())
					.and_then(|fb| fb.build())
					.map(|fb| Arc::new(fb))
					.map_err(|err| {
						match err { FramebufferCreationError::OomError(err) => err, err => unreachable!("{}", err) }
					})?;
				self.framebuffers[image_num] = Some((Arc::downgrade(&target.images()[image_num]), framebuffer.clone()));

				self.target_descs =
					Self::make_target_descs(target, &self.shared, framebuffer.width() as f32, framebuffer.height() as f32)?;

				framebuffer
			};

		let dimensions = [framebuffer.width() as f32, framebuffer.height() as f32];

		let mut command_buffer =
			AutoCommandBufferBuilder::primary_one_time_submit(self.shared.shaders.device.clone(), target.queue().family())?
				.begin_render_pass(framebuffer, true, vec![[0.1, 0.1, 0.1, 1.0].into()])
				.unwrap();

		for mesh in &mut self.meshes {
			command_buffer =
				unsafe {
					command_buffer
						.execute_commands(
							mesh.make_commands(&self.shared, &self.target_descs, target.queue().family(), dimensions,)?
						)
						.unwrap()
				};
		}

		Ok(
			command_buffer.end_render_pass().unwrap()
				.build()
				.map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?
		)
	}
}

pub struct SpriteBatchShared {
	shaders: Arc<SpriteBatchShaders>,
	subpass: Subpass<Arc<RenderPassAbstract + Send + Sync>>,
	color_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
	image_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
	color_sprite_desc_pool: Mutex<FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>>,
	image_sprite_desc_pool: Mutex<FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>>,
}
impl SpriteBatchShared {
	pub fn new(shaders: Arc<SpriteBatchShaders>, format: Format) -> Arc<Self> {
		let subpass =
			Subpass::from(
				Arc::new(
					single_pass_renderpass!(
						shaders.device.clone(),
						attachments: { color: { load: Clear, store: Store, format: format, samples: 1, } },
						pass: { color: [color], depth_stencil: {} }
					).expect("failed to create render pass")
				) as Arc<RenderPassAbstract + Send + Sync>,
				0
			).expect("failed to create subpass");

		let color_pipeline = Arc::new(
			GraphicsPipeline::start()
				.vertex_input_single_buffer::<SpriteVertex>()
				.vertex_shader(shaders.color_vertex_shader.main_entry_point(), ())
				.triangle_list()
				.viewports_dynamic_scissors_irrelevant(1)
				.fragment_shader(shaders.color_fragment_shader.main_entry_point(), ())
				.render_pass(subpass.clone())
				.build(shaders.device.clone())
				.expect("failed to create pipeline")
		);

		let image_pipeline = Arc::new(
			GraphicsPipeline::start()
				.vertex_input_single_buffer::<SpriteVertex>()
				.vertex_shader(shaders.image_vertex_shader.main_entry_point(), ())
				.triangle_list()
				.viewports_dynamic_scissors_irrelevant(1)
				.fragment_shader(shaders.image_fragment_shader.main_entry_point(), ())
				.render_pass(subpass.clone())
				.build(shaders.device.clone())
				.expect("failed to create pipeline")
		);

		Arc::new(Self {
			shaders: shaders,
			subpass: subpass,
			color_pipeline: color_pipeline.clone(),
			image_pipeline: image_pipeline.clone(),
			color_sprite_desc_pool: Mutex::new(FixedSizeDescriptorSetsPool::new(color_pipeline, 1)),
			image_sprite_desc_pool: Mutex::new(FixedSizeDescriptorSetsPool::new(image_pipeline, 1)),
		})
	}
}

pub struct SpriteBatchShaders {
	device: Arc<Device>,
	buffer: Arc<ImmutableBuffer<[SpriteVertex; 6]>>,
	color_vertex_shader: vs::Shader,
	color_fragment_shader: fs::Shader,
	image_vertex_shader: vs::Shader,
	image_fragment_shader: fs::Shader,
}
impl SpriteBatchShaders {
	pub fn new(window: &mut Window) -> Result<Arc<Self>, DeviceMemoryAllocError> {
		let (buffer, future) =
			ImmutableBuffer::from_data(
				[
					SpriteVertex { position: [0.0, 0.0] },
					SpriteVertex { position: [1.0, 0.0] },
					SpriteVertex { position: [0.0, 1.0] },
					SpriteVertex { position: [0.0, 1.0] },
					SpriteVertex { position: [1.0, 0.0] },
					SpriteVertex { position: [1.0, 1.0] },
				],
				BufferUsage::vertex_buffer(),
				window.queue().clone(),
			)?;

		window.join_future(Box::new(future));

		Ok(Arc::new(Self {
			device: window.device().clone(),
			buffer: buffer,
			color_vertex_shader: vs::Shader::load(window.device().clone())?,
			color_fragment_shader: fs::Shader::load(window.device().clone())?,
			image_vertex_shader: vs::Shader::load(window.device().clone())?,
			image_fragment_shader: fs::Shader::load(window.device().clone())?,
		}))
	}
}

pub struct Sprite {
	state: Arc<Atom<Box<SpriteState>>>,
	position: Arc<ImmutableBuffer<[f32; 2]>>,
}
impl Sprite {
	pub fn from_file_with_format<P>(window: &Window, path: P, format: ImageFormat) -> Self
	where P: AsRef<Path> + Send + 'static {
		let state = Arc::new(Atom::new(Box::new(SpriteState::LoadingCpu)));

		{
			let queue = window.queue().clone();
			let state = state.clone();
			FS_POOL
				.spawn_fn(move || {
					let mut bytes = vec![];
					File::open(path).unwrap().read_to_end(&mut bytes).unwrap();

					CPU_POOL
						.spawn_fn(move || {
							let img = image::load_from_memory_with_format(&bytes, format).unwrap().to_rgba();
							let (width, height) = img.dimensions();
							let img = img.into_raw();

							let (img, future) =
								ImmutableImage::from_iter(
									img.into_iter(),
									Dimensions::Dim2d { width: width, height: height },
									R8G8B8A8Srgb,
									queue,
								)
								.unwrap();
							let future = future.then_signal_fence_and_flush().unwrap();

							state.swap(Box::new(SpriteState::LoadingGpu(img, future)));

							Ok(()) as Result<(), ()>
						})
						.forget();

					Ok(()) as Result<(), ()>
				})
				.forget();
		}

		Self {
			state: state,
			position: ImmutableBuffer::from_data([10.0, 10.0], BufferUsage::uniform_buffer(), window.queue().clone())
				.unwrap()
				.0
		}
	}

	fn make_commands(
		&self,
		shared: &SpriteBatchShared,
		target_descs: &TargetDescs,
		queue_family: QueueFamily,
		dimensions: [f32; 2],
	) -> Result<AutoCommandBuffer, OomError> {
		let state = self.state.take().unwrap();
		let state =
			match &*state {
				SpriteState::LoadingGpu(img, future) => match future.wait(Some(Default::default())) {
					Ok(()) => Some(Box::new(SpriteState::Loaded(img.clone()))),
					_ => None
				},
				_ => None
			}
			.map_or(state, |new| new);

		let (pipeline, target_desc, sprite_desc) =
			if let SpriteState::Loaded(_) = &*state {
				(shared.image_pipeline.clone(), target_descs.image.clone(), &shared.image_sprite_desc_pool)
			} else {
				(shared.color_pipeline.clone(), target_descs.color.clone(), &shared.color_sprite_desc_pool)
			};
		let sprite_desc = sprite_desc.lock().unwrap()
			.next()
			.add_buffer(self.position.clone())
			.unwrap()
			.build()
			.unwrap();

		self.state.set_if_none(state);

		Ok(
			AutoCommandBufferBuilder::secondary_graphics_one_time_submit(shared.shaders.device.clone(), queue_family, shared.subpass.clone())?
				.draw(
					pipeline,
					DynamicState {
						line_width: None,
						viewports: Some(vec![
							Viewport {
								origin: [0.0, 0.0],
								dimensions: dimensions,
								depth_range: 0.0..1.0,
							}
						]),
						scissors: None,
					},
					vec![shared.shaders.buffer.clone()], (target_desc, sprite_desc), ()
				)
				.unwrap()
				.build()
				.map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?
		)
	}
}

enum SpriteState {
	LoadingCpu,
	LoadingGpu(Arc<ImmutableImage<R8G8B8A8Srgb>>, FenceSignalFuture<CommandBufferExecFuture<NowFuture, AutoCommandBuffer>>),
	Loaded(Arc<ImmutableImage<R8G8B8A8Srgb>>),
}

#[derive(Debug, Clone)]
struct SpriteVertex { position: [f32; 2] }
impl_vertex!(SpriteVertex, position);

struct TargetDescs {
	color: Arc<DescriptorSet + Send + Sync + 'static>,
	image: Arc<DescriptorSet + Send + Sync + 'static>,
}

mod vs {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "vertex"]
	#[src = "#version 450

layout(location = 0) in vec2 position;

layout(set = 0, binding = 0) uniform Target {
	vec2 size;
} target;

layout(set = 1, binding = 0) uniform Sprite {
	vec2 pos;
} sprite;

void main() {
	gl_Position = vec4(sprite.pos / target.size * 2 - 1 + position, 0.0, 1.0);
}
"]
	struct Dummy;
}

mod fs {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "fragment"]
	#[src = "
#version 450
layout(location = 0) out vec4 f_color;
void main() {
	f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"]
	struct Dummy;
}
