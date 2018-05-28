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
	sampler::{ Filter, MipmapMode, Sampler, SamplerAddressMode },
	sync::{ FenceSignalFuture, GpuFuture, NowFuture },
};

pub struct SpriteBatch {
	shared: Arc<SpriteBatchShared>,
	meshes: Vec<Sprite>,
	framebuffers:
		Vec<Option<(Weak<ImageViewAccess + Send + Sync + 'static>, Arc<FramebufferAbstract + Send + Sync + 'static>)>>,
	target_id: ObjectId,
	target_desc: Arc<DescriptorSet + Send + Sync + 'static>,
}
impl SpriteBatch {
	pub fn new(target: &mut RenderTarget, shared: Arc<SpriteBatchShared>) -> Result<Self, DeviceMemoryAllocError> {
		let dimensions = target.images()[0].dimensions();
		let target_descs =
			Self::make_target_desc(target, &shared, dimensions.width() as f32, dimensions.height() as f32)?;

		Ok(Self {
			shared: shared,
			meshes: vec![],
			framebuffers: vec![None; target.images().len()],
			target_id: target.id_root().make_id(),
			target_desc: target_descs,
		})
	}

	pub fn add_sprite(&mut self, sprite: Sprite) {
		self.meshes.push(sprite);
	}

	fn make_target_desc(
		target: &mut RenderTarget,
		shared: &SpriteBatchShared,
		width: f32,
		height: f32
	) -> Result<Arc<DescriptorSet + Send + Sync + 'static>, DeviceMemoryAllocError> {
		let (target_size, future) =
			ImmutableBuffer::from_data([width, height], BufferUsage::uniform_buffer(), target.queue().clone())?;
		target.join_future(Box::new(future));

		Ok(Arc::new(
			PersistentDescriptorSet::start(shared.pipeline.clone(), 0)
				.add_buffer(target_size.clone())
				.unwrap()
				.build()
				.unwrap()
		))
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

				self.target_desc =
					Self::make_target_desc(target, &self.shared, framebuffer.width() as f32, framebuffer.height() as f32)?;

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
							mesh.make_commands(&self.shared, &self.target_desc, target.queue().family(), dimensions,)?
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
	pipeline: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
	sprite_desc_pool: Mutex<FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>>,
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

		let pipeline = Arc::new(
			GraphicsPipeline::start()
				.vertex_input_single_buffer::<SpriteVertex>()
				.vertex_shader(shaders.vertex_shader.main_entry_point(), ())
				.triangle_list()
				.viewports_dynamic_scissors_irrelevant(1)
				.fragment_shader(shaders.fragment_shader.main_entry_point(), ())
				.render_pass(subpass.clone())
				.build(shaders.device.clone())
				.expect("failed to create pipeline")
		);

		Arc::new(Self {
			shaders: shaders,
			subpass: subpass,
			pipeline: pipeline.clone(),
			sprite_desc_pool: Mutex::new(FixedSizeDescriptorSetsPool::new(pipeline, 1)),
		})
	}
}

pub struct SpriteBatchShaders {
	device: Arc<Device>,
	vertices: Arc<ImmutableBuffer<[SpriteVertex; 6]>>,
	white_pixel: Arc<ImmutableImage<R8G8B8A8Srgb>>,
	vertex_shader: vs::Shader,
	fragment_shader: fs::Shader,
}
impl SpriteBatchShaders {
	pub fn new(window: &mut Window) -> Result<Arc<Self>, DeviceMemoryAllocError> {
		let (vertices, vertex_future) =
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

		let (white_pixel, white_pixel_future) =
			ImmutableImage::from_iter(
				[255, 255, 255, 255].iter().cloned(),
				Dimensions::Dim2d { width: 1, height: 1 },
				R8G8B8A8Srgb,
				window.queue().clone(),
			)
			.unwrap();

		window.join_future(Box::new(vertex_future.join(white_pixel_future)));

		Ok(Arc::new(Self {
			device: window.device().clone(),
			vertices: vertices,
			white_pixel: white_pixel,
			vertex_shader: vs::Shader::load(window.device().clone())?,
			fragment_shader: fs::Shader::load(window.device().clone())?,
		}))
	}
}

pub struct Sprite {
	state: Arc<Atom<Box<SpriteState>>>,
	static_desc: Arc<DescriptorSet + Send + Sync + 'static>,
	position: Arc<ImmutableBuffer<[f32; 2]>>,
}
impl Sprite {
	pub fn from_file_with_format<P>(window: &mut Window, shared: &SpriteBatchShared, path: P, format: ImageFormat) -> Self
	where P: AsRef<Path> + Send + 'static {
		let state = Arc::new(Atom::new(Box::new(SpriteState::LoadingCpu)));

		let (target_size, future) =
			ImmutableBuffer::from_data([100.0f32, 100.0f32], BufferUsage::uniform_buffer(), window.queue().clone()).unwrap();
		window.join_future(Box::new(future));

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
			static_desc:
				Arc::new(
					PersistentDescriptorSet::start(shared.pipeline.clone(), 2)
						.add_buffer(target_size.clone())
						.unwrap()
						.add_sampled_image(
							shared.shaders.white_pixel.clone(),
							Sampler::new(
								window.device().clone(),
								Filter::Linear,
								Filter::Linear, MipmapMode::Nearest,
								SamplerAddressMode::Repeat,
								SamplerAddressMode::Repeat,
								SamplerAddressMode::Repeat,
								0.0, 1.0, 0.0, 0.0
							).unwrap(),
						)
						.unwrap()
						.build()
						.unwrap()
				),
			position: ImmutableBuffer::from_data([10.0, 10.0], BufferUsage::uniform_buffer(), window.queue().clone())
				.unwrap()
				.0,
		}
	}

	fn make_commands(
		&self,
		shared: &SpriteBatchShared,
		target_desc: &Arc<DescriptorSet + Send + Sync + 'static>,
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

		self.state.set_if_none(state);

		Ok(
			AutoCommandBufferBuilder::secondary_graphics_one_time_submit(shared.shaders.device.clone(), queue_family, shared.subpass.clone())?
				.draw(
					shared.pipeline.clone(),
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
					vec![shared.shaders.vertices.clone()],
					(
						target_desc.clone(),
						shared.sprite_desc_pool.lock().unwrap()
							.next()
							.add_buffer(self.position.clone())
							.unwrap()
							.build()
							.unwrap(),
						self.static_desc.clone(),
					),
					()
				)
				.unwrap()
				.build()
				.map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?
		)
	}
}

enum SpriteState {
	LoadingCpu,
	LoadingGpu(
		Arc<ImmutableImage<R8G8B8A8Srgb>>,
		FenceSignalFuture<CommandBufferExecFuture<NowFuture, AutoCommandBuffer>>,
	),
	Loaded(Arc<ImmutableImage<R8G8B8A8Srgb>>),
}

#[derive(Debug, Clone)]
struct SpriteVertex { position: [f32; 2] }
impl_vertex!(SpriteVertex, position);

mod vs {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "vertex"]
	#[src = "#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;

layout(set = 0, binding = 0) uniform Target {
	vec2 size;
} target;

layout(set = 1, binding = 0) uniform SpriteDynamic {
	vec2 pos;
} sprite_dynamic;

layout(set = 2, binding = 0) uniform SpriteStatic {
	vec2 size;
} sprite_static;

void main() {
	tex_coords = position;
	gl_Position = vec4((2 * sprite_dynamic.pos + sprite_static.size * position - target.size) / target.size, 0.0, 1.0);
}
"]
	struct Dummy;
}

mod fs {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "fragment"]
	#[src = "#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 2, binding = 1) uniform sampler2D tex;

void main() {
	f_color = texture(tex, tex_coords);
}
"]
	struct Dummy;
}
