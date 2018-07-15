use { ImageFramebuffer, ObjectId, RenderTarget, window::Window };
use texture::Texture;
use std::sync::{ Arc, Mutex };
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	descriptor::{
		DescriptorSet,
		PipelineLayoutAbstract,
		descriptor_set::{ FixedSizeDescriptorSetsPool, PersistentDescriptorSet }
	},
	device::{ Device, Queue },
	format::Format,
	framebuffer::{ Framebuffer, FramebufferAbstract, FramebufferCreationError, RenderPassAbstract, Subpass },
	image::ImageViewAccess,
	instance::QueueFamily,
	memory::DeviceMemoryAllocError,
	pipeline::{ GraphicsPipeline, GraphicsPipelineAbstract, viewport::Viewport },
	sampler::{ Filter, MipmapMode, Sampler, SamplerAddressMode, SamplerCreationError },
	sync::GpuFuture,
};

pub struct SpriteBatch {
	shared: Arc<SpriteBatchShared>,
	meshes: Vec<Sprite>,
	framebuffers: Vec<ImageFramebuffer>,
	target_id: ObjectId,
	target_desc: Arc<DescriptorSet + Send + Sync + 'static>,
}
impl SpriteBatch {
	pub fn new(
		window: &Window,
		target: &RenderTarget,
		shared: Arc<SpriteBatchShared>
	) -> Result<(Self, impl GpuFuture), DeviceMemoryAllocError> {
		let dimensions = target.images()[0].dimensions();
		let (target_descs, future) =
			Self::make_target_desc(
				window.queue().clone(),
				shared.pipeline.clone(),
				dimensions.width(),
				dimensions.height()
			)?;

		let framebuffers =
			target.images().iter()
				.map(|image| {
					Framebuffer::start(shared.subpass.render_pass().clone())
						.add(image.clone())
						.and_then(|fb| fb.build())
						.map(|fb| ImageFramebuffer::new(Arc::downgrade(&image), Arc::new(fb)))
						.map_err(|err| match err {
							FramebufferCreationError::OomError(err) => err,
							err => unreachable!("{:?}", err),
						})
				})
				.collect::<Result<Vec<_>, _>>()?;

		Ok((
			Self {
				shared: shared,
				meshes: vec![],
				framebuffers: framebuffers,
				target_id: target.id_root().make_id(),
				target_desc: target_descs,
			},
			future
		))
	}

	pub fn add_sprite(&mut self, sprite: Sprite) {
		self.meshes.push(sprite);
	}

	fn make_target_desc(
		queue: Arc<Queue>,
		pipeline: impl PipelineLayoutAbstract + Send + Sync + 'static,
		width: u32,
		height: u32
	) -> Result<(Arc<DescriptorSet + Send + Sync + 'static>, impl GpuFuture), DeviceMemoryAllocError> {
		let (target_size, future) = ImmutableBuffer::from_data([width, height], BufferUsage::uniform_buffer(), queue)?;

		Ok((
			Arc::new(
				PersistentDescriptorSet::start(pipeline, 0)
					.add_buffer(target_size.clone())
					.unwrap()
					.build()
					.unwrap()
			),
			future
		))
	}

	pub fn commands(
		&mut self,
		window: &Window,
		target: &RenderTarget,
		image_num: usize,
	) -> Result<(AutoCommandBuffer, Option<impl GpuFuture>), DeviceMemoryAllocError> {
		assert!(self.target_id.is_child_of(target.id_root()));

		let framebuffer = self.framebuffers[image_num].image
			.upgrade()
			.iter()
			.filter(|old_image| Arc::ptr_eq(&target.images()[image_num], &old_image))
			.next()
			.map(|_| self.framebuffers[image_num].framebuffer.clone());
		let (framebuffer, future) =
			if let Some(framebuffer) = framebuffer {
				(framebuffer, None)
			} else {
				let framebuffer = Framebuffer::start(self.shared.subpass.render_pass().clone())
					.add(target.images()[image_num].clone())
					.and_then(|fb| fb.build())
					.map(|fb| Arc::new(fb))
					.map_err(|err| {
						match err { FramebufferCreationError::OomError(err) => err, err => unreachable!("{:?}", err) }
					})?;
				self.framebuffers[image_num] =
					ImageFramebuffer::new(Arc::downgrade(&target.images()[image_num]), framebuffer.clone());

				let (target_desc, future) =
					Self::make_target_desc(
						window.queue().clone(),
						self.shared.pipeline.clone(),
						framebuffer.width(),
						framebuffer.height()
					)?;

				self.target_desc = target_desc;

				(framebuffer as _, Some(future))
			};

		let dimensions = [framebuffer.width() as f32, framebuffer.height() as f32];

		let mut command_buffer =
			AutoCommandBufferBuilder::primary_one_time_submit(self.shared.shaders.device.clone(), window.queue().family())?
				.begin_render_pass(framebuffer, true, vec![[0.1, 0.1, 0.1, 1.0].into()])
				.unwrap();

		for mesh in &mut self.meshes {
			command_buffer =
				unsafe {
					command_buffer
						.execute_commands(
							mesh.make_commands(&self.shared, &self.target_desc, window.queue().family(), dimensions)?
						)
						.unwrap()
				};
		}

		Ok((
			command_buffer.end_render_pass().unwrap()
				.build()
				.map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?,
			future
		))
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
	vertex_shader: vs::Shader,
	fragment_shader: fs::Shader,
	sampler: Arc<Sampler>,
}
impl SpriteBatchShaders {
	pub fn new(window: &mut Window) -> Result<(Arc<Self>, impl GpuFuture), SpriteBatchShadersError> {
		let (vertices, future) =
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

		Ok((
			Arc::new(Self {
				device: window.device().clone(),
				vertices: vertices,
				vertex_shader: vs::Shader::load(window.device().clone())?,
				fragment_shader: fs::Shader::load(window.device().clone())?,
				sampler:
					Sampler::new(
						window.device().clone(),
						Filter::Linear,
						Filter::Linear, MipmapMode::Nearest,
						SamplerAddressMode::Repeat,
						SamplerAddressMode::Repeat,
						SamplerAddressMode::Repeat,
						0.0, 1.0, 0.0, 0.0
					)?,
			}),
			future
		))
	}
}

#[derive(Debug)]
pub enum SpriteBatchShadersError {
	DeviceMemoryAllocError(DeviceMemoryAllocError),
	OomError(OomError),
	TooManyObjects,
}
impl From<DeviceMemoryAllocError> for SpriteBatchShadersError {
	fn from(val: DeviceMemoryAllocError) -> Self {
		SpriteBatchShadersError::DeviceMemoryAllocError(val)
	}
}
impl From<OomError> for SpriteBatchShadersError {
	fn from(val: OomError) -> Self {
		SpriteBatchShadersError::OomError(val)
	}
}
impl From<SamplerCreationError> for SpriteBatchShadersError {
	fn from(val: SamplerCreationError) -> Self {
		match val {
			SamplerCreationError::OomError(err) => SpriteBatchShadersError::OomError(err),
			SamplerCreationError::TooManyObjects => SpriteBatchShadersError::TooManyObjects,
			_ => unreachable!(),
		}
	}
}

pub struct Sprite {
	static_desc: Arc<DescriptorSet + Send + Sync + 'static>,
	position: Arc<ImmutableBuffer<[f32; 2]>>,
}
impl Sprite {
	pub fn new(
		window: &Window,
		shared: &SpriteBatchShared,
		texture: &Texture,
		position: [f32; 2]
	) -> Result<(Self, impl GpuFuture), DeviceMemoryAllocError> {
		let (position, future) =
			ImmutableBuffer::from_data(position, BufferUsage::uniform_buffer(), window.queue().clone())?;

		Ok((
			Self {
				static_desc:
					Arc::new(
						PersistentDescriptorSet::start(shared.pipeline.clone(), 2)
							.add_sampled_image(texture.image().clone(), shared.shaders.sampler.clone())
							.unwrap()
							.build()
							.unwrap()
					),
				position: position
			},
			future
		))
	}

	fn make_commands(
		&mut self,
		shared: &SpriteBatchShared,
		target_desc: &Arc<DescriptorSet + Send + Sync + 'static>,
		queue_family: QueueFamily,
		dimensions: [f32; 2],
	) -> Result<AutoCommandBuffer, OomError> {
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

#[derive(Debug, Clone)]
struct SpriteVertex { position: [f32; 2] }
impl_vertex!(SpriteVertex, position);

mod vs {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "vertex"]
	#[src = "#version 450

layout(location = 0) in vec2 position;

layout(location = 0) out vec2 tex_coord;

layout(set = 0, binding = 0) uniform Target {
	uvec2 size;
} target;

layout(set = 1, binding = 0) uniform SpriteDynamic {
	vec2 pos;
} sprite_dynamic;

layout(set = 2, binding = 0) uniform sampler2D tex;

void main() {
	tex_coord = position;
	gl_Position = vec4(2 * (sprite_dynamic.pos + textureSize(tex, 0) * position) / target.size - 1, 0.0, 1.0);
}
"]
	struct Dummy;
}

mod fs {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "fragment"]
	#[src = "#version 450

layout(location = 0) in vec2 tex_coord;
layout(location = 0) out vec4 f_color;

layout(set = 2, binding = 0) uniform sampler2D tex;

void main() {
	f_color = texture(tex, tex_coord);
}
"]
	struct Dummy;
}
