use { Drawable, ObjectId, ObjectIdRoot, RenderTarget, window::Window };
use std::sync::{ Arc, Weak };
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	device::{ Device },
	format::Format,
	framebuffer::{ Framebuffer, FramebufferAbstract, FramebufferCreationError, RenderPassAbstract, Subpass },
	image::ImageViewAccess,
	instance::QueueFamily,
	memory::DeviceMemoryAllocError,
	pipeline::{ GraphicsPipeline, GraphicsPipelineAbstract, viewport::Viewport },
};

pub struct SpriteBatch {
	shared: Arc<SpriteBatchShared>,
	meshes: Vec<Triangle>,
	framebuffers:
		Vec<Option<(Weak<ImageViewAccess + Send + Sync + 'static>, Arc<FramebufferAbstract + Send + Sync + 'static>)>>,
	target_id: ObjectId,
}
impl SpriteBatch {
	pub fn new(shared: Arc<SpriteBatchShared>, target: &RenderTarget) -> Self {
		Self {
			shared: shared,
			meshes: vec![],
			framebuffers: vec![None; target.image_count()],
			target_id: target.id_root().make_id()
		}
	}

	pub fn add_triangle(&mut self, triangle: Triangle) {
		self.meshes.push(triangle);
	}
}
impl Drawable for SpriteBatch {
	fn commands(
		&mut self,
		target_id: &ObjectIdRoot,
		queue_family: QueueFamily,
		image_num: usize,
		image: &Arc<ImageViewAccess + Send + Sync + 'static>,
	) -> Result<AutoCommandBuffer, OomError> {
		assert!(self.target_id.is_child_of(target_id));

		let framebuffer = self.framebuffers[image_num].as_ref()
			.and_then(|(old_image, fb)| {
				old_image.upgrade().iter().filter(|old_image| Arc::ptr_eq(image, &old_image)).next().map(|_| fb.clone())
			});
		let framebuffer =
			if let Some(framebuffer) = framebuffer {
				framebuffer
			} else {
				let framebuffer = Framebuffer::start(self.shared.subpass.render_pass().clone())
					.add(image.clone())
					.and_then(|fb| fb.build())
					.map(|fb| Arc::new(fb))
					.map_err(|err| {
						match err { FramebufferCreationError::OomError(err) => err, err => unreachable!("{}", err) }
					})?;
				self.framebuffers[image_num] = Some((Arc::downgrade(image), framebuffer.clone()));
				framebuffer
			};

		let dimensions = [framebuffer.width() as f32, framebuffer.height() as f32];

		let mut command_buffer =
			AutoCommandBufferBuilder::primary_one_time_submit(self.shared.device.clone(), queue_family)?
				.begin_render_pass(framebuffer, true, vec![[0.1, 0.1, 0.1, 1.0].into()])
				.unwrap();

		for mesh in &self.meshes {
			command_buffer =
				unsafe {
					command_buffer
						.execute_commands(
							mesh.make_commands(
								self.shared.device.clone(),
								queue_family,
								self.shared.subpass.clone(),
								self.shared.pipeline.clone(),
								dimensions,
							)
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
	device: Arc<Device>,
	subpass: Subpass<Arc<RenderPassAbstract + Send + Sync>>,
	pipeline: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
}
impl SpriteBatchShared {
	pub fn new(shaders: &SpriteBatchShaders, format: Format) -> Arc<Self> {
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
				.vertex_input_single_buffer::<TriangleVertex>()
				.vertex_shader(shaders.vertex_shader.main_entry_point(), ())
				.triangle_list()
				.viewports_dynamic_scissors_irrelevant(1)
				.fragment_shader(shaders.fragment_shader.main_entry_point(), ())
				.render_pass(subpass.clone())
				.build(shaders.device.clone())
				.expect("failed to create pipeline")
		);

		Arc::new(Self {
			device: shaders.device.clone(),
			subpass: subpass,
			pipeline: pipeline
		})
	}
}

pub struct SpriteBatchShaders {
	device: Arc<Device>,
	vertex_shader: vs::Shader,
	fragment_shader: fs::Shader,
}
impl SpriteBatchShaders {
	pub fn new(window: &Window) -> Result<Self, OomError> {
		Ok(
			Self {
				device: window.device().clone(),
				vertex_shader: vs::Shader::load(window.device().clone())?,
				fragment_shader: fs::Shader::load(window.device().clone())?,
			}
		)
	}
}

pub struct Triangle {
	buffer: Arc<ImmutableBuffer<[TriangleVertex]>>,
}
impl Triangle {
	pub fn new(window: &mut Window) -> Result<Self, DeviceMemoryAllocError> {
		let (buffer, future) =
			ImmutableBuffer::from_iter(
				[
					TriangleVertex { position: [-0.5, -0.25] },
					TriangleVertex { position: [0.0, 0.5] },
					TriangleVertex { position: [0.25, -0.1] },
				].iter().cloned(),
				BufferUsage::vertex_buffer(),
				window.queue().clone(),
			)?;

		window.join_future(future);

		Ok(Self { buffer: buffer })
	}

	fn make_commands(
		&self,
		device: Arc<Device>,
		queue_family: QueueFamily,
		subpass: Subpass<impl RenderPassAbstract + Clone + Send + Sync + 'static>,
		pipeline: impl GraphicsPipelineAbstract + Send + Sync + 'static + Clone,
		dimensions: [f32; 2]
	) -> AutoCommandBuffer {
		AutoCommandBufferBuilder::secondary_graphics_one_time_submit(device, queue_family, subpass)
			.unwrap()
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
				vec![self.buffer.clone()], (), ()
			)
			.unwrap()
			.build()
			.unwrap()
	}
}

#[derive(Debug, Clone)]
struct TriangleVertex { position: [f32; 2] }
impl_vertex!(TriangleVertex, position);

mod vs {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "vertex"]
	#[src = "
#version 450
layout(location = 0) in vec2 position;
void main() {
	gl_Position = vec4(position, 0.0, 1.0);
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
