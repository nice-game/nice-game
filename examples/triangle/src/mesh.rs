use nice_game::{Drawable, RenderTarget};
use std::sync::{
	Arc,
	atomic::{AtomicBool, Ordering},
};
use vulkano::{
	buffer::{BufferUsage, ImmutableBuffer},
	command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, CommandBufferExecFuture, DynamicState},
	device::{Device, Queue},
	format::Format,
	framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
	image::ImageViewAccess,
	instance::QueueFamily,
	memory::DeviceMemoryAllocError,
	pipeline::{
		GraphicsPipeline,
		GraphicsPipelineAbstract,
		viewport::Viewport,
	},
	sync::NowFuture,
};

pub struct MeshBatch {
	shared: Arc<MeshBatchShared>,
	meshes: Vec<Triangle>,
	framebuffers: Vec<Option<Arc<FramebufferAbstract + Send + Sync + 'static>>>,
	recreated: Arc<AtomicBool>,
}
impl MeshBatch {
	pub fn new(shared: Arc<MeshBatchShared>, target: &mut RenderTarget) -> Self {
		let recreated = Arc::<AtomicBool>::default();
		target.register(recreated.clone());

		Self {
			shared: shared,
			meshes: vec![],
			framebuffers: (0..target.image_count()).map(|_| None).collect(),
			recreated: recreated,
		}
	}

	pub fn add_triangle(&mut self, triangle: Triangle) {
		self.meshes.push(triangle);
	}
}
impl Drawable for MeshBatch {
	fn commands(
		&mut self,
		queue_family: QueueFamily,
		image_num: usize,
		image: &Arc<ImageViewAccess + Send + Sync + 'static>,
	) -> AutoCommandBuffer {
		if self.recreated.swap(false, Ordering::Relaxed) {
			for framebuffer in &mut self.framebuffers {
				*framebuffer = None;
			}
		}

		let render_pass = self.shared.subpass.render_pass();
		let framebuffer = self.framebuffers[image_num].get_or_insert_with(|| {
			Arc::new(Framebuffer::start(render_pass.clone()).add(image.clone()).unwrap().build().unwrap())
		});

		let mut command_buffer =
			AutoCommandBufferBuilder::primary_one_time_submit(self.shared.device.clone(), queue_family)
				.unwrap()
				.begin_render_pass(framebuffer.clone(), true, vec![[0.0, 0.0, 1.0, 1.0].into()])
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
								[framebuffer.width() as f32, framebuffer.height() as f32],
							)
						)
						.unwrap()
				};
		}

		command_buffer.end_render_pass().unwrap().build().unwrap()
	}
}

pub struct MeshBatchShared {
	device: Arc<Device>,
	subpass: Subpass<Arc<RenderPassAbstract + Send + Sync>>,
	pipeline: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
}
impl MeshBatchShared {
	pub fn new(shaders: &MeshBatchShaders, format: Format) -> Arc<Self> {
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

pub struct MeshBatchShaders {
	device: Arc<Device>,
	vertex_shader: vs::Shader,
	fragment_shader: fs::Shader,
}
impl MeshBatchShaders {
	pub fn new(device: Arc<Device>) -> Self {
		Self {
			device: device.clone(),
			vertex_shader: vs::Shader::load(device.clone()).expect("failed to load shader module"),
			fragment_shader: fs::Shader::load(device).expect("failed to load shader module"),
		}
	}
}

pub struct Triangle {
	buffer: Arc<ImmutableBuffer<[TriangleVertex]>>,
}
impl Triangle {
	pub fn new(
		queue: Arc<Queue>,
	) -> Result<(Self, CommandBufferExecFuture<NowFuture, AutoCommandBuffer>), DeviceMemoryAllocError> {
		ImmutableBuffer::from_iter(
			[
				TriangleVertex { position: [-0.5, -0.25] },
				TriangleVertex { position: [0.0, 0.5] },
				TriangleVertex { position: [0.25, -0.1] },
			].iter().cloned(),
			BufferUsage::vertex_buffer(),
			queue,
		).map(|(buffer, future)| (Self { buffer: buffer }, future))
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
