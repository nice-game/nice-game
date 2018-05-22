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

		let render_pass = &self.shared.render_pass;
		let framebuffer = self.framebuffers[image_num].get_or_insert_with(|| {
			Arc::new(Framebuffer::start(render_pass.clone()).add(image.clone()).unwrap().build().unwrap())
		});

		let mut command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(self.shared.device.clone(), queue_family)
			.unwrap()
			.begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()])
			.unwrap();

		for mesh in &self.meshes {
			command_buffer = command_buffer
				.draw(
					self.shared.pipeline.clone(),
					DynamicState {
						line_width: None,
						viewports: Some(vec![
							Viewport {
								origin: [0.0, 0.0],
								dimensions: [framebuffer.width() as f32, framebuffer.height() as f32],
								depth_range: 0.0..1.0,
							}
						]),
						scissors: None,
					},
					vec![mesh.buffer.clone()], (), ()
				)
				.unwrap();
		}

		command_buffer.end_render_pass().unwrap().build().unwrap()
	}
}

pub struct MeshBatchShared {
	device: Arc<Device>,
	render_pass: Arc<RenderPassAbstract + Send + Sync>,
	pipeline: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
}
impl MeshBatchShared {
	pub fn new(device: Arc<Device>, format: Format) -> Arc<Self> {
		let render_pass = Arc::new(
			single_pass_renderpass!(
				device.clone(),
				attachments: { color: { load: Clear, store: Store, format: format, samples: 1 } },
				pass: { color: [color], depth_stencil: {} }
			).expect("failed to create render pass")
		);

		let pipeline = Arc::new(
			GraphicsPipeline::start()
				.vertex_input_single_buffer::<TriangleVertex>()
				.vertex_shader(
					vs::Shader::load(device.clone()).expect("failed to load shader module").main_entry_point(),
					()
				)
				.triangle_list()
				.viewports_dynamic_scissors_irrelevant(1)
				.fragment_shader(
					fs::Shader::load(device.clone()).expect("failed to load shader module"),
					()
				)
				.render_pass(Subpass::from(render_pass.clone(), 0).expect("failed to create subpass"))
				.build(device.clone())
				.expect("failed to create pipeline")
		);

		Arc::new(Self { device: device, render_pass: render_pass, pipeline: pipeline })
	}
}

pub struct Triangle {
	buffer: Arc<ImmutableBuffer<[TriangleVertex]>>,
}
impl Triangle {
	pub fn new(
		queue: Arc<Queue>
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
