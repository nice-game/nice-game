use { ObjectId, RenderTarget, window::Window };
use std::sync::{ Arc, Mutex, Weak };
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	descriptor::descriptor_set::FixedSizeDescriptorSetsPool,
	device::Device,
	format::Format,
	framebuffer::{ Framebuffer, FramebufferAbstract, FramebufferCreationError, RenderPassAbstract, Subpass },
	image::ImageViewAccess,
	instance::QueueFamily,
	memory::DeviceMemoryAllocError,
	pipeline::{ GraphicsPipeline, GraphicsPipelineAbstract, viewport::Viewport },
	sampler::SamplerCreationError,
};

pub struct MeshBatch {
	shared: Arc<MeshBatchShared>,
	meshes: Vec<Mesh>,
	framebuffers:
		Vec<Option<(Weak<ImageViewAccess + Send + Sync + 'static>, Arc<FramebufferAbstract + Send + Sync + 'static>)>>,
	target_id: ObjectId,
}
impl MeshBatch {
	pub fn new(target: &mut RenderTarget, shared: Arc<MeshBatchShared>) -> Result<Self, DeviceMemoryAllocError> {
		Ok(Self {
			shared: shared,
			meshes: vec![],
			framebuffers: vec![None; target.images().len()],
			target_id: target.id_root().make_id(),
		})
	}

	pub fn add_mesh(&mut self, mesh: Mesh) {
		self.meshes.push(mesh);
	}

	pub fn commands(
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
			if let Some(framebuffer) = framebuffer.as_ref() {
				framebuffer.clone()
			} else {
				let framebuffer = Framebuffer::start(self.shared.subpass.render_pass().clone())
					.add(target.images()[image_num].clone())
					.and_then(|fb| fb.build())
					.map(|fb| Arc::new(fb))
					.map_err(|err| {
						match err { FramebufferCreationError::OomError(err) => err, err => unreachable!("{:?}", err) }
					})?;
				self.framebuffers[image_num] = Some((Arc::downgrade(&target.images()[image_num]), framebuffer.clone()));

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
						.execute_commands(mesh.make_commands(&self.shared, target.queue().family(), dimensions)?)
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

pub struct MeshBatchShared {
	shaders: Arc<MeshBatchShaders>,
	subpass: Subpass<Arc<RenderPassAbstract + Send + Sync>>,
	pipeline: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
	mesh_desc_pool: Mutex<FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>>,
}
impl MeshBatchShared {
	pub fn new(shaders: Arc<MeshBatchShaders>, format: Format) -> Arc<Self> {
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
			)
			.expect("failed to create subpass");

		let pipeline = Arc::new(
			GraphicsPipeline::start()
				.vertex_input_single_buffer::<MeshVertex>()
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
			mesh_desc_pool: Mutex::new(FixedSizeDescriptorSetsPool::new(pipeline, 0)),
		})
	}
}

pub struct MeshBatchShaders {
	device: Arc<Device>,
	vertex_shader: vs::Shader,
	fragment_shader: fs::Shader,
}
impl MeshBatchShaders {
	pub fn new(window: &Window) -> Result<Arc<Self>, MeshBatchShadersError> {
		Ok(Arc::new(Self {
			device: window.device().clone(),
			vertex_shader: vs::Shader::load(window.device().clone())?,
			fragment_shader: fs::Shader::load(window.device().clone())?,
		}))
	}
}

#[derive(Debug)]
pub enum MeshBatchShadersError {
	DeviceMemoryAllocError(DeviceMemoryAllocError),
	OomError(OomError),
	TooManyObjects,
}
impl From<DeviceMemoryAllocError> for MeshBatchShadersError {
	fn from(val: DeviceMemoryAllocError) -> Self {
		MeshBatchShadersError::DeviceMemoryAllocError(val)
	}
}
impl From<OomError> for MeshBatchShadersError {
	fn from(val: OomError) -> Self {
		MeshBatchShadersError::OomError(val)
	}
}
impl From<SamplerCreationError> for MeshBatchShadersError {
	fn from(val: SamplerCreationError) -> Self {
		match val {
			SamplerCreationError::OomError(err) => MeshBatchShadersError::OomError(err),
			SamplerCreationError::TooManyObjects => MeshBatchShadersError::TooManyObjects,
			_ => unreachable!(),
		}
	}
}

pub struct Mesh {
	position: Arc<ImmutableBuffer<[f32; 2]>>,
	vertices: Arc<ImmutableBuffer<[MeshVertex]>>,
}
impl Mesh {
	pub fn new<D>(
		target: &mut RenderTarget,
		vertices: D,
		position: [f32; 2],
	) -> Result<Self, DeviceMemoryAllocError>
	where
		D: ExactSizeIterator<Item = MeshVertex>,
	{
		let (vertices, vertex_future) =
			ImmutableBuffer::from_iter(vertices, BufferUsage::vertex_buffer(), target.queue().clone())?;
		target.join_future(Box::new(vertex_future));

		let (position, future) =
			ImmutableBuffer::from_data(position, BufferUsage::uniform_buffer(), target.queue().clone())?;
		target.join_future(Box::new(future));

		Ok(Self { position: position, vertices: vertices })
	}

	fn make_commands(
		&mut self,
		shared: &MeshBatchShared,
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
					vec![self.vertices.clone()],
					shared.mesh_desc_pool.lock().unwrap()
						.next()
						.add_buffer(self.position.clone())
						.unwrap()
						.build()
						.unwrap(),
					()
				)
				.unwrap()
				.build()
				.map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?
		)
	}
}

#[derive(Debug, Clone)]
pub struct MeshVertex { pub position: [f32; 2] }
impl_vertex!(MeshVertex, position);

mod vs {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "vertex"]
	#[src = "#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;

layout(set = 0, binding = 0) uniform MeshDynamic {
	vec2 pos;
} mesh_dynamic;

void main() {
	tex_coords = position;
	gl_Position = vec4(2 * position - 1, 0.0, 1.0);
}"]
	struct Dummy;
}

mod fs {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "fragment"]
	#[src = "#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

void main() {
	f_color = vec4(tex_coords, 1, 1);
}"]
	struct Dummy;
}
