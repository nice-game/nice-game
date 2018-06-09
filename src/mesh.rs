use { ImageFramebuffer, ObjectId, RenderTarget, window::Window };
use cgmath::{ vec4, Quaternion, Vector3, Vector4 };
use std::{ f32::consts::PI, sync::Arc };
use vulkano::{
	OomError,
	buffer::{ BufferUsage, CpuBufferPool, ImmutableBuffer, cpu_pool::CpuBufferPoolSubbuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	descriptor::{ DescriptorSet, descriptor_set::{ FixedSizeDescriptorSetsPool, PersistentDescriptorSet } },
	device::Device,
	format::Format,
	framebuffer::{ Framebuffer, FramebufferAbstract, FramebufferCreationError, RenderPassAbstract, Subpass },
	image::{ AttachmentImage, ImageCreationError, ImageViewAccess },
	instance::QueueFamily,
	memory::{ DeviceMemoryAllocError, pool::StdMemoryPool },
	pipeline::{ GraphicsPipeline, GraphicsPipelineAbstract, viewport::Viewport },
	sync::GpuFuture,
};

const NORMAL_FORMAT: Format = Format::R32G32B32A32Sfloat;
const DEPTH_FORMAT: Format = Format::D16Unorm;

pub struct MeshBatch {
	shared: Arc<MeshBatchShared>,
	meshes: Vec<Mesh>,
	framebuffers: Vec<ImageFramebuffer>,
	target_id: ObjectId,
	image_color: Arc<AttachmentImage>,
	image_normal: Arc<AttachmentImage>,
	image_depth: Arc<AttachmentImage>,
	desc_target: Arc<DescriptorSet + Send + Sync + 'static>,
	camera_desc_pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
	mesh_desc_pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
}
impl MeshBatch {
	pub fn new(
		window: &Window,
		target: &RenderTarget,
		shared: Arc<MeshBatchShared>
	) -> Result<Self, DeviceMemoryAllocError> {
		let dimensions = target.images()[0].dimensions().width_height();
		let image_color = Self::make_transient_input_attachment(window.device().clone(), dimensions, window.format())?;
		let image_normal = Self::make_transient_input_attachment(window.device().clone(), dimensions, NORMAL_FORMAT)?;
		let image_depth = Self::make_transient_input_attachment(window.device().clone(), dimensions, DEPTH_FORMAT)?;

		let framebuffers =
			target.images().iter()
				.map(|image| {
					Framebuffer::start(shared.subpass_target.render_pass().clone())
						.add(image_color.clone())
						.and_then(|fb| fb.add(image_normal.clone()))
						.and_then(|fb| fb.add(image_depth.clone()))
						.and_then(|fb| fb.add(image.clone()))
						.and_then(|fb| fb.build())
						.map(|fb| ImageFramebuffer::new(Arc::downgrade(&image), Arc::new(fb)))
						.map_err(|err| match err {
							FramebufferCreationError::OomError(err) => err,
							err => unreachable!("{:?}", err),
						})
				})
				.collect::<Result<Vec<_>, _>>()?;

		let desc_target =
			Arc::new(
				PersistentDescriptorSet::start(shared.pipeline_target.clone(), 0)
					.add_image(image_color.clone())
					.unwrap()
					.add_image(image_normal.clone())
					.unwrap()
					.build()
					.unwrap()
			);

		let camera_desc_pool = FixedSizeDescriptorSetsPool::new(shared.pipeline_gbuffers.clone(), 0);
		let mesh_desc_pool = FixedSizeDescriptorSetsPool::new(shared.pipeline_gbuffers.clone(), 1);

		Ok(Self {
			shared: shared,
			meshes: vec![],
			framebuffers: framebuffers,
			target_id: target.id_root().make_id(),
			image_color: image_color,
			image_normal: image_normal,
			image_depth: image_depth,
			desc_target: desc_target,
			camera_desc_pool: camera_desc_pool,
			mesh_desc_pool: mesh_desc_pool,
		})
	}

	pub fn add_mesh(&mut self, mesh: Mesh) {
		self.meshes.push(mesh);
	}

	pub fn commands(
		&mut self,
		window: &Window,
		target: &RenderTarget,
		image_num: usize,
		camera: &Camera,
	) -> Result<AutoCommandBuffer, DeviceMemoryAllocError> {
		assert!(self.target_id.is_child_of(target.id_root()));

		let dimensions = target.images()[image_num].dimensions().width_height();

		let framebuffer = self.framebuffers[image_num].image
			.upgrade()
			.iter()
			.filter(|old_image| Arc::ptr_eq(&target.images()[image_num], &old_image))
			.next()
			.map(|_| self.framebuffers[image_num].framebuffer.clone());
		let framebuffer =
			if let Some(framebuffer) = framebuffer.as_ref() {
				framebuffer.clone()
			} else {
				self.image_color = Self::make_transient_input_attachment(window.device().clone(), dimensions, window.format())?;
				self.image_normal = Self::make_transient_input_attachment(window.device().clone(), dimensions, NORMAL_FORMAT)?;
				self.image_depth = Self::make_transient_input_attachment(window.device().clone(), dimensions, DEPTH_FORMAT)?;
				self.desc_target =
					Arc::new(
						PersistentDescriptorSet::start(self.shared.pipeline_target.clone(), 0)
							.add_image(self.image_color.clone())
							.unwrap()
							.add_image(self.image_normal.clone())
							.unwrap()
							.build()
							.unwrap()
					);

				let framebuffer = Framebuffer::start(self.shared.subpass_gbuffers.render_pass().clone())
					.add(self.image_color.clone())
					.and_then(|fb| fb.add(self.image_normal.clone()))
					.and_then(|fb| fb.add(self.image_depth.clone()))
					.and_then(|fb| fb.add(target.images()[image_num].clone()))
					.and_then(|fb| fb.build())
					.map(|fb| Arc::new(fb))
					.map_err(|err| {
						match err { FramebufferCreationError::OomError(err) => err, err => unreachable!("{:?}", err) }
					})?;
				self.framebuffers[image_num] =
					ImageFramebuffer::new(Arc::downgrade(&target.images()[image_num]), framebuffer.clone());

				framebuffer
			};

		let camera_desc =
			Arc::new(
				self.camera_desc_pool.next()
					.add_buffer(camera.position_buffer.clone())
					.unwrap()
					.add_buffer(camera.rotation_buffer.clone())
					.unwrap()
					.add_buffer(camera.projection_buffer.clone())
					.unwrap()
					.build()
					.unwrap()
			);

		let dimensions = [framebuffer.width() as f32, framebuffer.height() as f32];

		let mut command_buffer =
			AutoCommandBufferBuilder::primary_one_time_submit(self.shared.shaders.device.clone(), window.queue().family())?
				.begin_render_pass(
					framebuffer.clone(),
					true,
					vec![[0.0, 0.0, 0.0, 1.0].into(), [0.0; 4].into(), 1.0.into(), [0.0, 0.0, 0.0, 1.0].into()]
				)
				.unwrap();

		for mesh in &mut self.meshes {
			command_buffer =
				unsafe {
					command_buffer
						.execute_commands(
							mesh.make_commands(
								&self.shared,
								camera_desc.clone(),
								&mut self.mesh_desc_pool,
								window.queue().family(),
								dimensions
							)?
						)
						.unwrap()
				};
		}

		Ok(
			command_buffer.next_subpass(false)
				.unwrap()
				.draw(
					self.shared.pipeline_target.clone(),
					DynamicState {
						line_width: None,
						viewports:
							Some(vec![Viewport { origin: [0.0, 0.0], dimensions: dimensions, depth_range: 0.0..1.0 }]),
						scissors: None,
					},
					vec![self.shared.shaders.target_vertices.clone()],
					self.desc_target.clone(),
					()
				)
				.unwrap()
				.end_render_pass()
				.unwrap()
				.build()
				.map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?
		)
	}

	fn make_transient_input_attachment(
		device: Arc<Device>,
		dimensions: [u32; 2],
		format: Format,
	) -> Result<Arc<AttachmentImage>, DeviceMemoryAllocError> {
		AttachmentImage::transient_input_attachment(device, dimensions, format)
			.map_err(|err| match err { ImageCreationError::AllocError(err) => err, err => unreachable!(err) })
	}
}

pub struct MeshBatchShared {
	shaders: Arc<MeshBatchShaders>,
	subpass_gbuffers: Subpass<Arc<RenderPassAbstract + Send + Sync>>,
	subpass_target: Subpass<Arc<RenderPassAbstract + Send + Sync>>,
	pipeline_gbuffers: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
	pipeline_target: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
}
impl MeshBatchShared {
	pub fn new(shaders: Arc<MeshBatchShaders>, format: Format) -> Arc<Self> {
		let render_pass: Arc<RenderPassAbstract + Send + Sync> =
			Arc::new(
				ordered_passes_renderpass!(
					shaders.device.clone(),
					attachments: {
						color: { load: Clear, store: Store, format: format, samples: 1, },
						normal: { load: Clear, store: Store, format: NORMAL_FORMAT, samples: 1, },
						depth: { load: Clear, store: Store, format: DEPTH_FORMAT, samples: 1, },
						out: { load: Clear, store: Store, format: format, samples: 1, }
					},
					passes: [
						{ color: [color, normal], depth_stencil: {depth}, input: [] },
						{ color: [out], depth_stencil: {depth}, input: [color, normal] }
					]
				)
				.unwrap()
			);

		let subpass_gbuffers = Subpass::from(render_pass.clone(), 0).unwrap();
		let subpass_target = Subpass::from(render_pass, 1).unwrap();

		let pipeline_gbuffers =
			Arc::new(
				GraphicsPipeline::start()
					.vertex_input_single_buffer::<MeshVertex>()
					.vertex_shader(shaders.shader_gbuffers_vertex.main_entry_point(), ())
					.triangle_list()
					.viewports_dynamic_scissors_irrelevant(1)
					.fragment_shader(shaders.shader_gbuffers_fragment.main_entry_point(), ())
					.render_pass(subpass_gbuffers.clone())
					.build(shaders.device.clone())
					.expect("failed to create pipeline")
			);

		let pipeline_target =
			Arc::new(
				GraphicsPipeline::start()
					.vertex_input_single_buffer::<TargetVertex>()
					.vertex_shader(shaders.shader_target_vertex.main_entry_point(), ())
					.triangle_list()
					.viewports_dynamic_scissors_irrelevant(1)
					.fragment_shader(shaders.shader_target_fragment.main_entry_point(), ())
					.render_pass(subpass_target.clone())
					.build(shaders.device.clone())
					.expect("failed to create pipeline")
			);

		Arc::new(Self {
			shaders: shaders,
			subpass_gbuffers: subpass_gbuffers,
			subpass_target: subpass_target,
			pipeline_gbuffers: pipeline_gbuffers,
			pipeline_target: pipeline_target,
		})
	}
}

pub struct MeshBatchShaders {
	device: Arc<Device>,
	target_vertices: Arc<ImmutableBuffer<[TargetVertex; 6]>>,
	shader_gbuffers_vertex: vs_gbuffers::Shader,
	shader_gbuffers_fragment: fs_gbuffers::Shader,
	shader_target_vertex: vs_target::Shader,
	shader_target_fragment: fs_target::Shader,
}
impl MeshBatchShaders {
	pub fn new(window: &Window) -> Result<(Arc<Self>, impl GpuFuture), MeshBatchShadersError> {
		let (target_vertices, future) =
			ImmutableBuffer::from_data(
				[
					TargetVertex { position: [0.0, 0.0] },
					TargetVertex { position: [1.0, 0.0] },
					TargetVertex { position: [0.0, 1.0] },
					TargetVertex { position: [0.0, 1.0] },
					TargetVertex { position: [1.0, 0.0] },
					TargetVertex { position: [1.0, 1.0] },
				],
				BufferUsage::vertex_buffer(),
				window.queue().clone(),
			)?;

		Ok((
			Arc::new(Self {
				device: window.device().clone(),
				target_vertices: target_vertices,
				shader_gbuffers_vertex: vs_gbuffers::Shader::load(window.device().clone())?,
				shader_gbuffers_fragment: fs_gbuffers::Shader::load(window.device().clone())?,
				shader_target_vertex: vs_target::Shader::load(window.device().clone())?,
				shader_target_fragment: fs_target::Shader::load(window.device().clone())?,
			}),
			future
		))
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

pub struct Mesh {
	position: Arc<ImmutableBuffer<[f32; 3]>>,
	vertices: Arc<ImmutableBuffer<[MeshVertex]>>,
}
impl Mesh {
	pub fn new<D>(
		window: &Window,
		vertices: D,
		position: [f32; 3],
	) -> Result<(Self, impl GpuFuture), DeviceMemoryAllocError>
	where
		D: ExactSizeIterator<Item = MeshVertex>,
	{
		let (vertices, vertices_future) =
			ImmutableBuffer::from_iter(vertices, BufferUsage::vertex_buffer(), window.queue().clone())?;

		let (position, position_future) =
			ImmutableBuffer::from_data(position, BufferUsage::uniform_buffer(), window.queue().clone())?;

		Ok((Self { position: position, vertices: vertices }, vertices_future.join(position_future)))
	}

	fn make_commands(
		&mut self,
		shared: &MeshBatchShared,
		camera_desc: impl DescriptorSet + Send + Sync + 'static,
		mesh_desc_pool: &mut FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
		queue_family: QueueFamily,
		dimensions: [f32; 2],
	) -> Result<AutoCommandBuffer, OomError> {
		Ok(
			AutoCommandBufferBuilder::secondary_graphics_one_time_submit(shared.shaders.device.clone(), queue_family, shared.subpass_gbuffers.clone())?
				.draw(
					shared.pipeline_gbuffers.clone(),
					DynamicState {
						line_width: None,
						viewports:
							Some(vec![Viewport { origin: [0.0, 0.0], dimensions: dimensions, depth_range: 0.0..1.0 }]),
						scissors: None,
					},
					vec![self.vertices.clone()],
					(camera_desc, mesh_desc_pool.next().add_buffer(self.position.clone()).unwrap().build().unwrap()),
					()
				)
				.unwrap()
				.build()
				.map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?
		)
	}
}

pub struct Camera {
	position_pool: CpuBufferPool<Vector3<f32>>,
	rotation_pool: CpuBufferPool<Quaternion<f32>>,
	projection_pool: CpuBufferPool<Vector4<f32>>,
	position_buffer: CpuBufferPoolSubbuffer<Vector3<f32>, Arc<StdMemoryPool>>,
	rotation_buffer: CpuBufferPoolSubbuffer<Quaternion<f32>, Arc<StdMemoryPool>>,
	projection_buffer: CpuBufferPoolSubbuffer<Vector4<f32>, Arc<StdMemoryPool>>,
}
impl Camera {
	pub fn new(
		window: &Window,
		position: Vector3<f32>,
		rotation: Quaternion<f32>,
		aspect: f32,
		fovx: f32,
		znear: f32,
		zfar: f32
	) -> Result<Self, DeviceMemoryAllocError> {
		let position_pool = CpuBufferPool::uniform_buffer(window.device().clone());
		let rotation_pool = CpuBufferPool::uniform_buffer(window.device().clone());
		let projection_pool = CpuBufferPool::uniform_buffer(window.device().clone());

		let position_buffer = position_pool.next(position)?;
		let rotation_buffer = rotation_pool.next(rotation)?;
		let projection_buffer = projection_pool.next(Self::projection(aspect, fovx, znear, zfar))?;

		Ok(Self {
			position_pool: position_pool,
			rotation_pool: rotation_pool,
			projection_pool: projection_pool,
			position_buffer: position_buffer,
			rotation_buffer: rotation_buffer,
			projection_buffer: projection_buffer,
		})
	}

	pub fn set_position(
		&mut self,
		aspect: f32,
		fovx: f32,
		znear: f32,
		zfar: f32
	) -> Result<(), DeviceMemoryAllocError> {
		self.projection_buffer = self.projection_pool.next(Self::projection(aspect, fovx, znear, zfar))?;
		Ok(())
	}

	pub fn set_projection(&mut self, position: Vector3<f32>) -> Result<(), DeviceMemoryAllocError> {
		self.position_buffer = self.position_pool.next(position)?;
		Ok(())
	}

	pub fn set_rotation(&mut self, rotation: Quaternion<f32>) -> Result<(), DeviceMemoryAllocError> {
		self.rotation_buffer = self.rotation_pool.next(rotation)?;
		Ok(())
	}

	fn projection(aspect: f32, fovx: f32, znear: f32, zfar: f32) -> Vector4<f32> {
		let f = 1.0 / (fovx * (PI / 360.0)).tan();
		vec4(f / aspect, f, (zfar + znear) / (znear - zfar), 2.0 * zfar * znear / (znear - zfar))
	}
}

#[derive(Debug, Clone)]
pub struct MeshVertex {
	pub position: [f32; 3],
	pub normal: [f32; 3],
}
impl_vertex!(MeshVertex, position, normal);

#[derive(Debug, Clone)]
struct TargetVertex { position: [f32; 2] }
impl_vertex!(TargetVertex, position);

mod vs_gbuffers {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "vertex"]
	#[src = "#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 0) out vec3 out_normal;

layout(set = 0, binding = 0) uniform CameraPos { vec3 camera_pos; };
layout(set = 0, binding = 1) uniform CameraRot { vec4 camera_rot; };
layout(set = 0, binding = 2) uniform CameraProj { vec4 camera_proj; };

layout(set = 1, binding = 0) uniform MeshDynamic { vec3 mesh_pos; };

vec4 quat_inv(vec4 quat) {
	return vec4(-quat.xyz, quat.w) / dot(quat, quat);
}

vec3 quat_mul(vec4 quat, vec3 vec) {
	return cross(quat.xyz, cross(quat.xyz, vec) + vec * quat.w) * 2.0 + vec;
}

vec4 perspective(vec3 pos, vec4 proj) {
	return vec4(pos.xy * proj.xy, pos.z * proj.z + proj.w, -pos.z);
}

void main() {
	out_normal = quat_mul(quat_inv(camera_rot), normal);
	gl_Position = perspective(quat_mul(quat_inv(camera_rot), position + mesh_pos - camera_pos), camera_proj);
}"]
	struct Dummy;
}

mod fs_gbuffers {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "fragment"]
	#[src = "#version 450
layout(location = 0) in vec3 normal;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_normal;

void main() {
	out_color = vec4(1);
	out_normal = vec4(normal, 1);
}"]
	struct Dummy;
}

mod vs_target {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "vertex"]
	#[src = "#version 450
layout(location = 0) in vec2 position;

void main() {
	gl_Position = vec4(position * 2 - 1, 0.0, 1.0);
}
"]
	struct Dummy;
}

mod fs_target {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "fragment"]
	#[src = "#version 450
layout(location = 0) out vec4 out_color;

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput normal;

void main() {
	out_color = subpassLoad(color);
}
"]
	struct Dummy;
}
