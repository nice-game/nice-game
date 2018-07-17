mod mesh;
mod shaders;
mod render_pass;

pub use self::mesh::Mesh;
pub use self::shaders::{ MeshShaders, MeshShadersError };
pub use self::render_pass::MeshRenderPass;
use { ImageFramebuffer, ObjectId, RenderTarget, window::Window };
use camera::Camera;
use cgmath::vec4;
use std::sync::Arc;
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	descriptor::{ DescriptorSet, descriptor_set::{ FixedSizeDescriptorSetsPool, PersistentDescriptorSet } },
	device::Device,
	format::Format,
	framebuffer::{ Framebuffer, FramebufferAbstract, FramebufferCreationError, RenderPassAbstract },
	image::{ AttachmentImage, ImageCreationError, ImageViewAccess },
	memory::{ DeviceMemoryAllocError },
	pipeline::{ GraphicsPipelineAbstract, viewport::Viewport },
	sync::GpuFuture,
};

const ALBEDO_FORMAT: Format = Format::A2B10G10R10UnormPack32;
const NORMAL_FORMAT: Format = Format::R32G32B32A32Sfloat;
const DEPTH_FORMAT: Format = Format::D16Unorm;

pub struct MeshBatch {
	render_pass: Arc<MeshRenderPass>,
	meshes: Vec<Mesh>,
	target_id: ObjectId,
	gbuffers: GBuffers,
	camera_desc_pool_gbuffers: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
	camera_desc_pool_target: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
	mesh_desc_pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
}
impl MeshBatch {
	pub fn new(
		target: &RenderTarget,
		render_pass: Arc<MeshRenderPass>
	) -> Result<(Self, impl GpuFuture), DeviceMemoryAllocError> {
		let camera_desc_pool_gbuffers = FixedSizeDescriptorSetsPool::new(render_pass.pipeline_gbuffers.clone(), 0);
		let camera_desc_pool_target = FixedSizeDescriptorSetsPool::new(render_pass.pipeline_target.clone(), 1);
		let mesh_desc_pool = FixedSizeDescriptorSetsPool::new(render_pass.pipeline_gbuffers.clone(), 1);
		let (gbuffers, future) = Self::make_gbuffers(target, &render_pass)?;

		Ok((
			Self {
				render_pass: render_pass,
				meshes: vec![],
				target_id: target.id_root().make_id(),
				gbuffers: gbuffers,
				camera_desc_pool_gbuffers: camera_desc_pool_gbuffers,
				camera_desc_pool_target: camera_desc_pool_target,
				mesh_desc_pool: mesh_desc_pool,
			},
			future
		))
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
	) -> Result<(AutoCommandBuffer, Option<impl GpuFuture>), DeviceMemoryAllocError> {
		assert!(self.target_id.is_child_of(target.id_root()));

		let image = &target.images()[image_num];
		let gbuffers_future =
			if image.dimensions() != self.gbuffers.color.dimensions() {
				let (gbuffers, gbuffers_future) = Self::make_gbuffers(target, &self.render_pass)?;
				self.gbuffers = gbuffers;
				Some(gbuffers_future)
			} else {
				None
			};

		let camera_desc_gbuffers =
			Arc::new(
				self.camera_desc_pool_gbuffers.next()
					.add_buffer(camera.position_buffer.clone())
					.unwrap()
					.add_buffer(camera.rotation_buffer.clone())
					.unwrap()
					.add_buffer(camera.projection_buffer.clone())
					.unwrap()
					.build()
					.unwrap()
			);

		let dimensions = [image.dimensions().width() as f32, image.dimensions().height() as f32];

		let mut command_buffer =
			AutoCommandBufferBuilder
				::primary_one_time_submit(
					self.render_pass.shaders.target_vertices.device().clone(),
					window.queue().family()
				)?
				.begin_render_pass(
					Arc::new(
						Framebuffer::start(self.render_pass.render_pass().clone())
							.add(self.gbuffers.color.clone())
							.and_then(|fb| fb.add(self.gbuffers.normal.clone()))
							.and_then(|fb| fb.add(self.gbuffers.depth.clone()))
							.and_then(|fb| fb.add(image.clone()))
							.and_then(|fb| fb.build())
							.map_err(|err| match err {
								FramebufferCreationError::OomError(err) => err,
								err => unreachable!("{:?}", err),
							})?
					),
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
								&self.render_pass,
								camera_desc_gbuffers.clone(),
								&mut self.mesh_desc_pool,
								window.queue().family(),
								dimensions
							)?
						)
						.unwrap()
				};
		}

		Ok((
			command_buffer.next_subpass(false)
				.unwrap()
				.draw(
					self.render_pass.pipeline_target.clone(),
					DynamicState {
						line_width: None,
						viewports:
							Some(vec![Viewport { origin: [0.0, 0.0], dimensions: dimensions, depth_range: 0.0..1.0 }]),
						scissors: None,
					},
					vec![self.render_pass.shaders.target_vertices.clone()],
					(
						self.gbuffers.desc_target.clone(),
						self.camera_desc_pool_target.next()
							.add_buffer(camera.position_buffer.clone())
							.unwrap()
							.add_buffer(camera.rotation_buffer.clone())
							.unwrap()
							.add_buffer(camera.projection_buffer.clone())
							.unwrap()
							.build()
							.unwrap(),
					),
					()
				)
				.unwrap()
				.end_render_pass()
				.unwrap()
				.build()
				.map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?,
			gbuffers_future
		))
	}

	fn make_transient_input_attachment(
		device: Arc<Device>,
		dimensions: [u32; 2],
		format: Format,
	) -> Result<Arc<AttachmentImage>, DeviceMemoryAllocError> {
		AttachmentImage::transient_input_attachment(device, dimensions, format)
			.map_err(|err| match err { ImageCreationError::AllocError(err) => err, err => unreachable!(err) })
	}

	fn make_gbuffers(
		target: &RenderTarget,
		shared: &MeshRenderPass,
	) -> Result<(GBuffers, impl GpuFuture), DeviceMemoryAllocError> {
		let dimensions = target.images()[0].dimensions().width_height();
		let color =
			Self::make_transient_input_attachment(
				shared.shaders.target_vertices.device().clone(),
				dimensions,
				ALBEDO_FORMAT
			)?;
		let normal =
			Self::make_transient_input_attachment(
				shared.shaders.target_vertices.device().clone(),
				dimensions,
				NORMAL_FORMAT
			)?;
		let depth =
			Self::make_transient_input_attachment(
				shared.shaders.target_vertices.device().clone(),
				dimensions,
				DEPTH_FORMAT
			)?;

		let dimensions = [dimensions[0] as f32, dimensions[1] as f32];
		let (target_size, target_size_future) =
			ImmutableBuffer::from_data(
				vec4(
					dimensions[0],
					dimensions[1],
					2.0 / dimensions[0],
					2.0 / dimensions[1]
				),
				BufferUsage::uniform_buffer(),
				shared.shaders.queue.clone()
			)?;

		let desc_target =
			Arc::new(
				PersistentDescriptorSet::start(shared.pipeline_target.clone(), 0)
					.add_buffer(target_size)
					.unwrap()
					.add_image(color.clone())
					.unwrap()
					.add_image(normal.clone())
					.unwrap()
					.add_image(depth.clone())
					.unwrap()
					.build()
					.unwrap()
			);

		Ok((GBuffers { color: color, normal: normal, depth: depth, desc_target: desc_target }, target_size_future))
	}
}

#[derive(Clone)]
struct GBuffers {
	color: Arc<AttachmentImage>,
	normal: Arc<AttachmentImage>,
	depth: Arc<AttachmentImage>,
	desc_target: Arc<DescriptorSet + Send + Sync + 'static>,
}

#[derive(Debug, Clone)]
struct TargetVertex { position: [f32; 2] }
impl_vertex!(TargetVertex, position);
