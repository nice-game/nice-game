mod mesh;
mod shaders;
mod render_pass;

pub use self::mesh::Mesh;
pub use self::shaders::{ MeshShaders, MeshShadersError };
pub use self::render_pass::MeshRenderPass;
use { ObjectId, RenderTarget, window::Window };
use camera::Camera;
use cgmath::{ vec4, Vector4 };
use std::sync::Arc;
use vulkano::{
	buffer::{ BufferUsage, ImmutableBuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	descriptor::{ DescriptorSet, descriptor_set::{ FixedSizeDescriptorSetsPool, PersistentDescriptorSet } },
	device::Device,
	format::{ ClearValue, Format },
	framebuffer::{ Framebuffer, FramebufferCreationError },
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
	camera_desc_pool_history: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
	mesh_desc_pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
}
impl MeshBatch {
	pub fn new(
		target: &RenderTarget,
		render_pass: Arc<MeshRenderPass>
	) -> Result<(Self, impl GpuFuture), DeviceMemoryAllocError> {
		let camera_desc_pool_gbuffers = FixedSizeDescriptorSetsPool::new(render_pass.pipeline_gbuffers.clone(), 0);
		let camera_desc_pool_history = FixedSizeDescriptorSetsPool::new(render_pass.pipeline_history.clone(), 1);
		let mesh_desc_pool = FixedSizeDescriptorSetsPool::new(render_pass.pipeline_gbuffers.clone(), 1);
		let (gbuffers, future) = Self::make_gbuffers(target, &render_pass)?;

		Ok((
			Self {
				render_pass: render_pass,
				meshes: vec![],
				target_id: target.id_root().make_id(),
				gbuffers: gbuffers,
				camera_desc_pool_gbuffers: camera_desc_pool_gbuffers,
				camera_desc_pool_history: camera_desc_pool_history,
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

		let history_index = self.gbuffers.history_index as usize;
		self.gbuffers.history_index = !self.gbuffers.history_index;

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
							.and_then(|fb| fb.add(self.gbuffers.history[history_index].clone()))
							.and_then(|fb| fb.add(image.clone()))
							.and_then(|fb| fb.build())
							.map_err(|err| match err {
								FramebufferCreationError::OomError(err) => err,
								err => unreachable!("{:?}", err),
							})?
					),
					true,
					vec![[0.0, 0.0, 0.0, 1.0].into(), [0.0; 4].into(), 1.0.into(), ClearValue::None, ClearValue::None]
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

		let dynamic_state =
			DynamicState {
				line_width: None,
				viewports: Some(vec![Viewport { origin: [0.0, 0.0], dimensions: dimensions, depth_range: 0.0..1.0 }]),
				scissors: None,
			};

		let history_desc =
			if self.gbuffers.history_initialized {
				self.gbuffers.history_descs[history_index].clone()
			} else {
				Arc::new(
					PersistentDescriptorSet::start(self.render_pass.pipeline_history.clone(), 0)
						.add_buffer(self.gbuffers.size.clone())
						.unwrap()
						.add_sampled_image(self.render_pass.shaders.black_pixel.clone(), self.render_pass.shaders.sampler.clone())
						.unwrap()
						.add_image(self.gbuffers.color.clone())
						.unwrap()
						.add_image(self.gbuffers.normal.clone())
						.unwrap()
						.add_image(self.gbuffers.depth.clone())
						.unwrap()
						.build()
						.unwrap()
				)
			};
		let command_buffer = command_buffer.next_subpass(false)
			.unwrap()
			.draw(
				self.render_pass.pipeline_history.clone(),
				&dynamic_state,
				vec![self.render_pass.shaders.target_vertices.clone()],
				(
					history_desc,
					self.camera_desc_pool_history.next()
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
			.next_subpass(false)
			.unwrap()
			.draw(
				self.render_pass.pipeline_target.clone(),
				&dynamic_state,
				vec![self.render_pass.shaders.target_vertices.clone()],
				self.gbuffers.target_descs[history_index].clone(),
				()
			)
			.unwrap()
			.end_render_pass()
			.unwrap()
			.build()
			.map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?;

		Ok((command_buffer, gbuffers_future))
	}

	fn make_sampled_input_attachment(
		device: Arc<Device>,
		dimensions: [u32; 2],
		format: Format,
	) -> Result<Arc<AttachmentImage>, DeviceMemoryAllocError> {
		AttachmentImage::sampled_input_attachment(device, dimensions, format)
			.map_err(|err| match err { ImageCreationError::AllocError(err) => err, err => unreachable!(err) })
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
		let history =
			[
				Self::make_sampled_input_attachment(
					shared.shaders.target_vertices.device().clone(),
					dimensions,
					target.format()
				)?,
				Self::make_sampled_input_attachment(
					shared.shaders.target_vertices.device().clone(),
					dimensions,
					target.format()
				)?
			];

		let dimensions = [dimensions[0] as f32, dimensions[1] as f32];
		let (size, size_future) =
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

		let history_descs =
			[
				Arc::new(
					PersistentDescriptorSet::start(shared.pipeline_history.clone(), 0)
						.add_buffer(size.clone())
						.unwrap()
						.add_sampled_image(history[1].clone(), shared.shaders.sampler.clone())
						.unwrap()
						.add_image(color.clone())
						.unwrap()
						.add_image(normal.clone())
						.unwrap()
						.add_image(depth.clone())
						.unwrap()
						.build()
						.unwrap()
				) as _,
				Arc::new(
					PersistentDescriptorSet::start(shared.pipeline_history.clone(), 0)
						.add_buffer(size.clone())
						.unwrap()
						.add_sampled_image(history[0].clone(), shared.shaders.sampler.clone())
						.unwrap()
						.add_image(color.clone())
						.unwrap()
						.add_image(normal.clone())
						.unwrap()
						.add_image(depth.clone())
						.unwrap()
						.build()
						.unwrap()
				) as _
			];

		let target_descs =
			[
				Arc::new(
					PersistentDescriptorSet::start(shared.pipeline_target.clone(), 0)
						.add_image(history[0].clone())
						.unwrap()
						.build()
						.unwrap()
				) as _,
				Arc::new(
					PersistentDescriptorSet::start(shared.pipeline_target.clone(), 0)
						.add_image(history[1].clone())
						.unwrap()
						.build()
						.unwrap()
				) as _
			];

		Ok((
			GBuffers {
				size: size,
				color: color,
				normal: normal,
				depth: depth,
				history_descs: history_descs,
				target_descs: target_descs,
				history: history,
				history_index: false,
				history_initialized: false,
			},
			size_future
		))
	}
}

#[derive(Clone)]
struct GBuffers {
	size: Arc<ImmutableBuffer<Vector4<f32>>>,
	color: Arc<AttachmentImage>,
	normal: Arc<AttachmentImage>,
	depth: Arc<AttachmentImage>,
	history_descs: [Arc<DescriptorSet + Send + Sync + 'static>; 2],
	target_descs: [Arc<DescriptorSet + Send + Sync + 'static>; 2],
	history: [Arc<AttachmentImage>; 2],
	history_index: bool,
	history_initialized: bool,
}

#[derive(Debug, Clone)]
struct TargetVertex { position: [f32; 2] }
impl_vertex!(TargetVertex, position);
