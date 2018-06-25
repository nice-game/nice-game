mod mesh;
mod shaders;
mod shared;

pub use self::mesh::Mesh;
pub use self::shaders::{ MeshBatchShaders, MeshBatchShadersError };
pub use self::shared::MeshBatchShared;
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

const NORMAL_FORMAT: Format = Format::R32G32B32A32Sfloat;
const DEPTH_FORMAT: Format = Format::D16Unorm;

pub struct MeshBatch {
	shared: Arc<MeshBatchShared>,
	meshes: Vec<Mesh>,
	framebuffers: Vec<ImageFramebuffer>,
	target_id: ObjectId,
	desc_target: Arc<DescriptorSet + Send + Sync + 'static>,
	camera_desc_pool_gbuffers: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
	camera_desc_pool_target: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
	mesh_desc_pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
}
impl MeshBatch {
	pub fn new(
		target: &RenderTarget,
		shared: Arc<MeshBatchShared>
	) -> Result<(Self, impl GpuFuture), DeviceMemoryAllocError> {
		let camera_desc_pool_gbuffers = FixedSizeDescriptorSetsPool::new(shared.pipeline_gbuffers.clone(), 0);
		let camera_desc_pool_target = FixedSizeDescriptorSetsPool::new(shared.pipeline_target.clone(), 1);
		let mesh_desc_pool = FixedSizeDescriptorSetsPool::new(shared.pipeline_gbuffers.clone(), 1);
		let (gbuffers, desc_target, future) = Self::make_gbuffers(target, &shared)?;

		let framebuffers =
			target.images().iter()
				.map(|image| Self::make_framebuffer(
					shared.subpass_target.render_pass().clone(),
					gbuffers.clone(),
					image.clone(),
				))
				.collect::<Result<Vec<_>, _>>()?;

		Ok((
			Self {
				shared: shared,
				meshes: vec![],
				framebuffers: framebuffers,
				target_id: target.id_root().make_id(),
				desc_target: desc_target,
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

		let framebuffer = self.framebuffers[image_num].image
			.upgrade()
			.iter()
			.filter(|old_image| Arc::ptr_eq(&target.images()[image_num], &old_image))
			.next()
			.map(|_| self.framebuffers[image_num].framebuffer.clone());
		let (framebuffer, target_size_future) =
			if let Some(framebuffer) = framebuffer.as_ref() {
				(framebuffer.clone(), None)
			} else {
				let (gbuffers, desc_target, future) = Self::make_gbuffers(target, &self.shared)?;

				self.desc_target = desc_target;

				let image_framebuffer =
					Self::make_framebuffer(
						self.shared.subpass_target.render_pass().clone(),
						gbuffers,
						target.images()[image_num].clone()
					)?;
				let framebuffer = image_framebuffer.framebuffer.clone();
				self.framebuffers[image_num] = image_framebuffer;

				(framebuffer, Some(future))
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

		let dimensions = [framebuffer.width() as f32, framebuffer.height() as f32];

		let mut command_buffer =
			AutoCommandBufferBuilder
				::primary_one_time_submit(
					self.shared.shaders.target_vertices.device().clone(),
					window.queue().family()
				)?
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
					self.shared.pipeline_target.clone(),
					DynamicState {
						line_width: None,
						viewports:
							Some(vec![Viewport { origin: [0.0, 0.0], dimensions: dimensions, depth_range: 0.0..1.0 }]),
						scissors: None,
					},
					vec![self.shared.shaders.target_vertices.clone()],
					(
						self.desc_target.clone(),
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
			target_size_future
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

	fn make_framebuffer(
		render_pass: Arc<RenderPassAbstract + Sync + Send>,
		gbuffers: GBuffers,
		image: Arc<ImageViewAccess + Sync + Send + 'static>,
	) -> Result<ImageFramebuffer, OomError> {
		let GBuffers { image_color, image_normal, image_depth } = gbuffers;
		let weak_image = Arc::downgrade(&image);

		Framebuffer::start(render_pass)
			.add(image_color)
			.and_then(|fb| fb.add(image_normal))
			.and_then(|fb| fb.add(image_depth))
			.and_then(|fb| fb.add(image))
			.and_then(|fb| fb.build())
			.map(|fb| ImageFramebuffer::new(weak_image, Arc::new(fb)))
			.map_err(|err| match err {
				FramebufferCreationError::OomError(err) => err,
				err => unreachable!("{:?}", err),
			})
	}

	fn make_gbuffers(
		target: &RenderTarget,
		shared: &MeshBatchShared,
	) -> Result<(GBuffers, Arc<DescriptorSet + Send + Sync + 'static>, impl GpuFuture), DeviceMemoryAllocError> {
		let dimensions = target.images()[0].dimensions().width_height();
		let image_color =
			Self::make_transient_input_attachment(
				shared.shaders.target_vertices.device().clone(),
				dimensions,
				target.format()
			)?;
		let image_normal =
			Self::make_transient_input_attachment(
				shared.shaders.target_vertices.device().clone(),
				dimensions,
				NORMAL_FORMAT
			)?;
		let image_depth =
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
					.add_image(image_color.clone())
					.unwrap()
					.add_image(image_normal.clone())
					.unwrap()
					.add_image(image_depth.clone())
					.unwrap()
					.build()
					.unwrap()
			);

		Ok((
			GBuffers { image_color: image_color, image_normal: image_normal, image_depth: image_depth },
			desc_target,
			target_size_future
		))
	}
}

#[derive(Clone, Debug)]
struct GBuffers {
	image_color: Arc<AttachmentImage>,
	image_normal: Arc<AttachmentImage>,
	image_depth: Arc<AttachmentImage>,
}

#[derive(Debug, Clone)]
struct TargetVertex { position: [f32; 2] }
impl_vertex!(TargetVertex, position);
