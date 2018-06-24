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
	buffer::{ BufferUsage, ImmutableBuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	descriptor::{ DescriptorSet, descriptor_set::{ FixedSizeDescriptorSetsPool, PersistentDescriptorSet } },
	device::Device,
	format::Format,
	framebuffer::{ Framebuffer, FramebufferAbstract, FramebufferCreationError },
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
	camera_desc_pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
	mesh_desc_pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
}
impl MeshBatch {
	pub fn new(
		target: &RenderTarget,
		shared: Arc<MeshBatchShared>
	) -> Result<(Self, impl GpuFuture), DeviceMemoryAllocError> {
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
					.add_image(image_color)
					.unwrap()
					.add_image(image_normal)
					.unwrap()
					.add_image(image_depth)
					.unwrap()
					.build()
					.unwrap()
			);

		let camera_desc_pool = FixedSizeDescriptorSetsPool::new(shared.pipeline_gbuffers.clone(), 0);
		let mesh_desc_pool = FixedSizeDescriptorSetsPool::new(shared.pipeline_gbuffers.clone(), 1);

		Ok((
			Self {
				shared: shared,
				meshes: vec![],
				framebuffers: framebuffers,
				target_id: target.id_root().make_id(),
				desc_target: desc_target,
				camera_desc_pool: camera_desc_pool,
				mesh_desc_pool: mesh_desc_pool,
			},
			target_size_future
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

		let dimensions = target.images()[image_num].dimensions().width_height();

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
				let image_color =
					Self::make_transient_input_attachment(window.device().clone(), dimensions, target.format())?;
				let image_normal =
					Self::make_transient_input_attachment(window.device().clone(), dimensions, NORMAL_FORMAT)?;
				let image_depth =
					Self::make_transient_input_attachment(window.device().clone(), dimensions, DEPTH_FORMAT)?;

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
						self.shared.shaders.queue.clone()
					)?;

				self.desc_target =
					Arc::new(
						PersistentDescriptorSet::start(self.shared.pipeline_target.clone(), 0)
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

				let framebuffer = Framebuffer::start(self.shared.subpass_gbuffers.render_pass().clone())
					.add(image_color)
					.and_then(|fb| fb.add(image_normal))
					.and_then(|fb| fb.add(image_depth))
					.and_then(|fb| fb.add(target.images()[image_num].clone()))
					.and_then(|fb| fb.build())
					.map(|fb| Arc::new(fb))
					.map_err(|err| {
						match err { FramebufferCreationError::OomError(err) => err, err => unreachable!("{:?}", err) }
					})?;
				self.framebuffers[image_num] =
					ImageFramebuffer::new(Arc::downgrade(&target.images()[image_num]), framebuffer.clone());

				(framebuffer as _, Some(target_size_future))
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
			AutoCommandBufferBuilder::primary_one_time_submit(self.shared.shaders.target_vertices.device().clone(), window.queue().family())?
				.begin_render_pass(
					framebuffer.clone(),
					true,
					vec![
						[0.0, 0.0, 0.0, 1.0].into(),
						[0.0; 4].into(),
						1.0.into(),
						[0.0, 0.0, 0.0, 1.0].into()
					]
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
					self.desc_target.clone(),
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
}

#[derive(Debug, Clone)]
struct TargetVertex { position: [f32; 2] }
impl_vertex!(TargetVertex, position);
