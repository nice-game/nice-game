mod mesh;
mod shaders;
mod render_passes;

pub use self::mesh::Mesh;
pub use self::shaders::{ MeshShaders, MeshShadersError };
pub use self::render_passes::MeshRenderPasses;
use { ImageFramebuffer, ObjectId, RenderTarget, window::Window };
use camera::Camera;
use std::sync::Arc;
use vulkano::{
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	descriptor::{ DescriptorSet, descriptor_set::{ FixedSizeDescriptorSetsPool, PersistentDescriptorSet } },
	device::Device,
	format::Format,
	framebuffer::{ Framebuffer, FramebufferAbstract, FramebufferCreationError },
	image::{ AttachmentImage, ImageCreationError, ImageViewAccess },
	memory::{ DeviceMemoryAllocError },
	pipeline::{ GraphicsPipelineAbstract, viewport::Viewport },
};

const ALBEDO_FORMAT: Format = Format::A2B10G10R10UnormPack32;
const NORMAL_FORMAT: Format = Format::R32G32B32A32Sfloat;
const DEPTH_FORMAT: Format = Format::D16Unorm;

pub struct MeshBatch {
	render_passes: Arc<MeshRenderPasses>,
	meshes: Vec<Mesh>,
	target_id: ObjectId,
	gbuffers_framebuffer: Arc<FramebufferAbstract + Send + Sync + 'static>,
	gbuffers_camera_desc_pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
	target_framebuffers: Vec<ImageFramebuffer>,
	target_desc: Arc<DescriptorSet + Send + Sync + 'static>,
	target_camera_desc_pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
	mesh_desc_pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
}
impl MeshBatch {
	pub fn new(target: &RenderTarget, render_passes: Arc<MeshRenderPasses>) -> Result<Self, DeviceMemoryAllocError> {
		let gbuffers_camera_desc_pool = FixedSizeDescriptorSetsPool::new(render_passes.gbuffers_pipeline().clone(), 0);
		let target_camera_desc_pool = FixedSizeDescriptorSetsPool::new(render_passes.target_pipeline().clone(), 1);
		let mesh_desc_pool = FixedSizeDescriptorSetsPool::new(render_passes.gbuffers_pipeline().clone(), 1);
		let (gbuffers, target_desc) = Self::make_gbuffers(target, &render_passes)?;
		let GBuffers { image_color, image_normal, image_depth } = gbuffers;

		let gbuffers_framebuffer =
			Arc::new(
				Framebuffer::start(render_passes.gbuffers_render_pass().clone())
					.add(image_color)
					.and_then(|fb| fb.add(image_normal))
					.and_then(|fb| fb.add(image_depth))
					.and_then(|fb| fb.build())
					.map_err(|err| match err {
						FramebufferCreationError::OomError(err) => err,
						err => unreachable!("{:?}", err),
					})?
			);

		let target_framebuffers =
			target.images().iter()
				.map(|image| {
					Framebuffer::start(render_passes.target_render_pass().clone())
						.add(image.clone())
						.and_then(|fb| fb.build())
						.map(|fb| ImageFramebuffer::new(Arc::downgrade(image), Arc::new(fb)))
						.map_err(|err| match err {
							FramebufferCreationError::OomError(err) => err,
							err => unreachable!("{:?}", err),
						})
				})
				.collect::<Result<Vec<_>, _>>()?;

		Ok(
			Self {
				render_passes: render_passes,
				meshes: vec![],
				gbuffers_framebuffer: gbuffers_framebuffer,
				target_framebuffers: target_framebuffers,
				target_id: target.id_root().make_id(),
				target_desc: target_desc,
				gbuffers_camera_desc_pool: gbuffers_camera_desc_pool,
				target_camera_desc_pool: target_camera_desc_pool,
				mesh_desc_pool: mesh_desc_pool,
			}
		)
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

		let target_framebuffer = self.target_framebuffers[image_num].image
			.upgrade()
			.iter()
			.filter(|old_image| Arc::ptr_eq(&target.images()[image_num], &old_image))
			.next()
			.map(|_| self.target_framebuffers[image_num].framebuffer.clone());
		let target_framebuffer =
			if let Some(framebuffer) = target_framebuffer.as_ref() {
				framebuffer.clone()
			} else {
				let image = &target.images()[image_num];
				let target_framebuffer =
					Arc::new(
						Framebuffer::start(self.render_passes.target_render_pass().clone())
							.add(image.clone())
							.and_then(|fb| fb.build())
							.map_err(|err| match err {
								FramebufferCreationError::OomError(err) => err,
								err => unreachable!("{:?}", err),
							})?
					);
				self.target_framebuffers[image_num] =
					ImageFramebuffer::new(Arc::downgrade(image), target_framebuffer.clone());

				if target_framebuffer.width() != self.gbuffers_framebuffer.width() ||
					target_framebuffer.height() != self.gbuffers_framebuffer.height()
				{
					let (gbuffers, target_desc) = Self::make_gbuffers(target, &self.render_passes)?;
					let GBuffers { image_color, image_normal, image_depth } = gbuffers;

					self.target_desc = target_desc;
					self.gbuffers_framebuffer =
						Arc::new(
							Framebuffer::start(self.render_passes.gbuffers_render_pass().clone())
								.add(image_color)
								.and_then(|fb| fb.add(image_normal))
								.and_then(|fb| fb.add(image_depth))
								.and_then(|fb| fb.build())
								.map_err(|err| match err {
									FramebufferCreationError::OomError(err) => err,
									err => unreachable!("{:?}", err),
								})?
						);
				}

				target_framebuffer
			};

		let camera_desc_gbuffers =
			Arc::new(
				self.gbuffers_camera_desc_pool.next()
					.add_buffer(camera.position_buffer.clone())
					.unwrap()
					.add_buffer(camera.rotation_buffer.clone())
					.unwrap()
					.add_buffer(camera.projection_buffer.clone())
					.unwrap()
					.build()
					.unwrap()
			);

		let dimensions = [self.gbuffers_framebuffer.width() as f32, self.gbuffers_framebuffer.height() as f32];

		let mut command_buffer =
			AutoCommandBufferBuilder
				::primary_one_time_submit(
					self.render_passes.device().clone(),
					window.queue().family()
				)?
				.begin_render_pass(
					self.gbuffers_framebuffer.clone(),
					true,
					vec![[0.0, 0.0, 0.0, 1.0].into(), [0.0; 4].into(), 1.0.into()]
				)
				.unwrap();

		for mesh in &mut self.meshes {
			command_buffer =
				unsafe {
					command_buffer
						.execute_commands(
							mesh.make_commands(
								&self.render_passes,
								camera_desc_gbuffers.clone(),
								&mut self.mesh_desc_pool,
								window.queue().family(),
								dimensions
							)?
						)
						.unwrap()
				};
		}

		Ok(
			command_buffer
				.end_render_pass()
				.unwrap()
				.begin_render_pass(target_framebuffer, false, vec![[0.0, 0.0, 0.0, 1.0].into()])
				.unwrap()
				.draw(
					self.render_passes.target_pipeline().clone(),
					DynamicState {
						line_width: None,
						viewports:
							Some(vec![Viewport { origin: [0.0, 0.0], dimensions: dimensions, depth_range: 0.0..1.0 }]),
						scissors: None,
					},
					vec![self.render_passes.shaders().target_vertices.clone()],
					(
						self.target_desc.clone(),
						self.target_camera_desc_pool.next()
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
				.map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?
		)
	}

	fn make_sampled_attachment(
		device: Arc<Device>,
		dimensions: [u32; 2],
		format: Format,
	) -> Result<Arc<AttachmentImage>, DeviceMemoryAllocError> {
		AttachmentImage::sampled(device, dimensions, format)
			.map_err(|err| match err { ImageCreationError::AllocError(err) => err, err => unreachable!(err) })
	}

	fn make_gbuffers(
		target: &RenderTarget,
		shared: &MeshRenderPasses,
	) -> Result<(GBuffers, Arc<DescriptorSet + Send + Sync + 'static>), DeviceMemoryAllocError> {
		let dimensions = target.images()[0].dimensions().width_height();
		let image_color =
			Self::make_sampled_attachment(
				shared.device().clone(),
				dimensions,
				ALBEDO_FORMAT
			)?;
		let image_normal =
			Self::make_sampled_attachment(
				shared.device().clone(),
				dimensions,
				NORMAL_FORMAT
			)?;
		let image_depth =
			Self::make_sampled_attachment(
				shared.device().clone(),
				dimensions,
				DEPTH_FORMAT
			)?;

		let target_desc =
			Arc::new(
				PersistentDescriptorSet::start(shared.target_pipeline().clone(), 0)
					.add_sampled_image(image_color.clone(), shared.shaders().sampler.clone()).unwrap()
					.add_sampled_image(image_normal.clone(), shared.shaders().sampler.clone()).unwrap()
					.add_sampled_image(image_depth.clone(), shared.shaders().sampler.clone()).unwrap()
					.build().unwrap()
			);

		Ok((
			GBuffers { image_color: image_color, image_normal: image_normal, image_depth: image_depth },
			target_desc
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
