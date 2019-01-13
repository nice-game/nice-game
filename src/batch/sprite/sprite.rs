use super::Drawable2D;
use super::shared::SpriteBatchShared;
use crate::texture::Texture;
use std::sync::Arc;
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	descriptor::{ DescriptorSet, descriptor_set::PersistentDescriptorSet },
	device::Queue,
	instance::QueueFamily,
	memory::DeviceMemoryAllocError,
	pipeline::{ GraphicsPipelineAbstract, viewport::Viewport },
	sampler::Sampler,
	sync::GpuFuture,
};

pub struct Sprite {
	static_desc: Arc<DescriptorSet + Send + Sync + 'static>,
	position: Arc<ImmutableBuffer<[f32; 2]>>,
}
impl Sprite {
	pub(crate) fn new(
		queue: Arc<Queue>,
		pipeline: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
		sampler: Arc<Sampler>,
		texture: &Texture,
		position: [f32; 2]
	) -> Result<(Self, impl GpuFuture), DeviceMemoryAllocError> {
		let (position, future) = ImmutableBuffer::from_data(position, BufferUsage::uniform_buffer(), queue)?;

		Ok((
			Self {
				static_desc:
					Arc::new(
						PersistentDescriptorSet::start(pipeline, 2)
							.add_sampled_image(texture.image().clone(), sampler)
							.unwrap()
							.build()
							.unwrap()
					),
				position: position
			},
			future
		))
	}
}
impl Drawable2D for Sprite {
	fn make_commands(
		&mut self,
		shared: &SpriteBatchShared,
		target_desc: &Arc<DescriptorSet + Send + Sync + 'static>,
		queue_family: QueueFamily,
		dimensions: [f32; 2],
	) -> Result<AutoCommandBuffer, OomError> {
		Ok(
			AutoCommandBufferBuilder::secondary_graphics_one_time_submit(shared.shaders().device().clone(), queue_family, shared.subpass().clone())?
				.draw(
					shared.pipeline_sprite().clone(),
					&DynamicState {
						line_width: None,
						viewports:
							Some(vec![Viewport { origin: [0.0, 0.0], dimensions: dimensions, depth_range: 0.0..1.0 }]),
						scissors: None,
					},
					vec![shared.shaders().vertices().clone()],
					(
						target_desc.clone(),
						shared.sprite_desc_pool().lock().unwrap()
							.next()
							.add_buffer(self.position.clone())
							.unwrap()
							.build()
							.unwrap(),
						self.static_desc.clone(),
					),
					()
				)
				.unwrap()
				.build()
				.map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?
		)
	}
}
