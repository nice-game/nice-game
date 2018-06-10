use batch::mesh::MeshBatchShared;
use std::sync::Arc;
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	descriptor::{ DescriptorSet, descriptor_set::{ FixedSizeDescriptorSetsPool } },
	instance::QueueFamily,
	memory::DeviceMemoryAllocError,
	pipeline::{ GraphicsPipelineAbstract, viewport::Viewport },
	sync::GpuFuture,
};
use window::Window;

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

	pub(super) fn make_commands(
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

#[derive(Debug, Clone)]
pub struct MeshVertex {
	pub position: [f32; 3],
	pub normal: [f32; 3],
}
impl_vertex!(MeshVertex, position, normal);
