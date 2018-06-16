use batch::mesh::MeshBatchShared;
use cgmath::{ InnerSpace, Vector4 };
use codec::obj::Obj;
use cpu_pool::{ spawn_fs_then_cpu, DiskCpuFuture };
use decorum::hash_float_array;
use nom;
use std::{ collections::HashMap, fs::File, hash::{ Hash, Hasher }, io::prelude::*, iter::once, path::Path, sync::Arc };
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
	where D: ExactSizeIterator<Item = MeshVertex>,
	{
		let (vertices, vertices_future) =
			ImmutableBuffer::from_iter(vertices, BufferUsage::vertex_buffer(), window.queue().clone())?;

		let (position, position_future) =
			ImmutableBuffer::from_data(position, BufferUsage::uniform_buffer(), window.queue().clone())?;

		Ok((Self { position: position, vertices: vertices }, vertices_future.join(position_future)))
	}

	pub fn from_file<P>(
		window: &Window,
		position: [f32; 3],
		path: P
	) -> DiskCpuFuture<(Self, impl GpuFuture + Send + Sync + 'static), MeshFromFileError>
	where P: AsRef<Path> + Send + 'static
	{
		let queue = window.queue().clone();
		spawn_fs_then_cpu(
			|_| {
				let mut buf = String::new();
				File::open(path)?.read_to_string(&mut buf).map(|_| buf)
			},
			move |_, buf| {
				let obj = Obj::from_str(&buf)
					.map_err(|err| match err {
						nom::Err::Error(nom::Context::Code(loc, kind)) =>
							nom::Err::Error(nom::Context::Code(loc.to_owned(), kind)),
						nom::Err::Failure(nom::Context::Code(loc, kind)) =>
							nom::Err::Failure(nom::Context::Code(loc.to_owned(), kind)),
						err => unreachable!(err),
					})?;

				let mut vertices = vec![];
				for object in once(&obj.root_object).chain(obj.named_objects.iter().map(|(_, o)| o)) {
					for face in &object.faces {
						for triangle in triangulate(face.vertices.iter().map(|v| object.vertices[v.position])) {
							let positions = [
								object.vertices[face.vertices[triangle[0]].position].xyz(),
								object.vertices[face.vertices[triangle[1]].position].xyz(),
								object.vertices[face.vertices[triangle[2]].position].xyz(),
							];

							let mut triangle_normal =
								(positions[1] - positions[0]).cross(positions[2] - positions[0]).normalize();

							vertices.extend(
								(0..triangle.len())
									.map(|ti| MeshVertex::new(
										positions[ti].into(),
										face.vertices[triangle[ti]].normal.map(|ni| object.normals[ni])
											.unwrap_or(triangle_normal)
											.into()
									))
							);
						}
					}
				}

				let (vertices, vertices_future) =
					ImmutableBuffer::from_iter(vertices.into_iter(), BufferUsage::vertex_buffer(), queue.clone())?;

				let (position, position_future) =
					ImmutableBuffer::from_data(position, BufferUsage::uniform_buffer(), queue)?;

				Ok((Self { position: position, vertices: vertices }, vertices_future.join(position_future)))
			}
		)
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

#[derive(Debug)]
pub enum MeshFromFileError {
	Nom(nom::Err<String>),
	DeviceMemoryAllocError(DeviceMemoryAllocError),
}
impl From<nom::Err<String>> for MeshFromFileError{
	fn from(err: nom::Err<String>) -> Self {
		MeshFromFileError::Nom(err)
	}
}
impl From<DeviceMemoryAllocError> for MeshFromFileError{
	fn from(err: DeviceMemoryAllocError) -> Self {
		MeshFromFileError::DeviceMemoryAllocError(err)
	}
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshVertex {
	pub position: [f32; 3],
	pub normal: [f32; 3],
}
impl MeshVertex {
	pub fn new(position: [f32; 3], normal: [f32; 3]) -> Self {
		Self { position: position, normal: normal }
	}
}
impl Hash for MeshVertex {
	fn hash<H: Hasher>(&self, state: &mut H) {
		hash_float_array(&self.position, state);
		hash_float_array(&self.normal, state);
	}
}
impl_vertex!(MeshVertex, position, normal);

fn triangulate<'a>(mut vertices: impl ExactSizeIterator<Item = Vector4<f32>>) -> impl Iterator<Item = [usize; 3]> {
	let v0 = vertices.next().unwrap();
	let v1 = vertices.next().unwrap();
	vertices
		.enumerate()
		.scan(
			(v0, v1),
			|(v0, vprev), (i, vcur)| {
				let i = i + 2;
				let angle = (*v0 - *vprev).angle(*vprev - vcur);
				*vprev = vcur;

				if angle.0 > 0.0 {
					Some([0, i - 1, i])
				} else if angle.0 < 0.0 {
					Some([i, i - 1, 0])
				} else {
					panic!("Triangle must not be a line.");
				}
			}
		)
}
