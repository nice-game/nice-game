use batch::mesh::MeshBatchShared;
use byteorder::{LE, ReadBytesExt};
use cpu_pool::{ spawn_fs, CpuFuture };
use std::{
	fs::File,
	io::{ self, prelude::*, SeekFrom },
	mem::size_of,
	path::Path,
	sync::Arc,
	vec::IntoIter as VecIntoIter,
};
use vulkano::{
	OomError,
	buffer::{ BufferAccess, BufferUsage, CpuAccessibleBuffer, ImmutableBuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	descriptor::{ DescriptorSet, descriptor_set::{ FixedSizeDescriptorSetsPool } },
	device::Queue,
	format::Format,
	instance::QueueFamily,
	memory::DeviceMemoryAllocError,
	pipeline::{
		GraphicsPipelineAbstract,
		vertex::{ AttributeInfo, IncompatibleVertexDefinitionError, InputRate, VertexDefinition, VertexSource },
		viewport::Viewport
	},
	sync::GpuFuture,
};
use window::Window;

pub struct Mesh {
	position: Arc<ImmutableBuffer<[f32; 3]>>,
	positions: Arc<ImmutableBuffer<[[f32; 3]]>>,
	normals: Arc<ImmutableBuffer<[[f32; 3]]>>,
	texcoords_main: Arc<ImmutableBuffer<[[f32; 2]]>>,
	indices: Arc<ImmutableBuffer<[u32]>>,
}
impl Mesh {
	pub fn from_file<P>(
		window: &Window,
		position: [f32; 3],
		path: P
	) -> CpuFuture<(Mesh, impl GpuFuture + Send + Sync + 'static), MeshFromFileError>
	where P: AsRef<Path> + Send + 'static
	{
		let queue = window.queue().clone();
		let test = spawn_fs(move |_| {
			let (position, position_future) =
				ImmutableBuffer::from_data(position, BufferUsage::uniform_buffer(), queue.clone())?;

			let mut file = File::open(path)?;

			let mut magic_number = [0; 4];
			file.read_exact(&mut magic_number)?;
			assert_eq!(&magic_number, b"nmdl");

			file.seek(SeekFrom::Current(4))?;

			let vertex_count = file.read_u32::<LE>()? as usize;
			let positions_offset = file.read_u32::<LE>()? as u64;
			let normals_offset = file.read_u32::<LE>()? as u64;
			let texcoords_main_offset = file.read_u32::<LE>()? as u64;
			let _texcoords_lightmap_offset = file.read_u32::<LE>()? as u64;
			let index_count = file.read_u32::<LE>()? as usize;
			let indices_offset = file.read_u32::<LE>()? as u64;

			file.seek(SeekFrom::Start(positions_offset))?;
			let (positions, positions_future) =
				Self::buffer_from_file(
					queue.clone(),
					BufferUsage::vertex_buffer(),
					vertex_count,
					&mut || Ok([file.read_f32::<LE>()?, file.read_f32::<LE>()?, file.read_f32::<LE>()?])
				)?;

			file.seek(SeekFrom::Start(normals_offset))?;
			let (normals, normals_future) =
				Self::buffer_from_file(
					queue.clone(),
					BufferUsage::vertex_buffer(),
					vertex_count,
					&mut || Ok([file.read_f32::<LE>()?, file.read_f32::<LE>()?, file.read_f32::<LE>()?])
				)?;

			file.seek(SeekFrom::Start(texcoords_main_offset))?;
			let (texcoords_main, texcoords_main_future) =
				Self::buffer_from_file(
					queue.clone(),
					BufferUsage::vertex_buffer(),
					vertex_count,
					&mut || Ok([file.read_f32::<LE>()?, file.read_f32::<LE>()?])
				)?;

			file.seek(SeekFrom::Start(indices_offset))?;
			let (indices, indices_future) =
				Self::buffer_from_file(queue, BufferUsage::index_buffer(), index_count, &mut || file.read_u32::<LE>())?;

			Ok((
				Mesh {
					position: position,
					positions: positions,
					normals: normals,
					texcoords_main: texcoords_main,
					indices: indices
				},
				position_future
					.join(positions_future)
					.join(normals_future)
					.join(texcoords_main_future)
					.join(indices_future)
			))
		});

		test
	}

	fn buffer_from_file<T>(
		queue: Arc<Queue>,
		usage: BufferUsage,
		count: usize,
		read: &mut FnMut() -> io::Result<T>
	) -> Result<(Arc<ImmutableBuffer<[T]>>, impl GpuFuture), MeshFromFileError>
	where T: Send + Sync + 'static
	{
		let tmpbuf =
			unsafe {
				CpuAccessibleBuffer::uninitialized_array(queue.device().clone(), count, BufferUsage::transfer_source())?
			};
		{
			let mut tmpbuf_lock = tmpbuf.write().unwrap();
			for i in 0..count {
				tmpbuf_lock[i] = read()?;
			}
		}
		ImmutableBuffer::from_buffer(tmpbuf, usage, queue).map_err(|e| e.into())
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
				.draw_indexed(
					shared.pipeline_gbuffers.clone(),
					DynamicState {
						line_width: None,
						viewports:
							Some(vec![Viewport { origin: [0.0, 0.0], dimensions: dimensions, depth_range: 0.0..1.0 }]),
						scissors: None,
					},
					vec![self.positions.clone(), self.normals.clone(), self.texcoords_main.clone()],
					self.indices.clone(),
					(camera_desc, mesh_desc_pool.next().add_buffer(self.position.clone()).unwrap().build().unwrap()),
					()
				)
				.unwrap()
				.build()
				.map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?
		)
	}
}

pub struct MeshVertexDefinition {}
impl MeshVertexDefinition {
	pub fn new() -> Self {
		Self {}
	}
}
unsafe impl<I> VertexDefinition<I> for MeshVertexDefinition {
	type BuffersIter = VecIntoIter<(u32, usize, InputRate)>;
	type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;

	fn definition(
		&self,
		_interface: &I
	) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
		// TODO: validate against shader
		Ok((
			vec![
				(0, size_of::<[f32; 3]>(), InputRate::Vertex),
				(1, size_of::<[f32; 3]>(), InputRate::Vertex),
				(2, size_of::<[f32; 2]>(), InputRate::Vertex)
			].into_iter(),
			vec![
				(0, 0, AttributeInfo { offset: 0, format: Format::R32G32B32Sfloat }),
				(1, 1, AttributeInfo { offset: 0, format: Format::R32G32B32Sfloat }),
				(2, 2, AttributeInfo { offset: 0, format: Format::R32G32Sfloat })
			].into_iter()
		))
	}
}
unsafe impl VertexSource<Vec<Arc<BufferAccess + Send + Sync>>> for MeshVertexDefinition {
	#[inline]
	fn decode(
		&self,
		source: Vec<Arc<BufferAccess + Send + Sync>>
	) -> (Vec<Box<BufferAccess + Send + Sync>>, usize, usize) {
		assert_eq!(source.len(), 3);
		let len = source[0].size() / size_of::<[f32; 3]>();
		(source.into_iter().map(|x| Box::new(x) as _).collect(), len, 1)
	}
}

#[derive(Debug)]
pub enum MeshFromFileError {
	Io(io::Error),
	DeviceMemoryAllocError(DeviceMemoryAllocError),
}
impl From<io::Error> for MeshFromFileError{
	fn from(err: io::Error) -> Self {
		MeshFromFileError::Io(err)
	}
}
impl From<DeviceMemoryAllocError> for MeshFromFileError{
	fn from(err: DeviceMemoryAllocError) -> Self {
		MeshFromFileError::DeviceMemoryAllocError(err)
	}
}
