use batch::mesh::MeshBatchShared;
use byteorder::{LE, ReadBytesExt};
use cgmath::{ Quaternion, Vector3 };
use cpu_pool::{ spawn_fs, CpuFuture };
use futures::prelude::*;
use std::{
	fs::File,
	io::{ self, prelude::*, SeekFrom },
	mem::{ size_of, transmute },
	path::{ Path, PathBuf },
	sync::Arc,
	vec::IntoIter as VecIntoIter,
};
use vulkano::{
	OomError,
	buffer::{
		BufferAccess,
		BufferSlice,
		BufferUsage,
		CpuAccessibleBuffer,
		CpuBufferPool,
		ImmutableBuffer,
		cpu_pool::CpuBufferPoolSubbuffer,
	},
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	descriptor::{ DescriptorSet, descriptor_set::{ FixedSizeDescriptorSetsPool, PersistentDescriptorSet } },
	device::Queue,
	format::Format,
	instance::QueueFamily,
	memory::{ DeviceMemoryAllocError, pool::StdMemoryPool },
	pipeline::{
		GraphicsPipelineAbstract,
		vertex::{ AttributeInfo, IncompatibleVertexDefinitionError, InputRate, VertexDefinition, VertexSource },
		viewport::Viewport
	},
	sync::GpuFuture,
};
use window::Window;

pub struct Mesh {
	position_pool: CpuBufferPool<Vector3<f32>>,
	rotation_pool: CpuBufferPool<Quaternion<f32>>,
	position: CpuBufferPoolSubbuffer<Vector3<f32>, Arc<StdMemoryPool>>,
	rotation: CpuBufferPoolSubbuffer<Quaternion<f32>, Arc<StdMemoryPool>>,
	positions: Arc<ImmutableBuffer<[[f32; 3]]>>,
	normals: Arc<ImmutableBuffer<[[f32; 3]]>>,
	texcoords_main: Arc<ImmutableBuffer<[[f32; 2]]>>,
	materials: Vec<Material>,
	material_descs: Vec<Arc<DescriptorSet + Send + Sync + 'static>>,
}
impl Mesh {
	pub fn from_file<P>(
		window: &Window,
		shared: &MeshBatchShared,
		position: Vector3<f32>,
		rotation: Quaternion<f32>,
		path: P
	) -> impl Future<Item = (Mesh, impl GpuFuture + Send + Sync + 'static), Error = MeshFromFileError>
	where P: AsRef<Path> + Send + 'static
	{
		let device = window.device().clone();
		let queue = window.queue().clone();
		let pipeline_gbuffers = shared.pipeline_gbuffers.clone();

		spawn_fs(move |_| {
			let mut file = File::open(path)?;

			let mut magic_number = [0; 4];
			file.read_exact(&mut magic_number)?;
			assert_eq!(&magic_number, b"nmdl");

			// skip version for now
			file.seek(SeekFrom::Current(4))?;

			let vertex_count = file.read_u32::<LE>()? as usize;
			let positions_offset = file.read_u32::<LE>()? as u64;
			let normals_offset = file.read_u32::<LE>()? as u64;
			let texcoords_main_offset = file.read_u32::<LE>()? as u64;
			let _texcoords_lightmap_offset = file.read_u32::<LE>()? as u64;
			let index_count = file.read_u32::<LE>()? as usize;
			let indices_offset = file.read_u32::<LE>()? as u64;
			let material_count = file.read_u8()? as usize;
			let materials_offset = file.read_u32::<LE>()? as u64;

			debug!("vertex_count: {}", vertex_count);
			debug!("positions_offset: {}", positions_offset);
			debug!("normals_offset: {}", normals_offset);
			debug!("texcoords_main_offset: {}", texcoords_main_offset);
			debug!("_texcoords_lightmap_offset: {}", _texcoords_lightmap_offset);
			debug!("index_count: {}", index_count);
			debug!("indices_offset: {}", indices_offset);
			debug!("material_count: {}", material_count);
			debug!("materials_offset: {}", materials_offset);

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
				Self::buffer_from_file(
					queue.clone(),
					BufferUsage::index_buffer(),
					index_count,
					&mut || file.read_u32::<LE>()
				)?;

			file.seek(SeekFrom::Start(materials_offset))?;

			// round MaterialGpu size up to minimum alignment
			let mut material_stride =
				queue.device().physical_device().limits().min_uniform_buffer_offset_alignment() as usize;
			material_stride = (size_of::<MaterialGpu>() + material_stride - 1) / material_stride * material_stride;
			debug!("material stride: {}", material_stride);

			let mut materials = Vec::with_capacity(material_count);
			let material_buf =
				unsafe {
					CpuAccessibleBuffer::uninitialized_array(
						queue.device().clone(),
						material_count * material_stride,
						BufferUsage::transfer_source()
					)?
				};
			{
				let mut material_buf_lock = material_buf.write().unwrap();
				let mut index_start = 0;
				for i in 0..material_count {
					let index_count = file.read_u32::<LE>()? as usize;

					materials
						.push(Material {
							indices:
								indices.clone().into_buffer_slice().slice(index_start..index_start + index_count).unwrap(),
							texture1: {
								// skip texture for now
								file.seek(SeekFrom::Current(6))?;
								None
							},
							texture2: {
								// skip texture for now
								file.seek(SeekFrom::Current(6))?;
								None
							},
						});

					material_buf_lock[i * material_stride..i * material_stride + size_of::<MaterialGpu>()]
						.copy_from_slice(
							&unsafe {
								transmute::<_, [u8; size_of::<MaterialGpu>()]>(
									MaterialGpu {
										light_penetration: file.read_u8()? as u32,
										subsurface_scattering: file.read_u8()? as u32,
										emissive_brightness: file.read_u16::<LE>()? as u32,
										base_color: {
											let mut buf = [0; 3];
											file.read_exact(&mut buf)?;
											[
												(buf[0] as f32 / 255.0).powf(2.2),
												(buf[1] as f32 / 255.0).powf(2.2),
												(buf[2] as f32 / 255.0).powf(2.2)
											]
										},
									}
								)
							}
						);

					index_start += index_count;
				}
			}

			let (material_buf, material_buf_future) =
				ImmutableBuffer::from_buffer(material_buf, BufferUsage::uniform_buffer(), queue)?;
			let material_descs = (0..material_count)
				.map(|i| {
					let material_offset = material_stride * i;
					Arc::new(
						PersistentDescriptorSet::start(pipeline_gbuffers.clone(), 2)
							.add_buffer(
								material_buf.clone()
									.into_buffer_slice()
									.slice(material_offset..material_offset + size_of::<MaterialGpu>())
									.unwrap()
							)
							.unwrap()
							.build()
							.unwrap()
					) as _
				})
				.collect();

			let position_pool = CpuBufferPool::uniform_buffer(device.clone());
			let rotation_pool = CpuBufferPool::uniform_buffer(device);
			let position = position_pool.next(position)?;
			let rotation = rotation_pool.next(rotation)?;

			Ok((
				Mesh {
					position_pool: position_pool,
					rotation_pool: rotation_pool,
					position: position,
					rotation: rotation,
					positions: positions,
					normals: normals,
					texcoords_main: texcoords_main,
					materials: materials,
					material_descs: material_descs,
				},
				positions_future
					.join(normals_future)
					.join(texcoords_main_future)
					.join(indices_future)
					.join(material_buf_future)
			))
		})
	}

	pub fn set_position(&mut self, position: Vector3<f32>) -> Result<(), DeviceMemoryAllocError> {
		self.position = self.position_pool.next(position)?;
		Ok(())
	}

	pub fn set_rotation(&mut self, rotation: Quaternion<f32>) -> Result<(), DeviceMemoryAllocError> {
		self.rotation = self.rotation_pool.next(rotation)?;
		Ok(())
	}

	pub(super) fn make_commands(
		&mut self,
		shared: &MeshBatchShared,
		camera_desc: impl DescriptorSet + Clone + Send + Sync + 'static,
		mesh_desc_pool: &mut FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
		queue_family: QueueFamily,
		dimensions: [f32; 2],
	) -> Result<AutoCommandBuffer, OomError> {
		let mut cmd = AutoCommandBufferBuilder
			::secondary_graphics_one_time_submit(
				shared.shaders.target_vertices.device().clone(),
				queue_family,
				shared.subpass_gbuffers.clone()
			)?;

		for (i, mat) in self.materials.iter().enumerate() {
			cmd = cmd
				.draw_indexed(
					shared.pipeline_gbuffers.clone(),
					DynamicState {
						line_width: None,
						viewports:
							Some(vec![Viewport { origin: [0.0, 0.0], dimensions: dimensions, depth_range: 0.0..1.0 }]),
						scissors: None,
					},
					vec![self.positions.clone(), self.normals.clone(), self.texcoords_main.clone()],
					mat.indices.clone(),
					(
						camera_desc.clone(),
						mesh_desc_pool.next()
							.add_buffer(self.position.clone())
							.unwrap()
							.add_buffer(self.rotation.clone())
							.unwrap()
							.build()
							.unwrap(),
						self.material_descs[i].clone()
					),
					()
				)
				.unwrap();
		}

		Ok(cmd.build().map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?)
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

struct Material {
	indices: BufferSlice<[u32], Arc<ImmutableBuffer<[u32]>>>,
	texture1: Option<PathBuf>,
	texture2: Option<PathBuf>,
}

#[repr(C)]
struct MaterialGpu {
	light_penetration: u32,
	subsurface_scattering: u32,
	emissive_brightness: u32,
	base_color: [f32; 3],
}
