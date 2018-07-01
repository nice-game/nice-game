use atom::Atom;
use batch::mesh::MeshBatchShared;
use byteorder::{LE, ReadBytesExt};
use cgmath::{ Quaternion, Vector3 };
use cpu_pool::{ execute_future, spawn_fs, GpuFutureFuture };
use futures::prelude::*;
use std::{
	fs::File,
	io::{ self, prelude::*, SeekFrom },
	mem::{ size_of, transmute },
	path::{ Path, PathBuf },
	sync::Arc,
	vec::IntoIter as VecIntoIter,
};
use texture::{ ImageFormat, ImmutableTexture };
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
	device::{ Device, Queue },
	format::Format,
	image::ImageViewAccess,
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
	material_buf: Arc<ImmutableBuffer<[u8]>>,
	materials: Vec<Material>,
}
impl Mesh {
	pub fn from_file(
		window: &Window,
		shared: Arc<MeshBatchShared>,
		path: impl AsRef<Path> + Clone + Send + 'static,
		position: Vector3<f32>,
		rotation: Quaternion<f32>,
	) -> impl Future<Item = (Self, impl GpuFuture + Send + Sync + 'static), Error = MeshFromFileError>
	{
		let device = window.device().clone();
		let queue = window.queue().clone();
		spawn_fs(move |_| Self::from_file_impl(device, queue, shared, path, position, rotation))
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

		for mat in &self.materials {
			let desc = mat.desc.take().unwrap();

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
						desc.clone()
					),
					()
				)
				.unwrap();

			mat.desc.set_if_none(desc);
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

	fn from_file_impl(
		device: Arc<Device>,
		queue: Arc<Queue>,
		shared: Arc<MeshBatchShared>,
		path: impl AsRef<Path> + Clone + Send + 'static,
		position: Vector3<f32>,
		rotation: Quaternion<f32>,
	) -> Result<(Self, impl GpuFuture + Send + Sync + 'static), MeshFromFileError> {
		let mut file = File::open(path.clone())?;

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

		let material_buf =
			unsafe {
				CpuAccessibleBuffer::uninitialized_array(
					queue.device().clone(),
					material_count * material_stride,
					BufferUsage::transfer_source()
				)?
			};
		let mut index_counts = Vec::with_capacity(material_count);
		let mut mat_temp_datas = Vec::with_capacity(material_count);
		{
			let mut material_buf_lock = material_buf.write().unwrap();
			for i in 0..material_count {
				index_counts.push(file.read_u32::<LE>()?);
				mat_temp_datas
					.push(MaterialTextureInfo {
						texture1_name_size: file.read_u16::<LE>()?,
						texture1_name_offset: file.read_u32::<LE>()?,
						texture2_name_size: file.read_u16::<LE>()?,
						texture2_name_offset: file.read_u32::<LE>()?,
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
			}
		}

		let (material_buf, material_buf_future) =
			ImmutableBuffer::from_buffer(material_buf, BufferUsage::uniform_buffer(), queue.clone())?;

		let mut materials = Vec::with_capacity(material_count);
		let mut index_start = 0;
		for (i, index_count) in index_counts.into_iter().enumerate() {
			let index_count = index_count as usize;
			let material_offset = material_stride * i;
			materials
				.push(Material {
					indices: indices.clone().into_buffer_slice().slice(index_start..index_start + index_count).unwrap(),
					desc:
						Arc::new(Atom::new(Box::new(Arc::new(
							PersistentDescriptorSet::start(shared.pipeline_gbuffers.clone(), 2)
								.add_buffer(
									material_buf.clone()
										.into_buffer_slice()
										.slice(material_offset..material_offset + size_of::<MaterialGpu>())
										.unwrap()
								)
								.unwrap()
								.add_sampled_image(shared.shaders.white_pixel.clone(), shared.shaders.sampler.clone())
								.unwrap()
								.build()
								.unwrap()
						))))
				});

			index_start += index_count;
		}

		for (i, data) in mat_temp_datas.into_iter().enumerate() {
			if data.texture1_name_size != 0 {
				file.seek(SeekFrom::Start(data.texture1_name_offset as u64))?;
				let mut buf = vec![0; data.texture1_name_size as usize];
				file.read_exact(&mut buf)?;
				let path = path.as_ref().parent().unwrap().join(String::from_utf8(buf).unwrap());

				let desc = materials[i].desc.clone();
				let material_buf = material_buf.clone();
				let material_offset = material_stride * i;
				let pipeline_gbuffers = shared.pipeline_gbuffers.clone();
				let sampler = shared.shaders.sampler.clone();

				let future = ImmutableTexture
					::from_file_with_format_impl(queue.clone(), path.clone(), ImageFormat::TGA)
					.map_err(|err| error!("{:?}", err))
					.and_then(|(tex, future)| {
						GpuFutureFuture::new(future).map(|_| tex).map_err(|err| error!("{:?}", err))
					})
					.and_then(move |tex| {
						desc
							.swap(Box::new(Arc::new(
								PersistentDescriptorSet::start(pipeline_gbuffers.clone(), 2)
									.add_buffer(
										material_buf.clone()
											.into_buffer_slice()
											.slice(material_offset..material_offset + size_of::<MaterialGpu>())
											.unwrap()
									)
									.unwrap()
									.add_sampled_image(tex.image, sampler.clone())
									.unwrap()
									.build()
									.unwrap()
							)));
						Ok(())
					})
					.or_else::<Result<_, Never>, _>(move |err| { error!("{:?}: {:?}", path, err); Ok(()) });
				execute_future(future);
			}

			if data.texture2_name_size != 0 {
				file.seek(SeekFrom::Start(data.texture2_name_offset as u64))?;
				let mut buf = vec![0; data.texture2_name_size as usize];
				file.read_exact(&mut buf)?;
				let path = path.as_ref().parent().unwrap().join(String::from_utf8(buf).unwrap());

				let desc = materials[i].desc.clone();
				let material_buf = material_buf.clone();
				let material_offset = material_stride * i;
				let pipeline_gbuffers = shared.pipeline_gbuffers.clone();
				let sampler = shared.shaders.sampler.clone();

				let future = ImmutableTexture
					::from_file_with_format_impl(queue.clone(), path.clone(), ImageFormat::TGA)
					.map_err(|err| error!("{:?}", err))
					.and_then(|(tex, future)| {
						GpuFutureFuture::new(future).map(|_| tex).map_err(|err| error!("{:?}", err))
					})
					.and_then(move |tex| {
						desc
							.swap(Box::new(Arc::new(
								PersistentDescriptorSet::start(pipeline_gbuffers.clone(), 2)
									.add_buffer(
										material_buf.clone()
											.into_buffer_slice()
											.slice(material_offset..material_offset + size_of::<MaterialGpu>())
											.unwrap()
									)
									.unwrap()
									.add_sampled_image(tex.image, sampler.clone())
									.unwrap()
									.build()
									.unwrap()
							)));
						Ok(())
					})
					.or_else::<Result<_, Never>, _>(move |err| { error!("{:?}: {:?}", path, err); Ok(()) });
				execute_future(future);
			}
		}

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
				material_buf: material_buf,
				materials: materials,
			},
			positions_future
				.join(normals_future)
				.join(texcoords_main_future)
				.join(indices_future)
				.join(material_buf_future)
		))
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
	desc: Arc<Atom<Box<Arc<DescriptorSet + Sync + Send + 'static>>>>,
}

struct MaterialTextureInfo {
	texture1_name_size: u16,
	texture1_name_offset: u32,
	texture2_name_size: u16,
	texture2_name_offset: u32,
}

#[repr(C)]
struct MaterialGpu {
	light_penetration: u32,
	subsurface_scattering: u32,
	emissive_brightness: u32,
	base_color: [f32; 3],
}
