use crate::batch::mesh::{ MeshRenderPass, mesh::{ Material, MaterialTextureInfo, MaterialUniform, Mesh, MeshFromFileError } };
use crate::cpu_pool::{ execute_future, GpuFutureFuture };
use crate::texture::{ ImageFormat, ImmutableTexture, Texture };
use atom::Atom;
use byteorder::{LE, ReadBytesExt};
use cgmath::{ Quaternion, Vector3 };
use futures::{ FutureExt, future::ready, prelude::* };
use log::{ debug, log };
use std::{ fs::File, io::{ self, prelude::*, SeekFrom }, mem::{ size_of, transmute }, path::{ Path }, sync::Arc };
use vulkano::{
	buffer::{ BufferAccess, BufferUsage, CpuAccessibleBuffer, CpuBufferPool, ImmutableBuffer },
	descriptor::descriptor_set::PersistentDescriptorSet,
	device::{ Device, Queue },
	sync::GpuFuture,
};

pub fn from_nice_model(
	device: Arc<Device>,
	queue: Arc<Queue>,
	render_pass: Arc<MeshRenderPass>,
	path: impl AsRef<Path> + Clone + Send + 'static,
	position: Vector3<f32>,
	rotation: Quaternion<f32>,
) -> Result<(Mesh, impl GpuFuture + Send + Sync + 'static), MeshFromFileError> {
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
		buffer_from_file(
			queue.clone(),
			BufferUsage::vertex_buffer(),
			vertex_count,
			&mut || Ok([file.read_f32::<LE>()?, file.read_f32::<LE>()?, file.read_f32::<LE>()?])
		)?;

	file.seek(SeekFrom::Start(normals_offset))?;
	let (normals, normals_future) =
		buffer_from_file(
			queue.clone(),
			BufferUsage::vertex_buffer(),
			vertex_count,
			&mut || Ok([file.read_f32::<LE>()?, file.read_f32::<LE>()?, file.read_f32::<LE>()?])
		)?;

	file.seek(SeekFrom::Start(texcoords_main_offset))?;
	let (texcoords_main, texcoords_main_future) =
		buffer_from_file(
			queue.clone(),
			BufferUsage::vertex_buffer(),
			vertex_count,
			&mut || Ok([file.read_f32::<LE>()?, file.read_f32::<LE>()?])
		)?;

	file.seek(SeekFrom::Start(indices_offset))?;
	let (indices, indices_future) =
		buffer_from_file(
			queue.clone(),
			BufferUsage::index_buffer(),
			index_count,
			&mut || file.read_u32::<LE>()
		)?;

	file.seek(SeekFrom::Start(materials_offset))?;

	// round MaterialUniform size up to minimum alignment
	let mut material_stride =
		queue.device().physical_device().limits().min_uniform_buffer_offset_alignment() as usize;
	material_stride = (size_of::<MaterialUniform>() + material_stride - 1) / material_stride * material_stride;
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

			material_buf_lock[i * material_stride..i * material_stride + size_of::<MaterialUniform>()]
				.copy_from_slice(
					&unsafe {
						transmute::<_, [u8; size_of::<MaterialUniform>()]>(
							MaterialUniform {
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
						PersistentDescriptorSet::start(render_pass.pipeline_gbuffers.clone(), 2)
							.add_buffer(
								material_buf.clone()
									.into_buffer_slice()
									.slice(material_offset..material_offset + size_of::<MaterialUniform>())
									.unwrap()
							)
							.unwrap()
							.add_sampled_image(render_pass.shaders.texture1_default.clone(), render_pass.shaders.sampler.clone())
							.unwrap()
							.add_sampled_image(render_pass.shaders.texture2_default.clone(), render_pass.shaders.sampler.clone())
							.unwrap()
							.build()
							.unwrap()
					))))
			});

		index_start += index_count;
	}

	for (i, data) in mat_temp_datas.into_iter().enumerate() {
		let texture1_default = render_pass.shaders.texture1_default.clone();
		let future1: Box<Future<Output = _> + Send + Unpin> =
			if data.texture1_name_size != 0 {
				file.seek(SeekFrom::Start(data.texture1_name_offset as u64))?;
				let mut buf = vec![0; data.texture1_name_size as usize];
				file.read_exact(&mut buf)?;
				let path = path.as_ref().parent().unwrap().join(String::from_utf8(buf).unwrap());

				Box::new(
					ImmutableTexture
						::from_file_with_format_impl(queue.clone(), path.clone(), ImageFormat::PNG, true)
						.map(|result| result
							.map(|(tex, future)| {
								GpuFutureFuture::new(future).map(|_| tex.image().clone()).unwrap()
							})
							.unwrap_or_else(move |_| texture1_default)
						)
				)
			} else {
				Box::new(ready(texture1_default))
			};

		let texture2_default = render_pass.shaders.texture2_default.clone();
		let future2: Box<Future<Output = _> + Send + Unpin> =
			if data.texture2_name_size != 0 {
				file.seek(SeekFrom::Start(data.texture2_name_offset as u64))?;
				let mut buf = vec![0; data.texture2_name_size as usize];
				file.read_exact(&mut buf)?;
				let path = path.as_ref().parent().unwrap().join(String::from_utf8(buf).unwrap());

				Box::new(
					ImmutableTexture
						::from_file_with_format_impl(queue.clone(), path.clone(), ImageFormat::PNG, false)
						.map(|result| result
							.map(|(tex, future)| {
								GpuFutureFuture::new(future).map(|_| tex.image().clone()).unwrap()
							})
							.unwrap_or_else(move |_| texture2_default)
						)
				)
			} else {
				Box::new(ready(texture2_default))
			};

		let desc = materials[i].desc.clone();
		let material_buf = material_buf.clone();
		let material_offset = material_stride * i;
		let pipeline_gbuffers = render_pass.pipeline_gbuffers.clone();
		let sampler = render_pass.shaders.sampler.clone();

		let future = FutureExt::join(future1, future2)
			.map(move |(tex1, tex2)| {
				desc
					.swap(Box::new(Arc::new(
						PersistentDescriptorSet::start(pipeline_gbuffers.clone(), 2)
							.add_buffer(
								material_buf.clone()
									.into_buffer_slice()
									.slice(material_offset..material_offset + size_of::<MaterialUniform>())
									.unwrap()
							)
							.unwrap()
							.add_sampled_image(tex1, sampler.clone())
							.unwrap()
							.add_sampled_image(tex2, sampler.clone())
							.unwrap()
							.build()
							.unwrap()
					)));
			});

		execute_future(future);
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
			materials: materials,
		},
		positions_future
			.join(normals_future)
			.join(texcoords_main_future)
			.join(indices_future)
			.join(material_buf_future)
	))
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
