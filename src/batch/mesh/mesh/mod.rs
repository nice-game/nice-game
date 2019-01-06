mod codec;

use atom::Atom;
use batch::mesh::MeshRenderPass;
use cgmath::{ Quaternion, Vector3 };
use cpu_pool::spawn_fs;
use futures::prelude::*;
use std::{ io, mem::size_of, path::Path, sync::Arc, vec::IntoIter as VecIntoIter, };
use vulkano::{
	OomError,
	buffer::{ BufferAccess, BufferSlice, CpuBufferPool, ImmutableBuffer, cpu_pool::CpuBufferPoolSubbuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, DynamicState },
	descriptor::{ DescriptorSet, descriptor_set::FixedSizeDescriptorSetsPool },
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
}
impl Mesh {
	pub fn from_file(
		window: &Window,
		render_pass: Arc<MeshRenderPass>,
		path: impl AsRef<Path> + Clone + Send + 'static,
		position: Vector3<f32>,
		rotation: Quaternion<f32>,
	) -> impl Future<Item = (Self, impl GpuFuture + Send + Sync + 'static), Error = MeshFromFileError>
	{
		let device = window.device().device().clone();
		let queue = window.device().queue().clone();
		spawn_fs(move |_| codec::from_nice_model(device, queue, render_pass, path, position, rotation))
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
		render_pass: &MeshRenderPass,
		camera_desc: impl DescriptorSet + Clone + Send + Sync + 'static,
		mesh_desc_pool: &mut FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>,
		queue_family: QueueFamily,
		dimensions: [f32; 2],
	) -> Result<AutoCommandBuffer, OomError> {
		let mut cmd = AutoCommandBufferBuilder
			::secondary_graphics_one_time_submit(
				render_pass.shaders.target_vertices.device().clone(),
				queue_family,
				render_pass.subpass_gbuffers.clone()
			)?;

		let state =
			DynamicState {
				line_width: None,
				viewports: Some(vec![Viewport { origin: [0.0, 0.0], dimensions: dimensions, depth_range: 0.0..1.0 }]),
				scissors: None,
			};

		for mat in &self.materials {
			let desc = mat.desc.take().unwrap();

			cmd = cmd
				.draw_indexed(
					render_pass.pipeline_gbuffers.clone(),
					&state,
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
struct MaterialUniform {
	light_penetration: u32,
	subsurface_scattering: u32,
	emissive_brightness: u32,
	base_color: [f32; 3],
}
