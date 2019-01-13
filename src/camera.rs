use crate::window::Window;
use cgmath::{ vec4, Quaternion, Vector3, Vector4 };
use std::{ f32::consts::PI, sync::Arc };
use vulkano::{
	buffer::{ CpuBufferPool, cpu_pool::CpuBufferPoolSubbuffer },
	memory::{ DeviceMemoryAllocError, pool::StdMemoryPool },
};

pub struct Camera {
	position_pool: CpuBufferPool<Vector3<f32>>,
	rotation_pool: CpuBufferPool<Quaternion<f32>>,
	projection_pool: CpuBufferPool<Vector4<f32>>,
	pub(crate) position_buffer: CpuBufferPoolSubbuffer<Vector3<f32>, Arc<StdMemoryPool>>,
	pub(crate) rotation_buffer: CpuBufferPoolSubbuffer<Quaternion<f32>, Arc<StdMemoryPool>>,
	pub(crate) projection_buffer: CpuBufferPoolSubbuffer<Vector4<f32>, Arc<StdMemoryPool>>,
}
impl Camera {
	pub fn new(
		window: &Window,
		position: Vector3<f32>,
		rotation: Quaternion<f32>,
		aspect: f32,
		fovx: f32,
		znear: f32,
		zfar: f32,
	) -> Result<Self, DeviceMemoryAllocError> {
		let position_pool = CpuBufferPool::uniform_buffer(window.device().device().clone());
		let rotation_pool = CpuBufferPool::uniform_buffer(window.device().device().clone());
		let projection_pool = CpuBufferPool::uniform_buffer(window.device().device().clone());

		let position_buffer = position_pool.next(position)?;
		let rotation_buffer = rotation_pool.next(rotation)?;
		let projection_buffer = projection_pool.next(Self::projection(aspect, fovx, znear, zfar))?;

		Ok(Self {
			position_pool: position_pool,
			rotation_pool: rotation_pool,
			projection_pool: projection_pool,
			position_buffer: position_buffer,
			rotation_buffer: rotation_buffer,
			projection_buffer: projection_buffer,
		})
	}

	pub fn set_position(&mut self, position: Vector3<f32>) -> Result<(), DeviceMemoryAllocError> {
		self.position_buffer = self.position_pool.next(position)?;
		Ok(())
	}

	pub fn set_projection(
		&mut self,
		aspect: f32,
		fovx: f32,
		znear: f32,
		zfar: f32
	) -> Result<(), DeviceMemoryAllocError> {
		self.projection_buffer = self.projection_pool.next(Self::projection(aspect, fovx, znear, zfar))?;
		Ok(())
	}

	pub fn set_rotation(&mut self, rotation: Quaternion<f32>) -> Result<(), DeviceMemoryAllocError> {
		self.rotation_buffer = self.rotation_pool.next(rotation)?;
		Ok(())
	}

	fn projection(aspect: f32, fovx: f32, znear: f32, zfar: f32) -> Vector4<f32> {
		let f = 1.0 / (fovx * (PI / 360.0)).tan();
		vec4(f / aspect, f, (zfar + znear) / (znear - zfar), 2.0 * zfar * znear / (znear - zfar))
	}
}
