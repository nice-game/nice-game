use cgmath::{ Quaternion, Vector3 };
use std::sync::Arc;
use vulkano::{
	buffer::{ CpuBufferPool, cpu_pool::CpuBufferPoolSubbuffer },
	memory::{ DeviceMemoryAllocError, pool::StdMemoryPool }
};
use window::Window;

pub struct Camera {
	position_pool: CpuBufferPool<Vector3<f32>>,
	rotation_pool: CpuBufferPool<Quaternion<f32>>,
	position_buffer: CpuBufferPoolSubbuffer<Vector3<f32>, Arc<StdMemoryPool>>,
	rotation_buffer: CpuBufferPoolSubbuffer<Quaternion<f32>, Arc<StdMemoryPool>>,
}
impl Camera {
	pub fn new(window: &Window, position: Vector3<f32>, rotation: Quaternion<f32>) -> Result<Self, DeviceMemoryAllocError> {
		let position_pool = CpuBufferPool::uniform_buffer(window.device().clone());
		let rotation_pool = CpuBufferPool::uniform_buffer(window.device().clone());

		let position_buffer = position_pool.next(position)?;
		let rotation_buffer = rotation_pool.next(rotation)?;

		Ok(Self {
			position_pool: position_pool,
			rotation_pool: rotation_pool,
			position_buffer: position_buffer,
			rotation_buffer: rotation_buffer,
		})
	}

	pub fn set_position(&mut self, position: Vector3<f32>) -> Result<(), DeviceMemoryAllocError> {
		self.position_buffer = self.position_pool.next(position)?;
		Ok(())
	}

	pub fn set_rotation(&mut self, rotation: Quaternion<f32>) -> Result<(), DeviceMemoryAllocError> {
		self.rotation_buffer = self.rotation_pool.next(rotation)?;
		Ok(())
	}
}
