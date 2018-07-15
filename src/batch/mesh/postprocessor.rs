use texture::TargetTexture;
use vulkano::{ device::Device, format::Format, memory::DeviceMemoryAllocError };
use std::sync::Arc;

pub struct Postprocessor {
	targets: [TargetTexture; 2],
	active_target: bool,
}
impl Postprocessor {
	pub fn new(device: Arc<Device>, format: Format, dimensions: [u32; 2]) -> Result<Self, DeviceMemoryAllocError> {
		Ok(Self {
			targets:
				[
					TargetTexture::new_impl(device.clone(), format, dimensions)?,
					TargetTexture::new_impl(device, format, dimensions)?,
				],
			active_target: false,
		})
	}

	pub fn draw(&mut self) {
		self.active_target = !self.active_target;
	}
}
