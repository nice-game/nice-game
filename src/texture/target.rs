use { ObjectIdRoot, RenderTarget };
use std::sync::Arc;
use texture::Texture;
use vulkano::{
	device::Queue,
	format::Format,
	image::{ AttachmentImage, ImageCreationError, ImageViewAccess },
	memory::DeviceMemoryAllocError,
};
use window::Window;

pub struct TargetTexture {
	image: [Arc<ImageViewAccess + Send + Sync + 'static>; 1],
	id_root: ObjectIdRoot,
	queue: Arc<Queue>,
}
impl TargetTexture {
	pub fn new(window: &Window, dimensions: [u32; 2]) -> Result<Self, DeviceMemoryAllocError> {
		AttachmentImage::sampled(window.device().clone(), dimensions, window.format())
			.map(|image| Self {
				image: [image],
				id_root: ObjectIdRoot::new(),
				queue: window.queue().clone()
			})
			.map_err(|err| match err { ImageCreationError::AllocError(err) => err, _ => unreachable!() })
	}
}
impl RenderTarget for TargetTexture {
	fn format(&self) -> Format {
		self.image[0].format()
	}

	fn id_root(&self) -> &ObjectIdRoot {
		&self.id_root
	}

	fn images(&self) -> &[Arc<ImageViewAccess + Send + Sync + 'static>] {
		&self.image
	}

	fn queue(&self) -> &Arc<Queue> {
		&self.queue
	}
}
impl Texture for TargetTexture {
	fn image(&self) -> &Arc<ImageViewAccess + Send + Sync + 'static> {
		&self.image[0]
	}
}
