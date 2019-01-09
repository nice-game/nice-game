use cpu_pool::{ spawn_cpu, spawn_fs };
use futures::prelude::*;
use image::{ self, ImageError, ImageFormat };
use std::{ fs::File, io::{ self, prelude::* }, path::Path, sync::Arc };
use texture::Texture;
use vulkano::{
	OomError,
	device::Queue,
	format::{ AcceptsPixels, Format },
	image::{ Dimensions, ImageCreationError, ImageViewAccess, ImmutableImage },
	memory::DeviceMemoryAllocError,
	sync::{ FlushError, GpuFuture },
};
use window::Window;

#[derive(Clone)]
pub struct ImmutableTexture {
	image: Arc<ImageViewAccess + Send + Sync + 'static>,
}
impl ImmutableTexture {
	pub fn from_data<I, P>(window: &Window, data: I) -> Result<(Self, impl GpuFuture), TextureError>
	where I: ExactSizeIterator<Item = P>, P: Send + Sync + Clone + 'static, Format: AcceptsPixels<P> {
		let (image, future) =
			ImmutableImage::from_iter(
				data,
				Dimensions::Dim2d { width: 1, height: 1 },
				Format::R8G8B8A8Unorm,
				window.device().queue().clone(),
			)?;

		Ok((Self { image: image }, future))
	}

	pub fn from_file_with_format<P>(
		window: &Window,
		path: P,
		format: ImageFormat,
		srgb: bool,
	) -> impl Future<Output = Result<(Self, impl GpuFuture), TextureError>>
	where P: AsRef<Path> + Send + 'static {
		Self::from_file_with_format_impl(window.device().queue().clone(), path, format, srgb)
	}

	pub(crate) fn from_file_with_format_impl<P>(
		queue: Arc<Queue>,
		path: P,
		format: ImageFormat,
		srgb: bool,
	) -> impl Future<Output = Result<(Self, impl GpuFuture), TextureError>>
	where P: AsRef<Path> + Send + 'static {
		spawn_fs(|| {
			let mut bytes = vec![];
			File::open(path)?.read_to_end(&mut bytes)?;
			Ok(bytes)
		})
			.then(move |bytes: Result<Vec<u8>, io::Error>| spawn_cpu(move || {
				let bytes = bytes?;
				let img = image::load_from_memory_with_format(&bytes, format)?.to_rgba();
				let (width, height) = img.dimensions();
				let img = img.into_raw();

				let (img, future) =
					ImmutableImage::from_iter(
						img.into_iter(),
						Dimensions::Dim2d { width: width, height: height },
						if srgb { Format::R8G8B8A8Srgb } else { Format::R8G8B8A8Unorm },
						queue,
					)?;

				Ok((Self { image: img }, future))
			}))
	}

	pub(crate) fn from_image(image: Arc<ImageViewAccess + Send + Sync + 'static>) -> Self {
		Self { image: image }
	}
}
impl Texture for ImmutableTexture {
	fn image(&self) -> &Arc<ImageViewAccess + Send + Sync + 'static> {
		&self.image
	}
}

#[derive(Debug)]
pub enum TextureError {
	IoError(io::Error),
	ImageError(ImageError),
	DeviceLost,
	DeviceMemoryAllocError(DeviceMemoryAllocError),
	OomError(OomError),
}
impl From<FlushError> for TextureError {
	fn from(val: FlushError) -> Self {
		match val {
			FlushError::OomError(err) => TextureError::OomError(err),
			_ => unreachable!(),
		}
	}
}
impl From<ImageCreationError> for TextureError {
	fn from(val: ImageCreationError) -> Self {
		match val {
			ImageCreationError::AllocError(err) => TextureError::DeviceMemoryAllocError(err),
			_ => unreachable!(),
		}
	}
}
impl From<ImageError> for TextureError {
	fn from(val: ImageError) -> Self {
		TextureError::ImageError(val)
	}
}
impl From<io::Error> for TextureError {
	fn from(val: io::Error) -> Self {
		TextureError::IoError(val)
	}
}
