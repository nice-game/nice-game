use cpu_pool::{ spawn_cpu, spawn_fs, CpuFuture };
use futures::prelude::*;
use image::{ self, ImageError, ImageFormat };
use std::{ fs::File, io::{ self, prelude::* }, path::Path, sync::Arc };
use texture::Texture;
use vulkano::{
	OomError,
	command_buffer::{ AutoCommandBuffer, CommandBufferExecFuture },
	format::R8G8B8A8Srgb,
	image::{ Dimensions, ImageCreationError, ImageViewAccess, ImmutableImage },
	memory::DeviceMemoryAllocError,
	sync::{ FenceSignalFuture, FlushError, GpuFuture, NowFuture },
};
use window::Window;

pub struct ImmutableTexture {
	image: Arc<ImageViewAccess + Send + Sync + 'static>,
}
impl ImmutableTexture {
	pub fn from_file_with_format<P>(window: &Window, path: P, format: ImageFormat) -> TextureFuture
	where P: AsRef<Path> + Send + 'static {
		let queue = window.queue().clone();
		let future =
			spawn_fs(move |_| {
				let mut bytes = vec![];
				File::open(path)?.read_to_end(&mut bytes)?;

				Ok(spawn_cpu(move |_| {
					let img = image::load_from_memory_with_format(&bytes, format)?.to_rgba();
					let (width, height) = img.dimensions();
					let img = img.into_raw();

					let (img, future) =
						ImmutableImage::from_iter(
							img.into_iter(),
							Dimensions::Dim2d { width: width, height: height },
							R8G8B8A8Srgb,
							queue,
						)?;

					Ok(SpriteGpuData { image: img, future: future.then_signal_fence_and_flush()? })
				}))
			});

		TextureFuture { state: SpriteState::LoadingDisk(future) }
	}
}
impl Texture for ImmutableTexture {
	fn image(&self) -> &Arc<ImageViewAccess + Send + Sync + 'static> {
		&self.image
	}
}

pub struct TextureFuture {
	state: SpriteState,
}
impl Future for TextureFuture {
	type Item = ImmutableTexture;
	type Error = TextureError;

	fn poll(&mut self, cx: &mut task::Context) -> Poll<Self::Item, Self::Error> {
		let mut new_state = None;

		match &mut self.state {
			SpriteState::LoadingDisk(future) => match future.poll(cx)? {
				Async::Ready(subfuture) => new_state = Some(SpriteState::LoadingCpu(subfuture)),
				Async::Pending => return Ok(Async::Pending),
			},
			_ => (),
		}

		if let Some(new_state) = new_state.take() {
			self.state = new_state;
		}

		match &mut self.state {
			SpriteState::LoadingCpu(future) => match future.poll(cx)? {
				Async::Ready(data) => new_state = Some(SpriteState::LoadingGpu(data)),
				Async::Pending => return Ok(Async::Pending),
			},
			_ => (),
		}

		if let Some(new_state) = new_state.take() {
			self.state = new_state;
		}

		match &self.state {
			SpriteState::LoadingGpu(data) => match data.future.wait(Some(Default::default())) {
				Ok(()) => Ok(Async::Ready(ImmutableTexture { image: data.image.clone() })),
				Err(FlushError::Timeout) => {
					cx.waker().wake();
					Ok(Async::Pending)
				},
				Err(FlushError::DeviceLost) => Err(TextureError::DeviceLost),
				Err(err) => panic!("{}", err),
			},
			_ => unreachable!(),
		}
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

enum SpriteState {
	LoadingDisk(CpuFuture<CpuFuture<SpriteGpuData, TextureError>, io::Error>),
	LoadingCpu(CpuFuture<SpriteGpuData, TextureError>),
	LoadingGpu(SpriteGpuData),
}

struct SpriteGpuData {
	image: Arc<ImmutableImage<R8G8B8A8Srgb>>,
	future: FenceSignalFuture<CommandBufferExecFuture<NowFuture, AutoCommandBuffer>>,
}
