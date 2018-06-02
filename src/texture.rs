pub use image::ImageFormat;
use { CPU_POOL, FS_POOL };
use cpu_pool::CpuFuture;
use futures::prelude::*;
use image::{ self, ImageError };
use std::{ fs::File, io::{ self, prelude::* }, path::Path, sync::Arc };
use vulkano::{
	OomError,
	command_buffer::{ AutoCommandBuffer, CommandBufferExecFuture },
	device::Queue,
	format::R8G8B8A8Srgb,
	image::{ Dimensions, ImageCreationError, ImmutableImage },
	memory::DeviceMemoryAllocError,
	sync::{ FenceSignalFuture, FlushError, GpuFuture, NowFuture },
};

pub struct Texture {
	image: Arc<ImmutableImage<R8G8B8A8Srgb>>,
}
impl Texture {
	pub fn from_file_with_format<P>(queue: Arc<Queue>, path: P, format: ImageFormat) -> TextureFuture
	where P: AsRef<Path> + Send + 'static {
		let future = FS_POOL.lock().unwrap()
			.dispatch(move |_| {
				let mut bytes = vec![];
				File::open(path)?.read_to_end(&mut bytes)?;

				let future = CPU_POOL.lock().unwrap()
					.dispatch(move |_| {
						let img = image::load_from_memory_with_format(&bytes, format)?.to_rgba();
						let (width, height) = img.dimensions();
						let img = img.into_raw();

						let (img, future) =
							ImmutableImage::from_iter(
								img.into_iter(),
								Dimensions::Dim2d { width: width, height: height },
								R8G8B8A8Srgb,
								queue.clone(),
							)?;

						Ok(SpriteGpuData { image: img, future: future.then_signal_fence_and_flush()? })
					});

				Ok(future)
			});

		TextureFuture { state: SpriteState::LoadingDisk(future) }
	}

	pub(super) fn image(&self) -> &Arc<ImmutableImage<R8G8B8A8Srgb>> {
		&self.image
	}
}

pub struct TextureFuture {
	state: SpriteState,
}
impl Future for TextureFuture {
	type Item = Texture;
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
				Ok(()) => Ok(Async::Ready(Texture { image: data.image.clone() })),
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
