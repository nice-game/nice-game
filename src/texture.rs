pub use image::ImageFormat;
use { CPU_POOL, FS_POOL };
use cpu_pool::CpuFuture;
use futures::prelude::*;
use image;
use std::{ fs::File, io::prelude::*, path::Path, sync::Arc };
use vulkano::{
	command_buffer::{ AutoCommandBuffer, CommandBufferExecFuture },
	device::Queue,
	format::R8G8B8A8Srgb,
	image::{ Dimensions, immutable::ImmutableImage },
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
				File::open(path).unwrap().read_to_end(&mut bytes).unwrap();

				let future = CPU_POOL.lock().unwrap()
					.dispatch(move |_| {
						let img = image::load_from_memory_with_format(&bytes, format).unwrap().to_rgba();
						let (width, height) = img.dimensions();
						let img = img.into_raw();

						let (img, img_future) =
							ImmutableImage::from_iter(
								img.into_iter(),
								Dimensions::Dim2d { width: width, height: height },
								R8G8B8A8Srgb,
								queue.clone(),
							)
							.unwrap();

						Ok(SpriteGpuData {
							image: img,
							future: img_future.then_signal_fence_and_flush().unwrap(),
						})
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
	type Error = ();

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
				Err(err) => panic!("{}", err),
			},
			_ => unreachable!(),
		}
	}
}

enum SpriteState {
	LoadingDisk(CpuFuture<CpuFuture<SpriteGpuData, ()>, ()>),
	LoadingCpu(CpuFuture<SpriteGpuData, ()>),
	LoadingGpu(SpriteGpuData),
}

struct SpriteGpuData {
	image: Arc<ImmutableImage<R8G8B8A8Srgb>>,
	future: FenceSignalFuture<CommandBufferExecFuture<NowFuture, AutoCommandBuffer>>,
}
