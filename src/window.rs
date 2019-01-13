pub use winit::{ Event, MouseButton, MouseCursor, WindowEvent, WindowId, dpi::{ LogicalPosition, LogicalSize } };

use crate::{ ObjectIdRoot, RenderTarget };
use crate::device::DeviceCtx;
use std::{ iter::Iterator, sync::{ Arc, atomic::{ AtomicBool, Ordering } }};
use vulkano::{
	format::Format,
	image::ImageViewAccess,
	memory::DeviceMemoryAllocError,
	swapchain::{
		acquire_next_image,
		AcquireError,
		PresentMode,
		Surface,
		SurfaceTransform,
		Swapchain,
		SwapchainCreationError
	},
	sync::{ FlushError, GpuFuture },
};
use winit;

pub struct Window {
	surface: Arc<Surface<winit::Window>>,
	device: Arc<DeviceCtx>,
	swapchain: Arc<Swapchain<winit::Window>>,
	images: Vec<Arc<ImageViewAccess + Send + Sync + 'static>>,
	previous_frame_end: Option<Box<GpuFuture>>,
	resized: Arc<AtomicBool>,
	id_root: ObjectIdRoot,
}
impl Window {
	pub fn join_future(&mut self, future: impl GpuFuture + 'static) {
		if let Some(previous_frame_end) = self.previous_frame_end.take() {
			self.previous_frame_end = Some(Box::new(previous_frame_end.join(future)));
		} else {
			self.previous_frame_end = Some(Box::new(future));
		}
	}

	pub fn present<F>(
		&mut self,
		get_commands: impl FnOnce(&mut Self, usize, Box<GpuFuture>) -> F
	) -> Result<(), DeviceMemoryAllocError>
	where
		F: GpuFuture + 'static
	{
		if self.resized.swap(false, Ordering::Relaxed) {
			let dimensions = self.surface.capabilities(self.device.device().physical_device())
				.expect("failed to get surface capabilities")
				.current_extent
				.unwrap_or(
					self.surface.window()
						.get_inner_size()
						.map(|size| {
							let size: (u32, u32) = size.into();
							[size.0, size.1]
						})
						.unwrap()
				);

			let (swapchain, images) =
				match self.swapchain.recreate_with_dimension(dimensions) {
					Ok(ret) => ret,
					Err(SwapchainCreationError::UnsupportedDimensions) => {
						self.resized.store(true, Ordering::Relaxed);
						return Ok(());
					},
					Err(err) => unreachable!(err),
				};

			self.swapchain = swapchain;
			self.images = images.into_iter().map(|x| x as _).collect();
		}

		let (image_num, acquire_future) =
			match acquire_next_image(self.swapchain.clone(), None) {
				Ok(val) => val,
				Err(AcquireError::OutOfDate) => {
					self.resized.store(true, Ordering::Relaxed);
					return Ok(());
				},
				Err(err) => unreachable!(err)
			};

		let mut future: Box<GpuFuture> =
			if let Some(mut future) = self.previous_frame_end.take() {
				future.cleanup_finished();
				Box::new(future.join(acquire_future))
			} else {
				Box::new(acquire_future)
			};
		future = Box::new(get_commands(self, image_num, future));
		let future = future.then_swapchain_present(self.device.queue().clone(), self.swapchain.clone(), image_num)
			.then_signal_fence_and_flush();
		self.previous_frame_end =
			match future {
				Ok(future) => Some(Box::new(future)),
				Err(FlushError::OutOfDate) => {
					self.resized.store(true, Ordering::Relaxed);
					return Ok(());
				},
				Err(err) => unreachable!(err),
			};

		Ok(())
	}

	pub fn get_inner_size(&self) -> Option<LogicalSize> {
		self.surface.window().get_inner_size()
	}

	pub fn set_cursor(&self, cursor: MouseCursor) {
		self.surface.window().set_cursor(cursor)
	}

	pub fn set_cursor_position(&self, pos: LogicalPosition) -> Result<(), String> {
		self.surface.window().set_cursor_position(pos)
	}

	pub fn device(&self) -> &Arc<DeviceCtx> {
		&self.device
	}

	pub(crate) fn new(surface: Arc<Surface<winit::Window>>, device: Arc<DeviceCtx>, resized: Arc<AtomicBool>) -> Self {
		let (swapchain, images) = {
			let caps = surface.capabilities(device.device().physical_device()).expect("failed to get surface capabilities");
			Swapchain::new(
				device.device().clone(),
				surface.clone(),
				caps.min_image_count,
				Format::B8G8R8A8Srgb,
				caps.current_extent
					.unwrap_or(
						surface.window()
							.get_inner_size()
							.map(|size| {
								let size: (u32, u32) = size.into();
								[size.0, size.1]
							})
							.unwrap()
					),
				1,
				caps.supported_usage_flags,
				device.queue(),
				SurfaceTransform::Identity,
				caps.supported_composite_alpha.iter().next().unwrap(),
				PresentMode::Fifo,
				true,
				None
			).expect("failed to create swapchain")
		};
		let images = images.into_iter().map(|x| x as _).collect();

		Self {
			surface: surface,
			device: device,
			swapchain: swapchain,
			images: images,
			previous_frame_end: None,
			resized: resized,
			id_root: ObjectIdRoot::new(),
		}
	}
}
impl RenderTarget for Window {
	fn format(&self) -> Format {
		self.swapchain.format()
	}

	fn id_root(&self) -> &ObjectIdRoot {
		&self.id_root
	}

	fn images(&self) -> &[Arc<ImageViewAccess + Send + Sync + 'static>] {
		&self.images
	}
}
