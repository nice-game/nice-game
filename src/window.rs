pub use winit::{ Event, WindowEvent, WindowId };

use { Context, Drawable, ObjectIdRoot, RenderTarget };
use std::{ collections::HashMap, iter::Iterator, sync::{ Arc, atomic::{ AtomicBool, Ordering } }};
use vulkano::{
	device::{ Device, DeviceExtensions, Queue },
	format::Format,
	image::ImageViewAccess,
	instance::{ Features, PhysicalDevice },
	swapchain::{
		acquire_next_image,
		AcquireError,
		PresentMode,
		Surface,
		SurfaceTransform,
		Swapchain,
		SwapchainCreationError
	},
	sync::{ now, FlushError, GpuFuture },
};
use vulkano_win::VkSurfaceBuild;
use winit;

pub struct EventsLoop {
	events: winit::EventsLoop,
	resized: HashMap<WindowId, Arc<AtomicBool>>,
}
impl EventsLoop {
	pub fn new() -> Self {
		Self { events: winit::EventsLoop::new(), resized: HashMap::new() }
	}

	pub fn poll_events(&mut self, mut callback: impl FnMut(Event)) {
		let resized = &mut self.resized;
		self.events.poll_events(|event| {
			match event {
				Event::WindowEvent { event: WindowEvent::Closed, window_id } => {
					resized.remove(&window_id);
				},
				Event::WindowEvent { event: WindowEvent::Resized(_, _), window_id } => {
					resized[&window_id].store(true, Ordering::Relaxed);
				},
				_ => (),
			}

			callback(event);
		});
	}
}

pub struct Window {
	surface: Arc<Surface<winit::Window>>,
	device: Arc<Device>,
	queue: Arc<Queue>,
	swapchain: Arc<Swapchain<winit::Window>>,
	images: Vec<Arc<ImageViewAccess + Send + Sync + 'static>>,
	previous_frame_end: Option<Box<GpuFuture>>,
	resized: Arc<AtomicBool>,
	id_root: ObjectIdRoot,
}
impl Window {
	pub fn new<T: Into<String>>(ctx: &Context, events: &mut EventsLoop, title: T) -> Self {
		let pdevice = PhysicalDevice::enumerate(&ctx.instance).next().expect("no device available");
		println!("Using device: {} (type: {:?})", pdevice.name(), pdevice.ty());

		let surface = winit::WindowBuilder::new()
			.with_title(title)
			.build_vk_surface(&events.events, ctx.instance.clone())
			.expect("failed to create window");

		let qfam = pdevice.queue_families()
			.find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap())
			.expect("failed to find a graphical queue family");

		let (device, mut queues) = Device::new(
			pdevice,
			&Features::none(),
			&DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() },
			[(qfam, 1.0)].iter().cloned()
		).expect("failed to create device");
		let queue = queues.next().unwrap();

		let (swapchain, images) = {
			let caps = surface.capabilities(pdevice).expect("failed to get surface capabilities");
			Swapchain::new(
				device.clone(),
				surface.clone(),
				caps.min_image_count,
				caps.supported_formats[0].0,
				caps.current_extent.unwrap_or(surface.window().get_inner_size().map(|(x, y)| [x, y]).unwrap()),
				1,
				caps.supported_usage_flags,
				&queue,
				SurfaceTransform::Identity,
				caps.supported_composite_alpha.iter().next().unwrap(),
				PresentMode::Fifo,
				true,
				None
			).expect("failed to create swapchain")
		};
		let images = images.into_iter().map(|x| x as _).collect();

		let previous_frame_end = Some(Box::new(now(device.clone())) as Box<GpuFuture>);

		let resized = Arc::<AtomicBool>::default();
		events.resized.insert(surface.window().id(), resized.clone());

		Self {
			surface: surface,
			device: device,
			queue: queue,
			swapchain: swapchain,
			images: images,
			previous_frame_end: previous_frame_end,
			resized: resized,
			id_root: ObjectIdRoot::new(),
		}
	}

	pub fn present<'a>(&mut self, drawables: &mut [&'a mut Drawable]) {
		if self.resized.swap(false, Ordering::Relaxed) {
			let dimensions = self.surface.capabilities(self.device.physical_device())
				.expect("failed to get surface capabilities")
				.current_extent
				.unwrap_or(self.surface.window().get_inner_size().map(|(x, y)| [x, y]).unwrap());

			let (swapchain, images) = match self.swapchain.recreate_with_dimension(dimensions) {
				Ok(ret) => ret,
				Err(SwapchainCreationError::UnsupportedDimensions) => {
					self.resized.store(true, Ordering::Relaxed);
					return;
				},
				Err(err) => panic!("{:?}", err),
			};
			let images = images.into_iter().map(|x| x as _).collect();

			self.swapchain = swapchain;
			self.images = images;
		}

		let (image_num, acquire_future) = match acquire_next_image(self.swapchain.clone(), None) {
			Ok(val) => val,
			Err(AcquireError::OutOfDate) => {
				self.resized.store(true, Ordering::Relaxed);
				return;
			},
			Err(err) => panic!("{:?}", err)
		};

		let mut future = if let Some(mut future) = self.previous_frame_end.take() {
			future.cleanup_finished();
			Box::new(future.join(acquire_future)) as Box<GpuFuture>
		} else {
			Box::new(acquire_future)
		};
		for drawable in drawables {
			let commands = drawable.commands(&self.id_root, self.queue.family(), image_num, &self.images[image_num]);
			future = Box::new(future.then_execute(self.queue.clone(), commands).unwrap());
		}
		let future = future.then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_num)
			.then_signal_fence_and_flush();
		self.previous_frame_end = match future {
			Ok(future) => Some(Box::new(future)),
			Err(FlushError::OutOfDate) => {
				self.resized.store(true, Ordering::Relaxed);
				return;
			},
			Err(err) => panic!(err),
		};
	}

	pub fn device(&self) -> &Arc<Device> {
		&self.device
	}

	pub fn format(&self) -> Format {
		self.swapchain.format()
	}

	pub fn queue(&self) -> &Arc<Queue> {
		&self.queue
	}
}
impl RenderTarget for Window {
	fn id_root(&self) -> &ObjectIdRoot {
		&self.id_root
	}

	fn image_count(&self) -> usize {
		self.images.len()
	}
}
