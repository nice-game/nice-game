#![feature(proc_macro_gen)]

extern crate atom;
extern crate byteorder;
extern crate cgmath;
extern crate futures;
extern crate image;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
extern crate num_cpus;
extern crate rusttype;
#[macro_use]
extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

pub mod camera;
pub mod cpu_pool;
pub mod batch;
pub mod device;
pub mod texture;
pub mod window;

pub use vulkano::{ command_buffer::CommandBuffer, instance::Version, sync::GpuFuture };

use device::DeviceCtx;
use std::{ collections::HashMap, sync::{ Arc, Weak, atomic::{ AtomicBool, Ordering } } };
use vulkano::{
	device::{ Device, DeviceExtensions, Features },
	format::Format,
	framebuffer::FramebufferAbstract,
	image::ImageViewAccess,
	instance::{ ApplicationInfo, Instance, InstanceCreationError, PhysicalDevice },
	swapchain::Surface,
};
use vulkano_win::VkSurfaceBuild;
use window::Window;
use winit::{ Event, WindowEvent, WindowId };

/// Root struct for this library. Any windows that are created using the same context will share some resources.
pub struct Context {
	events: EventsLoop,
	instance: Arc<Instance>,
	devices: Vec<Arc<DeviceCtx>>,
}
impl Context {
	pub fn new(name: Option<&str>, version: Option<Version>) -> Result<Self, InstanceCreationError> {
		Ok(Self {
			events: EventsLoop::new(),
			instance:
				Instance::new(
					Some(&ApplicationInfo {
						application_name: name.map(|x| x.into()),
						application_version: version,
						engine_name: Some("nIce Game".into()),
						engine_version: Some(Version {
							major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
							minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
							patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
						}),
					}),
					&vulkano_win::required_extensions(),
					None
				)?,
			devices: vec![],
		})
	}

	pub fn create_window<T: Into<String>>(&mut self, title: T) -> Window {
		let surface = winit::WindowBuilder::new()
			.with_title(title)
			.build_vk_surface(&self.events.events, self.instance.clone())
			.expect("failed to create window");

		let device = self.get_device_for_surface(&surface);

		let resized = Arc::<AtomicBool>::default();
		self.events.resized.insert(surface.window().id(), resized.clone());

		Window::new(surface, device, resized)
	}

	pub fn poll_events<F: FnMut(Event)>(&mut self, callback: F) {
		self.events.poll_events(callback)
	}

	fn get_device_for_surface<T>(&mut self, surface: &Surface<T>) -> Arc<DeviceCtx> {
		for device in &self.devices {
			let qfam = device.queue().family();
			if qfam.supports_graphics() && surface.is_supported(qfam).unwrap() {
				return device.clone();
			}
		}

		let pdevice = PhysicalDevice::enumerate(&self.instance).next().expect("no device available");
		info!("Using device: {} ({:?})", pdevice.name(), pdevice.ty());

		let qfam = pdevice.queue_families()
			.find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap())
			.expect("failed to find a graphical queue family");

		let (device, mut queues) =
			Device::new(
				pdevice,
				&Features::none(),
				&DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() },
				[(qfam, 1.0)].iter().cloned()
			)
			.expect("failed to create device");
		let queue = queues.next().unwrap();

		let ret = DeviceCtx::new(device, queue);
		self.devices.push(ret.clone());
		ret
	}
}

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
				Event::WindowEvent { event: WindowEvent::CloseRequested, window_id } => {
					resized.remove(&window_id);
				},
				Event::WindowEvent { event: WindowEvent::Resized(_), window_id } => {
					resized[&window_id].store(true, Ordering::Relaxed);
				},
				_ => (),
			}

			callback(event);
		});
	}
}

pub struct ObjectId {
	val: Weak<()>,
}
impl ObjectId {
	pub fn is_child_of(&self, root: &ObjectIdRoot) -> bool {
		self.val.upgrade().map_or(false, |val| Arc::ptr_eq(&val, &root.val))
	}
}

pub struct ObjectIdRoot {
	val: Arc<()>,
}
impl ObjectIdRoot {
	fn new() -> Self {
		Self { val: Arc::default() }
	}

	pub fn make_id(&self) -> ObjectId {
		ObjectId { val: Arc::downgrade(&self.val) }
	}
}

#[derive(Clone)]
struct ImageFramebuffer {
	image: Weak<ImageViewAccess + Send + Sync + 'static>,
	framebuffer: Arc<FramebufferAbstract + Send + Sync + 'static>,
}
impl ImageFramebuffer {
	fn new(
		image: Weak<ImageViewAccess + Send + Sync + 'static>,
		framebuffer: Arc<FramebufferAbstract + Send + Sync + 'static>,
	) -> Self {
		Self { image: image, framebuffer: framebuffer }
	}
}

pub trait RenderTarget {
	fn format(&self) -> Format;
	fn id_root(&self) -> &ObjectIdRoot;
	fn images(&self) -> &[Arc<ImageViewAccess + Send + Sync + 'static>];
}
