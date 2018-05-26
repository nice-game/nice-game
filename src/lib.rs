extern crate vulkano;
extern crate vulkano_win;
extern crate winit;

pub mod window;

pub use vulkano::instance::Version;

use std::sync::Arc;
use vulkano::{
	command_buffer::AutoCommandBuffer,
	image::ImageViewAccess,
	instance::{ApplicationInfo, Instance, QueueFamily},
};

/// Root struct for this library. Any windows that are created using the same context will share some resources.
pub struct Context {
	instance: Arc<Instance>,
}
impl Context {
	pub fn new(name: Option<&str>, version: Option<Version>) -> Self {
		Self {
			instance: Instance::new(
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
			).unwrap()
		}
	}
}

pub trait RenderTarget {
	fn image_count(&self) -> usize;
}

pub trait Drawable {
	fn commands(
		&mut self,
		queue_family: QueueFamily,
		image_num: usize,
		image: &Arc<ImageViewAccess + Send + Sync + 'static>
	) -> AutoCommandBuffer;
}
