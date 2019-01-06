use batch::sprite::Font;
use std::{ collections::HashMap, io, path::{ Path, PathBuf }, sync::{ Arc, Mutex, Weak } };
use vulkano::device::{ Device, Queue };

pub struct DeviceCtx {
	device: Arc<Device>,
	queue: Arc<Queue>,
	fonts: Mutex<HashMap<PathBuf, Weak<Font>>>,
}
impl DeviceCtx {
	pub fn get_font<P: AsRef<Path>>(&self, path: P) -> Result<Arc<Font>, io::Error> {
		let mut fonts = self.fonts.lock().unwrap();
		fonts.get(path.as_ref())
			.and_then(|font| font.upgrade())
			.map(|font| Ok(font))
			.unwrap_or_else(|| {
				let ret = Font::from_file(self.queue.clone(), path.as_ref());
				if let Ok(ret) = &ret {
					fonts.insert(path.as_ref().to_path_buf(), Arc::downgrade(ret));
				}
				ret
			})
	}

	pub(crate) fn new(device: Arc<Device>, queue: Arc<Queue>) -> Arc<Self> {
		Arc::new(Self { device: device, queue: queue, fonts: Mutex::default() })
	}

	pub(crate) fn device(&self) -> &Arc<Device> {
		&self.device
	}

	pub fn queue(&self) -> &Arc<Queue> {
		&self.queue
	}
}
