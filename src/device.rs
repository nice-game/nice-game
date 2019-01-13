use crate::batch::sprite::Font;
use decorum::R32;
use std::{ collections::HashMap, fs, io, path::{ Path, PathBuf }, sync::{ Arc, Mutex, Weak } };
use vulkano::device::{ Device, Queue };

pub struct DeviceCtx {
	device: Arc<Device>,
	queue: Arc<Queue>,
	fonts: Mutex<HashMap<(PathBuf, R32), Weak<Font>>>,
}
impl DeviceCtx {
	pub fn get_font<P: AsRef<Path>>(&self, path: P, scale: f32) -> Result<Arc<Font>, io::Error> {
		let path = fs::canonicalize(path)?;
		let mut fonts = self.fonts.lock().unwrap();
		let path_scale = (path, scale.into());

		fonts.get(&path_scale)
			.and_then(|font| font.upgrade())
			.map(|font| Ok(font))
			.unwrap_or_else(|| {
				let ret = Font::from_file(self.queue.clone(), &path_scale.0, scale);
				if let Ok(ret) = &ret {
					fonts.insert(path_scale, Arc::downgrade(ret));
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
