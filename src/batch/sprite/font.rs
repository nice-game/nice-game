use cpu_pool::GpuFutureFuture;
use futures::executor::block_on;
use rusttype::{ Font as RtFont, Point, Scale };
use std::{ collections::HashMap, fs::File, io::{ self, prelude::* }, path::Path, rc::Rc, sync::Arc };
use vulkano::{
	command_buffer::{ AutoCommandBuffer, CommandBufferExecFuture },
	device::Queue,
	format::Format,
	image::{ Dimensions, ImageCreationError, ImmutableImage },
	memory::DeviceMemoryAllocError,
	sync::{ FenceSignalFuture, GpuFuture, NowFuture },
};
use window::Window;

pub struct Font {
	queue: Arc<Queue>,
	font: RtFont<'static>,
	glyphs: HashMap<char, Option<Arc<ImmutableImage<Format>>>>,
	futures: HashMap<char, GpuFutureFuture<CommandBufferExecFuture<NowFuture, AutoCommandBuffer>>>,
}
impl Font {
	pub fn from_file<P: AsRef<Path> + Send + 'static>(window: &Window, path: P) -> Result<Self, io::Error> {
		let mut bytes = vec![];
		File::open(path)?.read_to_end(&mut bytes)?;

		let font = RtFont::from_bytes(bytes).unwrap();

		Ok(Self { queue: window.queue().clone(), font: font, glyphs: HashMap::new(), futures: HashMap::new() })
	}

	pub fn load(&mut self, ch: char) -> Result<(), DeviceMemoryAllocError> {
		if !self.glyphs.contains_key(&ch) {
			if let Some((image, image_future)) = self.load_impl(ch)? {
				self.glyphs.insert(ch, Some(image));
				self.futures.insert(ch, GpuFutureFuture::new(image_future).unwrap());
			} else {
				self.glyphs.insert(ch, None);
			}
		}

		Ok(())
	}

	pub fn load_chars(&mut self, chars: impl Iterator<Item = char>) -> Result<(), DeviceMemoryAllocError> {
		for ch in chars {
			self.load(ch)?;
		}

		Ok(())
	}

	pub(crate) fn images_for(&mut self, chars: impl Iterator<Item = char>) -> Result<Vec<Option<Arc<ImmutableImage<Format>>>>, ()> {
		chars.map(|ch| {
			if let Some(image) = self.glyphs.get(&ch) {
				if let Some(future) = self.futures.remove(&ch) {
					block_on(future).unwrap();
				}

				return Ok(image.clone());
			}

			if let Some((image, future)) = self.load_impl(ch).unwrap() {
				self.glyphs.insert(ch, Some(image));
				block_on(GpuFutureFuture::new(future).unwrap()).unwrap();

				Ok(self.glyphs.get(&ch).unwrap().clone())
			} else {
				self.glyphs.insert(ch, None);
				Ok(None)
			}
		}).collect()
	}

	fn load_impl(
		&mut self,
		ch: char
	) -> Result<
		Option<(Arc<ImmutableImage<Format>>, CommandBufferExecFuture<NowFuture, AutoCommandBuffer>)>,
		DeviceMemoryAllocError
	> {
		let glyph = self.font.glyph(ch).scaled(Scale::uniform(14.0)).positioned(Point { x: 0.0, y: 0.0 });

		if let Some(bb) = glyph.pixel_bounding_box() {
			let mut pixels = Vec::with_capacity(bb.width() as usize * bb.height() as usize);

			glyph.draw(|x, y, v| {
				pixels[y as usize * bb.width() as usize + x as usize] = (255.0 * v) as u8;
			});

			let (image, image_future) =
				ImmutableImage
					::from_iter(
						pixels.into_iter(),
						Dimensions::Dim2d { width: bb.width() as u32, height: bb.height() as u32 },
						Format::R8Unorm,
						self.queue.clone(),
					)
					.map_err(|err| match err {
						ImageCreationError::AllocError(err) => err,
						_ => unreachable!(),
					})?;

			Ok(Some((image, image_future)))
		} else {
			Ok(None)
		}
	}
}

#[derive(Clone)]
pub(super) struct Glyph {
	image: Arc<ImmutableImage<Format>>,
}
