use batch::sprite::{ Drawable2D, SpriteBatchShared };
use rusttype::{ Font as RtFont, GlyphId, Point, Scale };
use std::{ collections::HashMap, fs::File, io::{ self, prelude::* }, path::Path, sync::Arc };
use texture::{ Texture, ImmutableTexture };
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, CommandBufferExecFuture, DynamicState },
	descriptor::{
		DescriptorSet,
		descriptor_set::PersistentDescriptorSet
	},
	device::Queue,
	format::Format,
	image::{ Dimensions, ImageCreationError, ImmutableImage },
	instance::QueueFamily,
	memory::DeviceMemoryAllocError,
	pipeline::viewport::Viewport,
	sync::{ now, FenceSignalFuture, FlushError, GpuFuture, NowFuture },
};
use window::Window;

pub struct Font {
	queue: Arc<Queue>,
	font: RtFont<'static>,
	glyphs: HashMap<GlyphId, Option<ImmutableTexture>>,
	futures: HashMap<GlyphId, Arc<FenceSignalFuture<CommandBufferExecFuture<NowFuture, AutoCommandBuffer>>>>,
}
impl Font {
	pub fn from_file<P: AsRef<Path> + Send + 'static>(window: &Window, path: P) -> Result<Self, io::Error> {
		let mut bytes = vec![];
		File::open(path)?.read_to_end(&mut bytes)?;

		let font = RtFont::from_bytes(bytes).unwrap();

		Ok(Self { queue: window.queue().clone(), font: font, glyphs: HashMap::new(), futures: HashMap::new() })
	}

	pub fn load(&mut self, ch: char) -> Result<(), DeviceMemoryAllocError> {
		let id = self.font.glyph(ch).id();

		if !self.glyphs.contains_key(&id) {
			if let Some((image, image_future)) = self.load_impl(id)? {
				self.glyphs.insert(id, Some(ImmutableTexture::from_image(image)));
				self.futures.insert(id, Arc::new(image_future.then_signal_fence_and_flush().unwrap()));
			} else {
				self.glyphs.insert(id, None);
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

	pub fn make_sprite(
		&self,
		text: &str,
		shared: &SpriteBatchShared,
		[x, y]: [f32; 2],
	) -> Result<(TextSprite, impl GpuFuture), DeviceMemoryAllocError> {
		let mut positions = vec![];
		let mut future: Box<GpuFuture> = Box::new(now(self.queue.device().clone()));

		let mut static_descs = HashMap::new();
		let mut glyph_futures = HashMap::new();

		for glyph in self.font.layout(text, Scale::uniform(32.0), Point { x: x, y: y }) {
			let id = glyph.id();

			let point = glyph.position();
			let (position, pos_future) =
				ImmutableBuffer::from_data([point.x, point.y], BufferUsage::uniform_buffer(), self.queue.clone())?;
			positions.push((id, position));
			future = Box::new(future.join(pos_future));

			if let Some(tex) = self.glyphs.get(&id).unwrap() {
				static_descs.entry(id)
					.or_insert_with(|| Arc::new(
						PersistentDescriptorSet::start(shared.pipeline().clone(), 2)
							.add_sampled_image(tex.image().clone(), shared.shaders().sampler().clone())
							.unwrap()
							.build()
							.unwrap()
					) as Arc<DescriptorSet + Send + Sync + 'static>);

				if let Some(fut) = self.futures.get(&id) {
					glyph_futures.insert(id, fut.clone());
				}
			}
		}

		Ok((TextSprite { static_descs: static_descs, positions: positions, futures: glyph_futures }, future))
	}

	fn load_impl(
		&mut self,
		id: GlyphId
	) -> Result<
		Option<(Arc<ImmutableImage<Format>>, CommandBufferExecFuture<NowFuture, AutoCommandBuffer>)>,
		DeviceMemoryAllocError
	> {
		let glyph = self.font.glyph(id).scaled(Scale::uniform(32.0)).positioned(Point { x: 0.0, y: 0.0 });

		if let Some(bb) = glyph.pixel_bounding_box() {
			println!("'{:?}' {}, {}", id, bb.width(), bb.height());
			let bblen = bb.width() as usize * bb.height() as usize;
			let mut pixels = Vec::with_capacity(bblen);
			unsafe { pixels.set_len(bblen); }

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

			println!("{:?}", image);

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

pub struct TextSprite {
	static_descs: HashMap<GlyphId, Arc<DescriptorSet + Send + Sync + 'static>>,
	positions: Vec<(GlyphId, Arc<ImmutableBuffer<[f32; 2]>>)>,
	futures: HashMap<GlyphId, Arc<FenceSignalFuture<CommandBufferExecFuture<NowFuture, AutoCommandBuffer>>>>,
}
impl Drawable2D for TextSprite {
	fn make_commands(
		&mut self,
		shared: &SpriteBatchShared,
		target_desc: &Arc<DescriptorSet + Send + Sync + 'static>,
		queue_family: QueueFamily,
		dimensions: [f32; 2],
	) -> Result<AutoCommandBuffer, OomError> {
		let mut cmds = AutoCommandBufferBuilder::secondary_graphics_one_time_submit(shared.shaders().device().clone(), queue_family, shared.subpass().clone())?;

		for (id, pos) in &self.positions {
			if let Some(future) = self.futures.get(&id).map(|f| f.clone()) {
				match future.wait(Some(Default::default())) {
					Ok(()) => { self.futures.remove(&id); },
					Err(FlushError::Timeout) => { continue; },
					Err(err) => panic!(err),
				};
			}

			cmds = cmds
				.draw(
					shared.pipeline().clone(),
					DynamicState {
						line_width: None,
						viewports: Some(vec![
							Viewport {
								origin: [0.0, 0.0],
								dimensions: dimensions,
								depth_range: 0.0..1.0,
							}
						]),
						scissors: None,
					},
					vec![shared.shaders().vertices().clone()],
					(
						target_desc.clone(),
						shared.sprite_desc_pool().lock().unwrap()
							.next()
							.add_buffer(pos.clone())
							.unwrap()
							.build()
							.unwrap(),
						self.static_descs.get(id).unwrap().clone(),
					),
					()
				)
				.unwrap();
		}

		Ok(cmds.build().map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?)
	}
}
