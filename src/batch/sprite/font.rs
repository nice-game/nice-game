use batch::sprite::{ Drawable2D, SpriteBatchShared };
use rusttype::{ Font as RtFont, GlyphId, Point, Scale };
use std::{ collections::HashMap, fs::File, io::{ self, prelude::* }, path::Path, sync::{ Arc, Mutex } };
use texture::{ Texture, ImmutableTexture };
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, BuildError, CommandBufferExecFuture, DynamicState },
	descriptor::{ DescriptorSet, descriptor_set::PersistentDescriptorSet },
	device::Queue,
	format::Format,
	image::{ Dimensions, ImageCreationError, ImmutableImage },
	instance::QueueFamily,
	memory::DeviceMemoryAllocError,
	pipeline::viewport::Viewport,
	sync::{ FenceSignalFuture, FlushError, GpuFuture, JoinFuture, NowFuture },
};

pub struct Font {
	queue: Arc<Queue>,
	scale: f32,
	font: RtFont<'static>,
	glyphs: Mutex<HashMap<GlyphId, Option<Glyph>>>,
	futures: Mutex<HashMap<GlyphId, Arc<FenceSignalFuture<GlyphFuture>>>>,
}
impl Font {
	pub fn make_sprite(
		&self,
		text: &str,
		shared: &SpriteBatchShared,
		[x, y]: [f32; 2],
	) -> Result<TextSprite, DeviceMemoryAllocError> {
		self.load_chars(text.chars())?;

		let mut positions = vec![];

		let mut static_descs = HashMap::new();
		let mut glyph_futures = HashMap::new();
		let glyphs = self.glyphs.lock().unwrap();
		let futures = self.futures.lock().unwrap();

		for glyph in self.font.layout(text, Scale::uniform(self.scale), Point { x: x, y: y }) {
			let id = glyph.id();

			let point = glyph.position();
			let (position, pos_future) =
				ImmutableBuffer::from_data([point.x, point.y], BufferUsage::uniform_buffer(), self.queue.clone())?;
			positions.push((id, position, Some(pos_future.then_signal_fence_and_flush().unwrap())));

			if let Some(glyph) = glyphs.get(&id).unwrap() {
				static_descs.entry(id)
					.or_insert_with(|| Arc::new(
						PersistentDescriptorSet::start(shared.pipeline_text().clone(), 2)
							.add_buffer(glyph.offset.clone())
							.unwrap()
							.add_sampled_image(glyph.texture.image().clone(), shared.shaders().text_sampler().clone())
							.unwrap()
							.build()
							.unwrap()
					) as Arc<DescriptorSet + Send + Sync + 'static>);

				if let Some(fut) = futures.get(&id) {
					glyph_futures.insert(id, fut.clone());
				}
			}
		}

		Ok(TextSprite { static_descs: static_descs, positions: positions, futures: glyph_futures })
	}

	pub(crate) fn from_file<P: AsRef<Path>>(queue: Arc<Queue>, path: P, scale: f32) -> Result<Arc<Self>, io::Error> {
		let mut bytes = vec![];
		File::open(path)?.read_to_end(&mut bytes)?;

		let font = RtFont::from_bytes(bytes).unwrap();

		Ok(Arc::new(Self {
			queue: queue,
			font: font,
			glyphs: Mutex::default(),
			futures: Mutex::default(),
			scale: scale
		}))
	}

	fn load_chars(&self, chars: impl Iterator<Item = char>) -> Result<(), DeviceMemoryAllocError> {
		let mut glyphs = self.glyphs.lock().unwrap();
		let mut futures = self.futures.lock().unwrap();

		for ch in chars {
			let id = self.font.glyph(ch).id();

			if !glyphs.contains_key(&id) {
				let glyph = self.font.glyph(id).scaled(Scale::uniform(self.scale)).positioned(Point { x: 0.0, y: 0.0 });

				if let Some(bb) = glyph.pixel_bounding_box() {
					let bblen = bb.width() as usize * bb.height() as usize;
					let mut pixels = Vec::with_capacity(bblen);
					unsafe { pixels.set_len(bblen); }

					glyph.draw(|x, y, v| {
						pixels[y as usize * bb.width() as usize + x as usize] = (255.0 * v) as u8;
					});

					let (position, pos_future) =
						ImmutableBuffer::from_data([bb.min.x, bb.min.y], BufferUsage::uniform_buffer(), self.queue.clone())?;

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

					glyphs.insert(id, Some(Glyph { texture: ImmutableTexture::from_image(image), offset: position }));
					futures.insert(id, Arc::new(pos_future.join(image_future).then_signal_fence_and_flush().unwrap()));
				} else {
					glyphs.insert(id, None);
				}
			}
		}

		Ok(())
	}
}

pub struct TextSprite {
	static_descs: HashMap<GlyphId, Arc<DescriptorSet + Send + Sync + 'static>>,
	positions: Vec<(
		GlyphId,
		Arc<ImmutableBuffer<[f32; 2]>>,
		Option<FenceSignalFuture<CommandBufferExecFuture<NowFuture, AutoCommandBuffer>>>
	)>,
	futures: HashMap<GlyphId, Arc<FenceSignalFuture<GlyphFuture>>>,
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

		let state =
			DynamicState {
				line_width: None,
				viewports: Some(vec![Viewport { origin: [0.0, 0.0], dimensions: dimensions, depth_range: 0.0..1.0 }]),
				scissors: None,
			};

		for (id, pos, future) in &mut self.positions {
			if let Some(inner) = future.take() {
				match inner.wait(Some(Default::default())) {
					Ok(()) => (),
					Err(FlushError::Timeout) => { *future = Some(inner); },
					Err(err) => panic!(err),
				}
			}

			if let Some(future) = self.futures.get(&id).map(|f| f.clone()) {
				match future.wait(Some(Default::default())) {
					Ok(()) => { self.futures.remove(&id); },
					Err(FlushError::Timeout) => { continue; },
					Err(err) => panic!(err),
				}
			}

			if let Some(static_desc) = self.static_descs.get(id) {
				cmds = cmds
					.draw(
						shared.pipeline_text().clone(),
						&state,
						vec![shared.shaders().vertices().clone()],
						(
							target_desc.clone(),
							shared.sprite_desc_pool().lock().unwrap()
								.next()
								.add_buffer(pos.clone())
								.unwrap()
								.build()
								.unwrap(),
							static_desc.clone(),
						),
						()
					)
					.unwrap();
			}
		}

		Ok(cmds.build().map_err(|err| match err { BuildError::OomError(err) => err, err => unreachable!("{}", err) })?)
	}
}

type GlyphFuture =
	JoinFuture<
		CommandBufferExecFuture<NowFuture, AutoCommandBuffer>,
		CommandBufferExecFuture<NowFuture, AutoCommandBuffer>
	>;

struct Glyph {
	texture: ImmutableTexture,
	offset: Arc<ImmutableBuffer<[i32; 2]>>,
}
