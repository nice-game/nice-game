extern crate futures;
extern crate nice_game;

use futures::executor::block_on;
use nice_game::{
	Context,
	GpuFuture,
	RenderTarget,
	Version,
	batch::sprite::{ Font, Sprite, SpriteBatch, SpriteBatchShaders, SpriteBatchShared },
	texture::{ ImageFormat, ImmutableTexture },
	window::{ Event, EventsLoop, Window, WindowEvent },
};

fn main() {
	let mut events = EventsLoop::new();

	let mut window =
		Window::new(
			&Context::new(
				Some("Triangle Example"),
				Some(Version {
					major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
					minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
					patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
				}),
			).unwrap(),
			&mut events,
			"nIce Game"
		);

	let (shaders, shaders_future) = SpriteBatchShaders::new(&mut window).unwrap();

	let sprite_batch_shared = SpriteBatchShared::new(shaders, window.format());

	let (texture, texture_future) =
		block_on(
			ImmutableTexture::from_file_with_format(
				&window,
				"examples/assets/colors.png",
				ImageFormat::PNG,
				true
			)
		).unwrap();
	let (sprite, sprite_future) = Sprite::new(&mut window, &sprite_batch_shared, &texture, [10.0, 42.0]).unwrap();

	let mut font = Font::from_file(&window, "examples/assets/DroidSans.ttf").unwrap();
	font.load_chars("test".chars()).unwrap();
	let (text, text_future) = font.make_sprite("test", &sprite_batch_shared, [10.0, 10.0]).unwrap();

	let (mut sprite_batch, sprite_batch_future) = SpriteBatch::new(&window, &window, sprite_batch_shared.clone()).unwrap();
	sprite_batch.add_sprite(Box::new(sprite));
	sprite_batch.add_sprite(Box::new(text));

	window.join_future(shaders_future.join(texture_future).join(sprite_future).join(text_future).join(sprite_batch_future));

	loop {
		let mut done = false;
		events.poll_events(|event| match event {
			Event::WindowEvent { event: WindowEvent::Closed, .. } => done = true,
			_ => (),
		});

		if done {
			break;
		}

		window
			.present(|window, image_num, mut future| {
				let (commands, commands_future) = sprite_batch.commands(window, window, image_num).unwrap();
				if let Some(commands_future) = commands_future {
					future = Box::new(future.join(commands_future));
				}

				future.then_execute(window.queue().clone(), commands).unwrap()
			})
			.unwrap();
	}
}
