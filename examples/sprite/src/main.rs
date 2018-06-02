extern crate futures;
extern crate nice_game;

use futures::executor::block_on;
use nice_game::{
	Context,
	RenderTarget,
	Version,
	sprite::{ Sprite, SpriteBatch, SpriteBatchShaders, SpriteBatchShared },
	texture::{ ImageFormat, Texture },
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

	let sprite_batch_shared = SpriteBatchShared::new(SpriteBatchShaders::new(&mut window).unwrap(), window.format());

	let texture =
		block_on(
			Texture::from_file_with_format(
				window.queue().clone(),
				"examples/sprite/assets/colors.png",
				ImageFormat::PNG
			)
		).unwrap();
	let sprite = Sprite::new(&mut window, &sprite_batch_shared, &texture, [10.0, 10.0]);

	let mut sprite_batch = SpriteBatch::new(&mut window, sprite_batch_shared).unwrap();
	sprite_batch.add_sprite(sprite);

	loop {
		let mut done = false;
		events.poll_events(|event| match event {
			Event::WindowEvent { event: WindowEvent::Closed, .. } => done = true,
			_ => (),
		});

		if done {
			break;
		}

		window.present(&mut [&mut sprite_batch]).unwrap();
	}
}
