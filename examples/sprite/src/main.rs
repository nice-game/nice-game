extern crate futures;
extern crate nice_game;

use futures::executor::LocalPool;
use nice_game::{
	Context,
	RenderTarget,
	Version,
	sprite::{ ImageFormat, Sprite, SpriteBatch, SpriteBatchShaders, SpriteBatchShared },
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

	let mut pool = LocalPool::new();
	let mut exec = pool.executor();
	let sprite_future = Sprite::from_file_with_format(&mut window, sprite_batch_shared.clone(), "examples/sprite/assets/colors.png", ImageFormat::PNG);
	let sprite = pool.run_until(sprite_future, &mut exec).unwrap();

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
