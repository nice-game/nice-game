extern crate futures;
extern crate nice_game;

use futures::executor::block_on;
use nice_game::{
	Context,
	GpuFuture,
	RenderTarget,
	Version,
	sprite::{ Sprite, SpriteBatch, SpriteBatchShaders, SpriteBatchShared },
	texture::{ ImageFormat, ImmutableTexture, TargetTexture },
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

	let mut target = TargetTexture::new(&window, [400, 400]).unwrap();

	let texture =
		block_on(ImmutableTexture::from_file_with_format(&window, "examples/assets/colors.png", ImageFormat::PNG))
			.unwrap();

	let texture_sprite = Sprite::new(&mut target, &sprite_batch_shared, &texture, [0.0, 0.0]).unwrap();

	let mut target_sprite_batch = SpriteBatch::new(&mut target, sprite_batch_shared.clone()).unwrap();
	target_sprite_batch.add_sprite(texture_sprite);

	let target_sprite = Sprite::new(&mut window, &sprite_batch_shared, &target, [10.0, 10.0]).unwrap();

	let mut window_sprite_batch = SpriteBatch::new(&mut window, sprite_batch_shared).unwrap();
	window_sprite_batch.add_sprite(target_sprite);

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
				if let Some(target_future) = target.take_future() {
					future = Box::new(future.join(target_future));
				}

				future
					.then_execute(window.queue().clone(), target_sprite_batch.commands(&mut target, 0).unwrap())
					.unwrap()
					.then_signal_semaphore()
					.then_execute(window.queue().clone(), window_sprite_batch.commands(window, image_num).unwrap())
					.unwrap()
			})
			.unwrap();
	}
}