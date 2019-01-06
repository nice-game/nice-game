extern crate futures;
extern crate nice_game;

use futures::executor::block_on;
use nice_game::{
	Context,
	GpuFuture,
	RenderTarget,
	Version,
	batch::sprite::{ Sprite, SpriteBatch, SpriteBatchShaders, SpriteBatchShared },
	texture::{ ImageFormat, ImmutableTexture },
	window::{ Event, WindowEvent },
};

fn main() {
	let mut ctx =
		Context::new(
			Some("Triangle Example"),
			Some(Version {
				major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
				minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
				patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
			}),
		)
		.unwrap();

	let mut window = ctx.create_window("nIce Game");

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

	let text = window.device().get_font("examples/assets/consola.ttf", 24.0).unwrap()
		.make_sprite("The quick brown fox jumped over the lazy dog. (╯°□°）╯︵ ┻━┻", &sprite_batch_shared, [10.0, 32.0])
		.unwrap();

	let (mut sprite_batch, sprite_batch_future) = SpriteBatch::new(&window, &window, sprite_batch_shared.clone()).unwrap();
	sprite_batch.add_sprite(Box::new(sprite));
	sprite_batch.add_sprite(Box::new(text));

	window.join_future(shaders_future.join(texture_future).join(sprite_future).join(sprite_batch_future));

	loop {
		let mut done = false;
		ctx.poll_events(|event| match event {
			Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => done = true,
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

				future.then_execute(window.device().queue().clone(), commands).unwrap()
			})
			.unwrap();
	}
}
