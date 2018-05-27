extern crate nice_game;

use nice_game::{
	Context,
	RenderTarget,
	Version,
	sprite::{ SpriteBatch, SpriteBatchShaders, SpriteBatchShared, Triangle },
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

	let triangle = Triangle::new(&mut window).unwrap();

	let mut mesh_batch =
		SpriteBatch::new(SpriteBatchShared::new(&SpriteBatchShaders::new(&window).unwrap(), window.format()), &window);
	mesh_batch.add_triangle(triangle);

	loop {
		let mut done = false;
		events.poll_events(|event| match event {
			Event::WindowEvent { event: WindowEvent::Closed, .. } => done = true,
			_ => (),
		});

		if done {
			break;
		}

		window.present(&mut [&mut mesh_batch]).unwrap();
	}
}
