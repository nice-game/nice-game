extern crate nice_game;
#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;

mod mesh;

use mesh::{MeshBatch, MeshBatchShared, Triangle};
use nice_game::{
	Context,
	Version,
	window::{
		Event,
		EventsLoop,
		Window,
		WindowEvent,
	}
};

fn main() {
	let mut events = EventsLoop::new();

	let mut window = Window::new(
		&Context::new(
			Some("Triangle Example"),
			Some(Version {
				major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
				minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
				patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
			}),
		),
		&mut events,
		"nIce Game"
	);

	let mesh_batch_shared = MeshBatchShared::new(window.device().clone(), window.format());
	let mut mesh_batch = MeshBatch::new(mesh_batch_shared, &mut window);
	mesh_batch.add_triangle(Triangle::new(window.queue().clone()).unwrap().0);

	loop {
		let mut done = false;
		events.poll_events(|event| match event {
			Event::WindowEvent { event: WindowEvent::Closed, .. } => done = true,
			_ => (),
		});

		if done {
			break;
		}

		window.present(&mut [&mut mesh_batch]);
	}
}
