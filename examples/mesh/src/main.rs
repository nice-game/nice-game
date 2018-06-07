extern crate futures;
extern crate nice_game;

use futures::executor::block_on;
use nice_game::{
	Context,
	RenderTarget,
	Version,
	mesh::{ Mesh, MeshBatch, MeshBatchShaders, MeshBatchShared },
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

	let mesh_batch_shared = MeshBatchShared::new(MeshBatchShaders::new(&mut window).unwrap(), window.format());

	let mesh = Mesh::new(&mut window, [10.0, 10.0]).unwrap();

	let mut mesh_batch = MeshBatch::new(&mut window, mesh_batch_shared).unwrap();
	mesh_batch.add_mesh(mesh);

	loop {
		let mut done = false;
		events.poll_events(|event| match event {
			Event::WindowEvent { event: WindowEvent::Closed, .. } => done = true,
			_ => (),
		});

		if done {
			break;
		}

		window.present(vec![(None, &mut [&mut mesh_batch])]).unwrap();
	}
}
