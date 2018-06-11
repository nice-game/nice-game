extern crate cgmath;
extern crate futures;
extern crate nice_game;
extern crate simplelog;

use cgmath::{ One, Quaternion, Vector3, Zero };
use futures::executor::block_on;
use nice_game::{
	Context,
	GpuFuture,
	RenderTarget,
	Version,
	batch::mesh::{ Mesh, MeshBatch, MeshBatchShaders, MeshBatchShared, MeshVertex },
	camera::Camera,
	window::{ Event, EventsLoop, Window, WindowEvent },
};
use simplelog::{ LevelFilter, SimpleLogger };

fn main() {
	SimpleLogger::init(LevelFilter::Warn, simplelog::Config::default()).unwrap();

	// test obj parser
	block_on(Mesh::from_file("examples/assets/sphere.obj")).map(|_| ()).unwrap_or_else(|err| println!("{:#?}", err));
	// end test

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

	let (mesh_batch_shaders, mesh_batch_shaders_future) = MeshBatchShaders::new(&mut window).unwrap();
	let mesh_batch_shared = MeshBatchShared::new(mesh_batch_shaders, window.format());

	let (mesh, mesh_future) = Mesh::new(
		&window,
		vec![
			MeshVertex { position: [-1.0, -1.0, 0.0], normal:  [0.0, 0.0, -1.0] },
			MeshVertex { position: [1.0, -1.0, 0.0], normal:  [0.0, 0.0, -1.0] },
			MeshVertex { position: [-1.0, 1.0, 0.0], normal:  [0.0, 0.0, -1.0] },
			MeshVertex { position: [-1.0, 1.0, 0.0], normal:  [0.0, 0.0, -1.0] },
			MeshVertex { position: [1.0, -1.0, 0.0], normal:  [0.0, 0.0, -1.0] },
			MeshVertex { position: [1.0, 1.0, 0.0], normal:  [0.0, 0.0, -1.0] },
		].into_iter(),
		[0.0, 0.0, 2.0]
	).unwrap();

	let mut mesh_batch = MeshBatch::new(&window, mesh_batch_shared).unwrap();
	mesh_batch.add_mesh(mesh);

	let mut camera = make_camera(&window);

	window.join_future(mesh_future.join(mesh_batch_shaders_future));

	loop {
		let mut done = false;
		events.poll_events(|event| match event {
			Event::WindowEvent { event: WindowEvent::Closed, .. } => done = true,
			Event::WindowEvent { event: WindowEvent::Resized(_, _), .. } => camera = make_camera(&window),
			_ => (),
		});

		if done {
			break;
		}

		window
			.present(|window, image_num, future| {
				future
					.then_execute(
						window.queue().clone(), mesh_batch.commands(window, window, image_num, &camera).unwrap()
					)
					.unwrap()
			})
			.unwrap();
	}
}

fn make_camera(window: &Window) -> Camera {
	let [width, height] = window.images()[0].dimensions().width_height();
	Camera::new(
		&window,
		Vector3::zero(),
		Quaternion::one(),
		width as f32 / height as f32,
		140.0,
		1.0,
		1000.0
	).unwrap()
}
