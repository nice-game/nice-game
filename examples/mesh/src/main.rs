extern crate cgmath;
extern crate futures;
extern crate nice_game;
extern crate simplelog;

use cgmath::{ One, Quaternion, vec3 };
use futures::executor::block_on;
use nice_game::{
	Context,
	GpuFuture,
	RenderTarget,
	Version,
	batch::mesh::{ Mesh, MeshBatch, MeshBatchShaders, MeshBatchShared },
	camera::Camera,
	window::{ CursorState, Event, EventsLoop, MouseButton, Window, WindowEvent },
};
use simplelog::{ LevelFilter, SimpleLogger };

fn main() {
	SimpleLogger::init(LevelFilter::Debug, simplelog::Config::default()).unwrap();

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

	let (mesh, mesh_future) =
		block_on(Mesh::from_file(&window, &mesh_batch_shared, [0.0, 0.0, 3.0], "examples/assets/de_rebelzone.nmd")).unwrap();

	let (mut mesh_batch, mesh_batch_future) = MeshBatch::new(&window, mesh_batch_shared).unwrap();
	mesh_batch.add_mesh(mesh);

	let mut camera = make_camera(&window);

	window.join_future(mesh_future.join(mesh_batch_shaders_future).join(mesh_batch_future));

	loop {
		let mut done = false;
		events.poll_events(|event| match event {
			Event::WindowEvent { event: WindowEvent::Closed, .. } => done = true,
			Event::WindowEvent { event: WindowEvent::Focused(false), .. } => {
				window.set_cursor_state(CursorState::Normal).unwrap();
			},
			Event::WindowEvent { event: WindowEvent::MouseInput{ button: MouseButton::Left, .. }, .. } => {
				window.set_cursor_state(CursorState::Grab).unwrap();
			},
			Event::WindowEvent { event: WindowEvent::Resized(_, _), .. } => camera = make_camera(&window),
			_ => (),
		});

		if done {
			break;
		}

		window
			.present(|window, image_num, mut future| {
				let (cmds, cmds_future) = mesh_batch.commands(window, window, image_num, &camera).unwrap();
				if let Some(cmds_future) = cmds_future {
					future = Box::new(future.join(cmds_future));
				}
				future.then_execute(window.queue().clone(), cmds).unwrap()
			})
			.unwrap();
	}
}

fn make_camera(window: &Window) -> Camera {
	let [width, height] = window.images()[0].dimensions().width_height();
	Camera::new(
		&window,
		vec3(14.5, -10.5, -34.5),
		Quaternion::one(),
		width as f32 / height as f32,
		100.0,
		0.05,
		1500.0
	).unwrap()
}
