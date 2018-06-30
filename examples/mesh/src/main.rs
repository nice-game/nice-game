extern crate cgmath;
extern crate futures;
extern crate multiinput;
extern crate nice_game;
extern crate simplelog;

use cgmath::{ prelude::*, Quaternion, Rad, vec2, vec3, Vector2, Vector3 };
use futures::executor::block_on;
use multiinput::{ DeviceType, KeyId, RawEvent, RawInputManager, State };
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
use std::f32::consts::PI;

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
		block_on(
			Mesh::from_file(
				&window,
				&mesh_batch_shared,
				vec3(0.0, 0.0, 3.0),
				Quaternion::one(),
				"examples/assets/de_rebelzone/de_rebelzone.nmd"
			).unwrap()
		).unwrap();

	let (mut mesh_batch, mesh_batch_future) = MeshBatch::new(&window, mesh_batch_shared).unwrap();
	mesh_batch.add_mesh(mesh);

	let mut character = Character::new();
	let [win_width, win_height] = window.images()[0].dimensions().width_height();
	let mut camera =
		Camera::new(
			&window,
			Vector3::zero(),
			Quaternion::one(),
			win_width as f32 / win_height as f32,
			100.0,
			0.05,
			1500.0,
		).unwrap();

	window.join_future(mesh_future.join(mesh_batch_shaders_future).join(mesh_batch_future));

	let mut controls_active = false;
	let mut w_down = false;
	let mut a_down = false;
	let mut s_down = false;
	let mut d_down = false;
	let mut space_down = false;
	let mut shift_down = false;

	let mut raw_input = RawInputManager::new().unwrap();
	raw_input.register_devices(DeviceType::Keyboards);
	raw_input.register_devices(DeviceType::Mice);

	loop {
		let mut done = false;

		events.poll_events(|event| match event {
			Event::WindowEvent { event: WindowEvent::AxisMotion { axis, value, .. } , .. } => {
				println!("axis {}, value {}", axis, value);
			},
			Event::WindowEvent { event: WindowEvent::Closed, .. } => done = true,
			Event::WindowEvent { event: WindowEvent::Focused(false), .. } => {
				window.set_cursor_state(CursorState::Normal).unwrap();
				controls_active = false;
			},
			Event::WindowEvent { event: WindowEvent::MouseInput{ button: MouseButton::Left, .. }, .. } => {
				window.set_cursor_state(CursorState::Grab).unwrap();
				controls_active = true;
			},
			Event::WindowEvent { event: WindowEvent::Resized(_, _), .. } => {
				camera.set_projection(win_width as f32 / win_height as f32, 100.0, 0.05, 1500.0).unwrap();
			},
			_ => (),
		});

		while let Some(event) = raw_input.get_event() {
			match event {
				RawEvent::KeyboardEvent(_,  KeyId::Escape, State::Pressed) => done = true,
				RawEvent::KeyboardEvent(_,  KeyId::W, State::Pressed) => w_down = true,
				RawEvent::KeyboardEvent(_,  KeyId::W, State::Released) => w_down = false,
				RawEvent::KeyboardEvent(_,  KeyId::A, State::Pressed) => a_down = true,
				RawEvent::KeyboardEvent(_,  KeyId::A, State::Released) => a_down = false,
				RawEvent::KeyboardEvent(_,  KeyId::S, State::Pressed) => s_down = true,
				RawEvent::KeyboardEvent(_,  KeyId::S, State::Released) => s_down = false,
				RawEvent::KeyboardEvent(_,  KeyId::D, State::Pressed) => d_down = true,
				RawEvent::KeyboardEvent(_,  KeyId::D, State::Released) => d_down = false,
				RawEvent::KeyboardEvent(_,  KeyId::Space, State::Pressed) => space_down = true,
				RawEvent::KeyboardEvent(_,  KeyId::Space, State::Released) => space_down = false,
				RawEvent::KeyboardEvent(_,  KeyId::Shift, State::Pressed) => shift_down = true,
				RawEvent::KeyboardEvent(_,  KeyId::Shift, State::Released) => shift_down = false,
				RawEvent::MouseMoveEvent(_, x, y) => if controls_active {
					character.rotation += vec2(x as f32 / 300.0, y as f32 / 300.0);

					if character.rotation.x > 2.0 {
						character.rotation.x -= 4.0;
					} else if character.rotation.x < -2.0 {
						character.rotation.x += 4.0;
					}

					if character.rotation.y > 1.0 {
						character.rotation.y = 1.0;
					} else if character.rotation.y < -1.0 {
						character.rotation.y = -1.0;
					}
				},
				_ => (),
			}
		}

		if done {
			break;
		}

		let yaw = Quaternion::from_angle_y(Rad(-character.rotation.x * PI / 2.0));

		if controls_active && w_down { character.position += yaw.rotate_vector(vec3(0.0, 0.0, -0.5)); }
		if controls_active && a_down { character.position += yaw.rotate_vector(vec3(-0.5, 0.0, 0.0)); }
		if controls_active && s_down { character.position += yaw.rotate_vector(vec3(0.0, 0.0, 0.5)); }
		if controls_active && d_down { character.position += yaw.rotate_vector(vec3(0.5, 0.0, 0.0)); }
		if controls_active && space_down { character.position.y -= 0.5; }
		if controls_active && shift_down { character.position.y += 0.5; }

		camera.set_position(character.position).unwrap();
		camera.set_rotation(yaw * Quaternion::from_angle_x(Rad(character.rotation.y * PI / 2.0))).unwrap();

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

	window.set_cursor_state(CursorState::Normal).unwrap();
}

struct Character {
	position: Vector3<f32>,
	rotation: Vector2<f32>,
}
impl Character {
	fn new() -> Self {
		Self { position: vec3(22.0, 10.0, -26.0), rotation: vec2(-1.5, 0.0) }
	}
}
