use cgmath::{ One, Quaternion, Vector3, Zero };

pub struct Camera {
	pub position: Vector3<f32>,
	pub rotation: Quaternion<f32>,
}
impl Camera {
	pub fn new(position: Vector3<f32>, rotation: Quaternion<f32>) -> Self {
		Self { position: position, rotation: rotation }
	}
}
impl Default for Camera {
	fn default() -> Self {
		Self::new(Vector3::zero(), Quaternion::one())
	}
}
