use batch::mesh::{ TargetVertex };
use std::sync::Arc;
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	device::Device,
	memory::DeviceMemoryAllocError,
	sync::GpuFuture,
};
use window::Window;

pub struct MeshBatchShaders {
	pub(super) device: Arc<Device>,
	pub(super) target_vertices: Arc<ImmutableBuffer<[TargetVertex; 6]>>,
	pub(super) shader_gbuffers_vertex: vs_gbuffers::Shader,
	pub(super) shader_gbuffers_fragment: fs_gbuffers::Shader,
	pub(super) shader_target_vertex: vs_target::Shader,
	pub(super) shader_target_fragment: fs_target::Shader,
}
impl MeshBatchShaders {
	pub fn new(window: &Window) -> Result<(Arc<Self>, impl GpuFuture), MeshBatchShadersError> {
		let (target_vertices, future) =
			ImmutableBuffer::from_data(
				[
					TargetVertex { position: [0.0, 0.0] },
					TargetVertex { position: [1.0, 0.0] },
					TargetVertex { position: [0.0, 1.0] },
					TargetVertex { position: [0.0, 1.0] },
					TargetVertex { position: [1.0, 0.0] },
					TargetVertex { position: [1.0, 1.0] },
				],
				BufferUsage::vertex_buffer(),
				window.queue().clone(),
			)?;

		Ok((
			Arc::new(Self {
				device: window.device().clone(),
				target_vertices: target_vertices,
				shader_gbuffers_vertex: vs_gbuffers::Shader::load(window.device().clone())?,
				shader_gbuffers_fragment: fs_gbuffers::Shader::load(window.device().clone())?,
				shader_target_vertex: vs_target::Shader::load(window.device().clone())?,
				shader_target_fragment: fs_target::Shader::load(window.device().clone())?,
			}),
			future
		))
	}
}

#[derive(Debug)]
pub enum MeshBatchShadersError {
	DeviceMemoryAllocError(DeviceMemoryAllocError),
	OomError(OomError),
	TooManyObjects,
}
impl From<DeviceMemoryAllocError> for MeshBatchShadersError {
	fn from(val: DeviceMemoryAllocError) -> Self {
		MeshBatchShadersError::DeviceMemoryAllocError(val)
	}
}
impl From<OomError> for MeshBatchShadersError {
	fn from(val: OomError) -> Self {
		MeshBatchShadersError::OomError(val)
	}
}

mod vs_gbuffers {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "vertex"]
	#[src = "#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord_main;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec2 out_texcoord_main;
layout(location = 2) out vec3 out_base_color;

layout(set = 0, binding = 0) uniform CameraPos { vec3 camera_pos; };
layout(set = 0, binding = 1) uniform CameraRot { vec4 camera_rot; };
layout(set = 0, binding = 2) uniform CameraProj { vec4 camera_proj; };

layout(set = 1, binding = 0) uniform MeshDynamic { vec3 mesh_pos; };

layout(set = 2, binding = 0) uniform Material {
	uint light_penetration;
	uint subsurface_scattering;
	uint emissive_brightness;
	vec3 base_color;
};

vec4 quat_inv(vec4 quat) {
	return vec4(-quat.xyz, quat.w) / dot(quat, quat);
}

vec3 quat_mul(vec4 quat, vec3 vec) {
	return cross(quat.xyz, cross(quat.xyz, vec) + vec * quat.w) * 2.0 + vec;
}

vec4 perspective(vec4 proj, vec3 pos) {
	return vec4(pos.xy * proj.xy, pos.z * proj.z + proj.w, -pos.z);
}

void main() {
	out_normal = quat_mul(quat_inv(camera_rot), normal);
	out_texcoord_main = texcoord_main;
	out_base_color = base_color;
	gl_Position = perspective(camera_proj, quat_mul(quat_inv(camera_rot), position + mesh_pos - camera_pos));
}"]
	struct Dummy;
}

mod fs_gbuffers {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "fragment"]
	#[src = "#version 450
layout(location = 0) in vec3 normal;
layout(location = 1) in vec2 texcoord_main;
layout(location = 2) in vec3 base_color;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_normal;

void main() {
	out_color = vec4(base_color, 1);
	out_normal = vec4(normal, 1);
}"]
	struct Dummy;
}

mod vs_target {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "vertex"]
	#[src = "#version 450
layout(location = 0) in vec2 position;

void main() {
	gl_Position = vec4(position * 2 - 1, 0.0, 1.0);
}
"]
	struct Dummy;
}

mod fs_target {
	#[allow(dead_code)]
	#[derive(VulkanoShader)]
	#[ty = "fragment"]
	#[src = "#version 450
layout(location = 0) out vec4 out_color;

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput normal;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput depth;

void main() {
	out_color = subpassLoad(depth);
}
"]
	struct Dummy;
}
