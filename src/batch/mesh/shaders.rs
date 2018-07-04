use batch::mesh::{ TargetVertex };
use std::sync::Arc;
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	device::Queue,
	format::Format,
	image::{ Dimensions, ImageCreationError, ImageViewAccess, ImmutableImage },
	memory::DeviceMemoryAllocError,
	sampler::{ Filter, MipmapMode, Sampler, SamplerAddressMode, SamplerCreationError },
	sync::GpuFuture,
};
use window::Window;

pub struct MeshBatchShaders {
	pub(super) queue: Arc<Queue>,
	pub(super) target_vertices: Arc<ImmutableBuffer<[TargetVertex; 6]>>,
	pub(super) shader_gbuffers_vertex: vs_gbuffers::Shader,
	pub(super) shader_gbuffers_fragment: fs_gbuffers::Shader,
	pub(super) shader_target_vertex: vs_target::Shader,
	pub(super) shader_target_fragment: fs_target::Shader,
	pub(super) texture1_default: Arc<ImageViewAccess + Send + Sync + 'static>,
	pub(super) texture2_default: Arc<ImageViewAccess + Send + Sync + 'static>,
	pub(super) sampler: Arc<Sampler>,
}
impl MeshBatchShaders {
	pub fn new(window: &Window) -> Result<(Arc<Self>, impl GpuFuture), MeshBatchShadersError> {
		let (target_vertices, target_vertices_future) =
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

		let (texture1_default, texture1_default_future) =
				ImmutableImage::from_iter(
					vec![(255u8, 255u8, 255u8, 0u8)].into_iter(),
					Dimensions::Dim2d { width: 1, height: 1 },
					Format::R8G8B8A8Srgb,
					window.queue().clone(),
				)?;

		let (texture2_default, texture2_default_future) =
				ImmutableImage::from_iter(
					vec![(128u8, 128u8, 0u8, 0u8)].into_iter(),
					Dimensions::Dim2d { width: 1, height: 1 },
					Format::R8G8B8A8Srgb,
					window.queue().clone(),
				)?;

		Ok((
			Arc::new(Self {
				queue: window.queue().clone(),
				target_vertices: target_vertices,
				shader_gbuffers_vertex: vs_gbuffers::Shader::load(window.device().clone())?,
				shader_gbuffers_fragment: fs_gbuffers::Shader::load(window.device().clone())?,
				shader_target_vertex: vs_target::Shader::load(window.device().clone())?,
				shader_target_fragment: fs_target::Shader::load(window.device().clone())?,
				texture1_default: texture1_default,
				texture2_default: texture2_default,
				sampler:
					Sampler::new(
						window.device().clone(),
						Filter::Linear,
						Filter::Linear, MipmapMode::Nearest,
						SamplerAddressMode::Repeat,
						SamplerAddressMode::Repeat,
						SamplerAddressMode::Repeat,
						0.0, 1.0, 0.0, 0.0
					)?,
			}),
			target_vertices_future.join(texture1_default_future).join(texture2_default_future)
		))
	}
}

#[derive(Debug)]
pub enum MeshBatchShadersError {
	DeviceMemoryAllocError(DeviceMemoryAllocError),
	ImageCreationError(ImageCreationError),
	OomError(OomError),
	SamplerCreationError(SamplerCreationError),
	TooManyObjects,
}
impl From<DeviceMemoryAllocError> for MeshBatchShadersError {
	fn from(val: DeviceMemoryAllocError) -> Self {
		MeshBatchShadersError::DeviceMemoryAllocError(val)
	}
}
impl From<ImageCreationError> for MeshBatchShadersError {
	fn from(val: ImageCreationError) -> Self {
		MeshBatchShadersError::ImageCreationError(val)
	}
}
impl From<OomError> for MeshBatchShadersError {
	fn from(val: OomError) -> Self {
		MeshBatchShadersError::OomError(val)
	}
}
impl From<SamplerCreationError> for MeshBatchShadersError {
	fn from(val: SamplerCreationError) -> Self {
		MeshBatchShadersError::SamplerCreationError(val)
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
layout(location = 2) out vec3 out_base_albedo;

layout(set = 0, binding = 0) uniform CameraPos { vec3 camera_pos; };
layout(set = 0, binding = 1) uniform CameraRot { vec4 camera_rot; };
layout(set = 0, binding = 2) uniform CameraProj { vec4 camera_proj; };

layout(set = 1, binding = 0) uniform MeshPos { vec3 mesh_pos; };
layout(set = 1, binding = 1) uniform MeshRot { vec4 mesh_rot; };

layout(set = 2, binding = 0) uniform Material {
	uint light_penetration;
	uint subsurface_scattering;
	uint emissive_brightness;
	vec3 base_albedo;
};
layout(set = 2, binding = 1) uniform sampler2D tex1;
layout(set = 2, binding = 2) uniform sampler2D tex2;

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
	// stupid math library puts w first, so we flip it here
	vec4 camera_rot = camera_rot.yzwx;
	vec4 mesh_rot = mesh_rot.yzwx;

	out_normal = quat_mul(quat_inv(camera_rot), normal);
	out_texcoord_main = texcoord_main;
	out_base_albedo = base_albedo;
	gl_Position = perspective(camera_proj, quat_mul(quat_inv(camera_rot), quat_mul(mesh_rot, position) + mesh_pos - camera_pos));
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
layout(location = 2) in vec3 base_albedo;

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal;

layout(set = 2, binding = 1) uniform sampler2D tex1;
layout(set = 2, binding = 2) uniform sampler2D tex2;

float softSq(float x, float y) {
	return tanh(sin(x * 6.283185307179586 * 4.0) * y);
}

void main() {
	vec3 albedo = texture(tex2, texcoord_main).rgb;
	//albedo = mix(base_albedo, albedo, albedo.a), 1);
	out_albedo = vec4(sqrt(albedo), 0);
	out_normal = vec4(normalize(normal), 1);
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

layout(set = 0, binding = 0) uniform Resolution { vec4 resolution; };
layout(set = 0, binding = 1, input_attachment_index = 0) uniform subpassInput albedo;
layout(set = 0, binding = 2, input_attachment_index = 1) uniform subpassInput normal;
layout(set = 0, binding = 3, input_attachment_index = 2) uniform subpassInput depth;
layout(set = 1, binding = 0) uniform CameraPos { vec3 camera_pos; };
layout(set = 1, binding = 1) uniform CameraRot { vec4 camera_rot; };
layout(set = 1, binding = 2) uniform CameraProj { vec4 camera_proj; };

vec3 quat_mul(vec4 q, vec3 v) {
	return cross(q.xyz, cross(q.xyz, v) + v * q.w) * 2.0 + v;
}

void main() {
	// stupid math library puts w first, so we flip it here
	vec4 camera_rot = camera_rot.yzwx;

	vec3 g_position_ds = vec3(gl_FragCoord.xy * resolution.zw, 2.0 * subpassLoad(depth).x) - 1.0;
	vec3 g_position_cs = vec3(g_position_ds.xy / camera_proj.xy, -1.0) * camera_proj.w / (g_position_ds.z + camera_proj.z);
	vec3 g_position_ws = quat_mul(camera_rot, g_position_cs) + camera_pos;

	vec3 g_normal_cs = subpassLoad(normal).xyz;
	vec3 g_normal_ws = quat_mul(camera_rot, g_normal_cs);

	vec3 g_albedo = subpassLoad(albedo).rgb;
	g_albedo *= g_albedo;

	vec3 light = vec3(0);

	// sunlight
	vec3 sunColor = vec3(1.0, 0.85, 0.7) * 0.5;
	vec3 sunDir = normalize(vec3(-1, -4, 2));
	light += sunColor * max(0, dot(g_normal_ws, sunDir));

	// point light
	float lightRadius = 5.0;
	vec3 lightColor = vec3(0.7, 0.85, 1.0) * sqrt(lightRadius);
	vec3 lightPos = vec3(14.5, -11, -28.5);
	float lightDistance = distance(lightPos, g_position_ws);
	vec3 lightDir = normalize(lightPos - g_position_ws);
	float lightIntensity = max(0, dot(g_normal_ws, lightDir));
	lightIntensity *= sqrt(max(0, (lightRadius - lightDistance) / lightRadius));
	light += lightColor * lightIntensity / (lightDistance * lightDistance);

	// ambient
	light = max(light, 0.001);

	float exposure = 1.618;
	vec3 out_hdr = g_albedo * light * exposure;
	vec3 out_tonemapped = out_hdr / (1 + out_hdr);
	out_color = vec4(out_tonemapped, 1);
}
"]
	struct Dummy;
}
