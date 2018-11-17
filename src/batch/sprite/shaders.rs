use std::sync::Arc;
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	device::Device,
	memory::DeviceMemoryAllocError,
	sampler::{ Filter, MipmapMode, Sampler, SamplerAddressMode, SamplerCreationError },
	sync::GpuFuture,
};
use window::Window;

pub struct SpriteBatchShaders {
	device: Arc<Device>,
	vertices: Arc<ImmutableBuffer<[SpriteVertex; 6]>>,
	vertex_shader: vs::Shader,
	fragment_shader: fs::Shader,
	sampler: Arc<Sampler>,
}
impl SpriteBatchShaders {
	pub fn new(window: &mut Window) -> Result<(Arc<Self>, impl GpuFuture), SpriteBatchShadersError> {
		let (vertices, future) =
			ImmutableBuffer::from_data(
				[
					SpriteVertex { position: [0.0, 0.0] },
					SpriteVertex { position: [1.0, 0.0] },
					SpriteVertex { position: [0.0, 1.0] },
					SpriteVertex { position: [0.0, 1.0] },
					SpriteVertex { position: [1.0, 0.0] },
					SpriteVertex { position: [1.0, 1.0] },
				],
				BufferUsage::vertex_buffer(),
				window.queue().clone(),
			)?;

		Ok((
			Arc::new(Self {
				device: window.device().clone(),
				vertices: vertices,
				vertex_shader: vs::Shader::load(window.device().clone())?,
				fragment_shader: fs::Shader::load(window.device().clone())?,
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
			future
		))
	}

	pub(crate) fn device(&self) -> &Arc<Device> {
		&self.device
	}

	pub(crate) fn vertices(&self) -> &Arc<ImmutableBuffer<[SpriteVertex; 6]>> {
		&self.vertices
	}

	pub(crate) fn vertex_shader(&self) -> &vs::Shader {
		&self.vertex_shader
	}

	pub(crate) fn fragment_shader(&self) -> &fs::Shader {
		&self.fragment_shader
	}

	pub(crate) fn sampler(&self) -> &Arc<Sampler> {
		&self.sampler
	}
}

#[derive(Debug)]
pub enum SpriteBatchShadersError {
	DeviceMemoryAllocError(DeviceMemoryAllocError),
	OomError(OomError),
	TooManyObjects,
}
impl From<DeviceMemoryAllocError> for SpriteBatchShadersError {
	fn from(val: DeviceMemoryAllocError) -> Self {
		SpriteBatchShadersError::DeviceMemoryAllocError(val)
	}
}
impl From<OomError> for SpriteBatchShadersError {
	fn from(val: OomError) -> Self {
		SpriteBatchShadersError::OomError(val)
	}
}
impl From<SamplerCreationError> for SpriteBatchShadersError {
	fn from(val: SamplerCreationError) -> Self {
		match val {
			SamplerCreationError::OomError(err) => SpriteBatchShadersError::OomError(err),
			SamplerCreationError::TooManyObjects => SpriteBatchShadersError::TooManyObjects,
			_ => unreachable!(),
		}
	}
}

#[derive(Debug, Clone)]
pub(crate) struct SpriteVertex { position: [f32; 2] }
impl_vertex!(SpriteVertex, position);

mod vs {
	::vulkano_shaders::shader!{
		ty: "vertex",
		src: "#version 450
layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;

layout(set = 0, binding = 0) uniform Target {
	uvec2 size;
} target;

layout(set = 1, binding = 0) uniform SpriteDynamic {
	vec2 pos;
} sprite_dynamic;

layout(set = 2, binding = 0) uniform sampler2D tex;

void main() {
	tex_coords = position;
	gl_Position = vec4(2 * (sprite_dynamic.pos + textureSize(tex, 0) * position) / target.size - 1, 0.0, 1.0);
}
"
	}
}

mod fs {
	::vulkano_shaders::shader!{
		ty: "fragment",
		src: "#version 450
layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 2, binding = 0) uniform sampler2D tex;

void main() {
	f_color = texture(tex, tex_coords);
}
"
	}
}
