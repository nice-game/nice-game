use std::sync::Arc;
use vulkano::{
	OomError,
	buffer::{ BufferUsage, ImmutableBuffer },
	device::Device,
	memory::DeviceMemoryAllocError,
	sampler::{ BorderColor, Filter, MipmapMode, Sampler, SamplerAddressMode, SamplerCreationError },
	sync::GpuFuture,
};
use window::Window;

pub struct SpriteBatchShaders {
	device: Arc<Device>,
	vertices: Arc<ImmutableBuffer<[SpriteVertex; 6]>>,
	sprite_vertex_shader: sprite_vs::Shader,
	sprite_fragment_shader: sprite_fs::Shader,
	sprite_sampler: Arc<Sampler>,
	text_vertex_shader: text_vs::Shader,
	text_fragment_shader: text_fs::Shader,
	text_sampler: Arc<Sampler>,
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
				sprite_vertex_shader: sprite_vs::Shader::load(window.device().clone())?,
				sprite_fragment_shader: sprite_fs::Shader::load(window.device().clone())?,
				sprite_sampler:
					Sampler::new(
						window.device().clone(),
						Filter::Linear,
						Filter::Linear, MipmapMode::Nearest,
						SamplerAddressMode::Repeat,
						SamplerAddressMode::Repeat,
						SamplerAddressMode::Repeat,
						0.0, 1.0, 0.0, 0.0
					)?,
				text_vertex_shader: text_vs::Shader::load(window.device().clone())?,
				text_fragment_shader: text_fs::Shader::load(window.device().clone())?,
				text_sampler:
					Sampler::new(
						window.device().clone(),
						Filter::Linear,
						Filter::Linear, MipmapMode::Nearest,
						SamplerAddressMode::ClampToBorder(BorderColor::FloatTransparentBlack),
						SamplerAddressMode::ClampToBorder(BorderColor::FloatTransparentBlack),
						SamplerAddressMode::ClampToBorder(BorderColor::FloatTransparentBlack),
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

	pub(crate) fn sprite_vertex_shader(&self) -> &sprite_vs::Shader {
		&self.sprite_vertex_shader
	}

	pub(crate) fn sprite_fragment_shader(&self) -> &sprite_fs::Shader {
		&self.sprite_fragment_shader
	}

	pub(crate) fn text_vertex_shader(&self) -> &text_vs::Shader {
		&self.text_vertex_shader
	}

	pub(crate) fn text_fragment_shader(&self) -> &text_fs::Shader {
		&self.text_fragment_shader
	}

	pub(crate) fn sprite_sampler(&self) -> &Arc<Sampler> {
		&self.sprite_sampler
	}

	pub(crate) fn text_sampler(&self) -> &Arc<Sampler> {
		&self.text_sampler
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

mod sprite_vs {
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

mod sprite_fs {
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

mod text_vs {
	::vulkano_shaders::shader!{
		ty: "vertex",
		src: "#version 450
layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;

layout(set = 0, binding = 0) uniform Target { uvec2 size; } target;
layout(set = 1, binding = 0) uniform SpriteDynamic { vec2 pos; } sprite_dynamic;
layout(set = 2, binding = 0) uniform GlyphStatic { ivec2 pos; } glyph_static;
layout(set = 2, binding = 1) uniform sampler2D tex;

void main() {
	tex_coords = position;
	gl_Position = vec4(2 * (sprite_dynamic.pos + glyph_static.pos + textureSize(tex, 0) * position) / target.size - 1, 0.0, 1.0);
}
"
	}
}

mod text_fs {
	::vulkano_shaders::shader!{
		ty: "fragment",
		src: "#version 450
layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 2, binding = 1) uniform sampler2D tex;

void main() {
	f_color = vec4(1, 1, 1, texture(tex, tex_coords).r);
}
"
	}
}
