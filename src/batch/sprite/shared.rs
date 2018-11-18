use super::shaders::{ SpriteBatchShaders, SpriteVertex };
use std::sync::{ Arc, Mutex };
use vulkano::{
	descriptor::descriptor_set::FixedSizeDescriptorSetsPool,
	format::Format,
	framebuffer::{ RenderPassAbstract, Subpass },
	pipeline::{ GraphicsPipeline, GraphicsPipelineAbstract },
};

pub struct SpriteBatchShared {
	shaders: Arc<SpriteBatchShaders>,
	subpass: Subpass<Arc<RenderPassAbstract + Send + Sync>>,
	pipeline_sprite: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
	pipeline_text: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
	sprite_desc_pool: Mutex<FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>>,
}
impl SpriteBatchShared {
	pub fn new(shaders: Arc<SpriteBatchShaders>, format: Format) -> Arc<Self> {
		let subpass =
			Subpass::from(
				Arc::new(
					single_pass_renderpass!(
						shaders.device().clone(),
						attachments: { color: { load: Clear, store: Store, format: format, samples: 1, } },
						pass: { color: [color], depth_stencil: {} }
					).expect("failed to create render pass")
				) as Arc<RenderPassAbstract + Send + Sync>,
				0
			).expect("failed to create subpass");

		let pipeline_sprite = Arc::new(
			GraphicsPipeline::start()
				.vertex_input_single_buffer::<SpriteVertex>()
				.vertex_shader(shaders.sprite_vertex_shader().main_entry_point(), ())
				.triangle_list()
				.viewports_dynamic_scissors_irrelevant(1)
				.fragment_shader(shaders.sprite_fragment_shader().main_entry_point(), ())
				.render_pass(subpass.clone())
				.build(shaders.device().clone())
				.expect("failed to create pipeline")
		);

		let pipeline_text = Arc::new(
			GraphicsPipeline::start()
				.vertex_input_single_buffer::<SpriteVertex>()
				.vertex_shader(shaders.text_vertex_shader().main_entry_point(), ())
				.triangle_list()
				.viewports_dynamic_scissors_irrelevant(1)
				.fragment_shader(shaders.text_fragment_shader().main_entry_point(), ())
				.render_pass(subpass.clone())
				.blend_alpha_blending()
				.build(shaders.device().clone())
				.expect("failed to create pipeline")
		);

		Arc::new(Self {
			shaders: shaders,
			subpass: subpass,
			pipeline_sprite: pipeline_sprite.clone(),
			pipeline_text: pipeline_text,
			sprite_desc_pool: Mutex::new(FixedSizeDescriptorSetsPool::new(pipeline_sprite, 1)),
		})
	}

	pub(crate) fn shaders(&self) -> &Arc<SpriteBatchShaders> {
		&self.shaders
	}

	pub(crate) fn subpass(&self) -> &Subpass<Arc<RenderPassAbstract + Send + Sync>> {
		&self.subpass
	}

	pub(crate) fn pipeline_sprite(&self) -> &Arc<GraphicsPipelineAbstract + Send + Sync + 'static> {
		&self.pipeline_sprite
	}

	pub(crate) fn pipeline_text(&self) -> &Arc<GraphicsPipelineAbstract + Send + Sync + 'static> {
		&self.pipeline_text
	}

	pub(crate) fn sprite_desc_pool(
		&self
	) -> &Mutex<FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>> {
		&self.sprite_desc_pool
	}
}
