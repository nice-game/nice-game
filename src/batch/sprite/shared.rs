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
	pipeline: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
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

		let pipeline = Arc::new(
			GraphicsPipeline::start()
				.vertex_input_single_buffer::<SpriteVertex>()
				.vertex_shader(shaders.vertex_shader().main_entry_point(), ())
				.triangle_list()
				.viewports_dynamic_scissors_irrelevant(1)
				.fragment_shader(shaders.fragment_shader().main_entry_point(), ())
				.render_pass(subpass.clone())
				.build(shaders.device().clone())
				.expect("failed to create pipeline")
		);

		Arc::new(Self {
			shaders: shaders,
			subpass: subpass,
			pipeline: pipeline.clone(),
			sprite_desc_pool: Mutex::new(FixedSizeDescriptorSetsPool::new(pipeline, 1)),
		})
	}

	pub(crate) fn shaders(&self) -> &Arc<SpriteBatchShaders> {
		&self.shaders
	}

	pub(crate) fn subpass(&self) -> &Subpass<Arc<RenderPassAbstract + Send + Sync>> {
		&self.subpass
	}

	pub(crate) fn pipeline(&self) -> &Arc<GraphicsPipelineAbstract + Send + Sync + 'static> {
		&self.pipeline
	}

	pub(crate) fn sprite_desc_pool(
		&self
	) -> &Mutex<FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync + 'static>>> {
		&self.sprite_desc_pool
	}
}
