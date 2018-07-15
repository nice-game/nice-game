use batch::mesh::{ ALBEDO_FORMAT, NORMAL_FORMAT, DEPTH_FORMAT, MeshShaders, TargetVertex, mesh::MeshVertexDefinition };
use std::sync::Arc;
use vulkano::{
	device::Device,
	format::Format,
	framebuffer::{ RenderPassAbstract, Subpass },
	pipeline::{ GraphicsPipeline, GraphicsPipelineAbstract },
};

pub struct MeshRenderPasses {
	shaders: Arc<MeshShaders>,
	gbuffers_subpass0: Subpass<Arc<RenderPassAbstract + Send + Sync>>,
	gbuffers_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
	target_subpass0: Subpass<Arc<RenderPassAbstract + Send + Sync>>,
	target_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync + 'static>,
}
impl MeshRenderPasses {
	pub fn new(shaders: Arc<MeshShaders>, format: Format) -> Arc<Self> {
		let gbuffers_render_pass: Arc<RenderPassAbstract + Send + Sync> =
			Arc::new(
				single_pass_renderpass!(
					shaders.target_vertices.device().clone(),
					attachments: {
						color: { load: Clear, store: Store, format: ALBEDO_FORMAT, samples: 1, },
						normal: { load: Clear, store: Store, format: NORMAL_FORMAT, samples: 1, },
						depth: { load: Clear, store: Store, format: DEPTH_FORMAT, samples: 1, }
					},
					pass: { color: [color, normal], depth_stencil: {depth} }
				)
				.unwrap()
			);
		let gbuffers_subpass0 = Subpass::from(gbuffers_render_pass, 0).unwrap();
		let gbuffers_pipeline =
			Arc::new(
				GraphicsPipeline::start()
					.vertex_input(MeshVertexDefinition::new())
					.vertex_shader(shaders.shader_gbuffers_vertex.main_entry_point(), ())
					.triangle_list()
					.viewports_dynamic_scissors_irrelevant(1)
					.fragment_shader(shaders.shader_gbuffers_fragment.main_entry_point(), ())
					.render_pass(gbuffers_subpass0.clone())
					.depth_stencil_simple_depth()
					.build(shaders.target_vertices.device().clone())
					.expect("failed to create pipeline")
			);

		let target_render_pass: Arc<RenderPassAbstract + Send + Sync> =
			Arc::new(
				single_pass_renderpass!(
					shaders.target_vertices.device().clone(),
					attachments: { out: { load: Clear, store: Store, format: format, samples: 1, } },
					pass: { color: [out], depth_stencil: {} }
				)
				.unwrap()
			);
		let target_subpass0 = Subpass::from(target_render_pass, 0).unwrap();
		let target_pipeline =
			Arc::new(
				GraphicsPipeline::start()
					.vertex_input_single_buffer::<TargetVertex>()
					.vertex_shader(shaders.shader_target_vertex.main_entry_point(), ())
					.triangle_list()
					.viewports_dynamic_scissors_irrelevant(1)
					.fragment_shader(shaders.shader_target_fragment.main_entry_point(), ())
					.render_pass(target_subpass0.clone())
					.build(shaders.target_vertices.device().clone())
					.expect("failed to create pipeline")
			);

		Arc::new(Self {
			shaders: shaders,
			gbuffers_subpass0: gbuffers_subpass0,
			gbuffers_pipeline: gbuffers_pipeline,
			target_subpass0: target_subpass0,
			target_pipeline: target_pipeline,
		})
	}

	pub(crate) fn device(&self) -> &Arc<Device> {
		self.shaders.target_vertices.device()
	}

	pub(crate) fn shaders(&self) -> &Arc<MeshShaders> {
		&self.shaders
	}

	pub(crate) fn gbuffers_render_pass(&self) -> &Arc<RenderPassAbstract + Send + Sync> {
		&self.gbuffers_subpass0.render_pass()
	}

	pub(crate) fn gbuffers_subpass0(&self) -> &Subpass<Arc<RenderPassAbstract + Send + Sync>> {
		&self.gbuffers_subpass0
	}

	pub(crate) fn gbuffers_pipeline(&self) -> &Arc<GraphicsPipelineAbstract + Send + Sync + 'static> {
		&self.gbuffers_pipeline
	}

	pub(crate) fn target_render_pass(&self) -> &Arc<RenderPassAbstract + Send + Sync> {
		&self.target_subpass0.render_pass()
	}

	pub(crate) fn target_subpass0(&self) -> &Subpass<Arc<RenderPassAbstract + Send + Sync>> {
		&self.target_subpass0
	}

	pub(crate) fn target_pipeline(&self) -> &Arc<GraphicsPipelineAbstract + Send + Sync + 'static> {
		&self.target_pipeline
	}
}
