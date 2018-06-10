mod mesh;
mod mesh_batch;
mod shaders;
mod shared;

pub use self::mesh::{ Mesh, MeshVertex };
pub use self::mesh_batch::MeshBatch;
pub use self::shaders::{ MeshBatchShaders, MeshBatchShadersError };
pub use self::shared::MeshBatchShared;
use vulkano::format::Format;

const NORMAL_FORMAT: Format = Format::R32G32B32A32Sfloat;
const DEPTH_FORMAT: Format = Format::D16Unorm;

#[derive(Debug, Clone)]
struct TargetVertex { position: [f32; 2] }
impl_vertex!(TargetVertex, position);
