use libc::c_void;

const GGD_API_VERSION: u64 = 0;

#[no_mangle]
pub extern fn GGD_DriverMain(X: *mut GGD_DriverContext) -> GGDriverStatus {
	let X = &*X;

	if X.Version == GGD_API_VERSION {
		(X.RegisterRenderEngine)(&mut RENDER_ENGINE);

		GGDriverStatus::GGD_STATUS_DRIVER_READY
	} else {
		GGDriverStatus::GGD_STATUS_VERSION_INVALID
	}
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub enum GGDriverStatus {
	GGD_STATUS_DRIVER_INVALID = 0,
	GGD_STATUS_DRIVER_READY = 1,
	GGD_STATUS_DRIVER_ERROR = 2,
	GGD_STATUS_VERSION_INVALID = 3,
}

#[allow(non_camel_case_types)]
#[repr(packed)]
pub struct GGD_DriverContext {
	Version: u64,
	RegisterRenderEngine: extern fn (*mut GGD_RenderEngine),
	RegisterPhysicsEngine: extern fn (*mut GGD_PhysicsEngine),
}

#[allow(non_camel_case_types)]
#[repr(packed)]
pub struct GGD_RenderEngine {
	Name: *const char,
	Priority: u64,
	Validate: Option<extern fn () -> i32>,
	Shutdown: Option<extern fn (*mut GGD_RenderEngine) -> i32>,

	Window_Alloc: extern fn (sdlwindow: *mut c_void) -> *mut GGD_Window,
	Window_Free: extern fn (*mut GGD_Window),
	Window_IsValid: extern fn (*mut GGD_Window) -> i32,
	Window_Resize: extern fn (*mut GGD_Window, w: u32, h: u32),
	Window_Draw: extern fn (*mut GGD_Window, *mut GGD_ImageData),

	MeshData_Alloc: extern fn () -> *mut GGD_MeshData,
	MeshData_Free: extern fn (*mut GGD_MeshData),
	MeshData_Prepare: extern fn (*mut GGD_MeshData),
	MeshData_SetCacheData: Option<extern fn (*mut GGD_MeshData, buffer: *const c_void, size: u32) -> i32>,
	MeshData_GetCacheData: Option<extern fn (*mut GGD_MeshData, buffer: *mut c_void, size: *mut u32) -> i32>,
	MeshData_SetDistanceData: Option<extern fn (*mut GGD_MeshData, buffer: *const c_void, x: u32, y: u32, z: u32, format: GGDistanceFormat)>,
	MeshData_GetDistanceData: Option<extern fn (*mut GGD_MeshData, buffer: *mut c_void, x: u32, y: u32, z: u32, format: *mut GGDistanceFormat)>,
	MeshData_SetVertexData: extern fn (*mut GGD_MeshData, buffer: *const c_void, count: u32, format: GGVertexFormat),
	MeshData_GetVertexData: extern fn (*mut GGD_MeshData, buffer: *mut c_void, count: *mut u32, format: *mut GGVertexFormat),
	MeshData_SetIndexData: extern fn (*mut GGD_MeshData, buffer: *const c_void, count: u32, format: GGIndexFormat),
	MeshData_GetIndexData: extern fn (*mut GGD_MeshData, buffer: *mut c_void, count: *mut u32, format: *mut GGIndexFormat),
	MeshData_UseIndexData: Option<extern fn (*mut GGD_MeshData, src: *mut GGD_MeshData)>,

	ImageData_Alloc: extern fn () ->  *mut GGD_ImageData,
	ImageData_Free: extern fn (*mut GGD_ImageData),
	ImageData_Prepare: extern fn (*mut GGD_ImageData),
	ImageData_SetCacheData: Option<extern fn (image: *mut GGD_ImageData, buffer: *const c_void, size: u32) -> i32>,
	ImageData_GetCacheData: Option<extern fn (image: *mut GGD_ImageData, buffer: *mut c_void, size: *mut u32) -> i32>,
	ImageData_SetPixelData: extern fn (image: *mut GGD_ImageData, buffer: *const c_void, x: u32, y: u32, z: u32, format: GGPixelFormat),
	ImageData_GetPixelData: extern fn (image: *mut GGD_ImageData, buffer: *mut c_void, x: *mut u32, y: *mut u32, z: *mut u32, format: *mut GGPixelFormat),
	ImageData_Blur: extern fn (dst: *mut GGD_ImageData, src: *mut GGD_ImageData, radius: f32),

	MeshBatch_Alloc: extern fn () -> *mut GGD_MeshBatch,
	MeshBatch_Free: extern fn (*mut GGD_MeshBatch),
	MeshBatch_SetCacheData: Option<extern fn (*mut GGD_MeshBatch, buffer: *const c_void, size: u32) -> i32>,
	MeshBatch_GetCacheData: Option<extern fn (*mut GGD_MeshBatch, buffer: *mut c_void, size: *mut u32) -> i32>,

	MeshInstance_Alloc: extern fn (*mut GGD_MeshBatch) -> *mut GGD_MeshInstance,
	MeshInstance_Free: extern fn (*mut GGD_MeshInstance),
	MeshInstance_SetCacheData: Option<extern fn (*mut GGD_MeshInstance, buffer: *const c_void, size: u32) -> i32>,
	MeshInstance_GetCacheData: Option<extern fn (*mut GGD_MeshInstance, buffer: *mut c_void, size: *mut u32) -> i32>,
	MeshInstance_SetMeshData: extern fn (*mut GGD_MeshInstance, mesh: *mut GGD_MeshData, index: u32),
	MeshInstance_SetImageData: extern fn (*mut GGD_MeshInstance, image: *mut GGD_ImageData, layer: GGMaterialLayer),
	MeshInstance_SetAnimation: extern fn (*mut GGD_MeshInstance, firstIndex: u32, lastIndex: u32, frameRate: f32),
	MeshInstance_SetTransform: extern fn (*mut GGD_MeshInstance, pose: *mut GGTransform),
	MeshInstance_SetBoneTransform: extern fn (*mut GGD_MeshInstance, bone: u32, pose: *mut GGTransform),

	Camera_Alloc: extern fn () -> *mut GGD_Camera,
	Camera_Free: extern fn (*mut GGD_Camera),
	Camera_SetPerspective: extern fn (*mut GGD_Camera, aspect: f32, fovx: f32, zNear: f32, zFar: f32),
	Camera_SetOrthographic: extern fn (*mut GGD_Camera, w: f32, h: f32, zNear: f32, zFar: f32),
	Camera_SetParabolic: extern fn (*mut GGD_Camera, scale: f32),
	Camera_SetMeshBatch: extern fn (*mut GGD_Camera, *mut GGD_MeshBatch),
	Camera_SetTransform: extern fn (*mut GGD_Camera, *mut GGTransform),
	Camera_Draw: extern fn (*mut GGD_Camera, output: *mut GGD_ImageData),
}

#[allow(non_camel_case_types)]
#[repr(packed)]
pub struct GGD_PhysicsEngine {
	Name: *const char,
	Priority: u64,
	Validate: Option<extern fn () -> i32>,
	Shutdown: Option<extern fn (*mut GGD_PhysicsEngine) -> i32>,

	Shape_Alloc: extern fn () -> *mut GGD_Shape,
	Shape_Free: extern fn (*mut GGD_Shape),
	Shape_SetCacheData: extern fn (*mut GGD_Shape, buffer: *const c_void, size: u32) -> i32,
	Shape_GetCacheData: extern fn (*mut GGD_Shape, buffer: *mut c_void, size: *mut u32) -> i32,
	Shape_SetBox: extern fn (*mut GGD_Shape, x: f32, y: f32, z: f32),
	Shape_SetSphere: extern fn (*mut GGD_Shape, radius: f32),
	Shape_SetCylinder: extern fn (*mut GGD_Shape, radius: f32, height: f32),
	Shape_SetConvexMesh: extern fn (*mut GGD_Shape, vertices: *const c_void, count: u32, format: GGVertexFormat),
	Shape_SetTriangleMesh: extern fn (*mut GGD_Shape, vertices: *const c_void, vcount: u32, vformat: GGVertexFormat,
												indices: *const c_void, icount: u32, iformat: GGIndexFormat),
	Shape_SetDistanceData: extern fn (*mut GGD_Shape, buffer: *const c_void, x: u32, y: u32, z: u32, format: GGDistanceFormat),

	Simulation_Alloc: extern fn () -> *mut GGD_Simulation,
	Simulation_Free: extern fn (*mut GGD_Simulation),
	Simulation_Gravity: extern fn (*mut GGD_Simulation, x: f32, y: f32, z: f32),
	Simulation_Update: extern fn (*mut GGD_Simulation, dt: f32),

	ShapeInstance_Alloc: extern fn (*mut GGD_Simulation, *mut GGD_Shape) -> *mut GGD_ShapeInstance,
	ShapeInstance_Free: extern fn (*mut GGD_ShapeInstance),
	ShapeInstance_SetMass: extern fn (*mut GGD_ShapeInstance, mass: f32),
	ShapeInstance_GetMass: extern fn (*mut GGD_ShapeInstance) -> f32,
	ShapeInstance_SetFriction: extern fn (*mut GGD_ShapeInstance, friction: f32),
	ShapeInstance_GetFriction: extern fn (*mut GGD_ShapeInstance) -> f32,
	ShapeInstance_SetVelocity: extern fn (*mut GGD_ShapeInstance, poseDt: *mut GGTransform),
	ShapeInstance_GetVelocity: extern fn (*mut GGD_ShapeInstance, poseDt: *mut GGTransform),
	ShapeInstance_SetTransform: extern fn (*mut GGD_ShapeInstance, pose: *mut GGTransform),
	ShapeInstance_GetTransform: extern fn (*mut GGD_ShapeInstance, pose: *mut GGTransform),
	ShapeInstance_SetVelocityPoi32er: extern fn (*mut GGD_ShapeInstance, poseDtPtr: *mut GGTransform),
	ShapeInstance_SetTransformPoi32er: extern fn (*mut GGD_ShapeInstance, posePtr: *mut GGTransform),
}
