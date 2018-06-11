use cgmath::{ vec3, vec4, Vector3, Vector4 };
use nom::{ self, alphanumeric, digit, float_s, line_ending, space, space0 };
use std::{ mem, path::Path };

pub struct Obj {
	root_object: Object,
	named_objects: Vec<(String, Object)>,
}
impl Obj {
	pub fn from_str(s: &str) -> Result<Self, nom::Err<&str>> {
		let (remaining, obj) = obj(s)?;
		if remaining.is_empty() {
			Ok(obj)
		} else {
			Err(nom::Err::Error(nom::Context::Code(remaining, nom::ErrorKind::Eof)))
		}
	}
}

#[derive(Debug)]
pub struct Face {
	vertices: Vec<FaceVertex>,
}
impl Face {
	fn new(vertices: Vec<FaceVertex>) -> Self {
		Self { vertices: vertices }
	}
}

#[derive(Debug)]
pub struct FaceVertex {
	position: usize,
	texture: Option<usize>,
	normal: Option<usize>,
}
impl FaceVertex {
	fn new(position: usize, texture: Option<usize>, normal: Option<usize>) -> Self {
		Self { position: position, texture: texture, normal: normal }
	}
}

pub struct Object {
	vertices: Vec<Vector4<f32>>,
	normals: Vec<Vector3<f32>>,
	faces: Vec<Face>,
}
impl Object {
	fn new() -> Self {
		Self { vertices: vec![], normals: vec![], faces: vec![] }
	}
}

struct ObjBuilder {
	root_object: Option<Object>,
	named_objects: Vec<(String, Object)>,
	active_object: (String, Object),
}
impl ObjBuilder {
	fn new() -> Self {
		Self { root_object: None, named_objects: vec![], active_object: ("".to_owned(), Object::new()) }
	}

	fn add_object(&mut self, name: &str) {
		let mut named_object = (name.to_owned(), Object::new());
		mem::swap(&mut self.active_object, &mut named_object);
		if self.root_object.is_some() {
			self.named_objects.push(named_object);
		} else {
			self.root_object = Some(named_object.1);
		};
	}

	fn add_face(&mut self, face: Face) {
		self.active_object.1.faces.push(face);
	}

	fn add_normal(&mut self, normal: Vector3<f32>) {
		self.active_object.1.normals.push(normal);
	}

	fn add_vertex(&mut self, vertex: Vector4<f32>) {
		self.active_object.1.vertices.push(vertex);
	}

	fn build(mut self) -> Obj {
		let (root_object, named_objects) =
			if let Some(root_object) = self.root_object {
				self.named_objects.push(self.active_object);
				(root_object, self.named_objects)
			} else {
				(self.active_object.1, self.named_objects)
			};
		Obj { root_object: root_object, named_objects: named_objects }
	}
}

enum Line<'a> {
	Empty,
	Face(Face),
	Mtllib(Vec<&'a Path>),
	ObjectName(&'a str),
	SmoothGroup(u8),
	Usemtl(&'a str),
	VertexNormal(Vector3<f32>),
	VertexPosition(Vector4<f32>),
}

named!(comment<&str, ()>,
	do_parse!(
		tag!("#") >>
		take_until!("\n") >>
		line_ending >>
		()
	)
);

named!(empty<&str, ()>,
	do_parse!(
		space0 >>
		line_ending >>
		(())
	)
);

named!(face<&str, Face>,
	do_parse!(
		tag!("f") >>
		face_vertices:
			many1!(do_parse!(
				space >>
				v: map!(return_error!(ErrorKind::Digit, digit), |v| v.parse().unwrap()) >>
				vtn:
					map!(
						opt!(do_parse!(
							tag!("/") >>
							vt:
								alt!(
									map!(digit, |vt| Some(vt.parse().unwrap())) |
									map!(peek!(tag!("/")), |_| None)
								) >>
							vn:
								opt!(do_parse!(
									tag!("/") >>
									vn: map!(digit, |vt| vt.parse().unwrap()) >>
									(vn)
								)) >>
							(vt, vn)
						)),
						|vtn| vtn.unwrap_or((None, None))
					) >>
				(FaceVertex::new(v, vtn.0, vtn.1))
			)) >>
		space0 >>
		line_ending >>
		(Face::new(face_vertices))
	)
);

named!(line<&str, Line>,
	do_parse!(
		line:
			alt!(
				map!(comment, |_| Line::Empty) |
				map!(empty, |_| Line::Empty) |
				map!(face, |face| Line::Face(face)) |
				map!(mtllib, |paths| Line::Mtllib(paths)) |
				map!(object_name, |name| Line::ObjectName(name)) |
				map!(smooth_group, |group| Line::SmoothGroup(group)) |
				map!(usemtl, |name| Line::Usemtl(name)) |
				map!(vertex_normal, |normal| Line::VertexNormal(normal)) |
				map!(vertex_position, |position| Line::VertexPosition(position))
			) >>
		(line)
	)
);

named!(mtllib<&str, Vec<&Path>>,
	do_parse!(
		tag!("mtllib") >>
		space >>
		paths: return_error!(ErrorKind::Many1, many1!(do_parse!(path: path >> opt!(space) >> (path)))) >>
		space0 >>
		line_ending >>
		(paths)
	)
);

named!(obj<&str, Obj>,
	do_parse!(
		obj:
			fold_many0!(
				complete!(line),
				ObjBuilder::new(),
				|mut builder: ObjBuilder, line| {
					match line {
						Line::Empty => (),
						Line::Face(face) => {
							debug!("parsed f {:?}", face);
							builder.add_face(face);
						}
						Line::Mtllib(paths) => {
							debug!("parsed mtllib {:?}", paths);
							warn!("ignoring mtllib because it's currently not supported");
						},
						Line::ObjectName(name) => {
							debug!("parsed o {:?}", name);
							builder.add_object(name)
						},
						Line::SmoothGroup(group) => {
							debug!("parsed s {:?}", group);
							warn!("ignoring s because it's currently not supported");
						},
						Line::Usemtl(name) => {
							debug!("parsed usemtl {:?}", name);
							warn!("ignoring usemtl because it's currently not supported");
						},
						Line::VertexNormal(vec) => {
							debug!("parsed vn {:?}", vec);
							builder.add_normal(vec);
						},
						Line::VertexPosition(vec) => {
							debug!("parsed v {:?}", vec);
							builder.add_vertex(vec);
						},
					}
					builder
				}
			) >>
		(obj.build())
	)
);

named!(object_name<&str, &str>,
	do_parse!(
		tag!("o") >>
		space >>
		name: return_error!(ErrorKind::AlphaNumeric, alphanumeric) >>
		space0 >>
		line_ending >>
		(name)
	)
);

named!(smooth_group<&str, u8>,
	do_parse!(
		tag!("s") >>
		space >>
		group:
			return_error!(
				ErrorKind::Alt,
				alt!(map!(digit, |group| group.parse().unwrap()) | map!(tag!("off"), |_| 0))
			) >>
		space0 >>
		line_ending >>
		(group)
	)
);

named!(usemtl<&str, &str>,
	do_parse!(
		tag!("usemtl") >>
		space >>
		name: return_error!(ErrorKind::AlphaNumeric, alphanumeric) >>
		space0 >>
		line_ending >>
		(name)
	)
);

named!(vertex_normal<&str, Vector3<f32>>,
	do_parse!(
		tag!("vn") >>
		space >>
		x: return_error!(ErrorKind::Alt, float_s) >>
		return_error!(ErrorKind::Space, space) >>
		y: return_error!(ErrorKind::Alt, float_s) >>
		return_error!(ErrorKind::Space, space) >>
		z: return_error!(ErrorKind::Alt, float_s) >>
		space0 >>
		line_ending >>
		(vec3(x, y, z))
	)
);

named!(vertex_position<&str, Vector4<f32>>,
	do_parse!(
		tag!("v") >>
		space >>
		x: return_error!(ErrorKind::Alt, float_s) >>
		return_error!(ErrorKind::Space, space) >>
		y: return_error!(ErrorKind::Alt, float_s) >>
		return_error!(ErrorKind::Space, space) >>
		z: return_error!(ErrorKind::Alt, float_s) >>
		w: opt!(do_parse!(space >> w: float_s >> (w))) >>
		space0 >>
		line_ending >>
		(vec4(x, y, z, w.unwrap_or(1.0)))
	)
);

// TODO: support escaped spaces, paths inside quotes
named!(path<&str, &Path>,
	do_parse!(
		path: take_until_either1!(" \n\t\r") >>
		(Path::new(path))
	)
);
