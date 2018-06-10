use { CPU_POOL, FS_POOL, cpu_pool::CpuFuture };
use futures::prelude::*;
use nom::{ is_space, space };
use std::{ fs::File, io::{ self, prelude::* }, path::{ Path, PathBuf } };

pub struct Obj {

}
impl Obj {
	fn new() -> Self {
		Self {}
	}

	pub fn from_file<P: AsRef<Path> + Send + 'static>(path: P) -> ObjFuture {
		let future = FS_POOL.lock().unwrap()
			.dispatch(move |_| {
				let mut buf = String::new();
				File::open(path)?.read_to_string(&mut buf)?;

				Ok(CPU_POOL.lock().unwrap().dispatch(move |_| Ok(obj(&buf).map_err(|err| format!("{}", err))?.1)))
			});

		ObjFuture { state: ObjState::LoadingDisk(future) }
	}
}

pub struct ObjFuture {
	state: ObjState,
}
impl Future for ObjFuture {
	type Item = Obj;
	type Error = ObjError;

	fn poll(&mut self, cx: &mut task::Context) -> Poll<Self::Item, Self::Error> {
		let mut new_state = None;

		match &mut self.state {
			ObjState::LoadingDisk(future) => match future.poll(cx)? {
				Async::Ready(subfuture) => new_state = Some(ObjState::LoadingCpu(subfuture)),
				Async::Pending => return Ok(Async::Pending),
			},
			_ => (),
		}

		if let Some(new_state) = new_state {
			self.state = new_state;
		}

		match &mut self.state {
			ObjState::LoadingCpu(future) => match future.poll(cx)? {
				Async::Ready(data) => Ok(Async::Ready(data)),
				Async::Pending => return Ok(Async::Pending),
			},
			_ => unreachable!(),
		}
	}
}

enum ObjState {
	LoadingDisk(CpuFuture<CpuFuture<Obj, String>, io::Error>),
	LoadingCpu(CpuFuture<Obj, String>),
}

#[derive(Debug)]
pub enum ObjError {
	IoError(io::Error),
	ParseError(String),
}
impl From<String> for ObjError {
	fn from(val: String) -> Self {
		ObjError::ParseError(val)
	}
}
impl From<io::Error> for ObjError {
	fn from(val: io::Error) -> Self {
		ObjError::IoError(val)
	}
}

enum Line {
	Comment,
	Mtllib(Vec<PathBuf>),
}

named!(obj<&str, Obj>,
	do_parse!(
		obj:
			fold_many0!(
				line,
				Obj::new(),
				|obj, line| {
					match line {
						Line::Comment => (),
						Line::Mtllib(_) => warn!("ignoring `mtllib` line because it's currently not supported"),
					}
					obj
				}
			) >>
		eof!() >>
		(obj)
	)
);

named!(line<&str, Line>,
	do_parse!(
		line:
			alt!(
				map!(comment, |_| Line::Comment) |
				map!(mtllib, |paths| Line::Mtllib(paths))
			) >>
		(line)
	)
);

named!(comment<&str, ()>,
	do_parse!(
		tag!("#") >>
		take_until_and_consume!("\n") >>
		()
	)
);

named!(mtllib<&str, Vec<PathBuf>>,
	do_parse!(
		tag!("mtllib") >>
		space >>
		paths: many1!(path) >>
		(paths)
	)
);

// TODO: support escaped spaces, maybe paths inside quotes
named!(path<&str, PathBuf>,
	do_parse!(
		path: take_while1!(|ch| !is_space(ch as u8)) >>
		(Path::new(path).to_owned())
	)
);
