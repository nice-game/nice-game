use { cpu_pool::{ spawn_fs_then_cpu, DiskCpuFuture } };
use nom::{ is_space, space };
use std::{ fs::File, io::prelude::*, path::{ Path, PathBuf } };

pub struct Obj {

}
impl Obj {
	fn new() -> Self {
		Self {}
	}

	pub fn from_file<P: AsRef<Path> + Send + 'static>(path: P) -> DiskCpuFuture<Obj, String> {
		spawn_fs_then_cpu(
			move |_| {
				let mut buf = String::new();
				File::open(path)?.read_to_string(&mut buf).map(|_| buf)
			},
			move |_, buf| Ok(obj(&buf).map_err(|err| format!("{}", err))?.1)
		)
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
