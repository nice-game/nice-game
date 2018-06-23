use futures::{ channel::oneshot, executor::ThreadPool, future::{ lazy, ok }, prelude::* };
use num_cpus;
use std::{ io, cmp::min, sync::Mutex };

lazy_static! {
	static ref CPU_POOL: Mutex<CpuPool> = Mutex::new(CpuPool::new(min(1, num_cpus::get() - 1)));
	static ref FS_POOL: Mutex<CpuPool> = Mutex::new(CpuPool::new(1));
}

pub fn spawn_cpu<T, E>(func: impl FnOnce(&mut task::Context) -> Result<T, E> + Send + 'static) -> CpuFuture<T, E>
where
	T: Send + 'static,
	E: Send + 'static
{
	CPU_POOL.lock().unwrap().dispatch(func)
}

pub fn spawn_fs<T, E>(func: impl FnOnce(&mut task::Context) -> Result<T, E> + Send + 'static) -> CpuFuture<T, E>
where
	T: Send + 'static,
	E: Send + 'static
{
	FS_POOL.lock().unwrap().dispatch(func)
}

pub fn spawn_fs_then_cpu<FT, CT, E>(
	func_fs: impl FnOnce(&mut task::Context) -> Result<FT, io::Error> + Send + 'static,
	func_cpu: impl FnOnce(&mut task::Context, FT) -> Result<CT, E> + Send + 'static,
) -> DiskCpuFuture<CT, E>
where
	FT: Send + 'static,
	CT: Send + 'static,
	E: Send + 'static
{
	let future =
		spawn_fs(move |cx| {
			let fs_result = func_fs(cx)?;
			Ok(spawn_cpu(move |cx| func_cpu(cx, fs_result)))
		});

	DiskCpuFuture::new(future)
}

pub struct CpuPool {
	pool: ThreadPool,
}
impl CpuPool {
	pub(super) fn new(thread_count: usize) -> Self {
		Self { pool: ThreadPool::builder().pool_size(thread_count).create().unwrap() }
	}

	pub fn dispatch<T, E>(&mut self, func: impl FnOnce(&mut task::Context) -> Result<T, E> + Send + 'static) -> CpuFuture<T, E>
	where
		T: Send + 'static,
		E: Send + 'static
	{
		let (send, recv) = oneshot::channel();

		#[allow(unused_must_use)]
		self.pool.spawn(Box::new(lazy(move |cx| {
			send.send(func(cx));
			ok(())
		}))).unwrap();

		CpuFuture { recv: recv }
	}
}

pub struct CpuFuture<T, E> {
	recv: oneshot::Receiver<Result<T, E>>,
}
impl<T, E> Future for CpuFuture<T, E> {
	type Item = T;
	type Error = E;

	fn poll(&mut self, cx: &mut task::Context) -> Poll<Self::Item, Self::Error> {
		match self.recv.poll(cx) {
			Ok(Async::Ready(ret)) => match ret {
				Ok(ret) => Ok(Async::Ready(ret)),
				Err(err) => Err(err)
			},
			Ok(Async::Pending) => Ok(Async::Pending),
			Err(_) => unreachable!(),
		}
	}
}

pub struct DiskCpuFuture<T, E> {
	state: DiskCpuState<T, E>,
}
impl<T, E> DiskCpuFuture<T, E> {
	pub fn new(future: CpuFuture<CpuFuture<T, E>, io::Error>) -> Self {
		Self { state: DiskCpuState::LoadingDisk(future) }
	}
}
impl<T, E> Future for DiskCpuFuture<T, E> {
	type Item = T;
	type Error = DiskCpuError<E>;

	fn poll(&mut self, cx: &mut task::Context) -> Poll<Self::Item, Self::Error> {
		let mut new_state = None;

		match &mut self.state {
			DiskCpuState::LoadingDisk(future) => match future.poll(cx)? {
				Async::Ready(subfuture) => new_state = Some(DiskCpuState::LoadingCpu(subfuture)),
				Async::Pending => return Ok(Async::Pending),
			},
			_ => (),
		}

		if let Some(new_state) = new_state {
			self.state = new_state;
		}

		match &mut self.state {
			DiskCpuState::LoadingCpu(future) => match future.poll(cx).map_err(|err| DiskCpuError::CpuError(err))? {
				Async::Ready(val) => Ok(Async::Ready(val)),
				Async::Pending => return Ok(Async::Pending),
			},
			_ => unreachable!(),
		}
	}
}

enum DiskCpuState<T, E> {
	LoadingDisk(CpuFuture<CpuFuture<T, E>, io::Error>),
	LoadingCpu(CpuFuture<T, E>),
}

#[derive(Debug)]
pub enum DiskCpuError<E> {
	DiskError(io::Error),
	CpuError(E),
}
impl<E> From<io::Error> for DiskCpuError<E> {
	fn from(val: io::Error) -> Self {
		DiskCpuError::DiskError(val)
	}
}
