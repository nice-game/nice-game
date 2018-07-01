use futures::{ channel::oneshot, executor::ThreadPool, future::{ lazy, ok }, prelude::* };
use num_cpus;
use std::{ cmp::min, sync::Mutex };

lazy_static! {
	static ref CPU_POOL: Mutex<CpuPool> = Mutex::new(CpuPool::new(min(1, num_cpus::get() - 1)));
	static ref EXECUTOR_POOL: Mutex<ThreadPool> = Mutex::new(ThreadPool::builder().pool_size(1).create().unwrap());
	static ref FS_POOL: Mutex<CpuPool> = Mutex::new(CpuPool::new(1));
}

pub fn execute_future(future: impl Future<Item = (), Error = Never> + Send + 'static) {
	EXECUTOR_POOL.lock().unwrap().spawn(Box::new(future)).unwrap();
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
