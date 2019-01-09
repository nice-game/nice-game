use futures::{
	channel::oneshot,
	executor::ThreadPool,
	future::lazy,
	prelude::*,
	task::{ LocalWaker, Poll, SpawnExt }
};
use num_cpus;
use std::{ cmp::min, pin::Pin, sync::Mutex };
use vulkano::sync::{ FenceSignalFuture, FlushError, GpuFuture };

lazy_static! {
	static ref CPU_POOL: Mutex<CpuPool> = Mutex::new(CpuPool::new(min(1, num_cpus::get() - 1)));
	static ref EXECUTOR_POOL: Mutex<ThreadPool> = Mutex::new(ThreadPool::builder().pool_size(1).create().unwrap());
	static ref FS_POOL: Mutex<CpuPool> = Mutex::new(CpuPool::new(1));
}

pub fn execute_future(future: impl Future<Output = ()> + Unpin + Send + 'static) {
	EXECUTOR_POOL.lock().unwrap().spawn(Box::new(future)).unwrap();
}

pub fn spawn_cpu<T, E>(func: impl FnOnce() -> Result<T, E> + Send + 'static) -> CpuFuture<T, E>
where
	T: Send + 'static,
	E: Send + 'static
{
	CPU_POOL.lock().unwrap().dispatch(func)
}

pub fn spawn_fs<T, E>(func: impl FnOnce() -> Result<T, E> + Send + 'static) -> CpuFuture<T, E>
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

	pub fn dispatch<T, E>(&mut self, func: impl FnOnce() -> Result<T, E> + Send + 'static) -> CpuFuture<T, E>
	where
		T: Send + 'static,
		E: Send + 'static
	{
		let (send, recv) = oneshot::channel();
		self.pool.spawn(lazy(|_| { send.send(func()).ok(); })).unwrap();
		CpuFuture { recv: recv }
	}
}

pub struct CpuFuture<T, E> {
	recv: oneshot::Receiver<Result<T, E>>,
}
impl<T, E> Future for CpuFuture<T, E> {
	type Output = Result<T, E>;

	fn poll(mut self: Pin<&mut Self>, lw: &LocalWaker) -> Poll<Self::Output> {
		oneshot::Receiver::poll(Pin::new(&mut self.recv), lw).map(|val| val.unwrap())
	}
}

pub struct GpuFutureFuture<T: GpuFuture> {
	future: FenceSignalFuture<T>
}
impl<T: GpuFuture> GpuFutureFuture<T> {
	pub fn new(future: T) -> Result<Self, FlushError> {
		Ok(Self { future: future.then_signal_fence_and_flush()? })
	}
}
impl<T: GpuFuture> Future for GpuFutureFuture<T> {
	type Output = Result<(), FlushError>;

	fn poll(self: Pin<&mut Self>, _lw: &LocalWaker) -> Poll<Self::Output> {
		match self.future.wait(Some(Default::default())) {
			Ok(()) => Poll::Ready(Ok(())),
			Err(FlushError::Timeout) => Poll::Pending,
			Err(err) => Poll::Ready(Err(err)),
		}
	}
}
