use futures::{ channel::oneshot, executor::ThreadPool, future::{ lazy, ok }, prelude::* };

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
