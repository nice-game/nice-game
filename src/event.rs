use std::collections::HashMap;

pub struct Notifier<T> {
	listeners: HashMap<usize, Box<FnMut(&T)>>,
	next_id: usize,
}
impl<T: Clone> Notifier<T> {
	pub fn new() -> Self {
		Self { listeners: HashMap::new(), next_id: 0 }
	}

	pub fn register(&mut self, listener: Box<FnMut(&T)>) -> usize {
		while self.listeners.contains_key(&self.next_id) {
			self.next_id += 1;
		}

		self.listeners.insert(self.next_id, listener);
		self.next_id
	}

	pub fn unregister(&mut self, key: &usize) -> Option<Box<FnMut(&T)>> {
		self.listeners.remove(key)
	}

	pub fn notify(&mut self, event: &T) {
		for listener in &mut self.listeners.values_mut() {
			listener(event);
		}
	}
}
