mod immutable;
mod target;

pub use self::immutable::{ ImmutableTexture, TextureError };
pub use self::target::TargetTexture;
pub use image::ImageFormat;
use std::sync::Arc;
use vulkano::image::ImageViewAccess;

pub trait Texture {
	fn image(&self) -> &Arc<ImageViewAccess + Send + Sync + 'static>;
}
