pub const BITS_IN: usize = 7;
pub const BITS_OUT: usize = 8;

pub mod data;
pub mod model;
pub mod train;

pub use data::*;
pub use model::*;
pub use train::train;
