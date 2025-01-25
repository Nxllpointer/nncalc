pub const ARTIFACT_DIR: &str = "./learn/";
pub const MODEL_FILE: &str = "model.json";

pub const BITS_IN: usize = 7;
pub const BITS_OUT: usize = 8;

pub mod data;
pub mod infer;
pub mod model;
pub mod output;
pub mod train;

pub use data::*;
pub use infer::infer;
pub use model::*;
pub use output::*;
pub use train::train;
