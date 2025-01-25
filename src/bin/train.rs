use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};

fn main() {
    sp_project::train::<Autodiff<NdArray>>(&NdArrayDevice::Cpu)
}
