use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};

fn main() {
    nncalc::train::<Autodiff<NdArray>>(&NdArrayDevice::Cpu)
}
