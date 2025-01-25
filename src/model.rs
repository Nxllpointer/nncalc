use burn::{
    module::Module,
    nn::{loss::MseLoss, Linear, LinearConfig, Sigmoid, Tanh},
    prelude::Backend,
    tensor::Tensor,
};

const HIDDEN_SIZE: usize = 30;

#[derive(Module, Debug)]
pub struct AddModel<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
}

impl<B: Backend> AddModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            lin1: LinearConfig::new(crate::BITS_IN * 2, HIDDEN_SIZE).init(device),
            lin2: LinearConfig::new(HIDDEN_SIZE, crate::BITS_OUT).init(device),
        }
    }

    pub fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        x = self.lin1.forward(x);
        x = Tanh.forward(x);
        x = self.lin2.forward(x);
        x = Sigmoid.forward(x);
        x
    }

    pub fn forward_output(&self, batch: crate::AddBatch<B>) -> crate::AddOutput<B> {
        let outputs = self.forward(batch.inputs.clone());
        let loss = MseLoss::new().forward_no_reduction(outputs.clone(), batch.outputs.clone());
        crate::AddOutput {
            batch,
            outputs,
            loss,
        }
    }
}
