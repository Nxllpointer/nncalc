use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use burn::{
    backend::{ndarray::NdArrayDevice, Autodiff, NdArray},
    data::{
        dataloader::{batcher::Batcher, DataLoader, DataLoaderBuilder},
        dataset::Dataset,
    },
    lr_scheduler::exponential::ExponentialLrSchedulerConfig,
    module::Module,
    nn::{loss::MseLoss, Linear, LinearConfig, Sigmoid, Tanh},
    optim::AdamConfig,
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Tensor},
    train::{
        metric::LossMetric, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};

const BITS_IN: usize = 7;
const BITS_OUT: usize = 8;
const HIDDEN_SIZE: usize = 30;

#[derive(Module, Debug)]
struct AddModel<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
}

impl<B: Backend> AddModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            lin1: LinearConfig::new(BITS_IN * 2, HIDDEN_SIZE).init(device),
            lin2: LinearConfig::new(HIDDEN_SIZE, BITS_OUT).init(device),
        }
    }

    fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        x = self.lin1.forward(x);
        x = Tanh.forward(x);
        x = self.lin2.forward(x);
        x = Sigmoid.forward(x);
        x
    }

    fn forward_regression(&self, batch: AddBatch<B>) -> (RegressionOutput<B>, Tensor<B, 2>) {
        let outputs = self.forward(batch.inputs);
        let loss = MseLoss::new().forward_no_reduction(outputs.clone(), batch.outputs.clone());
        let mean_loss = loss.clone().mean_dim(1).mean();
        let regression_output = RegressionOutput::new(mean_loss, outputs, batch.outputs);
        (regression_output, loss)
    }
}

impl<B: AutodiffBackend> TrainStep<AddBatch<B>, RegressionOutput<B>> for AddModel<B> {
    fn step(&self, item: AddBatch<B>) -> burn::train::TrainOutput<RegressionOutput<B>> {
        let (regression, loss) = self.forward_regression(item);
        TrainOutput::new(self, loss.backward(), regression)
    }
}

impl<B: Backend> ValidStep<AddBatch<B>, RegressionOutput<B>> for AddModel<B> {
    fn step(&self, item: AddBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(item).0
    }
}

struct AddDataset;
impl Dataset<usize> for AddDataset {
    fn get(&self, index: usize) -> Option<usize> {
        Some(index)
    }

    fn len(&self) -> usize {
        2_usize.pow(BITS_IN as u32).pow(2)
    }
}

fn bits_tensor<B: Backend>(mut number: usize, num_bits: usize, device: &B::Device) -> Tensor<B, 1> {
    let mut bits: Vec<f32> = Vec::new();

    while bits.len() < num_bits {
        bits.push((number & 1) as f32);
        number >>= 1;
    }

    Tensor::from_floats(bits.as_slice(), device)
}

#[derive(Clone, Debug)]
struct AddBatch<B: Backend> {
    inputs: Tensor<B, 2>,
    outputs: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
struct AddBatcher<B: Backend>(B::Device);
impl<B: Backend> Batcher<usize, AddBatch<B>> for AddBatcher<B> {
    fn batch(&self, items: Vec<usize>) -> AddBatch<B> {
        let items: Vec<(usize, usize)> = items
            .into_iter()
            .map(|index| {
                let mask = 2_usize.pow(BITS_IN as u32) - 1;
                let first = index & mask;
                let second = (index >> BITS_IN) & mask;
                (first, second)
            })
            .collect();

        let inputs = items
            .iter()
            .map(|(first, second)| {
                let first = bits_tensor(*first, BITS_IN, &self.0);
                let second = bits_tensor(*second, BITS_IN, &self.0);
                Tensor::cat(vec![first, second], 0).to_device(&self.0)
            })
            .collect();

        let outputs = items
            .iter()
            .map(|(first, second)| bits_tensor(first + second, BITS_OUT, &self.0))
            .collect();

        let inputs = Tensor::stack(inputs, 0);
        let outputs = Tensor::stack(outputs, 0);

        AddBatch { inputs, outputs }
    }
}

fn create_loader<B: Backend>(device: &B::Device) -> Arc<dyn DataLoader<AddBatch<B>>> {
    DataLoaderBuilder::new(AddBatcher(device.clone()))
        .batch_size(100)
        .shuffle(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        )
        .build(AddDataset)
}

fn train<B: AutodiffBackend>(device: &B::Device) {
    let model = AddModel::<B>::new(device);
    let optimizer = AdamConfig::new().init();
    let lr_scheduler = ExponentialLrSchedulerConfig::new(0.1, 0.999)
        .init()
        .unwrap();

    let loader_train = create_loader(device);
    let loader_valid = create_loader(device);

    let learner = LearnerBuilder::<B, _, _, _, _, _>::new("./learn")
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(10)
        .devices(vec![device.clone()])
        .build(model, optimizer, lr_scheduler);

    learner.fit(loader_train, loader_valid);
}

fn main() {
    type Backend = Autodiff<NdArray>;
    let device = NdArrayDevice::Cpu;

    train::<Backend>(&device);
}
