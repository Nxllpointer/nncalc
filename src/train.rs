use std::{io::Write, time::Duration};

use burn::{
    lr_scheduler::exponential::ExponentialLrSchedulerConfig,
    optim::AdamConfig,
    prelude::Backend,
    tensor::backend::AutodiffBackend,
    train::{
        metric::LossMetric, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};

impl<B: AutodiffBackend> TrainStep<crate::AddBatch<B>, RegressionOutput<B>>
    for crate::model::AddModel<B>
{
    fn step(&self, item: crate::AddBatch<B>) -> burn::train::TrainOutput<RegressionOutput<B>> {
        let (regression, loss) = self.forward_regression(item);
        TrainOutput::new(self, loss.backward(), regression)
    }
}

impl<B: Backend> ValidStep<crate::AddBatch<B>, RegressionOutput<B>> for crate::AddModel<B> {
    fn step(&self, item: crate::AddBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(item).0
    }
}

pub fn train<B: AutodiffBackend>(device: &B::Device) {
    let model = crate::AddModel::<B>::new(device);
    let optimizer = AdamConfig::new().init();
    let lr_scheduler = ExponentialLrSchedulerConfig::new(0.1, 0.999)
        .init()
        .unwrap();

    let loader_train = crate::create_loader(device);
    let loader_valid = crate::create_loader(device);

    let learner = LearnerBuilder::<B, _, _, _, _, _>::new("./learn")
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(10)
        .devices(vec![device.clone()])
        .build(model, optimizer, lr_scheduler);

    learner.fit(loader_train, loader_valid);
    std::io::stdout().flush().ok();
    std::thread::sleep(Duration::from_millis(100));
}
