use std::{io::Write, time::Duration};

use burn::{
    lr_scheduler::exponential::ExponentialLrSchedulerConfig,
    module::Module,
    optim::AdamConfig,
    prelude::Backend,
    record::{FullPrecisionSettings, PrettyJsonFileRecorder},
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, TrainOutput, TrainStep, ValidStep},
};

impl<B: AutodiffBackend> TrainStep<crate::AddBatch<B>, crate::AddOutput<B>>
    for crate::model::AddModel<B>
{
    fn step(&self, item: crate::AddBatch<B>) -> burn::train::TrainOutput<crate::AddOutput<B>> {
        let output = self.forward_output(item);
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<crate::AddBatch<B>, crate::AddOutput<B>> for crate::AddModel<B> {
    fn step(&self, item: crate::AddBatch<B>) -> crate::AddOutput<B> {
        self.forward_output(item)
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
        .metric_train_numeric(crate::output::BitLossMetric::default())
        .metric_valid_numeric(crate::output::BitLossMetric::default())
        .metric_train_numeric(crate::output::NumericValueLossMetric::default())
        .metric_valid_numeric(crate::output::NumericValueLossMetric::default())
        .num_epochs(10)
        .devices(vec![device.clone()])
        .build(model, optimizer, lr_scheduler);

    let model = learner.fit(loader_train, loader_valid);

    model
        .save_file(
            format!("{}{}", crate::ARTIFACT_DIR, crate::MODEL_FILE),
            &PrettyJsonFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("Unable to save model");

    std::io::stdout().flush().ok();
    std::thread::sleep(Duration::from_millis(100));
}
