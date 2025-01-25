use std::marker::PhantomData;

use burn::{
    prelude::Backend,
    tensor::{Int, Tensor},
    train::metric::{
        state::{FormatOptions, NumericMetricState},
        Adaptor, ItemLazy, Metric, Numeric,
    },
};

#[derive(Clone, Debug)]
pub struct AddOutput<B: Backend> {
    pub batch: crate::AddBatch<B>,
    pub outputs: Tensor<B, 2>,
    pub loss: Tensor<B, 2>,
}

impl<B: Backend> AddOutput<B> {
    pub fn mean_bit_loss(&self) -> Tensor<B, 1> {
        self.loss.clone().mean_dim(1).squeeze(1)
    }

    pub fn numeric_values(&self) -> Tensor<B, 1, Int> {
        crate::bits_to_numeric(self.outputs.clone().round().int())
    }

    pub fn batch_numeric_values(&self) -> Tensor<B, 1, Int> {
        crate::bits_to_numeric(self.batch.outputs.clone().round().int())
    }
}

impl<B: Backend> ItemLazy for AddOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: Backend> Adaptor<AddOutput<B>> for AddOutput<B> {
    fn adapt(&self) -> AddOutput<B> {
        self.clone()
    }
}

#[derive(Default)]
pub struct BitLossMetric<B: Backend>(NumericMetricState, PhantomData<B>);
impl<B: Backend> Metric for BitLossMetric<B> {
    const NAME: &'static str = "Mean loss per bit";

    type Input = AddOutput<B>;

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &burn::train::metric::MetricMetadata,
    ) -> burn::train::metric::MetricEntry {
        let mean_bit_loss = item.mean_bit_loss();
        let [batch_size] = mean_bit_loss.dims();
        let total_mean_bit_loss = mean_bit_loss
            .mean()
            .to_data()
            .iter()
            .next()
            .expect("No total mean bit loss");

        self.0.update(
            total_mean_bit_loss,
            batch_size,
            FormatOptions::new(Self::NAME).precision(4),
        )
    }

    fn clear(&mut self) {
        self.0.reset();
    }
}

impl<B: Backend> Numeric for BitLossMetric<B> {
    fn value(&self) -> f64 {
        self.0.value()
    }
}

#[derive(Default)]
pub struct NumericValueLossMetric<B: Backend>(NumericMetricState, PhantomData<B>);
impl<B: Backend> Metric for NumericValueLossMetric<B> {
    const NAME: &'static str = "Mean loss per numeric value";

    type Input = AddOutput<B>;

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &burn::train::metric::MetricMetadata,
    ) -> burn::train::metric::MetricEntry {
        let numeric = item.numeric_values();
        let batch_numeric = item.batch_numeric_values();
        let [batch_size] = numeric.dims();
        let loss = (numeric - batch_numeric).abs();
        let mean_numeric_loss = loss
            .mean()
            .to_data()
            .iter()
            .next()
            .expect("No mean numeric loss");

        self.0.update(
            mean_numeric_loss,
            batch_size,
            FormatOptions::new(Self::NAME).precision(4),
        )
    }

    fn clear(&mut self) {
        self.0.reset();
    }
}

impl<B: Backend> Numeric for NumericValueLossMetric<B> {
    fn value(&self) -> f64 {
        self.0.value()
    }
}
