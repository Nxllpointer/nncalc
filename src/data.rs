use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use burn::{
    data::{
        dataloader::{batcher::Batcher, DataLoader, DataLoaderBuilder},
        dataset::Dataset,
    },
    prelude::Backend,
    tensor::Tensor,
};

#[derive(Clone, Debug)]
pub struct AddBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub outputs: Tensor<B, 2>,
}

struct AddDataset;
impl Dataset<usize> for AddDataset {
    fn get(&self, index: usize) -> Option<usize> {
        Some(index)
    }

    fn len(&self) -> usize {
        2_usize.pow(crate::BITS_IN as u32).pow(2)
    }
}

#[derive(Clone, Debug)]
struct AddBatcher<B: Backend>(B::Device);
impl<B: Backend> Batcher<usize, AddBatch<B>> for AddBatcher<B> {
    fn batch(&self, items: Vec<usize>) -> AddBatch<B> {
        let items: Vec<(usize, usize)> = items
            .into_iter()
            .map(|index| {
                let mask = 2_usize.pow(crate::BITS_IN as u32) - 1;
                let first = index & mask;
                let second = (index >> crate::BITS_IN) & mask;
                (first, second)
            })
            .collect();

        let inputs = items
            .iter()
            .map(|(first, second)| {
                let first = bits_tensor(*first, crate::BITS_IN, &self.0);
                let second = bits_tensor(*second, crate::BITS_IN, &self.0);
                Tensor::cat(vec![first, second], 0).to_device(&self.0)
            })
            .collect();

        let outputs = items
            .iter()
            .map(|(first, second)| bits_tensor(first + second, crate::BITS_OUT, &self.0))
            .collect();

        let inputs = Tensor::stack(inputs, 0);
        let outputs = Tensor::stack(outputs, 0);

        AddBatch { inputs, outputs }
    }
}

pub fn create_loader<B: Backend>(device: &B::Device) -> Arc<dyn DataLoader<AddBatch<B>>> {
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

pub fn bits_tensor<B: Backend>(
    mut number: usize,
    num_bits: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let mut bits: Vec<f32> = Vec::new();

    while bits.len() < num_bits {
        bits.push((number & 1) as f32);
        number >>= 1;
    }

    Tensor::from_floats(bits.as_slice(), device)
}
