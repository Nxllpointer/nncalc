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
    tensor::{Int, Tensor},
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

        create_batch(&self.0, items)
    }
}

pub fn create_batch<B: Backend>(device: &B::Device, items: Vec<(usize, usize)>) -> AddBatch<B> {
    let inputs = items
        .iter()
        .map(|(first, second)| {
            let first = bits_tensor(*first, crate::BITS_IN, device);
            let second = bits_tensor(*second, crate::BITS_IN, device);
            Tensor::cat(vec![first, second], 0).to_device(device)
        })
        .collect();

    let outputs = items
        .iter()
        .map(|(first, second)| bits_tensor(first + second, crate::BITS_OUT, device))
        .collect();

    let inputs = Tensor::stack(inputs, 0);
    let outputs = Tensor::stack(outputs, 0);

    AddBatch { inputs, outputs }
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

pub fn bits_to_numeric<B: Backend>(bits: Tensor<B, 2, Int>) -> Tensor<B, 1, Int> {
    let [batch_size, bit_count] = bits.dims();
    let mult: Vec<u32> = (0..bit_count)
        .into_iter()
        .map(|bit| 2u32.pow(bit as u32))
        .collect();
    let mult: Tensor<B, 2, Int> = Tensor::<B, 1, Int>::from_ints(mult.as_slice(), &bits.device())
        .unsqueeze::<2>()
        .repeat_dim(0, batch_size);

    bits.mul(mult).sum_dim(1).squeeze(1)
}
