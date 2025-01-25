use burn::{
    module::Module,
    prelude::Backend,
    record::{FullPrecisionSettings, PrettyJsonFileRecorder},
};

pub fn infer<B: Backend>(device: &B::Device, first: usize, second: usize) -> usize {
    let model = crate::AddModel::<B>::new(device);

    let model = model
        .load_file(
            format!("{}{}", crate::ARTIFACT_DIR, crate::MODEL_FILE),
            &PrettyJsonFileRecorder::<FullPrecisionSettings>::new(),
            device,
        )
        .expect("Unable to load model from file");

    let batch = crate::create_batch(device, vec![(first, second)]);

    let output = model.forward_output(batch);

    output
        .numeric_values()
        .into_data()
        .iter::<u32>()
        .next()
        .unwrap() as usize
}
