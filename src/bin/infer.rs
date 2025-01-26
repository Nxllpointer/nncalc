use std::io::{BufRead, Write};

use burn::backend::{ndarray::NdArrayDevice, NdArray};

fn main() {
    let mut stdin = std::io::stdin().lock().lines();
    let mut prompt_number = |message: &str| {
        print!("{message}");
        std::io::stdout().flush().ok();
        stdin
            .next()
            .unwrap()
            .unwrap()
            .parse::<usize>()
            .expect("Not a number")
    };

    loop {
        let first: usize = prompt_number("First number: ");
        let second: usize = prompt_number("Second number: ");

        println!(
            "{first} + {second} = {}",
            nncalc::infer::<NdArray>(&NdArrayDevice::Cpu, first, second)
        );
    }
}
