//! microGPT training binary.

#![allow(clippy::cast_precision_loss)]

use aprender::autograd::{clear_graph, no_grad, Tensor};
use aprender::nn::loss::CrossEntropyLoss;
use microgpt::{
    sample, tokenize, Adam, MicroGPT, BLOCK_SIZE, LR, N_EMBD, N_HEAD, NUM_STEPS, TEMPERATURE,
    VOCAB_SIZE,
};

fn main() {
    let names: Vec<&str> = include_str!("../data/names.txt")
        .lines()
        .filter(|l| !l.is_empty())
        .collect();

    println!("microGPT — {VOCAB_SIZE} vocab, {N_EMBD} embed, {N_HEAD} heads, {BLOCK_SIZE} ctx");
    println!("Training on {} names for {NUM_STEPS} steps\n", names.len());

    let mut model = MicroGPT::new();
    let loss_fn = CrossEntropyLoss::new();
    let mut optimizer = Adam::new();

    println!("Parameters: {}", model.param_count());

    for step in 0..NUM_STEPS {
        let name = names[step % names.len()].to_lowercase();
        let tokens = tokenize(&name);
        let n = (tokens.len() - 1).min(BLOCK_SIZE);
        if n == 0 {
            continue;
        }

        let input = &tokens[..n];
        let targets: Vec<f32> = tokens[1..=n].iter().map(|&t| t as f32).collect();

        let logits = model.forward(input);
        let target_tensor = Tensor::from_slice(&targets);
        let loss = loss_fn.forward(&logits, &target_tensor);

        loss.backward();
        let lr = LR * (1.0 - step as f32 / NUM_STEPS as f32);
        let mut params = model.parameters_mut();
        optimizer.step(&mut params, lr);
        clear_graph();

        if step % 100 == 0 || step == NUM_STEPS - 1 {
            println!("step {:4} | loss {:.4} | lr {:.5}", step, loss.item(), lr);
        }
    }

    println!("\nGenerated names (temperature={TEMPERATURE}):");
    no_grad(|| {
        for i in 0..20 {
            let name = sample(&model);
            println!("  {:2}: {name}", i + 1);
        }
    });
}
