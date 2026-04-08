#![allow(clippy::wildcard_imports, clippy::cast_precision_loss, clippy::uninlined_format_args, clippy::print_literal)]
//! Inspect microGPT model structure — shows parameter shapes, counts, and
//! tensor roles using the same introspection patterns as `apr inspect`.
//!
//! ```text
//! cargo run --example inspect_model
//! ```

use aprender::autograd::Tensor;
use microgpt::*;

fn tensor_stats(name: &str, t: &Tensor) {
    let data = t.data();
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let std = (data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();
    println!(
        "  {name:30} {:>12?}  {:>6}  min={min:+.4}  max={max:+.4}  μ={mean:+.6}  σ={std:.6}",
        t.shape(),
        t.numel()
    );
}

fn main() {
    println!("microGPT Model Inspection");
    println!("══════════════════════════════════════════════════\n");

    let model = MicroGPT::new();

    println!("Architecture:");
    println!("  Family:          microGPT (character-level, 1-layer GPT)");
    println!("  Kernel Class:    Custom (MHA + RMSNorm + ReLU)");
    println!("  Hidden Size:     {N_EMBD}");
    println!("  Attention Heads: {N_HEAD} (head_dim={HEAD_DIM})");
    println!("  FFN Size:        {FF_DIM}");
    println!("  Vocab Size:      {VOCAB_SIZE}");
    println!("  Context Length:   {BLOCK_SIZE}");
    println!("  Parameters:      {}\n", model.param_count());

    // ── apr tensors equivalent ──────────────────────────────────────────
    println!("Tensors (cf. `apr tensors`):");
    println!("  {:<30} {:>12}  {:>6}  {}", "Name", "Shape", "Numel", "Stats");
    println!("  {}", "─".repeat(90));

    tensor_stats("wte (token embeddings)", &model.wte);
    tensor_stats("wpe (position embeddings)", &model.wpe);
    for h in 0..N_HEAD {
        tensor_stats(&format!("wq[{h}] (query proj head {h})"), &model.wq[h]);
    }
    for h in 0..N_HEAD {
        tensor_stats(&format!("wk[{h}] (key proj head {h})"), &model.wk[h]);
    }
    for h in 0..N_HEAD {
        tensor_stats(&format!("wv[{h}] (value proj head {h})"), &model.wv[h]);
    }
    for h in 0..N_HEAD {
        tensor_stats(&format!("wo[{h}] (output proj head {h})"), &model.wo[h]);
    }
    tensor_stats("w_fc1 (MLP up projection)", &model.w_fc1);
    tensor_stats("w_fc2 (MLP down projection)", &model.w_fc2);
    tensor_stats("w_lm (LM head)", &model.w_lm);

    // ── apr explain --kernel equivalent ─────────────────────────────────
    println!("\nKernel Pipeline (cf. `apr explain --kernel`):");
    println!("┌─────────────────────────┬─────────────────────────────────┬─────────────────────────┐");
    println!("│ Operation               │ Implementation                  │ Contract                │");
    println!("├─────────────────────────┼─────────────────────────────────┼─────────────────────────┤");
    println!("│ Embedding Lookup        │ one_hot @ wte (differentiable)  │ microgpt-v1 § one_hot   │");
    println!("│ RMSNorm                 │ x * (1/rms) per row             │ microgpt-v1 § rms_norm  │");
    println!("│ Attention (4-head MHA)  │ per-head Q/K/V/O matmul        │ attention-kernel-v1     │");
    println!("│ Causal Mask             │ lower-tri 0, upper -1e9         │ microgpt-v1 § causal    │");
    println!("│ Softmax                 │ aprender autograd softmax       │ softmax-kernel-v1       │");
    println!("│ MLP (ReLU)              │ fc1 → ReLU → fc2               │ activation-kernel-v1    │");
    println!("│ LM Head                 │ matmul to vocab logits          │ matmul-kernel-v1        │");
    println!("└─────────────────────────┴─────────────────────────────────┴─────────────────────────┘");

    // ── apr explain --tensor equivalent ─────────────────────────────────
    println!("\nTensor Roles (cf. `apr explain --tensor`):");
    for (name, role) in [
        ("wte", "Token embedding — maps token IDs to dense vectors"),
        ("wpe", "Position embedding — adds positional information"),
        ("wq[h]", "Query projection in multi-head attention"),
        ("wk[h]", "Key projection in multi-head attention"),
        ("wv[h]", "Value projection in multi-head attention"),
        ("wo[h]", "Output projection — combines head outputs"),
        ("w_fc1", "Feed-forward up projection (expand to 4x)"),
        ("w_fc2", "Feed-forward down projection (compress to 1x)"),
        ("w_lm", "Language model head — projects to vocabulary logits"),
    ] {
        println!("  {name:10} → {role}");
    }

    // ── Forward pass trace ──────────────────────────────────────────────
    println!("\nForward Pass Trace (cf. `apr trace --payload`):");
    let tokens = tokenize("maria");
    let input = &tokens[..tokens.len() - 1];
    println!("  Input: \"maria\" → tokens {:?}", input);

    aprender::autograd::no_grad(|| {
        let logits = model.forward(input);
        println!("  Output shape: {:?}", logits.shape());
        let last_logits = &logits.data()[(input.len() - 1) * VOCAB_SIZE..input.len() * VOCAB_SIZE];
        let max_idx = last_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let predicted = if max_idx == BOS {
            "BOS (end)".to_string()
        } else {
            format!("'{}' (idx {})", CHARS.chars().nth(max_idx - 1).unwrap(), max_idx)
        };
        println!("  Next token prediction (argmax): {predicted}");
    });

    aprender::autograd::clear_graph();
}
