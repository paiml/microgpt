#![allow(clippy::wildcard_imports, clippy::cast_precision_loss, clippy::uninlined_format_args)]
//! Explain attention mechanics — traces through the scaled dot-product
//! attention computation showing intermediate shapes and values.
//!
//! Equivalent to `apr trace --layer attn --verbose` on a microGPT model.
//!
//! ```text
//! cargo run --example explain_attention
//! ```

use aprender::autograd::{clear_graph, no_grad, Tensor};
use microgpt::*;

fn main() {
    println!("microGPT Attention Explainer");
    println!("══════════════════════════════════════════════════\n");

    println!("Architecture (cf. `apr explain --kernel gpt2 -v`):");
    println!("  Attention type:     MHA (multi-head attention)");
    println!("  Heads:              {N_HEAD}");
    println!("  Head dim:           {HEAD_DIM}");
    println!("  Total attn dim:     {N_EMBD} ({N_HEAD} × {HEAD_DIM})");
    println!("  Scale factor:       1/√{HEAD_DIM} = {:.4}\n", 1.0 / (HEAD_DIM as f32).sqrt());

    let model = MicroGPT::new();
    let tokens = &[BOS, 13, 1, 18]; // "mar"

    no_grad(|| {
        println!("Input: [BOS, 'm', 'a', 'r'] = {:?}", tokens);
        println!("Sequence length: {}\n", tokens.len());

        // Embedding
        let tok_oh = one_hot(tokens, VOCAB_SIZE);
        let pos_oh = one_hot(&[0, 1, 2, 3], BLOCK_SIZE);
        let x = tok_oh.matmul(&model.wte).add(&pos_oh.matmul(&model.wpe));
        let x = rms_norm(&x);
        println!("After embedding + RMSNorm:");
        println!("  Shape: {:?}", x.shape());
        print_row_norms(&x, tokens.len());

        // Attention computation per head
        let x_n = rms_norm(&x);
        let mask = causal_mask(tokens.len());
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        println!("\nPer-head attention (cf. `apr trace --layer attn`):");
        for h in 0..N_HEAD {
            let q_h = x_n.matmul(&model.wq[h]);
            let k_h = x_n.matmul(&model.wk[h]);
            let v_h = x_n.matmul(&model.wv[h]);

            println!("\n  Head {h}:");
            println!("    Q shape: {:?}", q_h.shape());
            println!("    K shape: {:?}", k_h.shape());
            println!("    V shape: {:?}", v_h.shape());

            let scores = q_h.matmul(&k_h.transpose()).mul_scalar(scale);
            let scores = scores.add(&mask);
            let weights = scores.softmax();

            println!("    Attention weights (each row sums to 1.0):");
            let wd = weights.data();
            let n = tokens.len();
            for i in 0..n {
                let row: Vec<String> = (0..n)
                    .map(|j| format!("{:.3}", wd[i * n + j]))
                    .collect();
                println!("      pos {i} → [{}]", row.join(", "));
            }

            let head_out = weights.matmul(&v_h);
            println!("    Head output shape: {:?}", head_out.shape());
        }

        // Causal mask explanation
        println!("\nCausal Mask (cf. `apr explain --tensor attn_mask`):");
        println!("  Each position can only attend to itself and earlier positions.");
        println!("  Future positions are masked with -1e9 → softmax drives them to ~0.");
        let md = mask.data();
        let n = tokens.len();
        for i in 0..n {
            let row: Vec<String> = (0..n)
                .map(|j| {
                    if md[i * n + j] < -1e8 {
                        "  -∞ ".to_string()
                    } else {
                        "  0  ".to_string()
                    }
                })
                .collect();
            println!("    pos {i}: [{}]", row.join(","));
        }
    });

    clear_graph();
}

fn print_row_norms(x: &Tensor, n: usize) {
    let data = x.data();
    let cols = N_EMBD;
    for i in 0..n {
        let row = &data[i * cols..(i + 1) * cols];
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("  pos {i}: L2 norm = {norm:.4}");
    }
}
