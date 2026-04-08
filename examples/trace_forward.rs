#![allow(clippy::wildcard_imports, clippy::cast_precision_loss, clippy::uninlined_format_args)]
//! Trace a forward pass layer-by-layer — shows tensor shapes and norms
//! at each stage of the computation.
//!
//! Equivalent to `apr trace --payload --verbose` on a microGPT model.
//!
//! ```text
//! cargo run --example trace_forward
//! ```

use aprender::autograd::{clear_graph, no_grad, Tensor};
use microgpt::*;

fn trace_tensor(label: &str, t: &Tensor) {
    let data = t.data();
    let l2: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    println!(
        "  {label:40} shape={:?}  L2={l2:.4}  range=[{min:.4}, {max:.4}]  μ={mean:.6}",
        t.shape()
    );
}

fn main() {
    println!("microGPT Forward Pass Trace");
    println!("══════════════════════════════════════════════════");
    println!("Equivalent to: apr trace --payload --verbose\n");

    let model = MicroGPT::new();
    let name = "emma";
    let tokens = tokenize(name);
    let input = &tokens[..tokens.len() - 1];
    println!("Input: \"{name}\" → tokens {input:?}\n");

    no_grad(|| {
        // ── Stage 1: Embedding ──────────────────────────────────────────
        println!("Stage 1: Embedding");
        let tok_oh = one_hot(input, VOCAB_SIZE);
        let pos_ids: Vec<usize> = (0..input.len()).collect();
        let pos_oh = one_hot(&pos_ids, BLOCK_SIZE);

        let tok_emb = tok_oh.matmul(&model.wte);
        trace_tensor("tok_emb = one_hot(tokens) @ wte", &tok_emb);

        let pos_emb = pos_oh.matmul(&model.wpe);
        trace_tensor("pos_emb = one_hot(pos) @ wpe", &pos_emb);

        let x = tok_emb.add(&pos_emb);
        trace_tensor("x = tok_emb + pos_emb", &x);

        let x = rms_norm(&x);
        trace_tensor("x = RMSNorm(x)", &x);

        // ── Stage 2: Self-Attention ─────────────────────────────────────
        println!("\nStage 2: Multi-Head Self-Attention");
        let x_res = x.clone();
        let x_n = rms_norm(&x);
        trace_tensor("x_n = RMSNorm(x) [pre-attn]", &x_n);

        let mask = causal_mask(input.len());
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let mut attn_out: Option<Tensor> = None;
        for h in 0..N_HEAD {
            let q_h = x_n.matmul(&model.wq[h]);
            let k_h = x_n.matmul(&model.wk[h]);
            let v_h = x_n.matmul(&model.wv[h]);

            let scores = q_h.matmul(&k_h.transpose()).mul_scalar(scale);
            let scores = scores.add(&mask);
            let weights = scores.softmax();
            let head_out = weights.matmul(&v_h);
            let projected = head_out.matmul(&model.wo[h]);

            trace_tensor(&format!("head_{h}_out @ wo[{h}]"), &projected);

            attn_out = Some(match attn_out {
                Some(acc) => acc.add(&projected),
                None => projected,
            });
        }
        let x = x_res.add(&attn_out.unwrap());
        trace_tensor("x = x_res + attn_out [residual]", &x);

        // ── Stage 3: MLP ────────────────────────────────────────────────
        println!("\nStage 3: MLP (FFN)");
        let x_res = x.clone();
        let x_n = rms_norm(&x);
        trace_tensor("x_n = RMSNorm(x) [pre-MLP]", &x_n);

        let h = x_n.matmul(&model.w_fc1);
        trace_tensor("h = x_n @ w_fc1 [up project]", &h);

        let h = h.relu();
        trace_tensor("h = ReLU(h)", &h);

        let mlp_out = h.matmul(&model.w_fc2);
        trace_tensor("mlp_out = h @ w_fc2 [down project]", &mlp_out);

        let x = x_res.add(&mlp_out);
        trace_tensor("x = x_res + mlp_out [residual]", &x);

        // ── Stage 4: LM Head ────────────────────────────────────────────
        println!("\nStage 4: Language Model Head");
        let logits = x.matmul(&model.w_lm);
        trace_tensor("logits = x @ w_lm", &logits);

        // Show predictions
        println!("\nPredictions:");
        let targets = &tokens[1..];
        for (pos, &target) in targets.iter().enumerate() {
            let row = &logits.data()[pos * VOCAB_SIZE..(pos + 1) * VOCAB_SIZE];
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            let target_char = if target == BOS {
                "EOS".to_string()
            } else {
                CHARS.chars().nth(target - 1).unwrap().to_string()
            };
            let pred_char = if max_idx == BOS {
                "EOS".to_string()
            } else {
                CHARS.chars().nth(max_idx - 1).unwrap().to_string()
            };

            // Softmax for probability
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f32> = row.iter().map(|v| (v - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let target_prob = exps[target] / sum;

            println!(
                "  pos {pos}: target='{target_char}' predicted='{pred_char}' P(target)={target_prob:.4}"
            );
        }
    });

    clear_graph();
}
