//! Contract trait enforcement tests — verify YAML contract invariants at runtime.
//!
//! Each test maps to a proof obligation in contracts/microgpt-v1.yaml.

use aprender::autograd::{clear_graph, no_grad, Tensor};
use microgpt::*;

// ── PARAM-COUNT-001: Model parameter count equals 4,192 ────────────────────

#[test]
fn contract_param_count_001() {
    let model = MicroGPT::new();
    assert_eq!(model.param_count(), 4192);
    clear_graph();
}

// ── ONEHOT-ROW-001: Each row sums to exactly 1.0 ──────────────────────────

#[test]
fn contract_onehot_row_001() {
    for n_classes in [3, 10, VOCAB_SIZE] {
        let indices: Vec<usize> = (0..n_classes).collect();
        let t = one_hot(&indices, n_classes);
        let d = t.data();
        for row in 0..n_classes {
            let sum: f32 = d[row * n_classes..(row + 1) * n_classes].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "row {row} sum = {sum} (expected 1.0)"
            );
        }
    }
}

// ── MASK-CAUSAL-001: Lower-triangular structure ────────────────────────────

#[test]
fn contract_mask_causal_001() {
    for size in [1, 4, BLOCK_SIZE] {
        let m = causal_mask(size);
        let d = m.data();
        for i in 0..size {
            for j in 0..size {
                if j <= i {
                    assert!(
                        (d[i * size + j]).abs() < f32::EPSILON,
                        "mask[{i}][{j}] should be 0, got {}",
                        d[i * size + j]
                    );
                } else {
                    assert!(
                        (d[i * size + j] - (-1e9)).abs() < 1.0,
                        "mask[{i}][{j}] should be -1e9, got {}",
                        d[i * size + j]
                    );
                }
            }
        }
    }
}

// ── TOKENIZE-ROUNDTRIP-001: decode(tokenize(s)) == s for [a-z]* ────────────

#[test]
fn contract_tokenize_roundtrip_001() {
    for name in ["a", "hello", "abcdefghijklmnopqrstuvwxyz", ""] {
        let tokens = tokenize(name);
        assert_eq!(tokens[0], BOS, "must start with BOS");
        assert_eq!(*tokens.last().unwrap(), BOS, "must end with BOS");
        let decoded = decode(&tokens[1..tokens.len() - 1]);
        assert_eq!(decoded, name, "roundtrip failed for '{name}'");
    }
}

// ── ADAM-MONOTONIC-001: Step counter advances by 1 ─────────────────────────

#[test]
fn contract_adam_monotonic_001() {
    let mut opt = Adam::new();
    let mut params: Vec<&mut Tensor> = vec![];
    for expected in 1..=10 {
        opt.step(&mut params, 0.01);
        assert_eq!(opt.t, expected);
    }
}

// ── ADAM-V-NONNEG-001: Second moment non-negative ──────────────────────────

#[test]
fn contract_adam_v_nonneg_001() {
    let mut p = Tensor::from_vec(vec![1.0, -2.0, 3.0], &[1, 3]).requires_grad();
    let loss = p.sum();
    loss.backward();

    let mut opt = Adam::new();
    let mut params = vec![&mut p];
    opt.step(&mut params, 0.01);

    for &vi in &opt.v[0] {
        assert!(vi >= 0.0, "second moment must be non-negative, got {vi}");
    }
    clear_graph();
}

// ── FORWARD-SHAPE-001: Output shape [n, VOCAB_SIZE] ────────────────────────

#[test]
fn contract_forward_shape_001() {
    no_grad(|| {
        let model = MicroGPT::new();
        for n in [1, 3, 8, BLOCK_SIZE] {
            let tokens: Vec<usize> = (0..n).collect();
            let logits = model.forward(&tokens);
            assert_eq!(
                logits.shape(),
                &[n, VOCAB_SIZE],
                "forward({n} tokens) shape mismatch"
            );
        }
    });
    clear_graph();
}
