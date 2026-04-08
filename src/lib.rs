//! microGPT — Karpathy's 200-line GPT ported to Rust with aprender.
//!
//! A 4,192-parameter GPT trained on 32K names using character-level tokenization.
//! Architecture: 1 layer, 4 heads, `embed_dim=16`, `block_size=16`.
//!
//! Reference: <https://karpathy.github.io/2026/02/12/microgpt/>

use aprender::autograd::{no_grad, Tensor};
use rand::Rng;

// ── Architecture constants ──────────────────────────────────────────────────

pub const VOCAB_SIZE: usize = 27;
pub const N_EMBD: usize = 16;
pub const N_HEAD: usize = 4;
pub const HEAD_DIM: usize = N_EMBD / N_HEAD;
pub const BLOCK_SIZE: usize = 16;
pub const FF_DIM: usize = 4 * N_EMBD;
pub const BOS: usize = 0;
pub const CHARS: &str = "abcdefghijklmnopqrstuvwxyz";

// ── Training hyperparameters ────────────────────────────────────────────────

pub const NUM_STEPS: usize = 5000;
pub const LR: f32 = 0.01;
pub const BETA1: f32 = 0.85;
pub const BETA2: f32 = 0.99;
pub const INIT_STD: f32 = 0.08;
pub const TEMPERATURE: f32 = 0.5;

// ── Model ───────────────────────────────────────────────────────────────────

pub struct MicroGPT {
    pub wte: Tensor,
    pub wpe: Tensor,
    pub wq: Vec<Tensor>,
    pub wk: Vec<Tensor>,
    pub wv: Vec<Tensor>,
    pub wo: Vec<Tensor>,
    pub w_fc1: Tensor,
    pub w_fc2: Tensor,
    pub w_lm: Tensor,
}

impl MicroGPT {
    pub fn new() -> Self {
        Self {
            wte: randn(&[VOCAB_SIZE, N_EMBD]),
            wpe: randn(&[BLOCK_SIZE, N_EMBD]),
            wq: (0..N_HEAD).map(|_| randn(&[N_EMBD, HEAD_DIM])).collect(),
            wk: (0..N_HEAD).map(|_| randn(&[N_EMBD, HEAD_DIM])).collect(),
            wv: (0..N_HEAD).map(|_| randn(&[N_EMBD, HEAD_DIM])).collect(),
            wo: (0..N_HEAD).map(|_| randn(&[HEAD_DIM, N_EMBD])).collect(),
            w_fc1: randn(&[N_EMBD, FF_DIM]),
            w_fc2: randn(&[FF_DIM, N_EMBD]),
            w_lm: randn(&[N_EMBD, VOCAB_SIZE]),
        }
    }

    /// Forward pass: tokens → logits `[n, VOCAB_SIZE]`.
    ///
    /// Contract: `forward_pass` — output shape `[n, VOCAB_SIZE]`, all finite.
    pub fn forward(&self, tokens: &[usize]) -> Tensor {
        let n = tokens.len();
        let tok_oh = one_hot(tokens, VOCAB_SIZE);
        let pos_oh = one_hot(&(0..n).collect::<Vec<_>>(), BLOCK_SIZE);
        let mut x = tok_oh.matmul(&self.wte).add(&pos_oh.matmul(&self.wpe));
        x = rms_norm(&x);

        // Multi-head self-attention
        let x_res = x.clone();
        let x_n = rms_norm(&x);
        let mask = causal_mask(n);
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let mut attn_out: Option<Tensor> = None;
        for h in 0..N_HEAD {
            let q_h = x_n.matmul(&self.wq[h]);
            let k_h = x_n.matmul(&self.wk[h]);
            let v_h = x_n.matmul(&self.wv[h]);

            let scores = q_h.matmul(&k_h.transpose()).mul_scalar(scale);
            let scores = scores.add(&mask);
            let weights = scores.softmax();
            let head_out = weights.matmul(&v_h);

            let projected = head_out.matmul(&self.wo[h]);
            attn_out = Some(match attn_out {
                Some(acc) => acc.add(&projected),
                None => projected,
            });
        }
        x = x_res.add(&attn_out.expect("at least 1 head"));

        // MLP: fc1 → ReLU → fc2
        let x_res = x.clone();
        let x_n = rms_norm(&x);
        let h = x_n.matmul(&self.w_fc1).relu();
        x = x_res.add(&h.matmul(&self.w_fc2));

        x.matmul(&self.w_lm)
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut p: Vec<&mut Tensor> = vec![
            &mut self.wte,
            &mut self.wpe,
            &mut self.w_fc1,
            &mut self.w_fc2,
            &mut self.w_lm,
        ];
        for t in &mut self.wq {
            p.push(t);
        }
        for t in &mut self.wk {
            p.push(t);
        }
        for t in &mut self.wv {
            p.push(t);
        }
        for t in &mut self.wo {
            p.push(t);
        }
        p
    }

    pub fn param_count(&self) -> usize {
        let mut n = self.wte.numel() + self.wpe.numel();
        n += self.w_fc1.numel() + self.w_fc2.numel() + self.w_lm.numel();
        for v in [&self.wq, &self.wk, &self.wv, &self.wo] {
            n += v.iter().map(Tensor::numel).sum::<usize>();
        }
        n
    }
}

impl Default for MicroGPT {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Gaussian random tensor (std=`INIT_STD`) with gradient tracking.
pub fn randn(shape: &[usize]) -> Tensor {
    let mut rng = rand::rng();
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|_| {
            let u1: f32 = rng.random::<f32>().max(1e-7);
            let u2: f32 = rng.random::<f32>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * INIT_STD
        })
        .collect();
    Tensor::from_vec(data, shape).requires_grad()
}

/// `RMSNorm`: `x / sqrt(mean(x^2) + eps)`, applied per row.
///
/// Contract: `rms_norm` — shape preserved, output finite for finite nonzero input.
pub fn rms_norm(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let cols = shape[shape.len() - 1];
    let rows = x.numel() / cols;
    let data = x.data();

    let mut scale_data = Vec::with_capacity(x.numel());
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        #[allow(clippy::cast_precision_loss)]
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv_rms = 1.0 / (ms + 1e-5).sqrt();
        scale_data.extend(std::iter::repeat_n(inv_rms, cols));
    }
    let scale = Tensor::from_vec(scale_data, shape);
    x.mul(&scale)
}

/// One-hot encode indices → `[n, num_classes]` matrix.
///
/// Contract: `one_hot` — exactly one 1.0 per row, all others 0.0.
pub fn one_hot(indices: &[usize], num_classes: usize) -> Tensor {
    let n = indices.len();
    let mut data = vec![0.0f32; n * num_classes];
    for (i, &idx) in indices.iter().enumerate() {
        data[i * num_classes + idx] = 1.0;
    }
    Tensor::new(&data, &[n, num_classes])
}

/// Lower-triangular causal mask (future positions → large negative).
///
/// Contract: `causal_mask` — diagonal and below are 0.0, above is `-1e9`.
/// Uses `-1e9` instead of `-inf` to satisfy upstream softmax preconditions
/// while achieving identical numerical behavior after exponentiation.
pub fn causal_mask(size: usize) -> Tensor {
    let mut data = vec![0.0f32; size * size];
    for i in 0..size {
        for j in (i + 1)..size {
            data[i * size + j] = -1e9;
        }
    }
    Tensor::new(&data, &[size, size])
}

/// Weighted random sampling from a probability distribution.
///
/// Contract: `weighted_sample` — returns index in `0..probs.len()`.
pub fn weighted_sample(probs: &[f32]) -> usize {
    let mut rng = rand::rng();
    let r: f32 = rng.random();
    weighted_sample_with_r(probs, r)
}

/// Deterministic weighted sampling (exposed for testing).
pub fn weighted_sample_with_r(probs: &[f32], r: f32) -> usize {
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r <= cumulative {
            return i;
        }
    }
    probs.len() - 1
}

// ── Tokenizer ───────────────────────────────────────────────────────────────

/// Tokenize a name: `[BOS, chars..., BOS]`. Non-lowercase chars are dropped.
///
/// Contract: `tokenize_decode` — roundtrip identity for lowercase alpha strings.
pub fn tokenize(name: &str) -> Vec<usize> {
    let chars: Vec<char> = CHARS.chars().collect();
    let mut tokens = vec![BOS];
    for ch in name.chars() {
        if let Some(idx) = chars.iter().position(|&c| c == ch) {
            tokens.push(idx + 1);
        }
    }
    tokens.push(BOS);
    tokens
}

/// Decode token sequence back to a string (strips BOS tokens).
pub fn decode(tokens: &[usize]) -> String {
    let chars: Vec<char> = CHARS.chars().collect();
    tokens
        .iter()
        .filter_map(|&t| if t == BOS { None } else { chars.get(t - 1).copied() })
        .collect()
}

// ── Adam optimizer ──────────────────────────────────────────────────────────

/// Adam optimizer (Kingma & Ba, 2015) operating directly on autograd Tensors.
///
/// Contract: `adam_step` — second moment `v >= 0`, step counter monotonically advances.
pub struct Adam {
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub t: usize,
}

impl Adam {
    pub fn new() -> Self {
        Self {
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    pub fn step(&mut self, params: &mut [&mut Tensor], lr: f32) {
        self.t += 1;
        if self.m.len() < params.len() {
            self.m.resize(params.len(), Vec::new());
            self.v.resize(params.len(), Vec::new());
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let t_i32 = self.t as i32;
        let bc1 = 1.0 - BETA1.powi(t_i32);
        let bc2 = 1.0 - BETA2.powi(t_i32);

        for (i, param) in params.iter_mut().enumerate() {
            let Some(grad) = aprender::autograd::get_grad(param.id()) else {
                continue;
            };
            let g = grad.data();
            let p = param.data_mut();

            if self.m[i].len() != p.len() {
                self.m[i] = vec![0.0; p.len()];
                self.v[i] = vec![0.0; p.len()];
            }

            let m = &mut self.m[i];
            let v = &mut self.v[i];

            for j in 0..p.len() {
                m[j] = BETA1 * m[j] + (1.0 - BETA1) * g[j];
                v[j] = BETA2 * v[j] + (1.0 - BETA2) * g[j] * g[j];
                let m_hat = m[j] / bc1;
                let v_hat = v[j] / bc2;
                p[j] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
            }
        }
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self::new()
    }
}

// ── Sampling ────────────────────────────────────────────────────────────────

/// Generate a name by sampling from the model autoregressively.
pub fn sample(model: &MicroGPT) -> String {
    no_grad(|| {
        let mut tokens = vec![BOS];
        for _ in 0..BLOCK_SIZE {
            let logits = model.forward(&tokens);
            let last = tokens.len() - 1;
            let logit_data = logits.data();
            let last_logits: Vec<f32> = logit_data[last * VOCAB_SIZE..(last + 1) * VOCAB_SIZE]
                .iter()
                .map(|&l| l / TEMPERATURE)
                .collect();
            let probs = aprender::nn::functional::softmax_1d(&last_logits);
            let token_id = weighted_sample(&probs);
            if token_id == BOS {
                break;
            }
            tokens.push(token_id);
        }
        decode(&tokens[1..])
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::autograd::clear_graph;

    // ── one_hot ─────────────────────────────────────────────────────────

    #[test]
    fn one_hot_shape() {
        let t = one_hot(&[0, 1, 2], 5);
        assert_eq!(t.shape(), &[3, 5]);
    }

    #[test]
    fn one_hot_values() {
        let t = one_hot(&[0, 2], 3);
        let d = t.data();
        // Row 0: [1,0,0]
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], 0.0);
        assert_eq!(d[2], 0.0);
        // Row 1: [0,0,1]
        assert_eq!(d[3], 0.0);
        assert_eq!(d[4], 0.0);
        assert_eq!(d[5], 1.0);
    }

    #[test]
    fn one_hot_row_sum() {
        let t = one_hot(&[3, 0, 5, 1], 8);
        let d = t.data();
        for row in 0..4 {
            let sum: f32 = d[row * 8..(row + 1) * 8].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "row {row} sum = {sum}");
        }
    }

    // ── causal_mask ─────────────────────────────────────────────────────

    #[test]
    fn causal_mask_size_1() {
        let m = causal_mask(1);
        assert_eq!(m.shape(), &[1, 1]);
        assert_eq!(m.data()[0], 0.0);
    }

    #[test]
    fn causal_mask_lower_zero() {
        let m = causal_mask(4);
        let d = m.data();
        for i in 0..4 {
            for j in 0..=i {
                assert_eq!(d[i * 4 + j], 0.0, "mask[{i}][{j}] should be 0");
            }
        }
    }

    #[test]
    fn causal_mask_upper_large_negative() {
        let m = causal_mask(4);
        let d = m.data();
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert!(d[i * 4 + j] < -1e8,
                    "mask[{i}][{j}] should be large negative, got {}", d[i * 4 + j]);
            }
        }
    }

    #[test]
    fn causal_mask_shape() {
        let m = causal_mask(BLOCK_SIZE);
        assert_eq!(m.shape(), &[BLOCK_SIZE, BLOCK_SIZE]);
    }

    // ── tokenize / decode ───────────────────────────────────────────────

    #[test]
    fn tokenize_wraps_bos() {
        let t = tokenize("abc");
        assert_eq!(t[0], BOS);
        assert_eq!(*t.last().unwrap(), BOS);
    }

    #[test]
    fn tokenize_values_in_range() {
        let t = tokenize("hello");
        for &tok in &t {
            assert!(tok < VOCAB_SIZE, "token {tok} >= VOCAB_SIZE");
        }
    }

    #[test]
    fn tokenize_decode_roundtrip() {
        for name in ["hello", "world", "a", "abcdefghijklmnopqrstuvwxyz"] {
            let tokens = tokenize(name);
            let decoded = decode(&tokens[1..tokens.len() - 1]);
            assert_eq!(decoded, name, "roundtrip failed for '{name}'");
        }
    }

    #[test]
    fn tokenize_strips_non_alpha() {
        let t = tokenize("a-b c");
        let decoded = decode(&t[1..t.len() - 1]);
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn decode_empty() {
        assert_eq!(decode(&[]), "");
    }

    #[test]
    fn decode_strips_bos() {
        assert_eq!(decode(&[BOS, 1, 2, BOS]), "ab");
    }

    // ── rms_norm ────────────────────────────────────────────────────────

    #[test]
    fn rms_norm_shape_preserved() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = rms_norm(&x);
        assert_eq!(y.shape(), &[2, 3]);
    }

    #[test]
    fn rms_norm_finite_output() {
        let x = Tensor::new(&[0.5, -1.0, 2.0, 0.3], &[2, 2]);
        let y = rms_norm(&x);
        for &v in y.data() {
            assert!(v.is_finite(), "rms_norm output not finite: {v}");
        }
    }

    #[test]
    fn rms_norm_known_values() {
        // Single row [3, 4]: rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        // normalized: [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
        let x = Tensor::new(&[3.0, 4.0], &[1, 2]);
        let y = rms_norm(&x);
        let d = y.data();
        assert!((d[0] - 0.8485).abs() < 0.01, "got {}", d[0]);
        assert!((d[1] - 1.1314).abs() < 0.01, "got {}", d[1]);
    }

    // ── weighted_sample ─────────────────────────────────────────────────

    #[test]
    fn weighted_sample_deterministic() {
        assert_eq!(weighted_sample_with_r(&[0.0, 0.0, 1.0], 0.5), 2);
        assert_eq!(weighted_sample_with_r(&[1.0, 0.0, 0.0], 0.5), 0);
    }

    #[test]
    fn weighted_sample_boundary() {
        // r=0.0 should return first nonzero
        assert_eq!(weighted_sample_with_r(&[0.5, 0.5], 0.0), 0);
        // r=1.0 exactly: falls through to last
        assert_eq!(weighted_sample_with_r(&[0.5, 0.5], 1.0), 1);
    }

    #[test]
    fn weighted_sample_fallback() {
        // All zeros: cumulative never reaches r, falls to last index
        assert_eq!(weighted_sample_with_r(&[0.0, 0.0, 0.0], 0.5), 2);
    }

    #[test]
    fn weighted_sample_bounds() {
        for _ in 0..100 {
            let idx = weighted_sample(&[0.25, 0.25, 0.25, 0.25]);
            assert!(idx < 4);
        }
    }

    // ── randn ───────────────────────────────────────────────────────────

    #[test]
    fn randn_shape() {
        let t = randn(&[3, 5]);
        assert_eq!(t.shape(), &[3, 5]);
        assert_eq!(t.numel(), 15);
    }

    #[test]
    fn randn_finite() {
        let t = randn(&[100]);
        for &v in t.data() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn randn_requires_grad() {
        let t = randn(&[4]);
        assert!(t.requires_grad_enabled());
    }

    // ── MicroGPT ────────────────────────────────────────────────────────

    #[test]
    fn model_param_count() {
        let model = MicroGPT::new();
        assert_eq!(model.param_count(), 4192);
        clear_graph();
    }

    #[test]
    fn model_parameters_mut_count() {
        let mut model = MicroGPT::new();
        // 5 base tensors + 4 heads × 4 projections = 21
        assert_eq!(model.parameters_mut().len(), 21);
        clear_graph();
    }

    #[test]
    fn model_forward_shape() {
        no_grad(|| {
            let model = MicroGPT::new();
            let logits = model.forward(&[0, 1, 2]);
            assert_eq!(logits.shape(), &[3, VOCAB_SIZE]);
        });
        clear_graph();
    }

    #[test]
    fn model_forward_single_token() {
        no_grad(|| {
            let model = MicroGPT::new();
            let logits = model.forward(&[0]);
            assert_eq!(logits.shape(), &[1, VOCAB_SIZE]);
        });
        clear_graph();
    }

    #[test]
    fn model_forward_finite() {
        no_grad(|| {
            let model = MicroGPT::new();
            let logits = model.forward(&[0, 5, 10]);
            for &v in logits.data() {
                assert!(v.is_finite(), "logit not finite: {v}");
            }
        });
        clear_graph();
    }

    #[test]
    fn model_forward_max_length() {
        no_grad(|| {
            let model = MicroGPT::new();
            let tokens: Vec<usize> = (0..BLOCK_SIZE).collect();
            let logits = model.forward(&tokens);
            assert_eq!(logits.shape(), &[BLOCK_SIZE, VOCAB_SIZE]);
        });
        clear_graph();
    }

    // ── Adam ────────────────────────────────────────────────────────────

    #[test]
    fn adam_new() {
        let opt = Adam::new();
        assert_eq!(opt.t, 0);
        assert!(opt.m.is_empty());
        assert!(opt.v.is_empty());
    }

    #[test]
    fn adam_step_counter() {
        let mut opt = Adam::new();
        let mut params: Vec<&mut Tensor> = vec![];
        opt.step(&mut params, 0.01);
        assert_eq!(opt.t, 1);
        opt.step(&mut params, 0.01);
        assert_eq!(opt.t, 2);
    }

    #[test]
    fn adam_updates_params() {
        // Create a simple parameter, give it a gradient, step
        let mut p = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).requires_grad();
        let target = Tensor::from_slice(&[0.0]);

        // Forward: sum → loss
        let loss = p.sum();
        loss.backward();

        let before: Vec<f32> = p.data().to_vec();
        let mut opt = Adam::new();
        let mut params = vec![&mut p];
        opt.step(&mut params, 0.01);
        let after: Vec<f32> = params[0].data().to_vec();

        assert_ne!(before, after, "Adam should update parameters");
        // v should be non-negative
        for &vi in &opt.v[0] {
            assert!(vi >= 0.0, "second moment must be non-negative");
        }
        let _ = target; // suppress warning
        clear_graph();
    }

    #[test]
    fn adam_no_grad_skip() {
        // Parameter with no gradient in the graph — should not panic
        let mut p = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).requires_grad();
        let before: Vec<f32> = p.data().to_vec();

        let mut opt = Adam::new();
        clear_graph(); // ensure no gradients
        let mut params = vec![&mut p];
        opt.step(&mut params, 0.01);
        let after: Vec<f32> = params[0].data().to_vec();

        assert_eq!(before, after, "no gradient → no update");
        clear_graph();
    }

    // ── sample ──────────────────────────────────────────────────────────

    #[test]
    fn sample_returns_string() {
        let model = MicroGPT::new();
        let name = sample(&model);
        // May be empty if first sample is BOS, but should not panic
        assert!(name.len() <= BLOCK_SIZE);
        clear_graph();
    }

    #[test]
    fn sample_only_lowercase() {
        let model = MicroGPT::new();
        let name = sample(&model);
        for ch in name.chars() {
            assert!(ch.is_ascii_lowercase(), "unexpected char: {ch}");
        }
        clear_graph();
    }

    // ── Default impls ───────────────────────────────────────────────────

    #[test]
    fn adam_default() {
        let opt = Adam::default();
        assert_eq!(opt.t, 0);
    }

    #[test]
    fn model_default() {
        let model = MicroGPT::default();
        assert_eq!(model.param_count(), 4192);
        clear_graph();
    }
}
