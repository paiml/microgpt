//! microGPT — Karpathy's 200-line GPT ported to Rust with aprender.
//!
//! A 4,192-parameter GPT trained on 32K names using character-level tokenization.
//! Architecture: 1 layer, 4 heads, embed_dim=16, block_size=16.
//!
//! Reference: https://karpathy.github.io/2026/02/12/microgpt/

use aprender::autograd::{clear_graph, no_grad, Tensor};
use aprender::nn::loss::CrossEntropyLoss;
use rand::Rng;

// ── Hyperparameters (matching Karpathy's microGPT) ──────────────────────────

const VOCAB_SIZE: usize = 27; // 26 lowercase + BOS
const N_EMBD: usize = 16;
const N_HEAD: usize = 4;
const HEAD_DIM: usize = N_EMBD / N_HEAD; // 4
const BLOCK_SIZE: usize = 16;
const FF_DIM: usize = 4 * N_EMBD; // 64
const BOS: usize = 0;
const CHARS: &str = "abcdefghijklmnopqrstuvwxyz";

const NUM_STEPS: usize = 5000;
const LR: f32 = 0.01;
const BETA1: f32 = 0.85;
const BETA2: f32 = 0.99;
const INIT_STD: f32 = 0.08;
const TEMPERATURE: f32 = 0.5;

// ── Model ───────────────────────────────────────────────────────────────────
// Raw weight tensors + direct matmuls. This mirrors the original Python code
// and ensures gradients flow correctly through every parameter.

struct MicroGPT {
    wte: Tensor,       // [VOCAB_SIZE, N_EMBD] — token embeddings
    wpe: Tensor,       // [BLOCK_SIZE, N_EMBD] — position embeddings
    wq: Vec<Tensor>,   // N_HEAD × [N_EMBD, HEAD_DIM]
    wk: Vec<Tensor>,   // N_HEAD × [N_EMBD, HEAD_DIM]
    wv: Vec<Tensor>,   // N_HEAD × [N_EMBD, HEAD_DIM]
    wo: Vec<Tensor>,   // N_HEAD × [HEAD_DIM, N_EMBD]
    w_fc1: Tensor,     // [N_EMBD, FF_DIM]
    w_fc2: Tensor,     // [FF_DIM, N_EMBD]
    w_lm: Tensor,      // [N_EMBD, VOCAB_SIZE]
}

impl MicroGPT {
    fn new() -> Self {
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

    fn forward(&self, tokens: &[usize]) -> Tensor {
        let n = tokens.len();

        // Embedding lookup via one-hot matmul (differentiable)
        let tok_oh = one_hot(tokens, VOCAB_SIZE);
        let pos_oh = one_hot(&(0..n).collect::<Vec<_>>(), BLOCK_SIZE);
        let mut x = tok_oh.matmul(&self.wte).add(&pos_oh.matmul(&self.wpe));
        x = rms_norm(&x);

        // ── Transformer block (1 layer) ─────────────────────────────────

        // Multi-head self-attention
        let x_res = x.clone();
        let x_n = rms_norm(&x);
        let mask = causal_mask(n);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let mut attn_out: Option<Tensor> = None;
        for h in 0..N_HEAD {
            let q_h = x_n.matmul(&self.wq[h]); // [n, HEAD_DIM]
            let k_h = x_n.matmul(&self.wk[h]);
            let v_h = x_n.matmul(&self.wv[h]);

            // Scaled dot-product attention with causal mask
            let scores = q_h.matmul(&k_h.transpose()).mul_scalar(scale);
            let scores = scores.add(&mask);
            let weights = scores.softmax();
            let head_out = weights.matmul(&v_h); // [n, HEAD_DIM]

            // Project head back to embed dim and accumulate
            let projected = head_out.matmul(&self.wo[h]); // [n, N_EMBD]
            attn_out = Some(match attn_out {
                Some(acc) => acc.add(&projected),
                None => projected,
            });
        }
        x = x_res.add(&attn_out.expect("at least 1 head")); // residual

        // MLP: fc1 → ReLU → fc2
        let x_res = x.clone();
        let x_n = rms_norm(&x);
        let h = x_n.matmul(&self.w_fc1).relu();
        x = x_res.add(&h.matmul(&self.w_fc2)); // residual

        // LM head → logits [n, VOCAB_SIZE]
        x.matmul(&self.w_lm)
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
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
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Gaussian random tensor (std=0.08) with gradient tracking.
fn randn(shape: &[usize]) -> Tensor {
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

/// RMSNorm: x / sqrt(mean(x²) + eps). Gradient flows through element-wise mul.
fn rms_norm(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let cols = shape[shape.len() - 1];
    let rows = x.numel() / cols;
    let data = x.data();

    let mut scale_data = Vec::with_capacity(x.numel());
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv_rms = 1.0 / (ms + 1e-5).sqrt();
        scale_data.extend(std::iter::repeat_n(inv_rms, cols));
    }
    let scale = Tensor::from_vec(scale_data, shape);
    x.mul(&scale)
}

/// One-hot encode indices → [len, num_classes] matrix.
fn one_hot(indices: &[usize], num_classes: usize) -> Tensor {
    let n = indices.len();
    let mut data = vec![0.0f32; n * num_classes];
    for (i, &idx) in indices.iter().enumerate() {
        data[i * num_classes + idx] = 1.0;
    }
    Tensor::new(&data, &[n, num_classes])
}

/// Lower-triangular causal mask (future positions → -inf).
fn causal_mask(size: usize) -> Tensor {
    let mut data = vec![0.0f32; size * size];
    for i in 0..size {
        for j in (i + 1)..size {
            data[i * size + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::new(&data, &[size, size])
}

/// Weighted random sampling from a probability distribution.
fn weighted_sample(probs: &[f32]) -> usize {
    let mut rng = rand::rng();
    let r: f32 = rng.random();
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

fn tokenize(name: &str) -> Vec<usize> {
    let chars: Vec<char> = CHARS.chars().collect();
    let mut tokens = vec![BOS];
    for ch in name.chars() {
        if let Some(idx) = chars.iter().position(|&c| c == ch) {
            tokens.push(idx + 1); // 0 = BOS, 1-26 = a-z
        }
    }
    tokens.push(BOS); // BOS doubles as EOS
    tokens
}

fn decode(tokens: &[usize]) -> String {
    let chars: Vec<char> = CHARS.chars().collect();
    tokens
        .iter()
        .filter_map(|&t| if t == BOS { None } else { chars.get(t - 1).copied() })
        .collect()
}

// ── Adam optimizer (manual, to operate directly on Tensors) ─────────────────

struct Adam {
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: usize,
}

impl Adam {
    fn new() -> Self {
        Self {
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    fn step(&mut self, params: &mut [&mut Tensor], lr: f32) {
        self.t += 1;
        if self.m.len() < params.len() {
            self.m.resize(params.len(), Vec::new());
            self.v.resize(params.len(), Vec::new());
        }

        let bc1 = 1.0 - BETA1.powi(self.t as i32);
        let bc2 = 1.0 - BETA2.powi(self.t as i32);

        for (i, param) in params.iter_mut().enumerate() {
            let grad = match aprender::autograd::get_grad(param.id()) {
                Some(g) => g,
                None => continue,
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

// ── Training ────────────────────────────────────────────────────────────────

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

    let n_params: usize = model.parameters_mut().iter().map(|t| t.numel()).sum();
    println!("Parameters: {n_params}");

    for step in 0..NUM_STEPS {
        let name = names[step % names.len()].to_lowercase();
        let tokens = tokenize(&name);
        let n = (tokens.len() - 1).min(BLOCK_SIZE);
        if n == 0 {
            continue;
        }

        let input = &tokens[..n];
        let targets: Vec<f32> = tokens[1..=n].iter().map(|&t| t as f32).collect();

        // Forward
        let logits = model.forward(input);
        let target_tensor = Tensor::from_slice(&targets);
        let loss = loss_fn.forward(&logits, &target_tensor);

        // Backward + Adam with linear LR decay
        loss.backward();
        let lr = LR * (1.0 - step as f32 / NUM_STEPS as f32);
        let mut params = model.parameters_mut();
        optimizer.step(&mut params, lr);
        clear_graph();

        if step % 100 == 0 || step == NUM_STEPS - 1 {
            println!("step {:4} | loss {:.4} | lr {:.5}", step, loss.item(), lr);
        }
    }

    // ── Sampling ────────────────────────────────────────────────────────

    println!("\nGenerated names (temperature={TEMPERATURE}):");
    no_grad(|| {
        for i in 0..20 {
            let name = sample(&model);
            println!("  {:2}: {name}", i + 1);
        }
    });
}

fn sample(model: &MicroGPT) -> String {
    let mut tokens = vec![BOS];
    for _ in 0..BLOCK_SIZE {
        let logits = model.forward(&tokens);

        // Last position's logits → temperature-scaled softmax → sample
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
}
