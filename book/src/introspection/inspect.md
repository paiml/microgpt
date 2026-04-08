# apr inspect

The `apr inspect` command reveals model structure, metadata, and tensor
statistics. For microGPT, the equivalent is `cargo run --example inspect_model`.

## Model metadata

```text
$ cargo run --example inspect_model

microGPT Model Inspection
══════════════════════════════════════════════════

Architecture:
  Family:          microGPT (character-level, 1-layer GPT)
  Kernel Class:    Custom (MHA + RMSNorm + ReLU)
  Hidden Size:     16
  Attention Heads: 4 (head_dim=4)
  FFN Size:        64
  Vocab Size:      27
  Context Length:   16
  Parameters:      4192
```

Compare with a production model:

```text
$ apr inspect qwen-1.5b-q4k.apr

  Architecture:
    Family: qwen2
    Hidden Size: 1536
    Intermediate Size: 8960
    Vocab Size: 151936
    Max Position: 32768
    RoPE Theta: 1000000
```

microGPT is 362,000x smaller than Qwen-1.5B but uses the same
fundamental operations.

## Tensor inventory

Equivalent to `apr tensors <file> --stats`:

```text
Tensors (cf. `apr tensors`):
  Name                                  Shape   Numel  Stats
  ──────────────────────────────────────────────────────────────
  wte (token embeddings)         [27, 16]     432  μ=+0.010  σ=0.083
  wpe (position embeddings)      [16, 16]     256  μ=-0.010  σ=0.079
  wq[0] (query proj head 0)      [16,  4]      64  μ=-0.010  σ=0.081
  ...
  w_fc1 (MLP up projection)      [16, 64]    1024  μ=+0.000  σ=0.082
  w_fc2 (MLP down projection)    [64, 16]    1024  μ=+0.003  σ=0.079
  w_lm (LM head)                 [16, 27]     432  μ=+0.003  σ=0.078
```

All weights are initialized from `N(0, 0.08²)`, matching Karpathy's
original Python implementation.

## Kernel pipeline

Equivalent to `apr explain --kernel`:

```text
┌─────────────────────────┬─────────────────────────────────┬─────────────────────────┐
│ Operation               │ Implementation                  │ Contract                │
├─────────────────────────┼─────────────────────────────────┼─────────────────────────┤
│ Embedding Lookup        │ one_hot @ wte (differentiable)  │ microgpt-v1 § one_hot   │
│ RMSNorm                 │ x * (1/rms) per row             │ microgpt-v1 § rms_norm  │
│ Attention (4-head MHA)  │ per-head Q/K/V/O matmul        │ attention-kernel-v1     │
│ Causal Mask             │ lower-tri 0, upper -1e9         │ microgpt-v1 § causal    │
│ Softmax                 │ aprender autograd softmax       │ softmax-kernel-v1       │
│ MLP (ReLU)              │ fc1 → ReLU → fc2               │ activation-kernel-v1    │
│ LM Head                 │ matmul to vocab logits          │ matmul-kernel-v1        │
└─────────────────────────┴─────────────────────────────────┴─────────────────────────┘
```

Every operation maps to a provable contract. The kernel pipeline for
a production LLaMA model (via `apr explain --kernel llama`) shows the
same structure at scale:

```text
$ apr explain --kernel llama --proof-status

Kernel Pipeline (9 ops):
  MatVec (Q4K)    → matvec-kernel-v1         ◉ Tested
  Softmax         → softmax-kernel-v1        ◉ Tested
  Attention (GQA) → element-wise-ops-v1      ◉ Tested
  Normalization   → normalization-kernel-v1  ◉ Tested
  Activation      → element-wise-ops-v1      ◉ Tested
  MLP             → element-wise-ops-v1      ◉ Tested
  Position (RoPE) → rope-kernel-v1           ◉ Tested
```

## Tensor roles

Equivalent to `apr explain --tensor <name>`:

| Tensor | Role |
|--------|------|
| `wte` | Token embedding — maps token IDs to dense vectors |
| `wpe` | Position embedding — adds positional information |
| `wq[h]` | Query projection in multi-head attention |
| `wk[h]` | Key projection in multi-head attention |
| `wv[h]` | Value projection in multi-head attention |
| `wo[h]` | Output projection — combines head outputs |
| `w_fc1` | Feed-forward up projection (expand to 4x) |
| `w_fc2` | Feed-forward down projection (compress to 1x) |
| `w_lm` | Language model head — projects to vocabulary logits |
