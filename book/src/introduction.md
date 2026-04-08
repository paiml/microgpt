# microGPT

A port of Karpathy's [microGPT](https://karpathy.github.io/2026/02/12/microgpt/)
to Rust using [aprender](https://github.com/paiml/aprender).

## What is microGPT?

microGPT is a 4,192-parameter GPT that generates human-like names after
training on 32K examples. It demonstrates the algorithmic essence of large
language models: tokenization, embeddings, multi-head attention with a KV
cache, residual connections, RMSNorm, MLPs, and cross-entropy loss with
backpropagation.

Everything else is just efficiency.

## Why Rust + aprender?

The original Python implementation uses a custom autograd engine operating
on scalar `Value` objects. This Rust port uses
[aprender-core](https://crates.io/crates/aprender-core) for
SIMD-accelerated tensor operations with automatic differentiation,
achieving the same architecture with production-grade numerics.

## Quick start

```bash
cargo run --release
```

Trains for 5,000 steps (~2 seconds on CPU), then samples 20 names:

```text
step    0 | loss 3.29
step 2500 | loss 2.56
step 4999 | loss 1.95

Generated names (temperature=0.5):
  karila, maria, misha, mayla, anara
```
