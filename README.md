# microGPT

Karpathy's [microGPT](https://karpathy.github.io/2026/02/12/microgpt/) ported to Rust with [aprender](https://github.com/paiml/aprender).

A 4,192-parameter GPT trained on 32K names using character-level tokenization.
Everything else is just efficiency.

## Architecture

| Component | Value |
|-----------|-------|
| Embedding dim | 16 |
| Attention heads | 4 (head_dim=4) |
| Layers | 1 |
| Context length | 16 |
| Vocab | 27 (a-z + BOS) |
| Parameters | 4,192 |
| Normalization | RMSNorm (per-row) |
| Activation | ReLU |
| Optimizer | Adam (beta1=0.85, beta2=0.99) |

## Install

```bash
cargo install --path .
```

## Usage

```bash
cargo run --release
```

Trains for 5,000 steps on CPU, then generates 20 names.
Loss converges from ~3.3 (random baseline) to ~2.0 producing
name-like outputs (e.g. "karila", "maria", "misha").

## Provable Contracts

See [`contracts/microgpt-v1.yaml`](contracts/microgpt-v1.yaml) for
formal invariants covering one-hot encoding, RMSNorm, causal masking,
tokenizer roundtrip, Adam optimizer, and forward-pass shape contracts.

## Dependencies

Only [aprender-core](https://crates.io/crates/aprender-core) (pure Rust ML
framework, published on crates.io) and `rand`. No path or git dependencies.

## License

MIT
