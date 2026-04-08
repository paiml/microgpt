[![CI](https://github.com/paiml/microgpt/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/microgpt/actions)
[![crate](https://img.shields.io/crates/v/microgpt.svg)](https://crates.io/crates/microgpt)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![book](https://img.shields.io/badge/book-mdBook-blue.svg)](https://paiml.github.io/microgpt/)

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

The README itself is under contract — `tests/readme_contract.rs`
validates that every architectural claim matches the source constants.

## Documentation

Full architecture walkthrough, attention design decisions, and contract
reference: **[paiml.github.io/microgpt](https://paiml.github.io/microgpt/)**

## Dependencies

Only [aprender-core](https://crates.io/crates/aprender-core) (pure Rust ML
framework, published on crates.io) and `rand`. No path or git dependencies.

## License

MIT
