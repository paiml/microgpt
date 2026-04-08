# microGPT

Karpathy's [microGPT](https://karpathy.github.io/2026/02/12/microgpt/) ported to Rust with [aprender](https://github.com/paiml/aprender).

A 4,192-parameter GPT trained on 32K names using character-level tokenization. Everything else is just efficiency.

## Architecture

| Component | Value |
|-----------|-------|
| Embedding dim | 16 |
| Attention heads | 4 |
| Layers | 1 |
| Context length | 16 |
| Vocab | 27 (a-z + BOS) |
| Parameters | 4,192 |

## Run

```bash
cargo run --release
```

Trains for 5,000 steps (~2 seconds on CPU), then generates 20 names:

```
step    0 | loss 3.2916
step 1000 | loss 2.2245
step 4999 | loss 1.9453

Generated names (temperature=0.5):
   1: janiy
   2: karila
   3: sirin
   4: maria
   5: misha
```

## Dependencies

Only [aprender-core](https://crates.io/crates/aprender-core) (pure Rust ML framework) and `rand`.

## License

MIT
