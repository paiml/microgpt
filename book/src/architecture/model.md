# Model

## Architecture

| Component | Value |
|-----------|-------|
| Embedding dim | 16 |
| Attention heads | 4 (head_dim=4) |
| Layers | 1 |
| Context length | 16 |
| Vocab | 27 (a-z + BOS) |
| Parameters | 4,192 |

## Parameter breakdown

| Weight | Shape | Count |
|--------|-------|-------|
| `wte` (token embeddings) | [27, 16] | 432 |
| `wpe` (position embeddings) | [16, 16] | 256 |
| `wq` (4 heads) | 4 x [16, 4] | 256 |
| `wk` (4 heads) | 4 x [16, 4] | 256 |
| `wv` (4 heads) | 4 x [16, 4] | 256 |
| `wo` (4 heads) | 4 x [4, 16] | 256 |
| `w_fc1` (MLP up) | [16, 64] | 1,024 |
| `w_fc2` (MLP down) | [64, 16] | 1,024 |
| `w_lm` (LM head) | [16, 27] | 432 |
| **Total** | | **4,192** |

## Forward pass

```text
tokens → one_hot → matmul(wte) + matmul(wpe) → RMSNorm
       → Attention(Q,K,V,O) + residual
       → RMSNorm → MLP(fc1 → ReLU → fc2) + residual
       → matmul(w_lm) → logits
```

## Embedding lookup

Embeddings use a one-hot matmul instead of a dedicated `Embedding` layer.
This ensures gradients flow through the standard `matmul` backward pass:

```text
tok_emb = one_hot(tokens, 27) @ wte   // [n, 16]
pos_emb = one_hot(0..n, 16)  @ wpe   // [n, 16]
x = tok_emb + pos_emb
```

## RMSNorm

Applied per-row with straight-through gradient estimation:

```text
RMSNorm(x)_i = x_i / sqrt(mean(x_i^2) + 1e-5)
```

Uses `-1e9` instead of `-inf` in the causal mask to satisfy the upstream
softmax precondition contract (`x.iter().all(|v| v.is_finite())`).
