# Attention

## Scaled dot-product attention

Each of the 4 heads computes:

```text
Q_h = X @ Wq_h     // [n, 4]
K_h = X @ Wk_h     // [n, 4]
V_h = X @ Wv_h     // [n, 4]

scores = Q_h @ K_h^T / sqrt(4)     // [n, n]
scores = scores + causal_mask       // mask future positions
weights = softmax(scores)           // [n, n]
head_out = weights @ V_h            // [n, 4]
```

## Multi-head accumulation

Instead of concatenating heads and projecting through a single `[16, 16]`
output matrix, each head projects back to the full embedding dimension and
the outputs are summed:

```text
attn_out = sum_h (head_out_h @ Wo_h)   // [n, 16]
```

This is mathematically equivalent to standard concat + output projection:

```text
concat(heads) @ Wo = h0 @ Wo[0:4,:] + h1 @ Wo[4:8,:] + h2 @ Wo[8:12,:] + h3 @ Wo[12:16,:]
```

The per-head approach avoids needing tensor slicing operations in the
autograd graph, where aprender's `matmul` only supports 2D tensors.

## Causal masking

The causal mask is lower-triangular:

```text
mask[i][j] = 0    if j <= i  (attend)
mask[i][j] = -1e9 if j > i   (block)
```

After adding the mask to attention scores, `softmax(scores)` drives masked
positions to effectively zero probability (`exp(-1e9) ≈ 0`).

## Why per-head projections?

aprender's autograd `Tensor::matmul` operates on 2D tensors only.
The standard approach of computing `Q = X @ Wq` as `[n, 16]` then
reshaping to `[n, 4, 4]` requires tensor slicing with gradient tracking,
which the autograd doesn't support. Per-head `[16, 4]` projections
achieve identical parameter count (1,024 total for Q/K/V/O) while keeping
all operations in the 2D matmul autograd path.
