# apr explain

The `apr explain` command explains architecture, tensor roles, and kernel
dispatch. For microGPT, the equivalent is `cargo run --example explain_attention`.

## Attention mechanics

```text
$ cargo run --example explain_attention

Architecture (cf. `apr explain --kernel gpt2 -v`):
  Attention type:     MHA (multi-head attention)
  Heads:              4
  Head dim:           4
  Total attn dim:     16 (4 × 4)
  Scale factor:       1/√4 = 0.5000
```

The attention mechanism computes:

```text
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

This is the same equation used in GPT-2, LLaMA, Qwen, and every other
transformer. The only difference is scale: microGPT uses `d_k=4`,
production models use `d_k=128`.

## Attention weight visualization

For input "mar" (tokens `[BOS, m, a, r]`), the attention weights at
random initialization show the causal structure (values vary per run,
but the structural invariants are constant):

```text
Head 0:
  pos 0 → [1.000, 0.000, 0.000, 0.000]   ← BOS attends only to itself
  pos 1 → [0.482, 0.518, 0.000, 0.000]   ← 'm' splits between BOS and self
  pos 2 → [0.302, 0.383, 0.315, 0.000]   ← 'a' attends to all prior
  pos 3 → [0.254, 0.247, 0.215, 0.284]   ← 'r' attends broadly

Head 3:
  pos 0 → [1.000, 0.000, 0.000, 0.000]
  pos 1 → [0.494, 0.506, 0.000, 0.000]
  pos 2 → [0.286, 0.339, 0.376, 0.000]   ← head 3 focuses more on 'a'
  pos 3 → [0.247, 0.223, 0.250, 0.280]   ← head 3 focuses more on 'r'
```

Structural invariants (hold for ANY initialization):
- **Row 0** is always `[1, 0, 0, 0]` — the BOS token can only see itself
- **Upper triangle is zero** — causal masking prevents attending to the future
- **Each row sums to 1.0** — softmax normalizes attention weights

The specific non-trivial values (e.g., 0.482 vs 0.518) reflect random
initialization, not learned patterns. After training, head specialization
emerges.

## Causal mask explained

Equivalent to `apr explain --tensor attn_mask`:

```text
Causal Mask:
  Each position can only attend to itself and earlier positions.
  Future positions are masked with -1e9 → softmax drives them to ~0.

    pos 0: [  0  ,  -∞ ,  -∞ ,  -∞ ]
    pos 1: [  0  ,  0  ,  -∞ ,  -∞ ]
    pos 2: [  0  ,  0  ,  0  ,  -∞ ]
    pos 3: [  0  ,  0  ,  0  ,  0  ]
```

The mask uses `-1e9` instead of true `-∞` to satisfy the upstream
softmax precondition contract (`x.iter().all(|v| v.is_finite())`).
Since `exp(-1e9) ≈ 0` in f32, this is numerically identical.

## Comparison with production models

```text
$ apr explain --kernel llama -v

Constraints (from family contract):
  attention_type:      gqa          ← microGPT uses mha (simpler)
  activation:          silu         ← microGPT uses relu (simpler)
  norm_type:           rmsnorm      ← same as microGPT ✓
  mlp_type:            swiglu       ← microGPT uses relu_mlp (simpler)
  positional_encoding: rope         ← microGPT uses learned (simpler)
  has_bias:            false        ← same as microGPT ✓
```

microGPT uses the simpler variants of each component. The contract
structure is identical — only the kernel implementations differ.
