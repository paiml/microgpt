# apr trace

The `apr trace` command traces data through each layer of the model,
showing tensor shapes and statistics at every stage. For microGPT,
the equivalent is `cargo run --example trace_forward`.

## Full forward pass trace

```text
$ cargo run --example trace_forward

Input: "emma" → tokens [0, 5, 13, 13, 1]

Stage 1: Embedding
  tok_emb = one_hot(tokens) @ wte          shape=[5, 16]  L2=0.72
  pos_emb = one_hot(pos) @ wpe             shape=[5, 16]  L2=0.74
  x = tok_emb + pos_emb                    shape=[5, 16]  L2=1.04
  x = RMSNorm(x)                           shape=[5, 16]  L2=8.94

Stage 2: Multi-Head Self-Attention
  x_n = RMSNorm(x) [pre-attn]              shape=[5, 16]  L2=8.94
  head_0_out @ wo[0]                       shape=[5, 16]  L2=0.50
  head_1_out @ wo[1]                       shape=[5, 16]  L2=0.46
  head_2_out @ wo[2]                       shape=[5, 16]  L2=0.31
  head_3_out @ wo[3]                       shape=[5, 16]  L2=0.38
  x = x_res + attn_out [residual]          shape=[5, 16]  L2=9.09

Stage 3: MLP (FFN)
  x_n = RMSNorm(x) [pre-MLP]               shape=[5, 16]  L2=8.94
  h = x_n @ w_fc1 [up project]             shape=[5, 64]  L2=5.75
  h = ReLU(h)                              shape=[5, 64]  L2=3.87  ← ~50% zeroed
  mlp_out = h @ w_fc2 [down project]       shape=[5, 16]  L2=1.17
  x = x_res + mlp_out [residual]           shape=[5, 16]  L2=9.10

Stage 4: Language Model Head
  logits = x @ w_lm                        shape=[5, 27]  L2=3.29
```

## What the trace reveals

### RMSNorm normalization

After RMSNorm, `L2≈8.94` regardless of input magnitude. This is a
structural invariant: RMSNorm normalizes RMS to 1.0 per row, so each
row's L2 = `√16 = 4.0`, and total L2 = `√5 × 4.0 ≈ 8.94` for 5 rows
of 16 dimensions. The pre-RMSNorm L2 (1.04 here) depends on random
initialization.

### Attention is a small perturbation

The attention output norms (`L2≈0.3-0.5`) are much smaller than the
residual stream (`L2≈9.0`). After the residual connection, `L2` barely
changes: `9.0 → 9.09`. The residual stream carries most of the signal;
attention makes incremental adjustments.

### ReLU sparsity

Before ReLU: `L2=5.75`. After: `L2=3.87`. ReLU zeros out ~50% of the
64 FFN dimensions (the negative activations), creating a sparse
intermediate representation. This is visible in the range changing from
`[-0.90, 1.15]` to `[0.00, 1.15]`.

### Untrained predictions

Before training, the model predicts wrong characters:

```text
Predictions:
  pos 0: target='e' predicted='r' P(target)=0.035
  pos 1: target='m' predicted='g' P(target)=0.041
  pos 2: target='m' predicted='j' P(target)=0.037
  pos 3: target='a' predicted='c' P(target)=0.046
  pos 4: target='EOS' predicted='f' P(target)=0.033
```

The target probability is ~1/27 ≈ 0.037 — essentially random.
After 5,000 training steps, these probabilities increase significantly.

## Comparison with apr trace

For a production model like Qwen-1.5B:

```text
$ apr trace qwen-1.5b-q4k.apr --verbose

Layer 0:  attn_norm → q_proj → k_proj → v_proj → attn → o_proj → residual
          → ffn_norm → gate_proj → up_proj → SiLU → ffn_down → residual
Layer 1:  ... (same pattern × 28 layers)
```

microGPT has the same `norm → attn → residual → norm → ffn → residual`
pattern, just with 1 layer instead of 28 and ReLU instead of SiLU/SwiGLU.
