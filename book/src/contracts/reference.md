# Contract Reference

## microgpt-v1.yaml

The canonical contract for microGPT. All equations, proof obligations,
and falsification tests are defined in
[`contracts/microgpt-v1.yaml`](https://github.com/paiml/microgpt/blob/main/contracts/microgpt-v1.yaml).

## Equations

| ID | Formula | Tested | Proved |
|----|---------|--------|--------|
| `one_hot` | `one_hot(i,C)[i][j] = 1 if j==i else 0` | L2 | L4 |
| `rms_norm` | `x / sqrt(mean(x^2) + eps)` | L2 | - |
| `causal_mask` | `0 if j<=i else -1e9` | L2 | L4 |
| `tokenize_decode` | `decode(tokenize(s)) == s` | L2 | L4 |
| `weighted_sample` | `P(return i) = probs[i]` | L2 | - |
| `adam_step` | Kingma & Ba (2015) update rule | L2 | L4 |
| `forward_pass` | `LMHead(MLP(Attn(Embed(tokens))))` | L2 | L4 |

## Proof obligations

| ID | Description | Level |
|----|-------------|-------|
| `PARAM-COUNT-001` | Model has exactly 4,192 parameters | L4 |
| `ONEHOT-ROW-001` | Each one-hot row sums to 1.0 | L2 |
| `MASK-CAUSAL-001` | Lower-triangular structure | L2 |
| `TOKENIZE-ROUNDTRIP-001` | Encode/decode roundtrip identity | L2 |
| `ADAM-MONOTONIC-001` | Step counter advances by 1 | L4 |
| `ADAM-V-NONNEG-001` | Second moment >= 0 | L4 |
| `FORWARD-SHAPE-001` | Output shape [n, 27] | L2 |

## Bindings

8 bindings in `contracts/binding.yaml`, all status: `implemented`.

| Equation | Function | Signature |
|----------|----------|-----------|
| `one_hot` | `one_hot` | `fn(indices, num_classes) -> Tensor` |
| `rms_norm` | `rms_norm` | `fn(x: &Tensor) -> Tensor` |
| `causal_mask` | `causal_mask` | `fn(size) -> Tensor` |
| `tokenize_decode` | `tokenize` | `fn(name) -> Vec<usize>` |
| `tokenize_decode` | `decode` | `fn(tokens) -> String` |
| `weighted_sample` | `weighted_sample_with_r` | `fn(probs, r) -> usize` |
| `adam_step` | `Adam::step` | `fn(&mut self, params, lr)` |
| `forward_pass` | `MicroGPT::forward` | `fn(&self, tokens) -> Tensor` |
