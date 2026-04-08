# Training

## Dataset

32,033 names from Karpathy's
[makemore](https://github.com/karpathy/makemore) dataset,
character-level tokenized with a 27-token vocabulary (26 lowercase
letters + BOS/EOS token at index 0).

## Tokenization

```text
"maria" → [0, 13, 1, 18, 9, 1, 0]
           BOS m   a  r   i  a  BOS
```

The BOS token doubles as EOS. The tokenizer drops non-lowercase characters
silently, and the roundtrip property holds for all `[a-z]*` strings:

```text
decode(tokenize(name)[1..-1]) == name
```

## Loss function

Cross-entropy loss from aprender-core, with teacher forcing:

```text
loss = CrossEntropyLoss(logits[0..n], targets[1..n+1])
```

The initial loss is ~3.3 (random baseline for 27 classes: `-ln(1/27) ≈ 3.30`).
After 5,000 steps, loss converges to ~2.0.

## Optimizer

Manual Adam (Kingma & Ba, 2015) operating directly on autograd tensors:

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 0.01 (linear decay to 0) |
| beta1 | 0.85 |
| beta2 | 0.99 |
| epsilon | 1e-8 |
| Steps | 5,000 |

The optimizer is manual (not `aprender::nn::Adam`) because aprender's
`Linear::forward` uses a cached weight transpose whose `TensorId` differs
from the original weight — the built-in Adam looks up gradients by ID and
misses the update. Raw weight tensors with direct `matmul` ensure every
parameter receives its gradient.

## Sampling

Autoregressive generation with temperature scaling:

```text
for pos in 0..BLOCK_SIZE:
    logits = model.forward(tokens)
    probs  = softmax(logits[last] / temperature)
    next   = weighted_sample(probs)
    if next == BOS: break
    tokens.push(next)
```

Temperature 0.5 produces conservative, name-like outputs.
