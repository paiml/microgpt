#!/usr/bin/env python3
"""Generate parity reference data for microGPT.

Implements Karpathy's exact forward pass (scalar ops, no autograd)
with a fixed seed. Outputs weights + logits + tokenizer results as JSON
for the Rust parity test to validate against.

Reference: https://karpathy.github.io/2026/02/12/microgpt/
"""

import json
import math
import random
import sys

# ── Hyperparameters (matching blog post) ─────────────────────────────────────

VOCAB_SIZE = 27
N_EMBD = 16
N_HEAD = 4
HEAD_DIM = N_EMBD // N_HEAD  # 4
N_LAYER = 1
BLOCK_SIZE = 16
FF_DIM = 4 * N_EMBD  # 64
BOS = 0
CHARS = "abcdefghijklmnopqrstuvwxyz"
SEED = 42

# ── Helpers (exact blog post code, Value → float) ───────────────────────────

def linear(x, w):
    """w @ x: each row of w dot-product with x."""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def relu(x):
    return max(0.0, x)

# ── Forward pass (exact blog post GPT function) ─────────────────────────────

def gpt_forward(token_id, pos_id, keys, values, state_dict):
    """Process one token, accumulating KV cache. Returns logits."""
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(N_LAYER):
        # Self-attention
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(N_HEAD):
            hs = h * HEAD_DIM
            q_h = q[hs:hs+HEAD_DIM]
            k_h = [ki[hs:hs+HEAD_DIM] for ki in keys[li]]
            v_h = [vi[hs:hs+HEAD_DIM] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / HEAD_DIM**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # MLP
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [relu(xi) for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# ── Tokenizer ────────────────────────────────────────────────────────────────

def tokenize(name):
    tokens = [BOS]
    for ch in name:
        idx = CHARS.find(ch)
        if idx >= 0:
            tokens.append(idx + 1)
    tokens.append(BOS)
    return tokens

# ── Weight initialization ────────────────────────────────────────────────────

def generate_weights(seed):
    random.seed(seed)
    def matrix(nout, nin, std=0.08):
        return [[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)]

    sd = {
        'wte': matrix(VOCAB_SIZE, N_EMBD),
        'wpe': matrix(BLOCK_SIZE, N_EMBD),
        'lm_head': matrix(VOCAB_SIZE, N_EMBD),
    }
    for i in range(N_LAYER):
        sd[f'layer{i}.attn_wq'] = matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wk'] = matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wv'] = matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wo'] = matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.mlp_fc1'] = matrix(FF_DIM, N_EMBD)
        sd[f'layer{i}.mlp_fc2'] = matrix(N_EMBD, FF_DIM)
    return sd

# ── Run forward pass on test cases ──────────────────────────────────────────

def run_sequence(tokens, state_dict):
    """Run gpt() for each token, return per-position logits."""
    keys = [[] for _ in range(N_LAYER)]
    values = [[] for _ in range(N_LAYER)]
    all_logits = []
    for pos_id, token_id in enumerate(tokens):
        logits = gpt_forward(token_id, pos_id, keys, values, state_dict)
        all_logits.append(logits)
    return all_logits

def main():
    sd = generate_weights(SEED)

    # Test cases
    test_names = ["emma", "a", "maria"]
    test_cases = []

    for name in test_names:
        tokens = tokenize(name)
        input_tokens = tokens[:-1]  # drop last (we predict next token)
        all_logits = run_sequence(input_tokens, sd)

        # Cross-entropy loss for each position
        targets = tokens[1:]
        losses = []
        for pos, (logits, target) in enumerate(zip(all_logits, targets)):
            probs = softmax(logits)
            loss = -math.log(max(probs[target], 1e-10))
            losses.append(loss)
        avg_loss = sum(losses) / len(losses)

        test_cases.append({
            "name": name,
            "input_tokens": input_tokens,
            "target_tokens": targets,
            "logits": all_logits,  # [seq_len][VOCAB_SIZE]
            "per_position_loss": losses,
            "avg_loss": avg_loss,
        })

    # Tokenizer validation
    tokenizer_cases = [
        {"input": "hello", "expected": tokenize("hello")},
        {"input": "a", "expected": tokenize("a")},
        {"input": "", "expected": tokenize("")},
        {"input": "a-b", "expected": tokenize("a-b")},
    ]

    # Transpose weights for Rust layout:
    # Python: linear(x, w) = w @ x, w is [out, in]
    # Rust:   x @ w, w is [in, out]
    # So rust_w = python_w.T
    def transpose(m):
        rows, cols = len(m), len(m[0])
        return [[m[r][c] for r in range(rows)] for c in range(cols)]

    # For attention: Python has one [16,16] matrix per projection.
    # Rust has 4 per-head [16,4] matrices (already transposed).
    # rust_wq[h][j][k] = python_attn_wq[h*4+k][j]
    # = transpose(python_attn_wq[h*4:(h+1)*4])
    def split_heads_and_transpose(w):
        """[N_EMBD, N_EMBD] → list of N_HEAD × [N_EMBD, HEAD_DIM] (transposed sub-blocks)."""
        heads = []
        for h in range(N_HEAD):
            sub = w[h*HEAD_DIM:(h+1)*HEAD_DIM]  # [HEAD_DIM, N_EMBD]
            heads.append(transpose(sub))  # [N_EMBD, HEAD_DIM]
        return heads

    # For output projection: Python attn_wo is [N_EMBD, N_EMBD]
    # Rust wo[h] is [HEAD_DIM, N_EMBD]
    # rust_wo[h][k][i] = python_attn_wo[i][h*4+k]
    # = transpose(column-sliced python_attn_wo[:, h*4:(h+1)*4])
    def split_wo_heads(w):
        """[N_EMBD, N_EMBD] → list of N_HEAD × [HEAD_DIM, N_EMBD]."""
        heads = []
        for h in range(N_HEAD):
            # Extract columns h*4:(h+1)*4 from each row, then transpose
            sub = [[row[h*HEAD_DIM + k] for k in range(HEAD_DIM)] for row in w]  # [N_EMBD, HEAD_DIM]
            heads.append(transpose(sub))  # [HEAD_DIM, N_EMBD]
        return heads

    rust_weights = {
        "wte": sd['wte'],  # [27, 16] — same layout (embedding lookup, not linear)
        "wpe": sd['wpe'],  # [16, 16] — same layout
        "wq": split_heads_and_transpose(sd['layer0.attn_wq']),
        "wk": split_heads_and_transpose(sd['layer0.attn_wk']),
        "wv": split_heads_and_transpose(sd['layer0.attn_wv']),
        "wo": split_wo_heads(sd['layer0.attn_wo']),
        "w_fc1": transpose(sd['layer0.mlp_fc1']),   # [64,16].T → [16,64]
        "w_fc2": transpose(sd['layer0.mlp_fc2']),   # [16,64].T → [64,16]
        "w_lm": transpose(sd['lm_head']),            # [27,16].T → [16,27]
    }

    reference = {
        "seed": SEED,
        "hyperparams": {
            "vocab_size": VOCAB_SIZE,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "head_dim": HEAD_DIM,
            "block_size": BLOCK_SIZE,
            "ff_dim": FF_DIM,
        },
        "rust_weights": rust_weights,
        "test_cases": test_cases,
        "tokenizer_cases": tokenizer_cases,
    }

    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/parity_reference.json"
    with open(out_path, "w") as f:
        json.dump(reference, f, indent=None, separators=(",", ":"))

    # Print summary
    for tc in test_cases:
        print(f"  {tc['name']:10s} | tokens={tc['input_tokens']} | loss={tc['avg_loss']:.4f}")
    print(f"\nSaved to {out_path} ({len(json.dumps(reference))} bytes)")

if __name__ == "__main__":
    print("microGPT parity reference generator (Python)")
    print(f"Seed: {SEED}, Params: {VOCAB_SIZE}×{N_EMBD} + attention + MLP\n")
    main()
