# microgpt

Karpathy's microGPT ported to Rust with aprender. Character-level GPT (4,192 params).

## Code Search Policy

**NEVER use grep/glob for code search. ALWAYS prefer `pmat query`.**

```bash
pmat query "attention forward" --limit 10
pmat query "tokenize" --include-source --limit 5
pmat query --regex "fn\s+test_\w+" --limit 20
pmat query "unwrap" --faults --exclude-tests
```

## Contracts

See `contracts/microgpt-v1.yaml` for formal invariants.

```bash
pmat comply check
pv score contracts/microgpt-v1.yaml
```

## Quality Gates

```bash
cargo test                    # 37 tests, 99.78% coverage
cargo clippy -- -D warnings   # zero warnings
cargo llvm-cov --lib --summary-only
```
