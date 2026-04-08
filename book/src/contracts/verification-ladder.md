# Verification Ladder

microGPT's contract (`microgpt-v1.yaml`) is verified at multiple levels:

| Level | Method | Status | Count |
|-------|--------|--------|-------|
| L2 | Property tests (`tests/contract_traits.rs`) | 7/7 | 7 |
| L3 | Bounded model checking (decidable in Lean `native_decide`) | 7/7 | 7 |
| L4 | Lean 4 proofs (`lean/MicroGPT.lean`, 0 sorry) | 7/7 | 7 |
| L5 | Lean proved + fully bound | pending | 0 |

## What each level means

- **L2 (Property tested)**: Every proof obligation has a corresponding
  Rust test in `tests/contract_traits.rs` that asserts the invariant at
  runtime. These tests run on every `cargo test`.

- **L3 (Bounded model checked)**: The obligations are concrete arithmetic
  or structural properties decidable by `native_decide` in Lean 4.

- **L4 (Lean proved)**: `lean/MicroGPT.lean` contains 6 theorems with
  0 `sorry` stubs, covering param count, Adam monotonicity, second-moment
  non-negativity, and tokenizer structure.

- **L5 (Fully bound)**: Requires build.rs pipeline integration to pass
  binding status to `compute_proof_level`. Not applicable for standalone
  educational repos without the full provable-contracts build pipeline.

## README under contract

The README itself is verified by `tests/readme_contract.rs` (15 tests):

- Heading hierarchy (no skips, exactly one H1)
- Required sections (Architecture, Install, Usage, License)
- Every table claim matches a source constant
- Badge URLs are well-formed
- Table column parity
- No XSS links

## Book under contract

This mdBook is verified by `tests/book_contract.rs`:

- SUMMARY.md heading hierarchy
- All chapter files referenced in SUMMARY.md exist
- No broken internal links
- Consistent heading structure per chapter
