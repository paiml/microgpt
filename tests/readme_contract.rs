//! README.md contract — validates that every architectural claim in the README
//! matches the actual source constants and runtime behavior.
//!
//! Contract: `document-integrity-v1` § `required_sections` + `heading_hierarchy`
//! + `table_column_parity` + `link_wellformedness`.

use aprender::autograd::clear_graph;
use microgpt::*;

const README: &str = include_str!("../README.md");

// ── Heading hierarchy (document-integrity-v1 § heading_hierarchy) ───────────

#[test]
fn readme_has_exactly_one_h1() {
    let h1_count = README.lines().filter(|l| l.starts_with("# ") && !l.starts_with("##")).count();
    assert_eq!(h1_count, 1, "README must have exactly one H1");
}

#[test]
fn readme_no_heading_skips() {
    let mut prev_level = 0u8;
    for line in README.lines() {
        if let Some(stripped) = line.strip_prefix('#') {
            #[allow(clippy::cast_possible_truncation)]
            let level = 1 + stripped.chars().take_while(|&c| c == '#').count() as u8;
            if prev_level > 0 {
                assert!(
                    level <= prev_level + 1,
                    "heading skip: H{prev_level} → H{level} in: {line}"
                );
            }
            prev_level = level;
        }
    }
}

// ── Required sections (document-integrity-v1 § required_sections) ───────────

#[test]
fn readme_required_sections() {
    for section in ["## Architecture", "## Install", "## Usage", "## License"] {
        assert!(README.contains(section), "README missing section: {section}");
    }
}

// ── Architectural claims match source constants ─────────────────────────────

#[test]
fn readme_claims_embedding_dim() {
    assert!(
        README.contains(&format!("| Embedding dim | {N_EMBD} |")),
        "README claims wrong embedding dim (expected {N_EMBD})"
    );
}

#[test]
fn readme_claims_attention_heads() {
    assert!(
        README.contains(&format!("| Attention heads | {N_HEAD} (head_dim={HEAD_DIM}) |")),
        "README claims wrong head count (expected {N_HEAD}, head_dim={HEAD_DIM})"
    );
}

#[test]
fn readme_claims_layers() {
    assert!(
        README.contains("| Layers | 1 |"),
        "README claims wrong layer count (model has 1 layer)"
    );
}

#[test]
fn readme_claims_context_length() {
    assert!(
        README.contains(&format!("| Context length | {BLOCK_SIZE} |")),
        "README claims wrong context length (expected {BLOCK_SIZE})"
    );
}

#[test]
fn readme_claims_vocab_size() {
    assert!(
        README.contains(&format!("| Vocab | {VOCAB_SIZE} (a-z + BOS) |")),
        "README claims wrong vocab (expected {VOCAB_SIZE})"
    );
}

#[test]
fn readme_claims_param_count() {
    let model = MicroGPT::new();
    let actual = model.param_count();
    assert_eq!(actual, 4192);
    // Table uses formatted "4,192"
    assert!(
        README.contains("| Parameters | 4,192 |"),
        "README table claims wrong param count"
    );
    // Prose uses "4,192-parameter"
    assert!(
        README.contains("4,192-parameter"),
        "README prose claims wrong param count"
    );
    clear_graph();
}

#[test]
fn readme_claims_activation() {
    assert!(
        README.contains("| Activation | ReLU |"),
        "README claims wrong activation (model uses ReLU)"
    );
}

#[test]
fn readme_claims_optimizer_betas() {
    assert!(
        README.contains(&format!(
            "| Optimizer | Adam (beta1={BETA1}, beta2={BETA2}) |"
        )),
        "README claims wrong optimizer betas"
    );
}

// ── Dependency claims ───────────────────────────────────────────────────────

#[test]
fn readme_claims_aprender_only() {
    assert!(
        README.contains("aprender-core"),
        "README must reference aprender-core"
    );
    assert!(
        README.contains("No path or git dependencies"),
        "README must state no path/git deps"
    );
}

// ── Link well-formedness (document-integrity-v1 § link_wellformedness) ──────

#[test]
fn readme_no_broken_link_syntax() {
    for (i, line) in README.lines().enumerate() {
        assert!(
            !line.contains("](javascript:"),
            "XSS link at README line {}: {line}",
            i + 1
        );
    }
}

// ── Badge contract (CB-1320 + document-integrity-v1 § badge_integrity) ──────

#[test]
fn readme_has_ci_badge() {
    assert!(
        README.contains("[![CI]"),
        "README must have CI badge"
    );
    assert!(
        README.contains("actions/workflows/ci.yml/badge.svg"),
        "CI badge must point to ci.yml workflow"
    );
    assert!(
        README.contains("github.com/paiml/microgpt/actions"),
        "CI badge must link to paiml/microgpt actions"
    );
}

#[test]
fn readme_has_crate_badge() {
    assert!(
        README.contains("[![crate]"),
        "README must have crate version badge"
    );
    assert!(
        README.contains("img.shields.io/crates/v/microgpt"),
        "Crate badge must point to microgpt on crates.io"
    );
}

#[test]
fn readme_has_license_badge() {
    assert!(
        README.contains("[![license]"),
        "README must have license badge"
    );
    assert!(
        README.contains("license-MIT-blue"),
        "License badge must show MIT"
    );
}

#[test]
fn readme_badges_before_h1() {
    // Badges must appear before the first H1 (standard layout)
    let h1_pos = README.find("# microGPT").expect("README must have H1");
    let badge_pos = README.find("[![").expect("README must have badges");
    assert!(
        badge_pos < h1_pos,
        "badges must appear before H1 (badge at {badge_pos}, H1 at {h1_pos})"
    );
}

#[test]
fn readme_badge_urls_use_https() {
    for line in README.lines() {
        if line.starts_with("[![") {
            assert!(
                !line.contains("http://"),
                "badge URLs must use HTTPS: {line}"
            );
        }
    }
}

// ── Table column parity (document-integrity-v1 § table_column_parity) ───────

#[test]
fn readme_table_columns_consistent() {
    let mut in_table = false;
    let mut expected_cols = 0;

    for (i, line) in README.lines().enumerate() {
        if line.starts_with('|') && line.ends_with('|') {
            let cols = line.chars().filter(|&c| c == '|').count() - 1;
            if in_table {
                assert_eq!(
                    cols, expected_cols,
                    "table column count mismatch at README line {} ({cols} vs {expected_cols})",
                    i + 1
                );
            } else {
                in_table = true;
                expected_cols = cols;
            }
        } else {
            in_table = false;
        }
    }
}
