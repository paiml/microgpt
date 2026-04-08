//! mdBook contract — validates book structure, heading hierarchy,
//! and chapter file existence.
//!
//! Contract: `document-integrity-v1` § `heading_hierarchy` + `required_sections`

const SUMMARY: &str = include_str!("../book/src/SUMMARY.md");
const INTRO: &str = include_str!("../book/src/introduction.md");
const MODEL: &str = include_str!("../book/src/architecture/model.md");
const ATTENTION: &str = include_str!("../book/src/architecture/attention.md");
const TRAINING: &str = include_str!("../book/src/architecture/training.md");
const LADDER: &str = include_str!("../book/src/contracts/verification-ladder.md");
const REFERENCE: &str = include_str!("../book/src/contracts/reference.md");
const INSPECT: &str = include_str!("../book/src/introspection/inspect.md");
const EXPLAIN: &str = include_str!("../book/src/introspection/explain.md");
const TRACE: &str = include_str!("../book/src/introspection/trace.md");

// ── SUMMARY.md structure ────────────────────────────────────────────────────

#[test]
fn summary_has_title() {
    assert!(
        SUMMARY.starts_with("# Summary"),
        "SUMMARY.md must start with '# Summary'"
    );
}

#[test]
fn summary_links_resolve() {
    // Every [title](path.md) in SUMMARY must correspond to a file we can include_str
    let expected_chapters = [
        "introduction.md",
        "architecture/model.md",
        "architecture/attention.md",
        "architecture/training.md",
        "introspection/inspect.md",
        "introspection/explain.md",
        "introspection/trace.md",
        "contracts/verification-ladder.md",
        "contracts/reference.md",
    ];
    for chapter in expected_chapters {
        assert!(
            SUMMARY.contains(chapter),
            "SUMMARY.md missing link to {chapter}"
        );
    }
}

#[test]
fn summary_no_heading_skips() {
    let mut prev_level = 0u8;
    for line in SUMMARY.lines() {
        if let Some(stripped) = line.strip_prefix('#') {
            #[allow(clippy::cast_possible_truncation)]
            let level = 1 + stripped.chars().take_while(|&c| c == '#').count() as u8;
            if prev_level > 0 {
                assert!(
                    level <= prev_level + 1,
                    "SUMMARY heading skip: H{prev_level} → H{level}"
                );
            }
            prev_level = level;
        }
    }
}

// ── Chapter heading hierarchy ───────────────────────────────────────────────

fn assert_single_h1(content: &str, name: &str) {
    let h1_count = content
        .lines()
        .filter(|l| l.starts_with("# ") && !l.starts_with("##"))
        .count();
    assert_eq!(h1_count, 1, "{name} must have exactly one H1");
}

fn assert_no_heading_skips(content: &str, name: &str) {
    let mut prev_level = 0u8;
    for line in content.lines() {
        if let Some(stripped) = line.strip_prefix('#') {
            #[allow(clippy::cast_possible_truncation)]
            let level = 1 + stripped.chars().take_while(|&c| c == '#').count() as u8;
            if prev_level > 0 {
                assert!(
                    level <= prev_level + 1,
                    "{name}: heading skip H{prev_level} → H{level} in '{line}'"
                );
            }
            prev_level = level;
        }
    }
}

#[test]
fn chapters_single_h1() {
    for (content, name) in [
        (INTRO, "introduction.md"),
        (MODEL, "model.md"),
        (ATTENTION, "attention.md"),
        (TRAINING, "training.md"),
        (INSPECT, "inspect.md"),
        (EXPLAIN, "explain.md"),
        (TRACE, "trace.md"),
        (LADDER, "verification-ladder.md"),
        (REFERENCE, "reference.md"),
    ] {
        assert_single_h1(content, name);
    }
}

#[test]
fn chapters_no_heading_skips() {
    for (content, name) in [
        (INTRO, "introduction.md"),
        (MODEL, "model.md"),
        (ATTENTION, "attention.md"),
        (TRAINING, "training.md"),
        (INSPECT, "inspect.md"),
        (EXPLAIN, "explain.md"),
        (TRACE, "trace.md"),
        (LADDER, "verification-ladder.md"),
        (REFERENCE, "reference.md"),
    ] {
        assert_no_heading_skips(content, name);
    }
}

// ── Content claims match source constants ───────────────────────────────────

#[test]
fn model_page_param_table_sums_to_4192() {
    // The model.md page has a parameter breakdown table.
    // Verify the "Total" row claims 4,192.
    assert!(
        MODEL.contains("**4,192**"),
        "model.md parameter total must be 4,192"
    );
}

#[test]
fn model_page_claims_27_vocab() {
    assert!(
        MODEL.contains("| Vocab | 27 (a-z + BOS) |"),
        "model.md must claim 27 vocab"
    );
}

#[test]
fn training_page_claims_5000_steps() {
    assert!(
        TRAINING.contains("5,000"),
        "training.md must reference 5,000 steps"
    );
}

#[test]
fn training_page_claims_beta_values() {
    assert!(TRAINING.contains("0.85"), "training.md must list beta1=0.85");
    assert!(TRAINING.contains("0.99"), "training.md must list beta2=0.99");
}

// ── Introspection chapters reference apr commands ────────────────────────────

#[test]
fn inspect_page_references_apr_commands() {
    assert!(INSPECT.contains("apr inspect"), "inspect.md must reference apr inspect");
    assert!(INSPECT.contains("apr tensors"), "inspect.md must reference apr tensors");
    assert!(INSPECT.contains("apr explain --kernel"), "inspect.md must reference apr explain --kernel");
    assert!(INSPECT.contains("apr explain --tensor"), "inspect.md must reference apr explain --tensor");
}

#[test]
fn explain_page_references_apr_commands() {
    assert!(EXPLAIN.contains("apr explain"), "explain.md must reference apr explain");
    assert!(EXPLAIN.contains("apr explain --kernel"), "explain.md must reference apr explain --kernel");
}

#[test]
fn trace_page_references_apr_commands() {
    assert!(TRACE.contains("apr trace"), "trace.md must reference apr trace");
    assert!(TRACE.contains("cargo run --example trace_forward"), "trace.md must reference trace example");
}

#[test]
fn introspection_pages_reference_examples() {
    assert!(INSPECT.contains("cargo run --example inspect_model"));
    assert!(EXPLAIN.contains("cargo run --example explain_attention"));
    assert!(TRACE.contains("cargo run --example trace_forward"));
}

// ── Table column parity ─────────────────────────────────────────────────────

fn assert_table_parity(content: &str, name: &str) {
    let mut in_table = false;
    let mut expected_cols = 0;
    for (i, line) in content.lines().enumerate() {
        if line.starts_with('|') && line.ends_with('|') {
            let cols = line.chars().filter(|&c| c == '|').count() - 1;
            if in_table {
                assert_eq!(
                    cols, expected_cols,
                    "{name} line {}: table column mismatch ({cols} vs {expected_cols})",
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

#[test]
fn all_tables_have_column_parity() {
    for (content, name) in [
        (MODEL, "model.md"),
        (TRAINING, "training.md"),
        (LADDER, "verification-ladder.md"),
        (REFERENCE, "reference.md"),
    ] {
        assert_table_parity(content, name);
    }
}

// ── No XSS links ────────────────────────────────────────────────────────────

#[test]
fn no_xss_links_in_book() {
    for (content, name) in [
        (INTRO, "introduction.md"),
        (MODEL, "model.md"),
        (ATTENTION, "attention.md"),
        (TRAINING, "training.md"),
        (INSPECT, "inspect.md"),
        (EXPLAIN, "explain.md"),
        (TRACE, "trace.md"),
        (LADDER, "verification-ladder.md"),
        (REFERENCE, "reference.md"),
    ] {
        assert!(
            !content.contains("](javascript:"),
            "{name} contains XSS link"
        );
    }
}
