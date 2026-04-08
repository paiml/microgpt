//! SVG contract — validates hero.svg against `document-integrity-v1` §
//! `svg_structural_safety` and `microgpt-v1` § `hero_svg`.

const HERO: &str = include_str!("../assets/hero.svg");

// ── Valid XML with SVG root ─────────────────────────────────────────────────

#[test]
fn svg_has_svg_root() {
    assert!(
        HERO.contains("<svg"),
        "hero.svg must have an <svg> root element"
    );
    assert!(
        HERO.contains("</svg>"),
        "hero.svg must have a closing </svg> tag"
    );
}

#[test]
fn svg_has_xmlns() {
    assert!(
        HERO.contains("xmlns=\"http://www.w3.org/2000/svg\""),
        "hero.svg must declare SVG namespace"
    );
}

#[test]
fn svg_has_viewbox() {
    assert!(
        HERO.contains("viewBox"),
        "hero.svg must have a viewBox attribute for responsive scaling"
    );
}

// ── XSS prevention ─────────────────────────────────────────────────────────

#[test]
fn svg_no_script() {
    assert!(
        !HERO.contains("<script"),
        "hero.svg must not contain <script> elements"
    );
}

#[test]
fn svg_no_foreign_object() {
    assert!(
        !HERO.contains("<foreignObject"),
        "hero.svg must not contain <foreignObject> elements"
    );
}

#[test]
fn svg_no_onclick() {
    assert!(
        !HERO.contains("onclick") && !HERO.contains("onload") && !HERO.contains("onerror"),
        "hero.svg must not contain event handler attributes"
    );
}

// ── Content accuracy ────────────────────────────────────────────────────────

#[test]
fn svg_shows_param_count() {
    assert!(
        HERO.contains("4,192"),
        "hero.svg must display the parameter count"
    );
}

#[test]
fn svg_shows_architecture() {
    assert!(HERO.contains("4 heads"), "hero.svg must mention 4 heads");
    assert!(HERO.contains("1 layer"), "hero.svg must mention 1 layer");
    assert!(
        HERO.contains("16-dim"),
        "hero.svg must mention 16-dim embeddings"
    );
}

#[test]
fn svg_shows_pipeline() {
    for stage in ["tokens", "embed", "RMSNorm", "attn", "MLP", "logits", "names"] {
        assert!(
            HERO.contains(stage),
            "hero.svg pipeline must include '{stage}'"
        );
    }
}

// ── README references hero ──────────────────────────────────────────────────

#[test]
fn readme_references_hero() {
    let readme = include_str!("../README.md");
    assert!(
        readme.contains("assets/hero.svg"),
        "README.md must reference assets/hero.svg"
    );
}
