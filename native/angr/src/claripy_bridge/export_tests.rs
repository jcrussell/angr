//! In-module unit tests for `claripy_bridge/export.rs` (angr-ph300.52).
//!
//! These pin the concrete-value `BVV` encoding decision (`ConcreteBvvEncoding`)
//! that the `Concrete` and `Constrained` export arms now share. The two arms
//! drifted once: the `Constrained` arm was missing the `width / 8 <= 16` clause,
//! so a byte-aligned width > 128 (e.g. 192) built a 16-byte `PyBytes` and handed
//! it to `BVV(bytes, 192)` — `ClaripyValueError` string/size mismatch — while the
//! `Concrete` twin fell through to the wide-int path and succeeded.
//!
//! Pure (no Python interpreter): the cargo-test env has no claripy, so we assert
//! the branch decision rather than round-trip through `claripy.BVV`.

use super::{BoolCoercion, ConcreteBvvEncoding, WidthFixup};

#[test]
fn width_le_64_is_int64() {
    assert_eq!(
        ConcreteBvvEncoding::for_width(1),
        ConcreteBvvEncoding::Int64
    );
    assert_eq!(
        ConcreteBvvEncoding::for_width(40),
        ConcreteBvvEncoding::Int64
    );
    assert_eq!(
        ConcreteBvvEncoding::for_width(64),
        ConcreteBvvEncoding::Int64
    );
}

#[test]
fn byte_aligned_up_to_128_uses_bytes() {
    // 65..=128, byte-aligned: exact big-endian bytes, byte_count = width / 8.
    assert_eq!(
        ConcreteBvvEncoding::for_width(72),
        ConcreteBvvEncoding::Bytes(9)
    );
    assert_eq!(
        ConcreteBvvEncoding::for_width(128),
        ConcreteBvvEncoding::Bytes(16)
    );
}

#[test]
fn non_byte_aligned_over_64_uses_pyint() {
    // width % 8 != 0 and width > 64: cannot be exact bytes -> Python int.
    assert_eq!(
        ConcreteBvvEncoding::for_width(65),
        ConcreteBvvEncoding::PyIntWide
    );
    assert_eq!(
        ConcreteBvvEncoding::for_width(96 + 1),
        ConcreteBvvEncoding::PyIntWide
    );
}

#[test]
fn byte_aligned_over_128_uses_pyint_not_bytes() {
    // The angr-ph300.52 regression: width = 192 is byte-aligned but > 128, so
    // its 24 bytes cannot come from the u128's 16 bytes. MUST be PyIntWide, not
    // Bytes(24) — the latter is what the un-guarded Constrained arm produced.
    assert_eq!(
        ConcreteBvvEncoding::for_width(192),
        ConcreteBvvEncoding::PyIntWide
    );
    assert_eq!(
        ConcreteBvvEncoding::for_width(136),
        ConcreteBvvEncoding::PyIntWide
    );
    assert_eq!(
        ConcreteBvvEncoding::for_width(256),
        ConcreteBvvEncoding::PyIntWide
    );
}

// --- WidthFixup (angr-c7xno.14) ---------------------------------------------
//
// The binary-op width-reconciliation step used to `ZeroExt` the narrower
// operand unconditionally. That is only value-preserving when the narrower
// operand is a Bool coerced to BV(1); for two real BVs of different widths it
// would hand back a plausible-looking but semantically wrong AST — sign-flipped
// for a signed op such as `Slt`/`SDiv`. These pin the decision seam.

#[test]
fn equal_widths_need_no_fixup() {
    for coercion in [
        BoolCoercion::Neither,
        BoolCoercion::Arg0,
        BoolCoercion::Arg1,
        BoolCoercion::Both,
    ] {
        assert_eq!(WidthFixup::decide(32, 32, coercion), WidthFixup::Agree);
        assert_eq!(WidthFixup::decide(1, 1, coercion), WidthFixup::Agree);
    }
}

#[test]
fn coerced_bool_operand_is_zero_extended() {
    // The reachable-today shapes: one operand was a claripy Bool (length None),
    // became BV(1), and the other is a real BV wider than 1.
    assert_eq!(
        WidthFixup::decide(1, 64, BoolCoercion::Arg0),
        WidthFixup::ZeroExtendArg0
    );
    assert_eq!(
        WidthFixup::decide(64, 1, BoolCoercion::Arg1),
        WidthFixup::ZeroExtendArg1
    );
    // `Both` covers either side (though in practice it yields 1 vs 1).
    assert_eq!(
        WidthFixup::decide(1, 8, BoolCoercion::Both),
        WidthFixup::ZeroExtendArg0
    );
    assert_eq!(
        WidthFixup::decide(8, 1, BoolCoercion::Both),
        WidthFixup::ZeroExtendArg1
    );
}

#[test]
fn mismatched_real_bv_widths_are_rejected() {
    // No coercion happened, so both operands are real BVs — an upstream
    // invariant violation, not something to paper over with ZeroExt.
    assert_eq!(
        WidthFixup::decide(32, 64, BoolCoercion::Neither),
        WidthFixup::Reject
    );
    assert_eq!(
        WidthFixup::decide(64, 32, BoolCoercion::Neither),
        WidthFixup::Reject
    );
}

#[test]
fn coercion_on_the_wider_side_does_not_license_extension() {
    // Arg0 was the coerced Bool, yet Arg1 is the narrower operand: whatever
    // produced this, the operand about to be widened is a real BV. Reject.
    assert_eq!(
        WidthFixup::decide(8, 4, BoolCoercion::Arg0),
        WidthFixup::Reject
    );
    assert_eq!(
        WidthFixup::decide(4, 8, BoolCoercion::Arg1),
        WidthFixup::Reject
    );
}

// --- Constrained-arm cache asymmetry (angr-03vl4.9) --------------------------

/// Pins the reachability claim the `RustBV::Constrained` export arm's rationale
/// leans on: snapshot deserialization is the only live constructor of that
/// variant, so the `get_claripy_ast(id)` lookup the `Symbolic` arm performs
/// would be a guaranteed miss here. A new live constructor is exactly the change
/// that would make the skipped lookup observable, so it should land as a failure
/// of this test — which names the arm to revisit — rather than as a silent
/// behaviour change.
///
/// Textual, like `callbacks::inspect_bits_tests::bits_match_python_inspect_event_specs`:
/// the property is "no such call site exists anywhere in the crate", which no
/// amount of runtime exercising can demonstrate.
#[test]
fn constrained_has_no_live_constructor_beyond_snapshot_load() {
    let src_root = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    let mut files = Vec::new();
    collect_rs_files(std::path::Path::new(src_root), &mut files);
    assert!(
        !files.is_empty(),
        "no .rs files under {src_root} -- the scan would pass vacuously"
    );

    let mut snapshot_sites = 0usize;
    let mut other_sites = Vec::new();
    for path in &files {
        // Unit tests build `Constrained` values freely; the invariant is about
        // production paths. Test modules live in sibling `*_tests.rs` files.
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("_tests.rs"))
        {
            continue;
        }
        // SILENT(cat-a): a file that cannot be read is not a call site; the
        // `files.is_empty()` guard above already rules out a wholly failed scan.
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        for (lineno, line) in src.lines().enumerate() {
            let Some(col) = line.find("RustBV::Constrained {") else {
                continue;
            };
            if line.trim_start().starts_with("//") {
                continue;
            }
            // Struct-literal position vs. pattern position. Patterns sit in
            // `match` arms, `matches!`, tuple patterns and `|` alternations, all
            // of which put the arrow (if any) *after* the occurrence; a literal
            // sits on the value side of `=>` or `=`.
            let before = line[..col].trim_end();
            let is_literal = before.ends_with("=>")
                || before.ends_with("return")
                || (before.ends_with('=')
                    && !before.ends_with("==")
                    && !before.ends_with("!=")
                    && !before.ends_with("<=")
                    && !before.ends_with(">="));
            if !is_literal {
                continue;
            }
            if before.contains("RustBVData::Constrained") {
                snapshot_sites += 1;
            } else {
                other_sites.push(format!("{}:{}", path.display(), lineno + 1));
            }
        }
    }

    assert!(
        other_sites.is_empty(),
        "new live constructor(s) of RustBV::Constrained: {other_sites:?} -- \
         revisit the Constrained arm of rustbv_to_claripy_memo, which skips the \
         identity-cache lookup on the grounds that no live path produces a \
         Constrained carrying a registered symbol id"
    );
    assert_eq!(
        snapshot_sites, 1,
        "expected exactly one RustBV::Constrained literal (the RustBVData \
         deserialization arm in symbolic/value.rs); the scan may have gone stale"
    );
}

fn collect_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    // SILENT(cat-a): an unreadable directory contributes no call sites; the
    // caller asserts the overall scan found files.
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}
