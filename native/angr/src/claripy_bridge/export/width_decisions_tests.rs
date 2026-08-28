//! In-module unit tests for `claripy_bridge/export/width_decisions.rs`
//! (angr-ph300.52; moved here alongside its subject by the angr-5mnx3.11 split).
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
