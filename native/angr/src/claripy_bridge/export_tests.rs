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

use super::ConcreteBvvEncoding;

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
