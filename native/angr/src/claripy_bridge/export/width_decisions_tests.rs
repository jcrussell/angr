//! In-module unit tests for `claripy_bridge/export/width_decisions.rs`
//! (angr-ph300.52), kept alongside its subject (angr-5mnx3.11).
//!
//! These pin the concrete-value `BVV` encoding decision (`ConcreteBvvEncoding`).
//! It once had a `bytes` arm, and a copy of it missed the `width / 8 <= 16` clause, so a byte-aligned
//! width > 128 (e.g. 192) built a 16-byte `PyBytes` and handed it to
//! `BVV(bytes, 192)` — `ClaripyValueError` string/size mismatch.
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
fn byte_aligned_up_to_128_uses_pyint_not_bytes() {
    // angr-vmi2l.247: `BVV(bytes, size)` warns on every call, so even an exact
    // 9- or 16-byte value goes out as a Python int.
    assert_eq!(
        ConcreteBvvEncoding::for_width(72),
        ConcreteBvvEncoding::PyIntWide
    );
    assert_eq!(
        ConcreteBvvEncoding::for_width(128),
        ConcreteBvvEncoding::PyIntWide
    );
}

#[test]
fn non_byte_aligned_over_64_uses_pyint() {
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
    // Regression (angr-ph300.52): width = 192 is byte-aligned but > 128, so
    // its 24 bytes cannot come from the u128's 16 bytes; a 24-byte `bytes`
    // value is what the encoding once produced without the width guard.
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
