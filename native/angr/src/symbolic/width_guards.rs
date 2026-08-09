//! Bound checks for externally-controlled bitvector widths and Extract bounds.
//!
//! Every value that reaches [`RustBV`](super::RustBV)'s constructors from
//! *outside* the engine — a claripy AST imported by `claripy_bridge::import`,
//! a width or bit range passed to the handle-based bypass API in
//! `solver::handle_api` — carries a width the caller chose. `RustBV::concrete`
//! / `RustBV::symbolic` / the structural ops validate none of it: the only
//! guards downstream (`value_ops::truncate_into`, `value_ops::drive_extract`)
//! are `debug_assert!`s, which are compiled out of the `release` profile the
//! extension actually ships. An out-of-range `Extract` then reaches
//! `Z3_mk_extract` as an invalid-argument call (abort under `panic="abort"`,
//! not a catchable Python exception), and an absurd width reaches Z3 as a
//! multi-gigabit sort allocation (angr-c7xno.92/.93/.94).
//!
//! So the checks live here, once, and both trust boundaries call them. They
//! return the message rather than a typed error because the two callers report
//! through different channels (`PyValueError` vs `BridgeError`).

/// Largest bitvector width accepted from an external caller.
///
/// 1 Mibit (128 KiB per value) is far above anything the engine builds for
/// real — the widest values in practice are page-sized memory chunks and the
/// 1024-bit BVs the solver round-trip tests use — while still refusing the
/// `u32::MAX`-shaped widths that would have Z3 allocate a half-gigabyte sort.
/// Width **0** is deliberately allowed: a zero-width concrete `RustBV` is a
/// supported degenerate value (see `test_create_concrete_zero_width_round_trip`
/// in `tests/engines/rust/test_plugins.py`).
pub const MAX_BV_WIDTH: u32 = 1 << 20;

/// Reject a width above [`MAX_BV_WIDTH`], naming `op` in the message.
pub fn check_bv_width(op: &str, width: u32) -> Result<(), String> {
    if width > MAX_BV_WIDTH {
        return Err(format!(
            "{op}: width {width} exceeds the maximum supported bitvector width {MAX_BV_WIDTH}"
        ));
    }
    Ok(())
}

/// Reject an `Extract` whose bit range is not inside `source_width`.
///
/// Both halves matter and neither implies the other: `low > high` makes the
/// result width `high - low + 1` wrap to near-`u32::MAX` (the `release`
/// profile has no overflow checks), and `high >= source_width` is a plain
/// out-of-range extract. Callers that also want an upper bound on the *result*
/// width get it for free — a valid range inside a `check_bv_width`-checked
/// source cannot exceed the source.
pub fn check_extract_bounds(
    op: &str,
    high: u32,
    low: u32,
    source_width: u32,
) -> Result<(), String> {
    if high >= source_width {
        return Err(format!("{op} high={high} >= width={source_width}"));
    }
    if low > high {
        return Err(format!("{op} low={low} > high={high}"));
    }
    Ok(())
}

test_submod!("width_guards_tests.rs" => tests);
