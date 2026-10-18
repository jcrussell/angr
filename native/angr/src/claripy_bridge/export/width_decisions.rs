//! Pure width decision seams for the RustBV -> claripy export.
//!
//! Split out of `export.rs` (angr-5mnx3.11). Every item here is a total
//! function of integers and enum values: no Python interpreter, no `Bound`
//! handles, so which `claripy.BVV` argument encoding a concrete width takes
//! ([`ConcreteBvvEncoding`], angr-ph300.52) stays unit-testable in the
//! claripy-less cargo-test environment. Keep that property: an item that needs
//! `py` belongs in [`super::ast_helpers`], not here.

/// Which `claripy.BVV` argument encoding a concrete value of a given bit width
/// takes. Pure decision seam, unit-tested in `width_decisions_tests.rs` without a Python
/// interpreter — the width-guard branch that regressed once (angr-ph300.52)
/// is exactly this decision.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ConcreteBvvEncoding {
    /// `width <= 64`: pass as `i64`.
    Int64,
    /// Byte-aligned and fits in the u128's 16 bytes: pass the big-endian bytes.
    Bytes(usize),
    /// Non-byte-aligned OR width > 128: pass a Python int (zero-padded).
    PyIntWide,
}

impl ConcreteBvvEncoding {
    pub(super) fn for_width(width: u32) -> Self {
        if width <= 64 {
            Self::Int64
        } else if width.is_multiple_of(8) && width as usize / 8 <= 16 {
            Self::Bytes(width as usize / 8)
        } else {
            Self::PyIntWide
        }
    }
}

test_submod!("width_decisions_tests.rs" => width_decisions_tests);
