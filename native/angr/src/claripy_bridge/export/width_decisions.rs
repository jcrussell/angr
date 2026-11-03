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
    /// `width > 64`: pass a Python int, which `BVV(int, width)` zero-pads for
    /// any width. Never `bytes`: `BVV(bytes, size)` warns on every call that it
    /// ignores `size` (angr-vmi2l.247), and a byte count can disagree with
    /// `width` (angr-ph300.52).
    PyIntWide,
}

impl ConcreteBvvEncoding {
    pub(super) fn for_width(width: u32) -> Self {
        if width <= 64 {
            Self::Int64
        } else {
            Self::PyIntWide
        }
    }
}

test_submod!("width_decisions_tests.rs" => width_decisions_tests);
