//! Pure width/coercion decision seams for the RustBV -> claripy export walk.
//!
//! Split out of `export.rs` (angr-5mnx3.11). Every item here is a `match`-free
//! total function of integers and enum values: no Python interpreter, no `Bound`
//! handles, so [`super::rustbv_to_claripy_memo`]'s two historically fragile
//! branch decisions — which `claripy.BVV` argument encoding a concrete width
//! takes ([`ConcreteBvvEncoding`], angr-ph300.52) and how a two-operand width
//! mismatch is reconciled ([`WidthFixup`], angr-c7xno.14) — stay unit-testable
//! in the claripy-less cargo-test environment. Keep that property: an item that
//! needs `py` belongs in [`super::ast_helpers`], not here.

/// Which `claripy.BVV` argument encoding a concrete value of a given bit width
/// takes. Pure decision seam, unit-tested in `width_decisions_tests.rs` without a Python
/// interpreter — the width-guard branch that regressed in the `Constrained`
/// arm (angr-ph300.52) is exactly this decision.
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

/// Which operands of a two-operand `BVOp` were `Bool`s coerced to `BV(1)`
/// before the width-reconciliation step in `rustbv_to_claripy_memo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BoolCoercion {
    /// Both operands arrived as real BVs; no coercion happened.
    Neither,
    /// Operand 0 was a `Bool`, now a `BV(1)`.
    Arg0,
    /// Operand 1 was a `Bool`, now a `BV(1)`.
    Arg1,
    /// Both operands were `Bool`s, now `BV(1)`s.
    Both,
}

impl BoolCoercion {
    /// Whether operand `idx` is a coerced `Bool` (so its value is 0 or 1).
    fn covers(self, idx: usize) -> bool {
        matches!(
            (self, idx),
            (Self::Both, _) | (Self::Arg0, 0) | (Self::Arg1, 1)
        )
    }
}

/// How `rustbv_to_claripy_memo` reconciles a two-operand `BVOp` whose claripy
/// operands ended up with different widths. Pure decision seam, unit-tested in
/// `width_decisions_tests.rs` without a Python interpreter.
///
/// The only legitimate width mismatch is the [`BoolCoercion`] one: a `Bool`
/// operand became a `BV(1)` holding 0 or 1, so widening it with `ZeroExt` is
/// unambiguously value-preserving. Two *real* BVs of different widths cannot
/// happen — every `RustBV` binary-op constructor in `symbolic::value_ops`
/// checks operand widths before building the node — so reaching that case means
/// an upstream invariant was violated (a `debug_assert_eq!` compiled out in
/// release, or a hand-built/deserialized `Expression`). Zero-extending there
/// would silently mint a semantically wrong AST — sign-flipped for a signed op
/// like `Slt`/`SDiv` — so this module's "fail loud rather than hand back a
/// plausible-looking wrong answer" rule applies and the case is rejected
/// (angr-c7xno.14).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WidthFixup {
    /// Widths already agree; pass the operands through untouched.
    Agree,
    /// `ZeroExt` operand 0 (a coerced `Bool`) up to operand 1's width.
    ZeroExtendArg0,
    /// `ZeroExt` operand 1 (a coerced `Bool`) up to operand 0's width.
    ZeroExtendArg1,
    /// Mismatched widths that no `Bool` coercion explains — fail loud.
    Reject,
}

impl WidthFixup {
    pub(super) fn decide(w0: u32, w1: u32, coercion: BoolCoercion) -> Self {
        // The narrower operand is the one that would be widened; it is only
        // safe to widen when it is a coerced Bool.
        let (narrower, fixup) = if w0 == w1 {
            return Self::Agree;
        } else if w0 < w1 {
            (0usize, Self::ZeroExtendArg0)
        } else {
            (1usize, Self::ZeroExtendArg1)
        };
        if coercion.covers(narrower) {
            fixup
        } else {
            Self::Reject
        }
    }
}

test_submod!("width_decisions_tests.rs" => width_decisions_tests);
