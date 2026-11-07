//! The [`OpError`] type returned by every `VEXOps` entry point.
//!
//! Extracted from `ops/mod.rs` (angr-9ke6b.170). Re-exported by `ops/mod.rs`
//! so the public path stays `vex::ops::OpError` and the `use super::OpError`
//! imports in the sibling modules keep resolving.

use crate::ir::{IROp, IRType};
use angr_bv::UnsupportedRoundingMode;

/// Errors from VEX operation execution.
///
/// Deliberately not `#[non_exhaustive]`, which was dropped when `vex/` became
/// the `angr-vex` crate (angr-mx00p.51.10): across a crate boundary it forces a
/// wildcard arm, and the engine's `op_error_to_typed` maps every variant
/// exhaustively so a new failure mode cannot slip through untyped.
#[derive(Debug, Clone, thiserror::Error)]
pub enum OpError {
    /// Operation is not a unary operation.
    #[error("operation {0:?} is not unary")]
    NotUnary(IROp),
    /// Operation is not a binary operation.
    #[error("operation {0:?} is not binary")]
    NotBinary(IROp),
    /// Operation is not a quaternary operation.
    #[error("operation {0:?} is not quaternary")]
    NotQuaternary(IROp),
    /// Type mismatch.
    ///
    /// Reserved/defensive variant:
    /// currently unconstructed (kept deliberately, angr-36vvn.5) for the
    /// forthcoming type-checked op paths that will validate operand widths.
    #[error("type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch { expected: IRType, got: IRType },
    /// Invalid float type.
    #[error("invalid float type: {0:?}")]
    InvalidFloatType(IRType),
    /// Unsupported vector operation.
    #[error("unsupported vector operation: {0}")]
    UnsupportedVectorOp(String),
    /// An operand's width does not match the width the opcode implies.
    ///
    /// Built only by `VEXOps::require_operand_width` (module-private, so no
    /// intra-doc link from this public variant), which every
    /// operand-*width* precondition in this crate's op helpers — packed-vector
    /// (angr-3fb7p) and scalar (angr-fs8kb.95) alike — routes through instead
    /// of `debug_assert_eq!`. See that method's rustdoc for why the check has
    /// to report in every profile rather than panic in dev only.
    ///
    /// `what` names the caller and the operand (`"vec_mull left"`,
    /// `"binop add_into right"`), so the stringified error identifies the site.
    /// Distinct from [`Self::UnsupportedVectorOp`], which the scalar family
    /// would have had to borrow: "unsupported vector operation" is a wrong
    /// description of a width-mismatched `Iop_Add64`.
    #[error("{what}: operand width {actual} does not match expected {expected}")]
    OperandWidthMismatch {
        what: &'static str,
        actual: u32,
        expected: u32,
    },
    /// NEON op that hasn't been implemented yet (angr-bkcs scaffold).
    ///
    /// Distinct from [`Self::UnsupportedVectorOp`] because the silent
    /// fresh-symbolic fallback in `interpreter::expressions` swallows
    /// generic `OpError`s — this variant is propagated explicitly so
    /// missing NEON coverage surfaces as a typed `RustUnsupportedVexOpError`
    /// (test path) or a stringified errored-stash record (live exploration)
    /// instead of producing wrong results that are hard to attribute.
    /// Scaffolding must surface as this error, never as a silent fallback.
    #[error("NEON op {name} not yet implemented")]
    UnsupportedNeon { name: &'static str },
    /// Opcode string with no entry in `parse_opcode` (angr-tkbr.2).
    ///
    /// Routed from `IROp::Unmapped(name)`. Like `UnsupportedNeon`, this
    /// is propagated explicitly past the silent fresh-symbolic fallback
    /// in `interpreter::expressions` so callers see the real op name
    /// (typed `RustUnsupportedVexOpError` on the test path) instead of
    /// getting a fresh-symbolic value of the wrong width. Live exploration
    /// re-runs the block in Python's VEX engine instead of erroring the
    /// state (`CbExecutionError::strategy`, angr-mx00p.49).
    #[error("unmapped VEX opcode: {op_name}")]
    UnsupportedVexOp { op_name: String },
    /// A concrete rounding-mode operand `VexRm::decode` rejects (see its
    /// type doc): the Rust analogue of the Python engine's
    /// `UnsupportedIROpError` from `irop.py::translate_rm`. It takes the
    /// generic arm of `VEXInterpreter::vex_op_fallback`: the errored stash for
    /// concrete operands, the Python engine (which raises too) otherwise.
    #[error(transparent)]
    UnsupportedRoundingMode(#[from] UnsupportedRoundingMode),
    /// Raw/unimplemented opcode.
    #[error("raw/unimplemented opcode: {0}")]
    RawOpcode(u32),
}
