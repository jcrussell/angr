//! The [`OpError`] type returned by every `VEXOps` entry point.
//!
//! Extracted from `ops/mod.rs` (angr-9ke6b.170). Re-exported by `ops/mod.rs`
//! so the public path stays `vex::ops::OpError` and the `use super::OpError`
//! imports in the sibling modules keep resolving.

use crate::vex::ir::{IROp, IRType};

/// Errors from VEX operation execution.
///
/// `#[non_exhaustive]` per angr-irwe: new variants land in minor
/// versions as more ops gain explicit failure modes (e.g. additional
/// NEON scaffold buckets). Match sites must include a wildcard arm.
#[non_exhaustive]
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
    /// Reserved/defensive variant on this public `#[non_exhaustive]` enum:
    /// currently unconstructed (kept-with-ticket per angr-36vvn.5) for the
    /// forthcoming type-checked op paths that will validate operand widths.
    #[error("type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch { expected: IRType, got: IRType },
    /// Invalid float type.
    #[error("invalid float type: {0:?}")]
    InvalidFloatType(IRType),
    /// Unsupported vector operation.
    #[error("unsupported vector operation: {0}")]
    UnsupportedVectorOp(String),
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
    /// (typed `RustUnsupportedVexOpError` on the test path; stringified
    /// into the errored stash in live exploration — see the taxonomy note
    /// in `errors.rs`) instead of getting a fresh-symbolic value of the
    /// wrong width.
    #[error("unmapped VEX opcode: {op_name}")]
    UnsupportedVexOp { op_name: String },
    /// Raw/unimplemented opcode.
    #[error("raw/unimplemented opcode: {0}")]
    RawOpcode(u32),
}
