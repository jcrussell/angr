//! Symbolic value system for the VEX execution engine.
//!
//! This module provides:
//! - `RustBV`: A bitvector type that can be concrete or symbolic
//! - `SymContext`: Z3 solver context with constraint management
//! - `RustBVHandle`: Python-facing opaque handle for bypassing claripy
//! - `RustSymbolTable`: Registry mapping handles to RustBV values
//! - `SymbolicIdentityRegistry`: Preserves symbolic identity across Python<->Rust

mod context;
mod handle;
pub mod registry;
mod table;
mod value;

pub use context::{
    ConstraintSyncError, SymContext, get_solver_stats, record_mem_ite_depth, reset_solver_stats,
};
pub use handle::RustBVHandle;
pub use registry::{SymbolInfo, SymbolicIdentityRegistry, clear_global_registry, global_registry};
pub use table::RustSymbolTable;
pub use value::{BVOp, BitWidth, FloatOpKind, FloatPrec, RustBV, Signedness};
