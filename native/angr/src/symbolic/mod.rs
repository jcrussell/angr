//! Symbolic value system for the VEX execution engine.
//!
//! This module provides:
//! - `RustBV`: A bitvector type that can be concrete or symbolic
//! - `SymContext`: Z3 solver context with constraint management
//! - `RustBVHandle`: Python-facing opaque handle for bypassing claripy
//! - `RustSymbolTable`: Registry mapping handles to RustBV values

mod context;
mod handle;
mod table;
mod value;

pub use context::SymContext;
pub use handle::RustBVHandle;
pub use table::RustSymbolTable;
pub use value::{BitWidth, BVOp, RustBV, Signedness};
