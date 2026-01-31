//! Symbolic value system for the VEX execution engine.
//!
//! This module provides:
//! - `RustBV`: A bitvector type that can be concrete or symbolic
//! - `SymContext`: Z3 solver context with constraint management

mod context;
mod value;

pub use context::SymContext;
pub use value::{BitWidth, RustBV, Signedness};
