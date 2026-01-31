//! VEX Intermediate Representation types and operations.
//!
//! This module provides:
//! - VEX IR types (statements, expressions, operations)
//! - VEX operation implementations
//! - VEX lifting interface

pub mod ir;
mod lifter;
pub mod ops;

pub use ir::*;
pub use lifter::{IRSBBuilder, LiftError, NativeVEXLifter, VEXLifter};
pub use ops::{OpError, VEXOps};
