//! VEX Intermediate Representation types and operations.
//!
//! This module provides:
//! - VEX IR types (statements, expressions, operations)
//! - VEX operation implementations
//! - VEX lifting interface
//! - pyvex IRSB serialization/deserialization
//! - Clean call (CCall) implementations for flag calculations

pub mod ccall;
pub mod ir;
mod lifter;
pub mod opcode_map;
pub mod ops;
pub mod pyvex_bridge;

pub use ir::*;
pub use lifter::{IRSBBuilder, LiftError, NativeVEXLifter, VEXLifter};
pub use opcode_map::{parse_endness, parse_jumpkind, parse_opcode, parse_type};
pub use ops::{OpError, VEXOps};
pub use pyvex_bridge::{deserialize_irsb, DeserializeError};
