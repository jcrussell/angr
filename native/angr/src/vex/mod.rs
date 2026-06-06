//! VEX Intermediate Representation types and operations.
//!
//! This module provides:
//! - VEX IR types (statements, expressions, operations)
//! - VEX operation implementations
//! - VEX lifting interface
//! - pyvex IRSB serialization/deserialization
//! - Clean call (CCall) implementations for flag calculations
//! - Native dirty helper implementations (CPUID, RDTSC, etc.)

pub mod ccall;
pub mod dirty;
pub mod ir;
mod lifter;
pub mod opcode_map;
pub mod ops;
pub mod pyvex_bridge;
pub mod transcendentals;

pub use dirty::{DirtyHelperDispatch, DirtyHelperResult};
pub use ir::*;
pub use lifter::{IRSBBuilder, LiftError, NativeVEXLifter, VEXLifter};
pub use opcode_map::{
    parse_endness, parse_jumpkind, parse_opcode, parse_opcode_from_u32, parse_type,
};
pub use ops::{OpError, VEXOps, iropclass};
pub use pyvex_bridge::{DeserializeError, deserialize_irsb};
