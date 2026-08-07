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
#[cfg(feature = "libvex-ffi")]
pub mod libvex_ffi;
#[cfg(feature = "libvex-ffi")]
pub mod libvex_lifter;
mod lifter;
pub mod opcode_map;
pub mod ops;
pub mod pyvex_bridge;
pub mod transcendentals;

/// pyvex `_lift`'s `max_bytes` default (`pyvex/lifting/libvex.py`): the largest
/// byte window any single-block lift consumes.
///
/// Shared so every lift path agrees on one budget. libVEX stops at the block
/// boundary, so trailing bytes beyond the block are simply unused — reading up
/// to this many is always safe, and reading *fewer* is what goes wrong: the SMC
/// dirty-bytes probe in `VEXInterpreter::get_or_lift_block` used to cap at one
/// page, so a block needing fresh post-store bytes past offset 4096 would have
/// been lifted from stale image bytes (angr-sqfj8.67).
pub(crate) const VEX_MAX_BYTES: usize = 5000;

pub use dirty::{DirtyHelperDispatch, DirtyHelperResult, DirtyHelperState};
pub use ir::*;
pub use lifter::{LiftError, VEXLifter};
pub use opcode_map::{parse_endness, parse_jumpkind, parse_opcode, parse_type};
pub use ops::{OpError, VEXOps, iropclass};
pub use pyvex_bridge::{DeserializeError, deserialize_irsb};
