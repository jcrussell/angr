//! VEX IR lifting interface.
//!
//! This module defines the [`VEXLifter`] trait and its [`LiftError`] taxonomy.
//! Concrete lifting is implemented elsewhere: `pyvex_bridge` (JSON callback
//! into pyvex) and `libvex_lifter` (`NativeLibVEXLifter`, direct libVEX FFI).
//! Both construct `IRSB`s via struct literals rather than through any builder
//! in this module.

use super::ir::{IRSB, VexArch};

/// Errors from VEX lifting.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LiftError {
    /// Invalid architecture.
    #[error("invalid architecture: {0}")]
    InvalidArch(String),
    /// Failed to lift the given bytes.
    #[error("lift failed at 0x{addr:x}: {reason}")]
    LiftFailed { addr: u64, reason: String },
}

/// VEX lifter trait.
///
/// This trait abstracts over different VEX lifting backends.
pub trait VEXLifter {
    /// Lift bytes at the given address to VEX IR.
    fn lift(&self, bytes: &[u8], addr: u64, arch: VexArch) -> Result<IRSB, LiftError>;
}
