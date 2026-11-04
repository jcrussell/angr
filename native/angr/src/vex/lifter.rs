//! VEX IR lifting interface.
//!
//! This module defines the [`VEXLifter`] trait and its [`LiftError`] taxonomy.
//! Concrete lifting is implemented in the PyO3 shell: `libvex_lifter`
//! (`NativeLibVEXLifter`, direct libVEX FFI, the one [`VEXLifter`] impl), and
//! `pyvex_reader`, which reads the IRSB the Python `lift_block` callback
//! returns. Both construct `IRSB`s via struct literals rather than through any
//! builder in this module. The interpreter holds the lifter as
//! `Arc<dyn VEXLifter>`, so the core never names the FFI lifter
//! (angr-mx00p.51.7).

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
/// This trait abstracts over different VEX lifting backends. `Send + Sync`
/// because one lifter is shared by every worker's interpreter; an impl
/// serializes whatever non-reentrant backend it wraps itself.
pub trait VEXLifter: Send + Sync + std::fmt::Debug {
    /// Lift bytes at the given address to VEX IR at `opt_level` (`None`
    /// takes the lifter's default).
    fn lift_at(
        &self,
        bytes: &[u8],
        addr: u64,
        arch: VexArch,
        opt_level: Option<i32>,
    ) -> Result<IRSB, LiftError>;

    /// Lift bytes at the given address to VEX IR at the default opt level.
    fn lift(&self, bytes: &[u8], addr: u64, arch: VexArch) -> Result<IRSB, LiftError> {
        self.lift_at(bytes, addr, arch, None)
    }
}
