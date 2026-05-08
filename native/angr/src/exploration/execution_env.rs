//! Execution environment for `RustExplorationManager`.
//!
//! Groups manager-owned per-binary fields the manager hands to each
//! interpreter on every step:
//!
//! * `arch_name` — architecture name string (used for arch_from_name lookups).
//! * `vex_arch` — VEX architecture enum (passed to the interpreter).
//! * `binary_regions` — code regions for native lifting (Arc-shared).
//! * `block_cache` — IRSB cache shared with each interpreter via swap.
//! * `calling_convention` — argument extraction / return-register protocol.
//! * `little_endian` — endianness override (None = arch default).
//! * `max_history` — per-state `history` / `detailed_history` cap.
//!
//! Same `pub(crate)` direct-field pattern as `ProfilingCollector`,
//! `ConstraintSolver`, and `MemoryConfiguration`: callsites read/write
//! the inner fields through a thin delegation. Do NOT add helper methods
//! as a separate cleanup; the parent angr-4j5u was deferred multiple
//! times for cosmetic gains.

use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;

use crate::arch::CallingConvention;
use crate::vex::{VexArch, IRSB};

pub(crate) struct ExecutionEnvironment {
    pub(crate) arch_name: String,
    pub(crate) vex_arch: VexArch,
    pub(crate) binary_regions: Vec<(u64, Arc<Vec<u8>>)>,
    pub(crate) block_cache: LruCache<u64, Arc<IRSB>>,
    pub(crate) calling_convention: Box<dyn CallingConvention>,
    pub(crate) little_endian: Option<bool>,
    pub(crate) max_history: usize,
}

impl ExecutionEnvironment {
    pub(crate) fn new(
        arch_name: String,
        vex_arch: VexArch,
        calling_convention: Box<dyn CallingConvention>,
        little_endian: Option<bool>,
    ) -> Self {
        Self {
            arch_name,
            vex_arch,
            binary_regions: Vec::new(),
            block_cache: LruCache::new(NonZeroUsize::new(4096).expect("nonzero literal")),
            calling_convention,
            little_endian,
            max_history: 1000,
        }
    }
}
