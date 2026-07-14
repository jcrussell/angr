//! Execution environment for `RustExplorationManager`.
//!
//! Groups manager-owned per-binary fields the manager hands to each
//! interpreter on every step:
//!
//! * `arch_name` — architecture name string (used for arch_from_name lookups).
//! * `os_name` — OS / SimOS name (lowercase: "linux" default, "cgc" for
//!   DECREE binaries). Drives syscall-table dispatch when the syscall
//!   ABI is OS-specific rather than arch-specific (CGC uses x86 syscall
//!   numbers 1-7 that collide with Linux i386 numbers). Set from Python
//!   via [`crate::exploration::RustExplorationManager::set_os_name`].
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
use crate::interpreter::BLOCK_CACHE_CAPACITY;
use crate::vex::{IRSB, VexArch};

pub(crate) struct ExecutionEnvironment {
    pub(crate) arch_name: String,
    pub(crate) os_name: String,
    pub(crate) vex_arch: VexArch,
    pub(crate) binary_regions: Vec<(u64, Arc<Vec<u8>>)>,
    pub(crate) block_cache: LruCache<u64, Arc<IRSB>>,
    pub(crate) calling_convention: Box<dyn CallingConvention>,
    pub(crate) little_endian: Option<bool>,
    pub(crate) max_history: usize,
    /// `[start, end)` of the main object's code regions, when known. Hooks in
    /// this range are user `proj.hook()` territory and always defer to Python.
    pub(crate) main_object_range: Option<(u64, u64)>,
    /// Opt-in (angr-a8epx / angr-gorvf.3.2): also prefer the native registry
    /// for hooks that land inside a NON-main loaded object.
    pub(crate) prefer_native_library_hooks: bool,
}

/// Native-dispatch gate for a hooked PC.
///
/// Default: native fires only for hooks OUTSIDE every loaded object — i.e. the
/// extern-object PLT stubs angr synthesizes for a statically-known symbol.
/// Hooks that land inside a loaded object's code go to Python, so a user
/// `proj.hook()` override always wins.
///
/// With `prefer_library_hooks` on, a hook inside a **non-main** loaded object
/// (libc &c., installed by `use_sim_procedures=True` on a dynamically-linked
/// binary) prefers the native registry instead of bouncing to Python. Main
/// object hooks still go to Python. See `docs/advanced-topics/rust_engine.rst`
/// for the residual symbolic-proc caveat behind the opt-in.
pub(crate) fn prefer_native_dispatch(
    binary_regions: &[(u64, Arc<Vec<u8>>)],
    main_object_range: Option<(u64, u64)>,
    prefer_library_hooks: bool,
    addr: u64,
) -> bool {
    let in_binary = binary_regions
        .iter()
        .any(|(base, data)| addr >= *base && addr < *base + data.len() as u64);
    if !in_binary {
        return true;
    }
    let in_main = main_object_range.is_some_and(|(start, end)| addr >= start && addr < end);
    prefer_library_hooks && !in_main
}

impl ExecutionEnvironment {
    /// See [`prefer_native_dispatch`].
    pub(crate) fn prefer_native_dispatch(&self, addr: u64) -> bool {
        prefer_native_dispatch(
            &self.binary_regions,
            self.main_object_range,
            self.prefer_native_library_hooks,
            addr,
        )
    }
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
            os_name: "linux".to_string(),
            vex_arch,
            binary_regions: Vec::new(),
            block_cache: LruCache::new(
                NonZeroUsize::new(BLOCK_CACHE_CAPACITY).expect("BLOCK_CACHE_CAPACITY is non-zero"),
            ),
            calling_convention,
            little_endian,
            max_history: 1000,
            main_object_range: None,
            prefer_native_library_hooks: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Main object at [0x400000, 0x401000), a loaded library at [0x500000,
    /// 0x501000); 0x600000 is outside every region (extern-object stub).
    fn regions() -> Vec<(u64, Arc<Vec<u8>>)> {
        vec![
            (0x400000, Arc::new(vec![0u8; 0x1000])),
            (0x500000, Arc::new(vec![0u8; 0x1000])),
        ]
    }

    const MAIN: Option<(u64, u64)> = Some((0x400000, 0x401000));

    #[test]
    fn extern_stubs_always_prefer_native() {
        for prefer_lib in [false, true] {
            assert!(prefer_native_dispatch(
                &regions(),
                MAIN,
                prefer_lib,
                0x600000
            ));
        }
    }

    #[test]
    fn main_object_hooks_never_prefer_native() {
        for prefer_lib in [false, true] {
            assert!(!prefer_native_dispatch(
                &regions(),
                MAIN,
                prefer_lib,
                0x400100
            ));
        }
    }

    #[test]
    fn library_hooks_follow_the_gate() {
        assert!(!prefer_native_dispatch(&regions(), MAIN, false, 0x500100));
        assert!(prefer_native_dispatch(&regions(), MAIN, true, 0x500100));
    }

    /// Without a main-object range every in-binary hook is treated as library
    /// code, so the gate alone decides — matching pre-flag behavior when off.
    #[test]
    fn unknown_main_range_falls_back_to_the_gate() {
        assert!(!prefer_native_dispatch(&regions(), None, false, 0x400100));
        assert!(prefer_native_dispatch(&regions(), None, true, 0x400100));
    }
}
