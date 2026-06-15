//! Native SimProcedure implementations for common libc functions.
//!
//! This module provides Rust-native implementations of common SimProcedures
//! (like strlen, memcpy, strcmp) to eliminate Python callback overhead.
//!
//! # Design
//!
//! - Each procedure implements the `NativeSimProcedure` trait
//! - Procedures operate directly on `RustSimState` memory/registers
//! - Symbolic arguments trigger fallback to Python
//! - `NativeProcedureRegistry` manages procedure lookup
//!
//! # Performance
//!
//! Native procedures provide significant speedup by:
//! - Eliminating Python callback overhead (~100μs per call)
//! - Direct memory access without serialization
//! - Keeping state entirely in Rust
//!
//! # Dispatch priority
//!
//! When PC reaches a hooked address that is registered as a SimProcedure
//! (via `register_simprocedure(addr, name, ...)`), the run-loop and the
//! stepping path decide between **native** and **Python** dispatch using
//! the following ordered checks. The first rule that matches wins; the
//! dispatcher does NOT chain native then Python in sequence except when
//! native execution returns a fallback error.
//!
//! 1. **User-placed in-binary hooks always go to Python.** If the
//!    hook PC falls inside a loaded binary region
//!    (`environment.binary_regions`), the native registry is skipped
//!    entirely and the Python SimProcedure fires. This matches the
//!    `proj.hook(addr, MyProc())` workflow: the user explicitly
//!    overrode a binary instruction, so we honor their Python
//!    implementation regardless of name. See `run_loop.rs` ~line 226
//!    and `stepping.rs` ~line 586 (`if !is_in_binary {`).
//!
//! 2. **Per-name Python override skips native.** If
//!    [`NativeProcedureRegistry::set_python_override`] has been called
//!    for `name`, [`NativeProcedureRegistry::get`] returns `None` and
//!    the dispatcher falls through to the Python SimProcedure path —
//!    even when the procedure is otherwise registered. Used internally
//!    by [`libc_start_main::NativeLibcStartMain`] to force the entry
//!    path through angr's Python machinery while still claiming the
//!    name in the registry.
//!
//! 3. **Per-name disable skips native.** If
//!    [`NativeProcedureRegistry::disable`] has been called for `name`,
//!    `get()` returns `None`. Same effect as a Python override but
//!    semantically labels the procedure as "not implemented in Rust
//!    yet" rather than "Python wins".
//!
//! 4. **Global disable skips native.** If
//!    [`NativeProcedureRegistry::disable_all`] has been called,
//!    `get()` returns `None` for every name. Used by differential
//!    tests that need to force the Python path end-to-end.
//!
//! 5. **Native runs; on error, Python takes over.** If a native
//!    implementation is registered and none of (1)–(4) bypass it, the
//!    dispatcher calls `native_proc.call(state, args)`. On `Ok` it
//!    advances PC to the return address and bumps `native_calls`. On
//!    `Err(ProcedureError::SymbolicArgument | NotImplemented | Other |
//!    MaxIterations | Memory)` it bumps `python_fallbacks` (bucketed
//!    by error variant in `*_fallbacks_by_name`) and emits a
//!    `need_simprocedure` event so Python can handle it. **The
//!    Python implementation is never invoked while a native
//!    implementation is being attempted — only on its failure.**
//!
//! 6. **No native implementation → Python.** If `name` is not in the
//!    `procedures` map, the dispatcher proceeds directly to the
//!    Python fallback path. The fallback is recorded under
//!    `simprocedure_python_fallback_count` and
//!    `simprocedure_fallback_by_name` but **not** under
//!    `native_proc_stats.python_fallbacks` (which only counts cases
//!    where native was tried and lost).
//!
//! Two SimProcedures are **never** chained or merged: at most one of
//! native-or-Python runs per dispatch, and the choice is taken once.

#[macro_use]
mod macros;

pub mod ctype;
pub mod exit;
pub mod fgets;
pub mod fileops;
pub mod format_common;
pub mod getenv;
pub mod libc_start_main;
pub mod malloc;
pub mod memcmp;
pub mod memcpy;
pub mod memset;
pub mod printf;
pub mod puts;
pub mod python_proc;
pub mod rand;
pub mod read;
pub mod scanf;
pub mod sprintf;
pub mod stdio;
pub mod strcat;
pub mod strchr;
pub mod strcmp;
pub mod strcpy;
pub mod strings;
pub mod strlen;
pub mod strset;
pub mod strstr;
pub mod strtod;
pub mod strtol;
pub mod write;

#[cfg(test)]
mod test_util;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::memory::MemoryError;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Per-prefix counters for minting unique symbolic-variable IDs across
/// native procedures. Lazily initialized; entries are created on first use.
static SYMBOL_COUNTERS: OnceLock<Mutex<HashMap<&'static str, u64>>> = OnceLock::new();

/// Returns a monotone-unique counter value for the given prefix.
///
/// Each prefix maintains its own counter starting at 0, so callers using
/// different prefixes get independent monotone sequences. Replaces the
/// per-procedure `static AtomicU64 *_COUNTER` pattern so all fresh-symbol
/// counters are coordinated in one place and easy to audit. Symbol-name
/// uniqueness still depends on the prefix the caller chooses.
pub fn symbol_counter(prefix: &'static str) -> u64 {
    let counters = SYMBOL_COUNTERS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = counters.lock().expect("symbol counter mutex poisoned");
    let entry = guard.entry(prefix).or_insert(0);
    let id = *entry;
    *entry += 1;
    id
}

/// Error during native procedure execution.
///
/// Errors trigger fallback to Python SimProcedure handling.
///
/// `#[non_exhaustive]` per angr-irwe: minor versions may add new
/// variants; intra-crate matches must include a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProcedureError {
    /// Argument is symbolic, need Python for constraint handling.
    #[error("symbolic argument: {0}")]
    SymbolicArgument(String),
    /// Memory operation failed. Carries the structured `MemoryError`
    /// so callers can pattern-match on the underlying cause
    /// (unmapped page, permission violation, symbolic address, ...).
    #[error("memory error: {0}")]
    Memory(#[from] MemoryError),
    /// Procedure not implemented in Rust.
    #[error("procedure not implemented")]
    NotImplemented,
    /// Max iteration limit reached (e.g., unbounded strlen).
    #[error("max iterations reached: {0}")]
    MaxIterations(usize),
    /// Generic error with message.
    #[error("{0}")]
    Other(String),
}

/// Extract a concrete u64 value from a procedure argument, or return SymbolicArgument error.
pub fn extract_concrete_arg(arg: &RustBV, name: &str) -> Result<u64, ProcedureError> {
    arg.as_u64()
        .ok_or_else(|| ProcedureError::SymbolicArgument(name.to_string()))
}

/// Trait for native SimProcedure implementations.
///
/// Implementors provide Rust-native execution of common library functions.
/// When execution fails (e.g., due to symbolic arguments), the procedure
/// returns an error and the exploration manager falls back to Python.
pub trait NativeSimProcedure: Send + Sync {
    /// Get the procedure name (e.g., "strlen", "memcpy").
    fn name(&self) -> &'static str;

    /// Get the number of arguments this procedure expects.
    fn num_args(&self) -> usize;

    /// Whether this procedure never returns (e.g., exit, abort).
    fn no_return(&self) -> bool {
        false
    }

    /// Execute the procedure.
    ///
    /// # Arguments
    /// * `state` - The simulation state to operate on
    /// * `args` - Extracted arguments (number matches `num_args()`)
    ///
    /// # Returns
    /// * `Ok(Some(value))` - Procedure completed, return value to store
    /// * `Ok(None)` - Procedure completed, no return value (void or no-return)
    /// * `Err(e)` - Procedure failed, fall back to Python
    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError>;
}

/// Registry of native SimProcedure implementations.
///
/// Maps procedure names to their native implementations.
/// Used by the exploration manager to dispatch procedures.
pub struct NativeProcedureRegistry {
    /// Name -> implementation mapping.
    procedures: HashMap<String, Arc<dyn NativeSimProcedure>>,
    /// Whether native procedures are globally enabled.
    enabled: bool,
    /// Procedures that are disabled (always fall back to Python).
    disabled: std::collections::HashSet<String>,
    /// Procedures with Python overrides.
    python_overrides: std::collections::HashSet<String>,
}

impl Default for NativeProcedureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeProcedureRegistry {
    /// Create a new registry with default procedures.
    pub fn new() -> Self {
        let mut registry = NativeProcedureRegistry {
            procedures: HashMap::new(),
            enabled: true,
            disabled: std::collections::HashSet::new(),
            python_overrides: std::collections::HashSet::new(),
        };

        // Register default native procedures (original set)
        registry.register(Arc::new(strlen::NativeStrlen));
        registry.register(Arc::new(memcpy::NativeMemcpy));
        registry.register(Arc::new(memcpy::NativeMemmove));
        registry.register(Arc::new(strcmp::NativeStrcmp));
        registry.register(Arc::new(strcmp::NativeStrncmp));
        registry.register(Arc::new(strcmp::NativeStrcasecmp));
        registry.register(Arc::new(puts::NativePuts));
        registry.register(Arc::new(printf::NativePrintf));
        // New procedures (verified safe — no exploration flow changes)
        registry.register(Arc::new(memset::NativeMemset));
        registry.register(Arc::new(strcpy::NativeStrcpy));
        registry.register(Arc::new(strcpy::NativeStrncpy));
        registry.register(Arc::new(strlen::NativeStrnlen));
        // exit/abort: terminal NO_RET procedures. The native dispatchers in
        // both stepping.rs (interpreter exit) and mod.rs (top-of-loop hook
        // check) recognize no_return and route the main state to STASH_DEADENDED
        // instead of advancing PC to the call's return address. Without that
        // check the state would re-execute past the call (which in fauxware
        // overlaps main's prologue, causing an infinite re-entry loop).
        registry.register(Arc::new(exit::NativeExit));
        registry.register(Arc::new(exit::NativeUnderscoreExit));
        registry.register(Arc::new(exit::NativeAbort));
        registry.register(Arc::new(exit::NativeStackChkFail));
        registry.register(Arc::new(rand::NativeRand));
        registry.register(Arc::new(rand::NativeSrand));
        // Heap procedures (bump allocator, matching SimHeapBrk)
        registry.register(Arc::new(malloc::NativeMalloc));
        registry.register(Arc::new(malloc::NativeFree));
        registry.register(Arc::new(malloc::NativeCalloc));
        registry.register(Arc::new(malloc::NativeRealloc));
        registry.register(Arc::new(malloc::NativeMemalign));
        registry.register(Arc::new(malloc::NativePosixMemalign));
        // Character classification (ctype.h)
        registry.register(Arc::new(ctype::NativeIsDigit));
        registry.register(Arc::new(ctype::NativeIsAlpha));
        registry.register(Arc::new(ctype::NativeIsSpace));
        registry.register(Arc::new(ctype::NativeIsAlnum));
        registry.register(Arc::new(ctype::NativeIsUpper));
        registry.register(Arc::new(ctype::NativeIsLower));
        registry.register(Arc::new(ctype::NativeIsXdigit));
        registry.register(Arc::new(ctype::NativeIsPrint));
        registry.register(Arc::new(ctype::NativeToLower));
        registry.register(Arc::new(ctype::NativeToUpper));
        // String/memory search
        registry.register(Arc::new(strchr::NativeStrchr));
        registry.register(Arc::new(strchr::NativeMemchr));
        registry.register(Arc::new(strchr::NativeStrrchr));
        // Byte-set search (angr-f16h.5)
        registry.register(Arc::new(strset::NativeStrpbrk));
        registry.register(Arc::new(strset::NativeStrspn));
        registry.register(Arc::new(strset::NativeStrcspn));
        // String-to-integer conversion
        registry.register(Arc::new(strtol::NativeStrtol));
        registry.register(Arc::new(strtol::NativeStrtoul));
        registry.register(Arc::new(strtol::NativeStrtoll));
        registry.register(Arc::new(strtol::NativeStrtoull));
        registry.register(Arc::new(strtol::NativeAtoi));
        registry.register(Arc::new(strtol::NativeAtol));
        // String-to-float conversion (amd64 only; xmm0 return register)
        registry.register(Arc::new(strtod::NativeStrtod));
        // String concatenation
        registry.register(Arc::new(strcat::NativeStrcat));
        registry.register(Arc::new(strcat::NativeStrncat));
        // Input procedures (stdin) — create symbolic bytes and track them
        // in state.stdin_symbols for posix.dumps(0) export.
        registry.register(Arc::new(fgets::NativeFgets));
        registry.register(Arc::new(fgets::NativeFgetc));
        registry.register(Arc::new(fgets::NativeGetchar));
        registry.register(Arc::new(fgets::NativeGetc));
        // Output procedures (stdout)
        registry.register(Arc::new(puts::NativePutchar));
        registry.register(Arc::new(puts::NativeFputc));
        registry.register(Arc::new(puts::NativePutc));
        // String duplication
        registry.register(Arc::new(strcpy::NativeStrdup));
        // Memory comparison
        registry.register(Arc::new(memcmp::NativeMemcmp));
        // String search
        registry.register(Arc::new(strstr::NativeStrstr));
        // Input parsing (scanf family)
        registry.register(Arc::new(scanf::NativeScanf));
        registry.register(Arc::new(scanf::NativeIsoc99Scanf));
        registry.register(Arc::new(scanf::NativeSscanf));
        // Environment variable access
        registry.register(Arc::new(getenv::NativeGetenv));
        registry.register(Arc::new(getenv::NativeSetenv));
        registry.register(Arc::new(getenv::NativePutenv));
        registry.register(Arc::new(getenv::NativeUnsetenv));
        registry.register(Arc::new(getenv::NativeClearenv));
        // String formatting (sprintf, snprintf)
        registry.register(Arc::new(sprintf::NativeSprintf));
        registry.register(Arc::new(sprintf::NativeSnprintf));
        // I/O procedures: re-enabled by angr-3tek.2. The Python-side cache
        // is now invalidate-and-replayed per dirty page in
        // `_create_state_for_callback` (rust_callback_dispatch.py +
        // rust_state_sync.py::_replay_rust_dirty_pages), so the stale-cache
        // issue described in angr-mme3 / angr-3tek no longer applies.
        registry.register(Arc::new(read::NativeRead));
        registry.register(Arc::new(write::NativeWrite));
        // stdio shims (angr-70no): fwrite resolves FILE._fileno → fd buffer;
        // fflush / setvbuf are no-ops returning 0 (match Python procs).
        registry.register(Arc::new(stdio::NativeFwrite));
        registry.register(Arc::new(stdio::NativeFflush));
        registry.register(Arc::new(stdio::NativeSetvbuf));
        // stdio status / write shims (angr-f16h.1): feof/ferror dispatch off
        // FILE._fileno and return concrete int flags; fputs reuses fwrite's
        // write_fd path with a NUL-terminated source string.
        registry.register(Arc::new(stdio::NativeFeof));
        registry.register(Arc::new(stdio::NativeFerror));
        registry.register(Arc::new(stdio::NativeFputs));
        // File operations: registered for fd tracking in FileSystem.
        registry.register(Arc::new(fileops::NativeOpen));
        registry.register(Arc::new(fileops::NativeClose));
        registry.register(Arc::new(fileops::NativeLseek));
        registry.register(Arc::new(fileops::NativeDup));
        registry.register(Arc::new(fileops::NativeDup2));
        registry.register(Arc::new(fileops::NativePipe));
        // stdio file ops (angr-karp): allocate _IO_FILE structs and dispatch
        // through the FILE._fileno field. fopen/fdopen heap-allocate, fclose/
        // fseek/ftell/rewind read fileno back out.
        registry.register(Arc::new(fileops::NativeFopen));
        registry.register(Arc::new(fileops::NativeFdopen));
        registry.register(Arc::new(fileops::NativeFclose));
        registry.register(Arc::new(fileops::NativeFseek));
        registry.register(Arc::new(fileops::NativeFtell));
        registry.register(Arc::new(fileops::NativeRewind));
        // __libc_start_main: handles the after_main continuation by
        // deadending. Python init (rust_manager._step_python_to_main) covers
        // the entry/run path before Rust takes over, so the only invocation
        // path during Rust exploration is after_main → exit(0). See
        // procedures/libc_start_main.rs for the safety argument.
        registry.register(Arc::new(libc_start_main::NativeLibcStartMain));

        registry
    }

    /// Create an empty registry (no default procedures).
    pub fn empty() -> Self {
        NativeProcedureRegistry {
            procedures: HashMap::new(),
            enabled: true,
            disabled: std::collections::HashSet::new(),
            python_overrides: std::collections::HashSet::new(),
        }
    }

    /// Register a native procedure.
    pub fn register(&mut self, proc: Arc<dyn NativeSimProcedure>) {
        self.procedures.insert(proc.name().to_string(), proc);
    }

    /// Get a procedure by name, honoring the dispatch-priority gate.
    ///
    /// Returns `Some(proc)` only when **all** of the following hold:
    ///
    /// 1. The registry is globally enabled (`enabled == true`).
    /// 2. The name has not been individually disabled via [`Self::disable`].
    /// 3. The name has no Python override set via
    ///    [`Self::set_python_override`].
    /// 4. A native implementation has been registered for the name.
    ///
    /// Returning `None` instructs the dispatcher to fall back to the
    /// Python SimProcedure path. See the module-level **Dispatch
    /// priority** section for the full chain (in particular the
    /// `is_in_binary` gate, which is checked by the dispatcher *before*
    /// calling this method, so user-placed hooks inside the binary
    /// never reach the registry at all).
    pub fn get(&self, name: &str) -> Option<&Arc<dyn NativeSimProcedure>> {
        // Check if globally disabled
        if !self.enabled {
            return None;
        }

        // Check if this specific procedure is disabled
        if self.disabled.contains(name) {
            return None;
        }

        // Check if Python override is set
        if self.python_overrides.contains(name) {
            return None;
        }

        self.procedures.get(name)
    }

    /// Check if a procedure has a native implementation.
    pub fn has_native(&self, name: &str) -> bool {
        self.procedures.contains_key(name)
    }

    /// Disable all native procedures (always use Python).
    pub fn disable_all(&mut self) {
        self.enabled = false;
    }

    /// Enable all native procedures.
    pub fn enable_all(&mut self) {
        self.enabled = true;
    }

    /// Check if native procedures are enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Disable a specific procedure (fall back to Python).
    pub fn disable(&mut self, name: &str) {
        self.disabled.insert(name.to_string());
    }

    /// Enable a specific procedure.
    pub fn enable(&mut self, name: &str) {
        self.disabled.remove(name);
    }

    /// Check if a specific procedure is disabled.
    pub fn is_disabled(&self, name: &str) -> bool {
        self.disabled.contains(name)
    }

    /// Set a Python override for a procedure.
    ///
    /// When a Python override is set, [`Self::get`] returns `None` for
    /// `name` so the dispatcher falls through to the Python
    /// SimProcedure path. The override is consulted on every dispatch
    /// — the native implementation is never called and the Python one
    /// always wins. Use [`Self::remove_python_override`] to restore
    /// the native path.
    ///
    /// Distinct from [`Self::disable`] only in intent: an override
    /// asserts "Python is canonical for this name", while disable
    /// asserts "the Rust implementation is not trustworthy right now".
    /// The runtime effect is identical (both bypass native).
    pub fn set_python_override(&mut self, name: &str) {
        self.python_overrides.insert(name.to_string());
    }

    /// Remove a Python override.
    pub fn remove_python_override(&mut self, name: &str) {
        self.python_overrides.remove(name);
    }

    /// Check if a procedure has a Python override.
    pub fn has_python_override(&self, name: &str) -> bool {
        self.python_overrides.contains(name)
    }

    /// Get all registered procedure names.
    pub fn procedure_names(&self) -> Vec<&str> {
        self.procedures.keys().map(|s| s.as_str()).collect()
    }

    /// Get statistics about the registry.
    pub fn stats(&self) -> RegistryStats {
        RegistryStats {
            total: self.procedures.len(),
            enabled: self.enabled,
            disabled_count: self.disabled.len(),
            override_count: self.python_overrides.len(),
        }
    }
}

/// Statistics about the procedure registry.
#[derive(Debug, Clone)]
pub struct RegistryStats {
    /// Total number of registered procedures.
    pub total: usize,
    /// Whether native procedures are globally enabled.
    pub enabled: bool,
    /// Number of individually disabled procedures.
    pub disabled_count: usize,
    /// Number of procedures with Python overrides.
    pub override_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = NativeProcedureRegistry::new();
        assert!(registry.is_enabled());
        assert!(registry.has_native("strlen"));
        assert!(registry.has_native("memcpy"));
        assert!(registry.has_native("strcmp"));
    }

    #[test]
    fn test_registry_disable_all() {
        let mut registry = NativeProcedureRegistry::new();
        registry.disable_all();
        assert!(!registry.is_enabled());
        assert!(registry.get("strlen").is_none());
    }

    #[test]
    fn test_registry_disable_specific() {
        let mut registry = NativeProcedureRegistry::new();
        registry.disable("strlen");
        assert!(registry.get("strlen").is_none());
        assert!(registry.get("memcpy").is_some());
    }

    #[test]
    fn test_python_override() {
        let mut registry = NativeProcedureRegistry::new();
        registry.set_python_override("memcpy");
        assert!(registry.has_python_override("memcpy"));
        assert!(registry.get("memcpy").is_none());
        assert!(registry.get("strlen").is_some());
    }
}
