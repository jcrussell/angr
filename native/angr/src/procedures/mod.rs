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

#[macro_use]
mod macros;

pub mod ctype;
pub mod exit;
pub mod fgets;
pub mod fileops;
pub mod getenv;
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
pub mod strlen;
pub mod strstr;
pub mod strtol;
pub mod write;

use std::collections::HashMap;
use std::sync::Arc;

use crate::memory::MemoryError;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Error during native procedure execution.
///
/// Errors trigger fallback to Python SimProcedure handling.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProcedureError {
    /// Argument is symbolic, need Python for constraint handling.
    #[error("symbolic argument: {0}")]
    SymbolicArgument(String),
    /// Memory operation failed.
    #[error("memory error: {0}")]
    MemoryError(String),
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

impl From<MemoryError> for ProcedureError {
    fn from(e: MemoryError) -> Self {
        ProcedureError::MemoryError(e.to_string())
    }
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
        registry.register(Arc::new(rand::NativeRand));
        registry.register(Arc::new(rand::NativeSrand));
        // Heap procedures (bump allocator, matching SimHeapBrk)
        registry.register(Arc::new(malloc::NativeMalloc));
        registry.register(Arc::new(malloc::NativeFree));
        registry.register(Arc::new(malloc::NativeCalloc));
        registry.register(Arc::new(malloc::NativeRealloc));
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
        // String-to-integer conversion
        registry.register(Arc::new(strtol::NativeStrtol));
        registry.register(Arc::new(strtol::NativeStrtoul));
        registry.register(Arc::new(strtol::NativeAtoi));
        registry.register(Arc::new(strtol::NativeAtol));
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
        // File operations: registered for fd tracking in FileSystem.
        registry.register(Arc::new(fileops::NativeOpen));
        registry.register(Arc::new(fileops::NativeClose));
        registry.register(Arc::new(fileops::NativeLseek));
        registry.register(Arc::new(fileops::NativeDup));
        registry.register(Arc::new(fileops::NativeDup2));
        registry.register(Arc::new(fileops::NativePipe));

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

    /// Get a procedure by name.
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
    /// When a Python override is set, the native implementation is
    /// never called - always falls back to Python.
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
