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

pub mod strlen;
pub mod memcpy;
pub mod strcmp;

use std::collections::HashMap;
use std::sync::Arc;

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::memory::MemoryError;

/// Error during native procedure execution.
///
/// Errors trigger fallback to Python SimProcedure handling.
#[derive(Debug, Clone)]
pub enum ProcedureError {
    /// Argument is symbolic, need Python for constraint handling.
    SymbolicArgument(String),
    /// Memory operation failed.
    MemoryError(String),
    /// Procedure not implemented in Rust.
    NotImplemented,
    /// Max iteration limit reached (e.g., unbounded strlen).
    MaxIterations(usize),
    /// Generic error with message.
    Other(String),
}

impl From<MemoryError> for ProcedureError {
    fn from(e: MemoryError) -> Self {
        ProcedureError::MemoryError(e.to_string())
    }
}

impl std::fmt::Display for ProcedureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcedureError::SymbolicArgument(name) => {
                write!(f, "symbolic argument: {}", name)
            }
            ProcedureError::MemoryError(msg) => {
                write!(f, "memory error: {}", msg)
            }
            ProcedureError::NotImplemented => {
                write!(f, "procedure not implemented")
            }
            ProcedureError::MaxIterations(n) => {
                write!(f, "max iterations reached: {}", n)
            }
            ProcedureError::Other(msg) => {
                write!(f, "{}", msg)
            }
        }
    }
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

        // Register default native procedures
        registry.register(Arc::new(strlen::NativeStrlen));
        registry.register(Arc::new(memcpy::NativeMemcpy));
        registry.register(Arc::new(memcpy::NativeMemmove));
        registry.register(Arc::new(strcmp::NativeStrcmp));
        registry.register(Arc::new(strcmp::NativeStrncmp));
        registry.register(Arc::new(strcmp::NativeStrcasecmp));

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
