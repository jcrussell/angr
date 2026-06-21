//! Python-facing opaque handle for RustBV values.
//!
//! RustBVHandle provides a way for Python to reference Rust symbolic values
//! without converting to/from claripy ASTs. This bypasses the expensive
//! claripy_to_rustbv() conversion on the hot path.
//!
//! Key benefits:
//! - No Python object allocation for intermediate expressions
//! - No AST traversal for constraint conversion
//! - Direct Rust-Z3 path without round-trips

use pyo3::prelude::*;

/// Opaque handle to a RustBV value stored in the symbol table.
///
/// Python code can use these handles to perform symbolic operations
/// entirely in Rust, only extracting concrete values when needed.
///
/// Example usage from Python:
/// ```python
/// handle_a = solver.BVS("x", 32)   # Returns RustBVHandle
/// handle_b = solver.BVV(42, 32)    # Returns RustBVHandle
/// handle_c = handle_a + handle_b   # Operation happens in Rust
/// result = solver.eval(handle_c)   # Extract concrete value
/// ```
#[pyclass]
#[derive(Clone)]
pub struct RustBVHandle {
    /// Unique identifier for this value in the symbol table.
    id: u64,
    /// Bit width of the value.
    width: u32,
    /// Cached concrete value, if this handle is fully constrained.
    concrete: Option<u128>,
}

// Rust-only constructors and mutators (not exposed to Python).
impl RustBVHandle {
    /// Create a new handle for a symbolic value.
    pub fn new_symbolic(id: u64, width: u32) -> Self {
        RustBVHandle {
            id,
            width,
            concrete: None,
        }
    }

    /// Create a new handle for a concrete value.
    pub fn new_concrete(id: u64, value: u128, width: u32) -> Self {
        RustBVHandle {
            id,
            width,
            concrete: Some(value),
        }
    }
}

// Accessors and Python dunders — single source of truth for both Rust and
// Python callers. `#[getter]` exposes these as Python properties; Rust code
// can still call them as plain methods.
#[pymethods]
impl RustBVHandle {
    /// Unique identifier of this handle within the symbol table.
    #[getter]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Bit width of the value.
    #[getter]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Whether the handle carries a cached concrete value.
    #[getter]
    pub fn is_concrete(&self) -> bool {
        self.concrete.is_some()
    }

    /// Cached concrete value if available (returns None for symbolic).
    pub fn concrete(&self) -> Option<u128> {
        self.concrete
    }

    /// String representation for debugging.
    pub fn __repr__(&self) -> String {
        if let Some(v) = self.concrete {
            format!(
                "<RustBVHandle id={} width={} concrete=0x{:x}>",
                self.id, self.width, v
            )
        } else {
            format!(
                "<RustBVHandle id={} width={} symbolic>",
                self.id, self.width
            )
        }
    }

    /// Hash based on ID for use in Python dicts/sets.
    pub fn __hash__(&self) -> u64 {
        self.id
    }

    /// Equality check based on ID.
    pub fn __eq__(&self, other: &RustBVHandle) -> bool {
        self.id == other.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_creation() {
        let h = RustBVHandle::new_symbolic(1, 32);
        assert_eq!(h.id(), 1);
        assert_eq!(h.width(), 32);
        assert!(!h.is_concrete());
    }

    #[test]
    fn test_concrete_handle() {
        let h = RustBVHandle::new_concrete(2, 42, 32);
        assert_eq!(h.id(), 2);
        assert_eq!(h.width(), 32);
        assert!(h.is_concrete());
        assert_eq!(h.concrete(), Some(42));
    }
}
