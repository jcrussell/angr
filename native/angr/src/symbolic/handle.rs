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

    /// Get the handle ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get the bit width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the cached concrete value if available.
    pub fn concrete_value(&self) -> Option<u128> {
        self.concrete
    }

    /// Check if this handle has a cached concrete value.
    pub fn is_concrete(&self) -> bool {
        self.concrete.is_some()
    }

    /// Set the concrete value (used when a symbolic value becomes constrained).
    pub fn set_concrete(&mut self, value: u128) {
        self.concrete = Some(value);
    }
}

#[pymethods]
impl RustBVHandle {
    /// Get the handle ID (for debugging).
    #[getter]
    pub fn get_id(&self) -> u64 {
        self.id
    }

    /// Get the bit width.
    #[getter]
    pub fn get_width(&self) -> u32 {
        self.width
    }

    /// Get the length in bits (alias for width, for claripy compatibility).
    #[getter]
    pub fn length(&self) -> u32 {
        self.width
    }

    /// Check if this handle represents a concrete value.
    #[getter]
    pub fn get_is_concrete(&self) -> bool {
        self.is_concrete()
    }

    /// Check if this handle represents a symbolic value.
    #[getter]
    pub fn symbolic(&self) -> bool {
        !self.is_concrete()
    }

    /// Get the concrete value if available (returns None for symbolic).
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

    // =========================================================================
    // Arithmetic Operations (return new handles via solver context)
    // These are implemented as instance methods that delegate to the solver.
    // =========================================================================

    /// Addition: self + other
    /// Note: Actual implementation is in solver.rs via op_add().
    /// This is a placeholder that will raise an error if called directly.
    #[pyo3(name = "__add__")]
    pub fn py_add(&self, _other: &RustBVHandle) -> PyResult<RustBVHandle> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "RustBVHandle arithmetic must be performed via solver context",
        ))
    }

    /// Subtraction: self - other
    #[pyo3(name = "__sub__")]
    pub fn py_sub(&self, _other: &RustBVHandle) -> PyResult<RustBVHandle> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "RustBVHandle arithmetic must be performed via solver context",
        ))
    }

    /// Multiplication: self * other
    #[pyo3(name = "__mul__")]
    pub fn py_mul(&self, _other: &RustBVHandle) -> PyResult<RustBVHandle> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "RustBVHandle arithmetic must be performed via solver context",
        ))
    }

    /// Bitwise AND: self & other
    #[pyo3(name = "__and__")]
    pub fn py_and(&self, _other: &RustBVHandle) -> PyResult<RustBVHandle> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "RustBVHandle arithmetic must be performed via solver context",
        ))
    }

    /// Bitwise OR: self | other
    #[pyo3(name = "__or__")]
    pub fn py_or(&self, _other: &RustBVHandle) -> PyResult<RustBVHandle> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "RustBVHandle arithmetic must be performed via solver context",
        ))
    }

    /// Bitwise XOR: self ^ other
    #[pyo3(name = "__xor__")]
    pub fn py_xor(&self, _other: &RustBVHandle) -> PyResult<RustBVHandle> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "RustBVHandle arithmetic must be performed via solver context",
        ))
    }

    /// Bitwise NOT: ~self
    #[pyo3(name = "__invert__")]
    pub fn py_invert(&self) -> PyResult<RustBVHandle> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "RustBVHandle arithmetic must be performed via solver context",
        ))
    }

    /// Left shift: self << other
    #[pyo3(name = "__lshift__")]
    pub fn py_lshift(&self, _other: &RustBVHandle) -> PyResult<RustBVHandle> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "RustBVHandle arithmetic must be performed via solver context",
        ))
    }

    /// Right shift: self >> other
    #[pyo3(name = "__rshift__")]
    pub fn py_rshift(&self, _other: &RustBVHandle) -> PyResult<RustBVHandle> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "RustBVHandle arithmetic must be performed via solver context",
        ))
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
        assert_eq!(h.concrete_value(), Some(42));
    }
}
