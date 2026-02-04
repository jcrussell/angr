//! Python callback infrastructure for Rust VEX engine.
//!
//! This module provides the callback holder that allows Rust to call back into Python
//! for operations like memory access, hook execution, and syscall handling.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

use crate::symbolic::{RustBV, SymContext};

/// Result of a memory load callback.
#[derive(Debug, Clone)]
pub struct MemoryLoadResult<'ctx> {
    /// The concrete bytes loaded.
    pub data: Vec<u8>,
    /// Whether the value is symbolic (has an associated AST).
    pub is_symbolic: bool,
    /// The symbolic AST (if symbolic). This is a Python object reference.
    pub symbolic_ast: Option<PyObject>,
    /// The RustBV representation for the engine.
    pub value: RustBV<'ctx>,
}

/// Result of running the execution loop.
#[derive(Debug, Clone)]
pub enum RunResult {
    /// Reached max blocks limit - continue later.
    MaxBlocks { pc: u64 },
    /// Hit a hook address - need Python to handle.
    Hook { addr: u64 },
    /// Syscall encountered - need Python to handle.
    Syscall { num: u64, pc: u64 },
    /// Symbolic branch - need Python to fork states.
    SymbolicBranch {
        condition_id: u64,
        true_target: u64,
        false_target: u64,
    },
    /// Normal block end.
    BlockEnd { next_addr: u64, jumpkind: String },
    /// Error during execution.
    Error { message: String, addr: u64 },
    /// Need to lift a block at the given address.
    NeedLift { addr: u64 },
}

/// Python callback holder for the Rust VEX engine.
///
/// This struct holds references to Python callback functions that the Rust
/// engine calls during execution for memory access, hooks, syscalls, etc.
#[pyclass]
#[derive(Clone)]
pub struct PythonCallbacks {
    /// Callback for memory loads: fn(addr: u64, size: u32) -> (bytes, is_symbolic, symbolic_ast?)
    pub memory_load: Option<PyObject>,
    /// Callback for memory stores: fn(addr: u64, data: bytes) -> None
    pub memory_store: Option<PyObject>,
    /// Callback for hook execution: fn(addr: u64) -> new_pc
    pub on_hook: Option<PyObject>,
    /// Callback for syscall handling: fn(num: u64) -> None
    pub on_syscall: Option<PyObject>,
    /// Callback for lifting a block: fn(addr: u64) -> irsb_json
    pub lift_block: Option<PyObject>,
    /// Callback for getting register value: fn(offset: u32, size: u32) -> (bytes, is_symbolic, symbolic_ast?)
    pub get_register: Option<PyObject>,
    /// Callback for setting register value: fn(offset: u32, data: bytes) -> None
    pub put_register: Option<PyObject>,
}

#[pymethods]
impl PythonCallbacks {
    /// Create a new empty callback holder.
    #[new]
    pub fn new() -> Self {
        PythonCallbacks {
            memory_load: None,
            memory_store: None,
            on_hook: None,
            on_syscall: None,
            lift_block: None,
            get_register: None,
            put_register: None,
        }
    }

    /// Set the memory load callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int, size: int) -> tuple[bytes, bool, object | None]`
    ///
    /// Returns (concrete_bytes, is_symbolic, symbolic_ast_or_none).
    pub fn set_memory_load(&mut self, cb: PyObject) {
        self.memory_load = Some(cb);
    }

    /// Set the memory store callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int, data: bytes) -> None`
    pub fn set_memory_store(&mut self, cb: PyObject) {
        self.memory_store = Some(cb);
    }

    /// Set the hook execution callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int) -> int`
    ///
    /// Returns the new PC after hook execution.
    pub fn set_on_hook(&mut self, cb: PyObject) {
        self.on_hook = Some(cb);
    }

    /// Set the syscall handling callback.
    ///
    /// The callback should have signature:
    /// `fn(num: int) -> None`
    pub fn set_on_syscall(&mut self, cb: PyObject) {
        self.on_syscall = Some(cb);
    }

    /// Set the block lifting callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int) -> str`
    ///
    /// Returns the IRSB as a JSON string.
    pub fn set_lift_block(&mut self, cb: PyObject) {
        self.lift_block = Some(cb);
    }

    /// Set the register get callback.
    ///
    /// The callback should have signature:
    /// `fn(offset: int, size: int) -> tuple[bytes, bool, object | None]`
    ///
    /// Returns (concrete_bytes, is_symbolic, symbolic_ast_or_none).
    pub fn set_get_register(&mut self, cb: PyObject) {
        self.get_register = Some(cb);
    }

    /// Set the register put callback.
    ///
    /// The callback should have signature:
    /// `fn(offset: int, data: bytes) -> None`
    pub fn set_put_register(&mut self, cb: PyObject) {
        self.put_register = Some(cb);
    }

    /// Check if all required callbacks are set.
    pub fn is_ready(&self) -> bool {
        self.memory_load.is_some()
            && self.memory_store.is_some()
            && self.lift_block.is_some()
    }
}

impl Default for PythonCallbacks {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonCallbacks {
    /// Call the memory load callback.
    ///
    /// Returns (data_bytes, is_symbolic, symbolic_ast).
    pub fn call_memory_load(
        &self,
        py: Python<'_>,
        addr: u64,
        size: u32,
    ) -> PyResult<(Vec<u8>, bool, Option<PyObject>)> {
        let cb = self.memory_load.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("memory_load callback not set")
        })?;

        let result = cb.call1(py, (addr, size))?;
        let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;

        // Extract (bytes, is_symbolic, symbolic_ast?)
        let data_obj = tuple.get_item(0)?;
        let data: Vec<u8> = data_obj.extract()?;
        let is_symbolic: bool = tuple.get_item(1)?.extract()?;

        let symbolic_ast = if tuple.len() > 2 {
            let ast_obj = tuple.get_item(2)?;
            if ast_obj.is_none() {
                None
            } else {
                Some(ast_obj.unbind())
            }
        } else {
            None
        };

        Ok((data, is_symbolic, symbolic_ast))
    }

    /// Call the memory store callback.
    pub fn call_memory_store(&self, py: Python<'_>, addr: u64, data: &[u8]) -> PyResult<()> {
        let cb = self.memory_store.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("memory_store callback not set")
        })?;

        let py_bytes = PyBytes::new(py, data);
        cb.call1(py, (addr, py_bytes))?;
        Ok(())
    }

    /// Call the hook execution callback.
    ///
    /// Returns the new PC after hook execution.
    pub fn call_on_hook(&self, py: Python<'_>, addr: u64) -> PyResult<u64> {
        let cb = self.on_hook.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("on_hook callback not set")
        })?;

        let result = cb.call1(py, (addr,))?;
        result.extract(py)
    }

    /// Call the syscall handling callback.
    pub fn call_on_syscall(&self, py: Python<'_>, num: u64) -> PyResult<()> {
        let cb = self.on_syscall.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("on_syscall callback not set")
        })?;

        cb.call1(py, (num,))?;
        Ok(())
    }

    /// Call the block lifting callback.
    ///
    /// Returns the IRSB as a JSON string.
    pub fn call_lift_block(&self, py: Python<'_>, addr: u64) -> PyResult<String> {
        let cb = self.lift_block.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("lift_block callback not set")
        })?;

        let result = cb.call1(py, (addr,))?;
        result.extract(py)
    }

    /// Call the register get callback.
    ///
    /// Returns (data_bytes, is_symbolic, symbolic_ast).
    pub fn call_get_register(
        &self,
        py: Python<'_>,
        offset: u32,
        size: u32,
    ) -> PyResult<(Vec<u8>, bool, Option<PyObject>)> {
        let cb = self.get_register.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("get_register callback not set")
        })?;

        let result = cb.call1(py, (offset, size))?;
        let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;

        let data: Vec<u8> = tuple.get_item(0)?.extract()?;
        let is_symbolic: bool = tuple.get_item(1)?.extract()?;

        let symbolic_ast = if tuple.len() > 2 {
            let ast_obj = tuple.get_item(2)?;
            if ast_obj.is_none() {
                None
            } else {
                Some(ast_obj.unbind())
            }
        } else {
            None
        };

        Ok((data, is_symbolic, symbolic_ast))
    }

    /// Call the register put callback.
    pub fn call_put_register(&self, py: Python<'_>, offset: u32, data: &[u8]) -> PyResult<()> {
        let cb = self.put_register.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("put_register callback not set")
        })?;

        let py_bytes = PyBytes::new(py, data);
        cb.call1(py, (offset, py_bytes))?;
        Ok(())
    }
}

/// Execution event returned to Python from run_loop.
#[pyclass]
#[derive(Debug, Clone)]
pub struct LoopExecutionEvent {
    /// Type of event: "max_blocks", "hook", "syscall", "symbolic_branch", "block_end", "error", "need_lift"
    #[pyo3(get)]
    pub event_type: String,
    /// Current/next PC address.
    #[pyo3(get)]
    pub pc: Option<u64>,
    /// Hook/target address.
    #[pyo3(get)]
    pub addr: Option<u64>,
    /// Syscall number.
    #[pyo3(get)]
    pub syscall_num: Option<u64>,
    /// True target for symbolic branch.
    #[pyo3(get)]
    pub true_target: Option<u64>,
    /// False target for symbolic branch.
    #[pyo3(get)]
    pub false_target: Option<u64>,
    /// Jump kind string.
    #[pyo3(get)]
    pub jumpkind: Option<String>,
    /// Error message.
    #[pyo3(get)]
    pub error: Option<String>,
    /// Number of blocks executed this loop.
    #[pyo3(get)]
    pub blocks_executed: u32,
}

impl LoopExecutionEvent {
    pub fn from_run_result(result: RunResult, blocks_executed: u32) -> Self {
        match result {
            RunResult::MaxBlocks { pc } => LoopExecutionEvent {
                event_type: "max_blocks".to_string(),
                pc: Some(pc),
                addr: None,
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: None,
                blocks_executed,
            },
            RunResult::Hook { addr } => LoopExecutionEvent {
                event_type: "hook".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: None,
                blocks_executed,
            },
            RunResult::Syscall { num, pc } => LoopExecutionEvent {
                event_type: "syscall".to_string(),
                pc: Some(pc),
                addr: None,
                syscall_num: Some(num),
                true_target: None,
                false_target: None,
                jumpkind: Some("Ijk_Sys_syscall".to_string()),
                error: None,
                blocks_executed,
            },
            RunResult::SymbolicBranch {
                true_target,
                false_target,
                ..
            } => LoopExecutionEvent {
                event_type: "symbolic_branch".to_string(),
                pc: None,
                addr: None,
                syscall_num: None,
                true_target: Some(true_target),
                false_target: Some(false_target),
                jumpkind: None,
                error: None,
                blocks_executed,
            },
            RunResult::BlockEnd { next_addr, jumpkind } => LoopExecutionEvent {
                event_type: "block_end".to_string(),
                pc: Some(next_addr),
                addr: Some(next_addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: Some(jumpkind),
                error: None,
                blocks_executed,
            },
            RunResult::Error { message, addr } => LoopExecutionEvent {
                event_type: "error".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: Some(message),
                blocks_executed,
            },
            RunResult::NeedLift { addr } => LoopExecutionEvent {
                event_type: "need_lift".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: None,
                blocks_executed,
            },
        }
    }
}

/// Thread-safe wrapper for Python callbacks.
///
/// This allows the callbacks to be shared across interpreter instances
/// during a single execution loop.
pub struct CallbacksRef {
    inner: Arc<PythonCallbacks>,
}

impl CallbacksRef {
    pub fn new(callbacks: PythonCallbacks) -> Self {
        CallbacksRef {
            inner: Arc::new(callbacks),
        }
    }

    pub fn get(&self) -> &PythonCallbacks {
        &self.inner
    }
}

impl Clone for CallbacksRef {
    fn clone(&self) -> Self {
        CallbacksRef {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callbacks_creation() {
        pyo3::prepare_freethreaded_python();
        let callbacks = PythonCallbacks::new();
        assert!(!callbacks.is_ready());
    }

    #[test]
    fn test_loop_execution_event() {
        let event = LoopExecutionEvent::from_run_result(
            RunResult::BlockEnd {
                next_addr: 0x1000,
                jumpkind: "Ijk_Boring".to_string(),
            },
            5,
        );
        assert_eq!(event.event_type, "block_end");
        assert_eq!(event.pc, Some(0x1000));
        assert_eq!(event.blocks_executed, 5);
    }
}
