//! The `ExplorationStateSnapshot` pyclass: Python's view of a
//! [`StateExport`].
//!
//! Each pymethod forwards to the [`StateExport`] accessor of the same name;
//! which ones the Python bridge depends on is recorded on that accessor
//! block in `state/export.rs` (angr-hv4lt.11).

use pyo3::prelude::*;

use crate::state::StateExport;

/// Complete state snapshot for exploration export, as Python sees it.
///
/// A [`StateExport`] (`RustSimState::flush_and_export_full`) wrapped for
/// Python; everything an angr `SimState` is rebuilt from.
#[pyclass(name = "ExplorationStateSnapshot")]
pub struct ExplorationStateSnapshot {
    inner: StateExport,
}

impl From<StateExport> for ExplorationStateSnapshot {
    fn from(inner: StateExport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl ExplorationStateSnapshot {
    /// Unique state identifier.
    #[getter]
    pub fn state_id(&self) -> u64 {
        self.inner.state_id
    }

    /// Parent state ID (for fork tracking).
    #[getter]
    pub fn parent_id(&self) -> Option<u64> {
        self.inner.parent_id
    }

    /// Program counter.
    #[getter]
    pub fn pc(&self) -> u64 {
        self.inner.pc
    }

    /// Architecture name.
    #[getter]
    pub fn arch_name(&self) -> String {
        self.inner.arch_name.clone()
    }

    /// Number of constraints in the solver.
    #[getter]
    pub fn constraint_count(&self) -> usize {
        self.inner.constraint_count
    }

    /// Inspection enabled bitmask.
    #[getter]
    pub fn inspection_enabled(&self) -> u8 {
        self.inner.inspection_enabled
    }

    /// See [`StateExport::get_registers_raw`].
    pub fn get_registers_raw(&self) -> Vec<u8> {
        self.inner.get_registers_raw()
    }

    /// See [`StateExport::get_registers_named`].
    pub fn get_registers_named(&self) -> std::collections::HashMap<String, (u128, u32)> {
        self.inner.get_registers_named()
    }

    /// See [`StateExport::get_symbolic_register_names`].
    pub fn get_symbolic_register_names(&self) -> Vec<String> {
        self.inner.get_symbolic_register_names()
    }

    /// See [`StateExport::get_history`].
    pub fn get_history(&self) -> Vec<u64> {
        self.inner.get_history()
    }

    /// See [`StateExport::get_call_stack`].
    pub fn get_call_stack(&self) -> Vec<(u64, u64, u64, u64)> {
        self.inner.get_call_stack()
    }

    /// See [`StateExport::get_call_stack_depth`].
    pub fn get_call_stack_depth(&self) -> usize {
        self.inner.get_call_stack_depth()
    }

    /// See [`StateExport::get_detailed_history`].
    pub fn get_detailed_history(&self) -> Vec<(u64, u8, u64)> {
        self.inner.get_detailed_history()
    }

    /// See [`StateExport::get_detailed_history_str`].
    pub fn get_detailed_history_str(&self) -> Vec<(u64, String, u64)> {
        self.inner.get_detailed_history_str()
    }

    /// See [`StateExport::page_count`].
    pub fn page_count(&self) -> usize {
        self.inner.page_count()
    }

    /// See [`StateExport::get_page`].
    pub fn get_page(&self, index: usize) -> Option<(u64, Vec<u8>, u8, Vec<u16>)> {
        self.inner.get_page(index)
    }

    /// See [`StateExport::page_addresses`].
    pub fn page_addresses(&self) -> Vec<u64> {
        self.inner.page_addresses()
    }

    /// See [`StateExport::memory_load`].
    pub fn memory_load(&self, addr: u64, size: usize) -> Option<Vec<u8>> {
        self.inner.memory_load(addr, size)
    }

    /// See [`StateExport::get_heap_allocated`].
    pub fn get_heap_allocated(&self) -> Vec<(u64, u64)> {
        self.inner.get_heap_allocated()
    }

    /// See [`StateExport::get_heap_freed`].
    pub fn get_heap_freed(&self) -> Vec<u64> {
        self.inner.get_heap_freed()
    }

    /// See [`StateExport::get_heap_alloc_count`].
    pub fn get_heap_alloc_count(&self) -> usize {
        self.inner.get_heap_alloc_count()
    }

    /// See [`StateExport::get_heap_free_count`].
    pub fn get_heap_free_count(&self) -> usize {
        self.inner.get_heap_free_count()
    }

    /// See [`StateExport::get_open_fds`].
    pub fn get_open_fds(&self) -> Vec<(u32, String, u64, u32, usize, bool)> {
        self.inner.get_open_fds()
    }

    /// See [`StateExport::get_fd_count`].
    pub fn get_fd_count(&self) -> usize {
        self.inner.get_fd_count()
    }

    /// See [`StateExport::get_inspection_counts`].
    pub fn get_inspection_counts(&self) -> Vec<(String, u64)> {
        self.inner.get_inspection_counts()
    }

    /// See [`StateExport::get_symbolic_offsets`].
    pub fn get_symbolic_offsets(&self, page_addr: u64) -> Vec<u16> {
        self.inner.get_symbolic_offsets(page_addr)
    }
}
