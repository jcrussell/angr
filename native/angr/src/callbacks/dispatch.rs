//! Engine I/O dispatch: memory / register / hook / syscall / lift callbacks.
//!
//! Split out of `callbacks.rs` (god-module decomposition, angr-zel8z.2).
//! Inherent-impl block only; the single `#[pymethods]` block stays in `mod.rs`.

use super::*;

/// Decode a `(bytes, is_symbolic, symbolic_ast?)` tuple returned by a Python
/// data callback.
///
/// The trailing AST element is optional on the Python side: a 2-tuple, and a
/// 3-tuple whose last element is `None`, both mean "no symbolic AST". Shared
/// by `call_memory_load`, `call_memory_load_batch`, `call_get_register` and
/// `call_dirty_call` so a fix to the decoding lands on all four at once.
fn extract_data_tuple(tuple: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<BatchLoadEntry> {
    let data: Vec<u8> = tuple.get_item(0)?.extract()?;
    let is_symbolic: bool = tuple.get_item(1)?.extract()?;

    let symbolic_ast = if tuple.len() > 2 {
        let ast_obj = tuple.get_item(2)?;
        // SILENT(cat-a): a Python `None` in slot 2 is the documented
        // "concrete result, no AST" encoding, not a lost value.
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

impl PythonCallbacks {
    /// Call the memory load callback.
    ///
    /// Returns (data_bytes, is_symbolic, symbolic_ast).
    pub(crate) fn call_memory_load(
        &self,
        addr: u64,
        size: u32,
    ) -> PyResult<(Vec<u8>, bool, Option<Py<PyAny>>)> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::MemoryLoad,
            );
            let cb = self.memory_load.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("memory_load callback not set")
            })?;

            let result = cb.call1(py, (addr, size))?;
            let tuple = result.cast_bound::<pyo3::types::PyTuple>(py)?;
            extract_data_tuple(tuple)
        })
    }

    /// Call the memory store callback.
    pub(crate) fn call_memory_store(&self, addr: u64, data: &[u8]) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::MemoryStore,
            );
            let cb = self.memory_store.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("memory_store callback not set")
            })?;

            let py_bytes = PyBytes::new(py, data);
            cb.call1(py, (addr, py_bytes))?;
            Ok(())
        })
    }

    /// Call the batched memory store callback.
    ///
    /// This sends multiple stores in a single callback for efficiency.
    /// Falls back to individual stores if batch callback is not set.
    pub(crate) fn call_memory_store_batch(&self, stores: &[(u64, Vec<u8>)]) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::MemoryStoreBatch,
            );
            if stores.is_empty() {
                return Ok(());
            }

            // Try batch callback first
            if let Some(cb) = &self.memory_store_batch {
                // Convert stores to Python list of tuples
                let py_stores: Vec<(u64, Py<PyBytes>)> = stores
                    .iter()
                    .map(|(addr, data)| (*addr, PyBytes::new(py, data).unbind()))
                    .collect();
                cb.call1(py, (py_stores,))?;
                return Ok(());
            }

            // Fallback: call individual stores
            for (addr, data) in stores {
                self.call_memory_store(*addr, data)?;
            }
            Ok(())
        })
    }

    /// Call the batched memory load callback.
    ///
    /// This sends multiple load requests in a single callback for efficiency.
    /// Falls back to individual loads if batch callback is not set.
    ///
    /// Returns a vector of (data_bytes, is_symbolic, symbolic_ast) tuples,
    /// one for each load request.
    pub(crate) fn call_memory_load_batch(
        &self,
        loads: &[(u64, u32)], // (address, size) pairs
    ) -> PyResult<Vec<BatchLoadEntry>> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::MemoryLoadBatch,
            );
            if loads.is_empty() {
                return Ok(Vec::new());
            }

            // Try batch callback first
            if let Some(cb) = &self.memory_load_batch {
                // Convert loads to Python list of tuples
                let py_loads: Vec<(u64, u32)> = loads.to_vec();
                let result = cb.call1(py, (py_loads,))?;

                // Parse the result list
                let result_list = result.cast_bound::<pyo3::types::PyList>(py)?;
                let mut results = Vec::with_capacity(loads.len());

                for item in result_list.iter() {
                    let tuple = item.cast::<pyo3::types::PyTuple>()?;
                    results.push(extract_data_tuple(tuple)?);
                }

                return Ok(results);
            }

            // Fallback: call individual loads
            let mut results = Vec::with_capacity(loads.len());
            for &(addr, size) in loads {
                let (data, is_sym, ast) = self.call_memory_load(addr, size)?;
                results.push((data, is_sym, ast));
            }
            Ok(results)
        })
    }

    /// Call the symbolic memory store callback.
    ///
    /// This is called when the address is symbolic and concretizes to multiple values.
    /// The callback should perform conditional stores to each possible address.
    pub(crate) fn call_memory_store_symbolic(
        &self,
        addrs: &[u64],
        data: &RustBV,
        addr_ast: &RustBV,
    ) -> PyResult<()> {
        Python::attach(|_py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::MemoryStoreSymbolic,
            );

            // Prefer the full symbolic callback whenever it is wired. It handles
            // both symbolic and *concrete* data over a symbolic address range by
            // handing the address AST + data to Python's memory model, which
            // builds the correct conditional stores across every concretized
            // candidate. Gating this on `data.is_symbolic()` (the old behavior)
            // let concrete data fall through to the first-address-only fallback
            // below, silently dropping stores to addrs[1..] — the angr-ph300.64
            // divergent-memory bug for `table[x]=const` with 17+ concretizations.
            if self.memory_store_symbolic_full.is_some() {
                return self.call_memory_store_symbolic_full(addr_ast, data);
            }

            // Hard-error rather than silently storing only addrs[0] (module
            // invariant 1, `avoid-silent-no-op-callback-fallbacks`). The older
            // fallbacks below the `_full` dispatch above stored to addrs.first()
            // and returned Ok(()), silently dropping addrs[1..] — the exact
            // angr-ph300.64 divergent-memory bug class. `_full` is wired
            // unconditionally by rust_manager.py::_setup_callbacks, so this is
            // only reachable under Python/.so version skew (a stale .so predating
            // the setter); surface that loudly instead of diverging memory.
            let _ = addrs;
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "memory_store_symbolic_full callback not set",
            ))
        })
    }

    /// Call the hook execution callback.
    ///
    /// Returns the new PC after hook execution.
    ///
    /// Reference pattern for the `avoid-silent-no-op-callback-fallbacks`
    /// invariant (module-level invariant 1). When [`Self::on_hook`] is
    /// `None`, hard-error rather than no-op — see the module-level docs
    /// for why silent fallbacks mask wiring bugs.
    /// **No Rust caller** (angr-9ke6b.214): the Python side still registers
    /// `get_register` / `put_register` (`rust_manager.py::_setup_callbacks`),
    /// and `set_on_hook` / `set_on_syscall` remain on the `#[pymethods]`
    /// surface, but the engine reads and writes registers through
    /// `RustSimState` and routes hooks/syscalls through
    /// `ExplorationEvent` bounces instead of these direct callbacks. Retiring
    /// the path means dropping the pyclass setters and the Python
    /// registration too, so it is flagged rather than deleted here.
    #[allow(dead_code)]
    pub(crate) fn call_on_hook(&self, addr: u64) -> PyResult<u64> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::OnHook,
            );
            let cb = self.on_hook.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("on_hook callback not set")
            })?;

            let result = cb.call1(py, (addr,))?;
            result.extract(py)
        })
    }

    /// Call the syscall handling callback.
    /// **No Rust caller** (angr-9ke6b.214): the Python side still registers
    /// `get_register` / `put_register` (`rust_manager.py::_setup_callbacks`),
    /// and `set_on_hook` / `set_on_syscall` remain on the `#[pymethods]`
    /// surface, but the engine reads and writes registers through
    /// `RustSimState` and routes hooks/syscalls through
    /// `ExplorationEvent` bounces instead of these direct callbacks. Retiring
    /// the path means dropping the pyclass setters and the Python
    /// registration too, so it is flagged rather than deleted here.
    #[allow(dead_code)]
    pub(crate) fn call_on_syscall(&self, num: u64) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::OnSyscall,
            );
            let cb = self.on_syscall.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("on_syscall callback not set")
            })?;

            cb.call1(py, (num,))?;
            Ok(())
        })
    }

    /// Call the block lifting callback.
    ///
    /// Returns the IRSB as a JSON string.
    /// If `opt_level` is `Some`, passes it as the second positional arg.
    /// If `dirty_bytes` is `Some`, passes it as the third positional arg
    /// (Python callback uses these as `byte_string=` for SMC fresh-bytes lift).
    pub(crate) fn call_lift_block(
        &self,
        addr: u64,
        opt_level: Option<i32>,
        dirty_bytes: Option<&[u8]>,
    ) -> PyResult<String> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::LiftBlock,
            );
            let cb = self.lift_block.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("lift_block callback not set")
            })?;

            let result = match (opt_level, dirty_bytes) {
                (None, None) => cb.call1(py, (addr,))?,
                (Some(level), None) => cb.call1(py, (addr, level))?,
                (None, Some(bytes)) => {
                    let py_bytes = PyBytes::new(py, bytes);
                    cb.call1(py, (addr, py.None(), py_bytes))?
                }
                (Some(level), Some(bytes)) => {
                    let py_bytes = PyBytes::new(py, bytes);
                    cb.call1(py, (addr, level, py_bytes))?
                }
            };
            result.extract(py)
        })
    }

    /// Call the register get callback.
    ///
    /// Returns (data_bytes, is_symbolic, symbolic_ast).
    /// **No Rust caller** (angr-9ke6b.214): the Python side still registers
    /// `get_register` / `put_register` (`rust_manager.py::_setup_callbacks`),
    /// and `set_on_hook` / `set_on_syscall` remain on the `#[pymethods]`
    /// surface, but the engine reads and writes registers through
    /// `RustSimState` and routes hooks/syscalls through
    /// `ExplorationEvent` bounces instead of these direct callbacks. Retiring
    /// the path means dropping the pyclass setters and the Python
    /// registration too, so it is flagged rather than deleted here.
    #[allow(dead_code)]
    pub(crate) fn call_get_register(
        &self,
        offset: u32,
        size: u32,
    ) -> PyResult<(Vec<u8>, bool, Option<Py<PyAny>>)> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::GetRegister,
            );
            let cb = self.get_register.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("get_register callback not set")
            })?;

            let result = cb.call1(py, (offset, size))?;
            let tuple = result.cast_bound::<pyo3::types::PyTuple>(py)?;
            extract_data_tuple(tuple)
        })
    }

    /// Call the register put callback.
    /// **No Rust caller** (angr-9ke6b.214): the Python side still registers
    /// `get_register` / `put_register` (`rust_manager.py::_setup_callbacks`),
    /// and `set_on_hook` / `set_on_syscall` remain on the `#[pymethods]`
    /// surface, but the engine reads and writes registers through
    /// `RustSimState` and routes hooks/syscalls through
    /// `ExplorationEvent` bounces instead of these direct callbacks. Retiring
    /// the path means dropping the pyclass setters and the Python
    /// registration too, so it is flagged rather than deleted here.
    #[allow(dead_code)]
    pub(crate) fn call_put_register(&self, offset: u32, data: &[u8]) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::PutRegister,
            );
            let cb = self.put_register.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("put_register callback not set")
            })?;

            let py_bytes = PyBytes::new(py, data);
            cb.call1(py, (offset, py_bytes))?;
            Ok(())
        })
    }

    /// Call the dirty call callback for VEX helper functions.
    ///
    /// This handles dirty calls like CPUID, RDTSC, x87 operations, etc.
    /// Returns (data_bytes, is_symbolic, symbolic_ast).
    pub(crate) fn call_dirty_call(
        &self,
        name: &str,
        args: &[u64],
        ret_ty_bits: u32,
    ) -> PyResult<(Vec<u8>, bool, Option<Py<PyAny>>)> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::DirtyCall,
            );
            let cb = self.dirty_call.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("dirty_call callback not set")
            })?;

            // Convert args to Python list
            let args_list: Vec<u64> = args.to_vec();

            let result = cb.call1(py, (name, args_list, ret_ty_bits))?;
            let tuple = result.cast_bound::<pyo3::types::PyTuple>(py)?;
            extract_data_tuple(tuple)
        })
    }

    /// Check if dirty call callback is available.
    pub(crate) fn has_dirty_call(&self) -> bool {
        self.dirty_call.is_some()
    }

    /// Check if fetch_page callback is available.
    pub(crate) fn has_fetch_page(&self) -> bool {
        self.fetch_page.is_some()
    }

    /// Whether Python could serve `page_addr` at all (angr-gorvf.4.6).
    ///
    /// False means "Python would decline this page" — the caller must skip the
    /// crossing entirely rather than pay a GIL attach to be told no. True when
    /// no snapshot was installed (unknown → ask Python, the legacy behaviour).
    pub(crate) fn python_can_serve_page(&self, page_addr: u64) -> bool {
        match self.python_servable_pages.read() {
            Ok(guard) => match &*guard {
                Some(pages) => pages.contains(&page_addr),
                None => true,
            },
            Err(_) => true,
        }
    }

    /// Whether Python holds any page object at `page_addr` (angr-gorvf.4.7).
    ///
    /// Distinct from `python_can_serve_page`: this asks whether Python has
    /// *data* there at all, not whether the whole page can be fetched as
    /// concrete bytes. Fails open (assume Python has it) when no snapshot is
    /// installed, so an unknown page keeps crossing exactly as before.
    pub(crate) fn python_has_page(&self, page_addr: u64) -> bool {
        match self.python_page_universe.read() {
            Ok(guard) => match &*guard {
                Some(pages) => pages.contains(&page_addr),
                None => true,
            },
            Err(_) => true,
        }
    }

    /// Call the page fetch callback to load a single 4KB page.
    ///
    /// Returns (page_data, permissions, is_mapped).
    /// - page_data: 4096 bytes of page content
    /// - permissions: permission bits (R=4, W=2, X=1)
    /// - is_mapped: whether the page exists in Python memory
    pub(crate) fn call_fetch_page(&self, page_addr: u64) -> PyResult<(Vec<u8>, u8, bool)> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::FetchPage,
            );
            let cb = self.fetch_page.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("fetch_page callback not set")
            })?;

            let result = cb.call1(py, (page_addr,))?;
            let tuple = result.cast_bound::<pyo3::types::PyTuple>(py)?;

            let data: Vec<u8> = tuple.get_item(0)?.extract()?;
            let permissions: u8 = tuple.get_item(1)?.extract()?;
            let is_mapped: bool = tuple.get_item(2)?.extract()?;

            Ok((data, permissions, is_mapped))
        })
    }

    /// Call the batched page fetch callback to load multiple 4KB pages.
    ///
    /// Returns a list of (page_data, permissions, is_mapped) for each page.
    pub(crate) fn call_batch_fetch_pages(
        &self,
        page_addrs: &[u64],
    ) -> PyResult<Vec<(Vec<u8>, u8, bool)>> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::BatchFetchPages,
            );
            if page_addrs.is_empty() {
                return Ok(Vec::new());
            }

            // Try batch callback first
            if let Some(cb) = &self.batch_fetch_pages {
                let addrs_list: Vec<u64> = page_addrs.to_vec();
                let result = cb.call1(py, (addrs_list,))?;

                let result_list = result.cast_bound::<pyo3::types::PyList>(py)?;
                let mut results = Vec::with_capacity(page_addrs.len());

                for item in result_list.iter() {
                    let tuple = item.cast::<pyo3::types::PyTuple>()?;
                    let data: Vec<u8> = tuple.get_item(0)?.extract()?;
                    let permissions: u8 = tuple.get_item(1)?.extract()?;
                    let is_mapped: bool = tuple.get_item(2)?.extract()?;
                    results.push((data, permissions, is_mapped));
                }

                return Ok(results);
            }

            // Fallback: call individual fetches
            let mut results = Vec::with_capacity(page_addrs.len());
            for &page_addr in page_addrs {
                let (data, perms, mapped) = self.call_fetch_page(page_addr)?;
                results.push((data, perms, mapped));
            }
            Ok(results)
        })
    }

    /// Store a symbolic value to memory with full expression tree preservation.
    ///
    /// This method converts the RustBV expression tree to a claripy AST and
    /// calls Python to store it. This preserves symbolic expressions like
    /// `x + 10 ^ 0x42` instead of losing them to zeros.
    ///
    /// # Arguments
    /// * `py` - Python GIL token
    /// * `addr` - The memory address to store to
    /// * `value` - The symbolic RustBV value with expression tree
    ///
    /// # Returns
    /// Ok(()) on success. With the callback unset, a *concrete* value falls
    /// back to the byte-level `memory_store`; a *symbolic* value is an error
    /// (see the fallback comment below).
    pub(crate) fn call_memory_store_symbolic_value(
        &self,
        addr: u64,
        value: &RustBV,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::MemoryStoreSymbolicValue,
            );
            use crate::claripy_bridge::rustbv_to_claripy;

            // If the symbolic value callback is set, use it
            if let Some(cb) = &self.memory_store_symbolic_value {
                // Import claripy module
                let claripy_mod = py.import("claripy")?;

                // Convert RustBV expression tree to claripy AST
                let ast = rustbv_to_claripy(py, value, &claripy_mod)?;
                cb.call1(py, (addr, ast))?;
                return Ok(());
            }

            // Fallback: use the standard memory_store with byte representation.
            // Sound only for a concrete value, where `bv_to_bytes` is exact.
            // For a symbolic value `bv_to_bytes` returns all zeros, so this
            // fallback would report success while overwriting memory with 0 —
            // a wrong answer, not a degraded one (angr-9ke6b.19). Hard-error
            // instead, matching `call_memory_store_symbolic_full` and module
            // invariant 1 (`avoid-silent-no-op-callback-fallbacks`). Every
            // production call site guards with
            // `has_memory_store_symbolic_value()` first and
            // `rust_manager.py::_setup_callbacks` always registers the
            // callback, so this is only reachable from a future unguarded
            // caller or a minimal embedding — surface it loudly.
            if value.is_symbolic() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "memory_store_symbolic_value callback not set; refusing to \
                     zero-fill symbolic store at 0x{addr:x}"
                )));
            }
            let data_bytes = bv_to_bytes(value);
            self.call_memory_store(addr, &data_bytes)
        })
    }

    /// Check if symbolic value store callback is available.
    pub(crate) fn has_memory_store_symbolic_value(&self) -> bool {
        self.memory_store_symbolic_value.is_some()
    }

    /// True when the callback-memory-proxy gate is on, i.e. the memory-store
    /// callbacks are no-ops and Rust must keep the store itself (angr-5rjbq).
    pub(crate) fn memory_is_rust_proxy(&self) -> bool {
        self.memory_is_rust_proxy
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Call the full symbolic store callback (symbolic address + symbolic value).
    /// Used when the address cannot be concretized to a single value or small set.
    pub(crate) fn call_memory_store_symbolic_full(
        &self,
        addr_val: &RustBV,
        data_val: &RustBV,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::MemoryStoreSymbolicFull,
            );
            use crate::claripy_bridge::rustbv_to_claripy;

            if let Some(cb) = &self.memory_store_symbolic_full {
                let claripy_mod = py.import("claripy")?;
                let addr_ast = rustbv_to_claripy(py, addr_val, &claripy_mod)?;
                let data_ast = rustbv_to_claripy(py, data_val, &claripy_mod)?;
                cb.call1(py, (addr_ast, data_ast))?;
                return Ok(());
            }

            // Hard-error rather than silently no-op (module invariant 1,
            // `avoid-silent-no-op-callback-fallbacks`): a silent Ok(()) here
            // would drop the store and diverge Rust↔Python memory. Every
            // production call site guards with has_memory_store_symbolic_full(),
            // so this is only reachable if a future unguarded caller (or a
            // teardown that nulls the callback) hits it — surface it loudly.
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "memory_store_symbolic_full callback not set",
            ))
        })
    }

    /// Check if full symbolic store callback is available.
    pub(crate) fn has_memory_store_symbolic_full(&self) -> bool {
        self.memory_store_symbolic_full.is_some()
    }

    /// Load from memory at a symbolic address (full AST delegation).
    ///
    /// This is called when the address range is too large to concretize.
    /// Python will use angr's memory model to handle the symbolic address,
    /// which may build ITE chains or use address concretization strategies.
    ///
    /// # Arguments
    /// * `py` - Python GIL token
    /// * `addr_val` - The symbolic address as a RustBV
    /// * `size` - Number of bytes to load
    ///
    /// # Returns
    /// The loaded claripy AST from Python's memory model.
    pub(crate) fn call_memory_load_symbolic_full(
        &self,
        addr_val: &RustBV,
        size: u32,
    ) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::MemoryLoadSymbolicFull,
            );
            use crate::claripy_bridge::rustbv_to_claripy;

            if let Some(cb) = &self.memory_load_symbolic_full {
                let claripy_mod = py.import("claripy")?;
                let addr_ast = rustbv_to_claripy(py, addr_val, &claripy_mod)?;
                return cb.call1(py, (addr_ast, size));
            }
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "memory_load_symbolic_full callback not set",
            ))
        })
    }

    /// Check if full symbolic load callback is available.
    pub(crate) fn has_memory_load_symbolic_full(&self) -> bool {
        self.memory_load_symbolic_full.is_some()
    }

    /// Call the resolve_function callback to dynamically resolve unmodeled function calls.
    ///
    /// This is called when Rust encounters a CALL to an address that isn't hooked.
    /// Python can check its procedure registries and return procedure info if available.
    ///
    /// # Arguments
    /// * `py` - Python GIL token
    /// * `addr` - Address of the unmodeled function
    /// * `symbol_name` - Symbol name if known from binary, None otherwise
    ///
    /// # Returns
    /// * `Ok(Some((name, num_args, no_return)))` - Function resolved, register and retry
    /// * `Ok(None)` - Function cannot be resolved, deadend the state
    /// * `Err(...)` - Callback error
    pub(crate) fn call_resolve_function(
        &self,
        addr: u64,
        symbol_name: Option<&str>,
    ) -> PyResult<Option<(String, usize, bool)>> {
        Python::attach(|py| {
            let _gil = crate::gil_profile::GilWorkGuard::enter_site(
                crate::gil_profile::CallbackSite::ResolveFunction,
            );
            let cb = self.resolve_function.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("resolve_function callback not set")
            })?;

            let result = cb.call1(py, (addr, symbol_name))?;

            // Check if result is None
            if result.is_none(py) {
                return Ok(None);
            }

            // Extract tuple (name, num_args, no_return)
            let tuple = result.cast_bound::<pyo3::types::PyTuple>(py)?;
            if tuple.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "resolve_function must return (name, num_args, no_return) or None",
                ));
            }

            let name: String = tuple.get_item(0)?.extract()?;
            let num_args: usize = tuple.get_item(1)?.extract()?;
            let no_return: bool = tuple.get_item(2)?.extract()?;

            Ok(Some((name, num_args, no_return)))
        })
    }

    /// Check if resolve_function callback is available.
    pub(crate) fn has_resolve_function(&self) -> bool {
        self.resolve_function.is_some()
    }
}

/// Convert a RustBV to bytes (little-endian).
fn bv_to_bytes(bv: &RustBV) -> Vec<u8> {
    let width = bv.width();
    let num_bytes = width.div_ceil(8) as usize;

    if let Some(value) = bv.as_u128() {
        let mut bytes = vec![0u8; num_bytes];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = (value >> (i * 8)) as u8;
        }
        bytes
    } else {
        // For symbolic values, return zeros (the callback will handle it)
        vec![0u8; num_bytes]
    }
}

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod dispatch_tests;
