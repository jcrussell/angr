//! Engine I/O dispatch: memory / register / hook / syscall / lift callbacks.
//!
//! Split out of `callbacks.rs` (god-module decomposition, angr-zel8z.2).
//! Inherent-impl block only; the single `#[pymethods]` block stays in `mod.rs`.
//!
//! # Picking a symbolic-store variant
//!
//! `call_memory_store_symbolic`, `..._symbolic_value` and `..._symbolic_full`
//! are told apart by *which side* — address or data — is symbolic, and the
//! names do not say which (angr-9ke6b.29). The rule, shortest first:
//!
//! - **concrete address, value worth keeping as an AST** →
//!   [`PythonCallbacks::call_memory_store_symbolic_value`]. Takes `addr: u64`.
//!   Symbolic *data* is what the `_value` refers to; a concrete value is
//!   accepted too and degrades to the byte-level `memory_store` when the
//!   callback is unwired.
//! - **symbolic address** → [`PythonCallbacks::call_memory_store_symbolic_full`].
//!   Takes `addr_val: &RustBV`. "Full" = both sides cross as claripy ASTs, so
//!   the data may be symbolic *or* concrete; Python's memory model builds the
//!   conditional stores over every concretization.
//! - **symbolic address the interpreter already concretized to a small
//!   candidate set** → [`PythonCallbacks::call_memory_store_symbolic`]. This is
//!   not a third mode: it forwards to `_full` (passing the address AST, not the
//!   candidates) and hard-errors when `_full` is unwired, because storing to
//!   `addrs[0]` alone silently drops `addrs[1..]` (angr-ph300.64).
//!
//! Only `_value` and `_full` are real Python callback slots; each has a
//! matching `has_*` accessor that every production call site checks first.

use super::*;

/// Decode a `(bytes, is_symbolic, symbolic_ast?)` tuple returned by a Python
/// data callback.
///
/// The trailing AST element is optional on the Python side: a 2-tuple, and a
/// 3-tuple whose last element is `None`, both mean "no symbolic AST". Shared
/// by `call_memory_load`, `call_memory_load_batch` and `call_dirty_call` so a
/// fix to the decoding lands on all three at once.
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

/// Decode a `(page_data, permissions, is_mapped)` tuple returned by a Python
/// page-fetch callback.
///
/// Shared by `call_fetch_page` and the batch loop in `call_batch_fetch_pages`
/// so the single-page and batched paths cannot drift apart on element order or
/// element types.
fn extract_page_tuple(tuple: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<(Vec<u8>, u8, bool)> {
    let data: Vec<u8> = tuple.get_item(0)?.extract()?;
    let permissions: u8 = tuple.get_item(1)?.extract()?;
    let is_mapped: bool = tuple.get_item(2)?.extract()?;

    Ok((data, permissions, is_mapped))
}

/// One-shot latch for the `python_servable_pages` poison warning.
///
/// See [`warn_page_set_poisoned`] for why the warning is latched rather than
/// emitted per call. Each snapshot gets its own latch so a poisoned
/// `python_servable_pages` never masks a later poisoned `python_page_universe`.
pub(super) static SERVABLE_POISON_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// One-shot latch for the `python_page_universe` poison warning.
pub(super) static UNIVERSE_POISON_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Report — at most once per snapshot per process — that a page-set lock is
/// poisoned, i.e. a thread panicked while holding its write side.
///
/// Poison is impossible in a production wheel (`panic = "abort"`, workspace
/// `Cargo.toml`) but reachable in an unwinding test build, and nothing else on
/// these paths would ever reveal it — hence the warning.
///
/// Latched via `warned` because the reader ([`page_set_contains`]) sits in the
/// per-candidate-page prefetch loop (`prefetch::fetch_page` and
/// `prefetch::fetch_pages_batch`): a lock stays poisoned forever, so an
/// unlatched `log::warn!` would emit one line per page probe for the rest of
/// the process.
///
/// The read and write paths ([`page_set_contains`] and [`store_page_set`])
/// share one latch per snapshot: they report the same underlying fault and the
/// same degradation, so whichever side notices first is the one that speaks.
fn warn_page_set_poisoned(
    which: &str,
    warned: &std::sync::atomic::AtomicBool,
    detail: impl FnOnce() -> String,
) {
    if !warned.swap(true, std::sync::atomic::Ordering::Relaxed) {
        log::warn!(
            "{which} lock is poisoned (a thread panicked holding the write side); \
             page-serve filtering is disabled for the rest of this process — {}",
            detail()
        );
    }
}

/// Install (`Some`) or drop (`None`) a page-set snapshot.
///
/// Write-side twin of [`page_set_contains`] (angr-sqfj8.148), shared by the
/// four `py_{set,clear}_python_{servable_pages,page_universe}` setters in
/// `callbacks/mod.rs`, which differ only in which snapshot they write.
pub(super) fn store_page_set(
    lock: &std::sync::RwLock<Option<std::collections::HashSet<u64>>>,
    which: &str,
    warned: &std::sync::atomic::AtomicBool,
    value: Option<std::collections::HashSet<u64>>,
) {
    match lock.write() {
        Ok(mut guard) => *guard = value,
        // SILENT(cat-b): the update is dropped, but it cannot strand a *stale*
        // snapshot in front of the reader: `page_set_contains` sees the same
        // poison and fails open, so a snapshot that failed to install or clear
        // is never consulted again. The loss is the optimization, not the
        // answer — still a real fault, so it is logged.
        Err(_) => warn_page_set_poisoned(which, warned, || {
            "this update was dropped, and every page probe now crosses to Python \
             as it did before the snapshot existed"
                .to_string()
        }),
    }
}

/// Ask a page-set snapshot whether it contains `page_addr`, failing *open*.
///
/// Shared by [`PythonCallbacks::python_can_serve_page`] and
/// [`PythonCallbacks::python_has_page`], which differ only in which snapshot
/// they consult (angr-sqfj8.14). Both are pure *optimizations* — a `true`
/// answer only means "cross to Python and ask", which is the behaviour that
/// predates the snapshots — so every uncertain case answers `true`.
///
/// The two uncertain cases are deliberately told apart:
///
/// - **no snapshot installed** (`None`) is the normal steady state whenever
///   Python cannot prove a verdict for every page; it is not an error.
/// - **poisoned lock** means a thread panicked while holding the write side —
///   see [`warn_page_set_poisoned`] for why that is worth a (latched) log line.
fn page_set_contains(
    lock: &std::sync::RwLock<Option<std::collections::HashSet<u64>>>,
    which: &str,
    warned: &std::sync::atomic::AtomicBool,
    page_addr: u64,
) -> bool {
    match lock.read() {
        // SILENT(cat-a): `None` is the documented "Python installed no
        // snapshot" state, not a lost value — fail open and ask Python.
        Ok(guard) => guard
            .as_ref()
            .is_none_or(|pages| pages.contains(&page_addr)),
        // SILENT(cat-b): the fallback costs at most a needless GIL crossing
        // (the answer degrades to the pre-snapshot behaviour, never to a wrong
        // one), but it is a real fault, so it is logged rather than folded
        // into the `None` case above.
        Err(_) => {
            warn_page_set_poisoned(which, warned, || {
                format!(
                    "every page probe (first: {page_addr:#x}) now crosses to Python as it \
                     did before the snapshot existed"
                )
            });
            true
        }
    }
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
            let cb = require_callback!(self.memory_load);

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
            let cb = require_callback!(self.memory_store);

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
                // Converted to a Python list of tuples by PyO3's `IntoPyObject for &[T]`,
                // so the slice goes straight across without an intermediate `Vec` copy.
                let result = cb.call1(py, (loads,))?;

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

    /// Store over a **symbolic address that already concretized** to the
    /// `addrs` candidate set (data: either mode).
    ///
    /// Despite the bare name this is not a store mode of its own — see the
    /// variant table in the module docs. It exists so the interpreter can hand
    /// over both the concretization it computed and the original address AST;
    /// the store itself is delegated to `call_memory_store_symbolic_full`,
    /// which performs the conditional stores to each possible address.
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

            // Unconditionally hand off to the full symbolic callback. It
            // handles both symbolic and *concrete* data over a symbolic address
            // range by giving the address AST + data to Python's memory model,
            // which builds the correct conditional stores across every
            // concretized candidate. Gating this on `data.is_symbolic()` (the
            // old behavior) let concrete data fall through to a
            // first-address-only fallback that silently dropped stores to
            // addrs[1..] — the angr-ph300.64 divergent-memory bug for
            // `table[x]=const` with 17+ concretizations.
            //
            // With `_full` unwired the callee hard-errors (module invariant 1,
            // `avoid-silent-no-op-callback-fallbacks`) and that error is this
            // method's answer too — deliberately, rather than re-deriving a
            // duplicate "not set" message here. `_full` is wired
            // unconditionally by rust_manager.py::_setup_callbacks, so the
            // unset path is only reachable under Python/.so version skew (a
            // stale .so predating the setter).
            //
            // `addrs` is the concretization the interpreter already computed;
            // Python's memory model re-derives it from `addr_ast`, so it is
            // accepted for call-site symmetry but not forwarded.
            let _ = addrs;
            self.call_memory_store_symbolic_full(addr_ast, data)
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
            let cb = require_callback!(self.lift_block);

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
            let cb = require_callback!(self.dirty_call);

            // `args` becomes a Python list via `IntoPyObject for &[T]` — no copy needed.
            let result = cb.call1(py, (name, args, ret_ty_bits))?;
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
        page_set_contains(
            &self.python_servable_pages,
            "python_servable_pages",
            &SERVABLE_POISON_WARNED,
            page_addr,
        )
    }

    /// Whether Python holds any page object at `page_addr` (angr-gorvf.4.7).
    ///
    /// Distinct from `python_can_serve_page`: this asks whether Python has
    /// *data* there at all, not whether the whole page can be fetched as
    /// concrete bytes. Fails open (assume Python has it) when no snapshot is
    /// installed, so an unknown page keeps crossing exactly as before.
    pub(crate) fn python_has_page(&self, page_addr: u64) -> bool {
        page_set_contains(
            &self.python_page_universe,
            "python_page_universe",
            &UNIVERSE_POISON_WARNED,
            page_addr,
        )
    }

    /// Write side of [`Self::python_can_serve_page`]'s snapshot: `Some` installs,
    /// `None` drops it. Backs `py_set_python_servable_pages` /
    /// `py_clear_python_servable_pages`.
    pub(super) fn store_servable_pages(&self, value: Option<std::collections::HashSet<u64>>) {
        store_page_set(
            &self.python_servable_pages,
            "python_servable_pages",
            &SERVABLE_POISON_WARNED,
            value,
        );
    }

    /// Write side of [`Self::python_has_page`]'s snapshot: `Some` installs,
    /// `None` drops it. Backs `py_set_python_page_universe` /
    /// `py_clear_python_page_universe`.
    pub(super) fn store_page_universe(&self, value: Option<std::collections::HashSet<u64>>) {
        store_page_set(
            &self.python_page_universe,
            "python_page_universe",
            &UNIVERSE_POISON_WARNED,
            value,
        );
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
            let cb = require_callback!(self.fetch_page);

            let result = cb.call1(py, (page_addr,))?;
            let tuple = result.cast_bound::<pyo3::types::PyTuple>(py)?;
            extract_page_tuple(tuple)
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
                // `IntoPyObject for &[T]` builds the Python list directly from the slice.
                let result = cb.call1(py, (page_addrs,))?;

                let result_list = result.cast_bound::<pyo3::types::PyList>(py)?;
                let mut results = Vec::with_capacity(page_addrs.len());

                for item in result_list.iter() {
                    let tuple = item.cast::<pyo3::types::PyTuple>()?;
                    results.push(extract_page_tuple(tuple)?);
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

    /// Store to a **concrete address** a value whose expression tree is worth
    /// preserving (the `_value` in the name = the *data* side is the symbolic
    /// one). Contrast `call_memory_store_symbolic_full`, which is the
    /// symbolic-*address* variant; see the module docs for the full table.
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

    /// Store to a **symbolic address**, with both sides crossing to Python as
    /// claripy ASTs ("full" = full expression trees for address *and* data, so
    /// the data may be symbolic or concrete). Contrast
    /// `call_memory_store_symbolic_value`, which is the concrete-address /
    /// symbolic-data variant; see the module docs for the full table.
    ///
    /// Used when the address cannot be concretized to a single value, and (via
    /// `call_memory_store_symbolic`) when it concretized to a small set —
    /// Python's memory model turns the address AST into conditional stores.
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

            // Hard-error rather than silently no-op (module invariant 1,
            // `avoid-silent-no-op-callback-fallbacks`): a silent Ok(()) here
            // would drop the store and diverge Rust↔Python memory. Every
            // *direct* production call site guards with
            // has_memory_store_symbolic_full(); the one indirect path, via
            // `call_memory_store_symbolic`, is deliberately unguarded because
            // rust_manager.py::_setup_callbacks wires this callback
            // unconditionally (rationale in that method's body comment). So
            // this is only
            // reachable under Python/.so version skew, from a future unguarded
            // caller, or from a teardown that nulls the callback — surface it
            // loudly.
            let cb = require_callback!(self.memory_store_symbolic_full);

            let claripy_mod = py.import("claripy")?;
            let addr_ast = rustbv_to_claripy(py, addr_val, &claripy_mod)?;
            let data_ast = rustbv_to_claripy(py, data_val, &claripy_mod)?;
            cb.call1(py, (addr_ast, data_ast))?;
            Ok(())
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

            let cb = require_callback!(self.memory_load_symbolic_full);

            let claripy_mod = py.import("claripy")?;
            let addr_ast = rustbv_to_claripy(py, addr_val, &claripy_mod)?;
            cb.call1(py, (addr_ast, size))
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
            let cb = require_callback!(self.resolve_function);

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
