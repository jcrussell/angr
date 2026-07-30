//! PyO3 `#[pymethods]` wrappers for `PyRustSimState`.
//!
//! Thin delegation layer: each method forwards to an inner `RustSimState`
//! accessor (defined across the sibling `state/*.rs` impl files). Peeled out
//! of `mod.rs` to keep that file focused on the struct definitions and the
//! core `RustSimState` mechanics (angr-0mqkc.5). This is the single
//! `#[pymethods]` block for `PyRustSimState`; keep it that way so we don't
//! need PyO3's `multiple-pymethods` feature.
use super::*;
use crate::memory::Permission;
use crate::symbolic::RustBV;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAny, PyDict};
use std::sync::Arc;

#[pymethods]
impl PyRustSimState {
    /// Create a new state for the given architecture.
    #[new]
    #[pyo3(signature = (arch="amd64", little_endian=None))]
    pub fn new(arch: &str, little_endian: Option<bool>) -> PyResult<Self> {
        let inner =
            RustSimState::new_with_endian(arch, little_endian).map_err(PyValueError::new_err)?;
        Ok(PyRustSimState { inner })
    }

    /// Get the state ID.
    #[getter]
    pub fn state_id(&self) -> u64 {
        self.inner.state_id()
    }

    /// Get the parent state ID.
    #[getter]
    pub fn parent_id(&self) -> Option<u64> {
        self.inner.parent_id()
    }

    /// Get the program counter.
    #[getter]
    pub fn pc(&self) -> u64 {
        self.inner.pc()
    }

    /// Set the program counter.
    #[setter]
    pub fn set_pc(&mut self, pc: u64) {
        self.inner.set_pc(pc);
    }

    /// Get the architecture name.
    #[getter]
    pub fn arch_name(&self) -> &str {
        self.inner.arch().name()
    }

    /// Get the POSIX brk pointer (mirrors Python's `state.posix.brk`).
    #[getter]
    pub fn posix_brk(&self) -> u64 {
        self.inner.posix_brk()
    }

    /// Set the POSIX brk pointer. Used by the Python wrapper at state-creation
    /// time to push `state.posix.brk` (which the angr loader sets based on the
    /// binary's last address) into Rust so subsequent native brk syscalls
    /// start from the correct base.
    #[setter]
    pub fn set_posix_brk(&mut self, addr: u64) {
        self.inner.set_posix_brk(addr);
    }

    /// Get the getopt(3) `optind` cursor (mirrors `state.libc.getopt_optind`).
    #[getter]
    pub fn getopt_optind(&self) -> u32 {
        self.inner.getopt_cursor().0
    }

    /// Set the getopt(3) `optind` cursor.
    #[setter]
    pub fn set_getopt_optind(&mut self, value: u32) {
        let (_, optchar) = self.inner.getopt_cursor();
        self.inner.set_getopt_cursor(value, optchar);
    }

    /// Get the getopt(3) `optchar` cursor (mirrors `state.libc.getopt_optchar`).
    #[getter]
    pub fn getopt_optchar(&self) -> u32 {
        self.inner.getopt_cursor().1
    }

    /// Set the getopt(3) `optchar` cursor.
    #[setter]
    pub fn set_getopt_optchar(&mut self, value: u32) {
        let (optind, _) = self.inner.getopt_cursor();
        self.inner.set_getopt_cursor(optind, value);
    }

    /// Get the heap brk pointer (mirrors Python's `state.heap.heap_location`,
    /// the malloc bump allocator).
    #[getter]
    pub fn heap_brk(&self) -> u64 {
        self.inner.heap_brk()
    }

    /// Set the heap brk pointer. Used by the Python wrapper at state-creation
    /// time to push `state.heap.heap_location` into Rust so subsequent native
    /// allocations don't collide with a Python-side allocation (angr-um39j).
    #[setter]
    pub fn set_heap_brk(&mut self, addr: u64) {
        self.inner.set_heap_brk(addr);
    }

    /// Push Python's `state.libc.ctype_b_loc_table_ptr` into Rust so the native
    /// `__ctype_b_loc` proc can return it without a Python round-trip.
    #[setter]
    pub fn set_ctype_b_loc_table_ptr(&mut self, addr: u64) {
        let mut ptrs = self.inner.ctype_loc();
        ptrs.b = Some(addr);
        self.inner.set_ctype_loc(ptrs);
    }

    /// Push Python's `state.libc.ctype_tolower_loc_table_ptr` into Rust.
    #[setter]
    pub fn set_ctype_tolower_loc_table_ptr(&mut self, addr: u64) {
        let mut ptrs = self.inner.ctype_loc();
        ptrs.tolower = Some(addr);
        self.inner.set_ctype_loc(ptrs);
    }

    /// Push Python's `state.libc.ctype_toupper_loc_table_ptr` into Rust.
    #[setter]
    pub fn set_ctype_toupper_loc_table_ptr(&mut self, addr: u64) {
        let mut ptrs = self.inner.ctype_loc();
        ptrs.toupper = Some(addr);
        self.inner.set_ctype_loc(ptrs);
    }

    /// Push the loader-resolved guest address of the getopt(3) `optind` extern
    /// global into Rust so the native getopt proc can write the cursor back to
    /// guest memory. Mirrors the ctype table-ptr init-push channel.
    #[setter]
    pub fn set_getopt_optind_addr(&mut self, addr: u64) {
        let mut addrs = self.inner.getopt_extern();
        addrs.optind = Some(addr);
        self.inner.set_getopt_extern(addrs);
    }

    /// Push the loader-resolved guest address of the getopt(3) `optarg` extern
    /// global into Rust. See `set_getopt_optind_addr`.
    #[setter]
    pub fn set_getopt_optarg_addr(&mut self, addr: u64) {
        let mut addrs = self.inner.getopt_extern();
        addrs.optarg = Some(addr);
        self.inner.set_getopt_extern(addrs);
    }

    /// Push the loader-resolved guest address of the getopt(3) `optopt` extern
    /// global into Rust. See `set_getopt_optind_addr`.
    #[setter]
    pub fn set_getopt_optopt_addr(&mut self, addr: u64) {
        let mut addrs = self.inner.getopt_extern();
        addrs.optopt = Some(addr);
        self.inner.set_getopt_extern(addrs);
    }

    /// Read back the pushed getopt(3) `optind` extern address (`None` until
    /// the init-push runs). Exposed for round-trip verification.
    #[getter]
    pub fn get_getopt_optind_addr(&self) -> Option<u64> {
        self.inner.getopt_extern().optind
    }

    /// Read back the pushed getopt(3) `optarg` extern address. See
    /// `get_getopt_optind_addr`.
    #[getter]
    pub fn get_getopt_optarg_addr(&self) -> Option<u64> {
        self.inner.getopt_extern().optarg
    }

    /// Read back the pushed getopt(3) `optopt` extern address. See
    /// `get_getopt_optind_addr`.
    #[getter]
    pub fn get_getopt_optopt_addr(&self) -> Option<u64> {
        self.inner.getopt_extern().optopt
    }

    /// Register a symlink so native `readlink` / `readlinkat` resolve
    /// `link` to `target` (raw bytes, as `readlink(2)` returns — NOT
    /// NUL-terminated). Mirrors `FileSystem::add_symlink`; intended for a
    /// Python harness to seed pre-existing symlinks at state-creation time
    /// before handing off to Rust (`RustExplorationManager(..., symlinks=...)`).
    ///
    /// Python `state.fs` symlinks are NOT auto-mirrored — same explicit
    /// trade-off as `register_known_path` (angr-11djq.6.2 / angr-m7s7y).
    /// Forks inherit the entry via the `Arc<HashMap>` clone in
    /// `FileSystem::fork`, so seeding the initial state covers all
    /// descendants.
    pub fn register_symlink(&mut self, link: String, target: Vec<u8>) {
        self.inner.file_system().add_symlink(link, target);
    }

    /// Get the history (basic block addresses).
    pub fn history(&self) -> Vec<u64> {
        self.inner.history().iter().copied().collect()
    }

    /// Get a register value by name.
    pub fn get_register(&self, name: &str) -> PyResult<u128> {
        self.inner
            .get_register(name)
            .and_then(|bv| bv.as_u128())
            .ok_or_else(|| PyValueError::new_err(format!("cannot read register {name}")))
    }

    /// Set a register value by name.
    pub fn set_register(&mut self, name: &str, value: u128) -> PyResult<()> {
        let size = self
            .inner
            .arch()
            .register_size(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {name}")))?;
        let bv = RustBV::concrete(value, size * 8);
        if self.inner.set_register(name, bv) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "failed to set register: {name}"
            )))
        }
    }

    /// Set multiple registers in a single FFI call.
    /// Takes a dict of {name: value} pairs.
    ///
    /// **Invariant I5 (cross-mixin):** the caller — Python's
    /// `rust_state_sync.py` — must pre-filter the dict to
    /// `_supported_register_names` for the active architecture. The disk
    /// init cache pickles ALL `arch.register_names.values()` (cr0..8,
    /// ymm0..15, fs_seg, ds_seg, cmstart, cmlen, fpreg, ...), but only a
    /// subset has a slot in `RegisterFile` for amd64/x86/arm/etc. If an
    /// unsupported name leaks through, this method returns
    /// `PyValueError("unknown register: <name>")` rather than silently
    /// dropping the write. Do NOT "fix" by extending `arch/amd64.rs`
    /// unless the interpreter actually consumes the new register. See
    /// module-level invariant I5 in this file.
    pub fn set_registers_bulk(&mut self, registers: &Bound<'_, PyDict>) -> PyResult<()> {
        for (key, val) in registers.iter() {
            let name: String = key.extract()?;
            let value: u128 = val.extract()?;
            let size = self
                .inner
                .arch()
                .register_size(&name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {name}")))?;
            // I5 cross-check: register_size returning Some implies the
            // register has a RegisterFile slot. This debug assert documents
            // intent and would catch a regression where arch lookup and
            // RegisterFile membership drift apart.
            #[cfg(debug_assertions)]
            debug_assert!(
                size > 0,
                "I5: register {name} has zero size — arch table is malformed"
            );
            let bv = RustBV::concrete(value, size * 8);
            // Propagate a slotless-register failure the same way scalar
            // set_register does. register_size returning Some does not
            // guarantee a RegisterFile slot (the I5 size-known-but-slotless
            // drift case the debug_assert worries about), so a false return
            // here is a silent dropped write unless we surface it.
            if !self.inner.set_register(&name, bv) {
                return Err(PyValueError::new_err(format!(
                    "failed to set register: {name}"
                )));
            }
        }
        Ok(())
    }

    /// Get all register bytes.
    pub fn get_registers_raw(&self) -> Vec<u8> {
        self.inner.get_registers_raw()
    }

    /// Set all register bytes.
    pub fn set_registers_raw(&mut self, bytes: &[u8]) {
        self.inner.set_registers_raw(bytes);
    }

    /// Set a register to a symbolic value from a raw Z3 AST pointer.
    ///
    /// The Z3 AST must be a BitVec in the shared Z3 context.
    /// Used to import symbolic register values (e.g., BVS in rax) from Python.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_register_symbolic(
        &mut self,
        name: &str,
        z3_ast_ptr: usize,
        width: u32,
    ) -> PyResult<()> {
        use z3::ast::Ast;
        let size = self
            .inner
            .arch()
            .register_size(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {name}")))?;
        if width != size * 8 {
            return Err(PyValueError::new_err(format!(
                "width mismatch: register {} is {} bits, got {} bits",
                name,
                size * 8,
                width
            )));
        }
        // Reconstruct z3::ast::BV from raw pointer.
        // Reject a null pointer explicitly — this is a `#[pymethods]` entry
        // callable from Python with an arbitrary integer, so a 0 (or otherwise
        // absent) AST must surface as a `ValueError`, not `NonNull::new_unchecked`
        // UB / a Z3_inc_ref segfault on a bogus pointer.
        let raw = std::ptr::NonNull::new(z3_ast_ptr as *mut _).ok_or_else(|| {
            PyValueError::new_err(format!(
                "set_register_symbolic: null Z3 AST pointer for register {name}"
            ))
        })?;
        // SAFETY: `raw` is non-null (checked above); caller guarantees it is a
        // BV-sorted `Z3_ast` in the active thread-local context (z3-rs 0.19+
        // shares the process-global context with claripy's z3 backend).
        // `BV::wrap` takes its own ref. `width` was validated above to match the
        // register's bit width — i.e. the AST's BV sort width. A non-null but
        // otherwise garbage pointer remains the caller's responsibility.
        let z3_bv = unsafe {
            let ctx = z3::Context::thread_local();
            z3::ast::BV::wrap(&ctx, raw)
        };
        // Allocate a fresh, globally-unique symbol id rather than the constant
        // `0`. `NEXT_SYMBOL_ID` starts at 0, so the first symbol minted anywhere
        // in the process legitimately owns id 0; hardcoding `0` here made this
        // wrapped register alias whatever AST is registered under id 0 (a
        // cross-symbol identity collision, not merely the documented loss of
        // leaf-symbol identity — angr-ph300.50). A fresh id from the same global
        // allocator every other minted symbol uses keeps this register's export
        // independent of any unrelated leaf.
        let id = self.inner.solver().borrow().next_id();
        let bv = RustBV::Symbolic {
            id,
            ast: z3_bv,
            width,
            name: Arc::from(name),
        };
        if self.inner.set_register(name, bv) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "failed to set register: {name}"
            )))
        }
    }

    /// Set a register to a symbolic value from a full claripy AST.
    ///
    /// Unlike [`set_register_symbolic`] (which wraps a raw Z3 pointer as an
    /// opaque `RustBV::Symbolic` with a freshly-minted id and so loses
    /// leaf-symbol identity on export), this routes the claripy AST through
    /// `claripy_to_rustbv`. That interns every leaf BVS into the shared
    /// claripy<->Rust symbol cache, so a subregister set like
    /// `state.regs.ecx = BVS('ecx', 32)` round-trips back to the user's
    /// original symbol on export instead of minting a fresh `rcx_N`. This is
    /// the Layer 2 fix for angr-4ju9e / angr-21vi5 — it mirrors the
    /// identity-preserving import path that memory-sourced symbols already use
    /// (`import_symbolic_to_state`).
    pub fn set_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let size = self
            .inner
            .arch()
            .register_size(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {name}")))?;
        let solver = self.inner.solver().clone();
        let bv = {
            let ctx = solver.borrow();
            crate::claripy_bridge::claripy_to_rustbv(py, ast, &ctx)
                .map_err(|e| PyValueError::new_err(format!("AST conversion: {e}")))?
        };
        if bv.width() != size * 8 {
            return Err(PyValueError::new_err(format!(
                "width mismatch: register {} is {} bits, got {} bits",
                name,
                size * 8,
                bv.width()
            )));
        }
        if self.inner.set_register(name, bv) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "failed to set register: {name}"
            )))
        }
    }

    /// Map a memory region.
    #[pyo3(signature = (addr, size, permissions=7))]
    pub fn map_memory(&mut self, addr: u64, size: u64, permissions: u8) {
        self.inner
            .map_memory(addr, size, Permission::from_bits(permissions));
    }

    /// Map memory with initial data.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn map_memory_data(&mut self, addr: u64, data: &[u8], permissions: u8) {
        self.inner
            .map_memory_data(addr, data, Permission::from_bits(permissions));
    }

    /// Map multiple memory pages in a single FFI call.
    /// pages is a list of (addr, data, permissions) tuples.
    pub fn map_memory_batch(&mut self, pages: Vec<(u64, Vec<u8>, u8)>) {
        for (addr, data, permissions) in pages {
            self.inner
                .map_memory_data(addr, &data, Permission::from_bits(permissions));
        }
    }

    /// Enable or disable strict memory permission enforcement on load/store.
    /// Mirrors angr's STRICT_PAGE_ACCESS option. Default off.
    #[pyo3(name = "set_enforce_permissions")]
    pub fn py_set_enforce_permissions(&mut self, enabled: bool) {
        self.inner.set_enforce_permissions(enabled);
    }

    /// Whether strict memory permission enforcement is enabled.
    #[pyo3(name = "enforce_permissions")]
    pub fn py_enforce_permissions(&self) -> bool {
        self.inner.enforce_permissions()
    }

    /// Enable or disable non-executable page enforcement on instruction fetch.
    /// Mirrors angr's ENABLE_NX option. The X check fires only when this and
    /// `enforce_permissions` are both on. Default off.
    #[pyo3(name = "set_enforce_nx")]
    pub fn py_set_enforce_nx(&mut self, enabled: bool) {
        self.inner.set_enforce_nx(enabled);
    }

    /// Whether non-executable page enforcement is enabled.
    #[pyo3(name = "enforce_nx")]
    pub fn py_enforce_nx(&self) -> bool {
        self.inner.enforce_nx()
    }

    /// Suppress IP concretization for symbolic jump targets.
    /// Mirrors angr's NO_IP_CONCRETIZATION option. When set, a symbolic IP
    /// at block boundary routes the state to the unconstrained stash
    /// silently instead of being enumerated. Default off.
    #[pyo3(name = "set_no_ip_concretization")]
    pub fn py_set_no_ip_concretization(&mut self, enabled: bool) {
        self.inner.set_no_ip_concretization(enabled);
    }

    /// Whether NO_IP_CONCRETIZATION is active on this state.
    #[pyo3(name = "no_ip_concretization")]
    pub fn py_no_ip_concretization(&self) -> bool {
        self.inner.no_ip_concretization()
    }

    /// Suppress resolution of symbolic jump targets.
    /// Mirrors angr's NO_SYMBOLIC_JUMP_RESOLUTION option. When set, a symbolic
    /// jump target routes the state to the unconstrained stash before
    /// AddressConcretizer enumeration is attempted. Default off.
    #[pyo3(name = "set_no_symbolic_jump_resolution")]
    pub fn py_set_no_symbolic_jump_resolution(&mut self, enabled: bool) {
        self.inner.set_no_symbolic_jump_resolution(enabled);
    }

    /// Whether NO_SYMBOLIC_JUMP_RESOLUTION is active on this state.
    #[pyo3(name = "no_symbolic_jump_resolution")]
    pub fn py_no_symbolic_jump_resolution(&self) -> bool {
        self.inner.no_symbolic_jump_resolution()
    }

    /// Preserve the symbolic IP across block boundaries.
    /// Mirrors angr's KEEP_IP_SYMBOLIC option. When set, the engine still
    /// concretizes the next pc, but each fork's IP register is left holding
    /// the original symbolic expression and no `target == addr` narrowing
    /// constraint is added. Default off.
    #[pyo3(name = "set_keep_ip_symbolic")]
    pub fn py_set_keep_ip_symbolic(&mut self, enabled: bool) {
        self.inner.set_keep_ip_symbolic(enabled);
    }

    /// Whether KEEP_IP_SYMBOLIC is active on this state.
    #[pyo3(name = "keep_ip_symbolic")]
    pub fn py_keep_ip_symbolic(&self) -> bool {
        self.inner.keep_ip_symbolic()
    }

    /// Mirror a symex-relevant SimOption onto this state so native
    /// SimProcedures can branch on it (angr-kzjv6). `name` is the angr option
    /// string (e.g. `"SHORT_READS"`); `enabled=False` removes it. Wired from
    /// `_add_rust_state` for the small symex-relevant option subset.
    #[pyo3(name = "set_option")]
    pub fn py_set_option(&mut self, name: &str, enabled: bool) {
        self.inner.set_option(name, enabled);
    }

    /// Whether the named SimOption is active on this state.
    #[pyo3(name = "has_option")]
    pub fn py_has_option(&self, name: &str) -> bool {
        self.inner.has_option(name)
    }

    /// Opt this state's solver context in to fork-time
    /// `SharedLineageSolver` materialization (angr-3ms1 step 1b).
    ///
    /// Inherited by every descendant via `fork()` so a single call on a
    /// seed state suffices. Default off — the slice-1c materialization
    /// gate stays inert on plain `RustExplorationManager` runs to keep
    /// the v5a5-slice-4c.3-retry-failed-bfs-thrash-fundamental
    /// regression (defcon2016quals_baby-re ~10x slowdown under default
    /// BFS) out of CI.
    #[cfg(feature = "vex-engine-z3")]
    #[pyo3(name = "set_use_shared_lineage_solver")]
    pub fn py_set_use_shared_lineage_solver(&self, enabled: bool) {
        self.inner.set_use_shared_lineage_solver(enabled);
    }

    /// Whether fork-time `SharedLineageSolver` materialization is opted
    /// in on this state (angr-3ms1 step 1b).
    #[cfg(feature = "vex-engine-z3")]
    #[pyo3(name = "use_shared_lineage_solver")]
    pub fn py_use_shared_lineage_solver(&self) -> bool {
        self.inner.use_shared_lineage_solver()
    }

    /// Load from memory.
    ///
    /// angr-ph300.53: split into 16-byte sub-loads mirroring `memory_store`.
    /// `RustBV::Concrete` is u128-backed so `as_u128()` caps at 16 bytes; a
    /// single load of a wider (but fully concrete) region would otherwise error
    /// as "symbolic" even though every byte is concrete. Loading in 16-byte
    /// chunks keeps each `as_u128()` within range and reassembles the bytes.
    pub fn memory_load(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        let mut bytes = Vec::with_capacity(size as usize);
        let mut offset = 0u32;
        while offset < size {
            let chunk_size = (size - offset).min(16);
            let bv = self
                .inner
                .memory_load(addr + offset as u64, chunk_size)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let value = bv.as_u128().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "memory_load at 0x{:x} returned a symbolic value; cannot convert to concrete bytes",
                    addr + offset as u64
                ))
            })?;
            for i in 0..chunk_size as usize {
                bytes.push((value >> (i * 8)) as u8);
            }
            offset += chunk_size;
        }
        Ok(bytes)
    }

    /// Store to memory.
    ///
    /// angr-5aj8: split into 16-byte chunks because RustBV::Concrete is
    /// backed by a u128. Packing >16 bytes into a single concrete BV would
    /// shift-overflow during construction and then make store_concrete emit
    /// a 16-byte-cycle pattern over the full claimed width.
    pub fn memory_store(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        let mut offset = 0usize;
        while offset < data.len() {
            let remaining = data.len() - offset;
            let chunk_size = remaining.min(16);
            let chunk = &data[offset..offset + chunk_size];
            let width = (chunk_size * 8) as u32;
            let mut value: u128 = 0;
            for (i, &b) in chunk.iter().enumerate() {
                value |= (b as u128) << (i * 8);
            }
            let bv = RustBV::concrete(value, width);
            self.inner
                .memory_store(addr + offset as u64, bv)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            offset += chunk_size;
        }
        Ok(())
    }

    /// Behavioral probe for AVOID_MULTIVALUED_READS (angr-vkkny).
    ///
    /// Builds a structurally-symbolic 64-bit address pinned (via a solver
    /// constraint) to `addr`, performs a symbolic-address load through the
    /// same `memory_load_symbolic` path the interpreter uses, and returns the
    /// `(min, max)` solver bounds of the loaded value. The address stays a BVS
    /// (`as_u64()` is None) so `AddressConcretizer::should_avoid_multivalued_read`
    /// fires when the option is set, even though the constraint pins it to a
    /// single concrete location.
    ///
    /// Contract:
    ///   * `avoid_multivalued_reads` ON  -> the load returns a fresh
    ///     UNCONSTRAINED value, so `max > min` (NOT pinned to the concrete
    ///     backer byte at `addr`).
    ///   * `avoid_multivalued_reads` OFF -> the address concretizes to `addr`
    ///     and the load reads the backer, so `min == max`.
    ///
    /// Deleting the `should_avoid_multivalued_read` branch in
    /// `memory/load.rs::load_symbolic_unified` collapses the ON case to
    /// `min == max` — exactly the regression this surface lets a Python test
    /// catch. Returns `(min, max)` as a u128 pair (sufficient for loads up to
    /// 8 bytes wide).
    pub fn probe_symbolic_load_value_range(
        &mut self,
        addr: u64,
        size: u32,
    ) -> PyResult<(u128, u128)> {
        let sym_addr = {
            let ctx = self.inner.solver().borrow();
            let a = RustBV::symbolic(&ctx, "avoid_mv_probe_addr", 64);
            let eq = a.eq(&RustBV::concrete(addr as u128, 64), &ctx);
            ctx.assume_true(&eq);
            a
        };
        let loaded = self
            .inner
            .memory_load_symbolic(sym_addr, size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let lo = self
            .inner
            .min(&loaded, false)
            .ok_or_else(|| PyValueError::new_err("min: unsatisfiable"))?;
        let hi = self
            .inner
            .max(&loaded, false)
            .ok_or_else(|| PyValueError::new_err("max: unsatisfiable"))?;
        Ok((lo, hi))
    }

    /// Add a lazy region.
    pub fn add_lazy_region(&mut self, start_addr: u64, size: u64) {
        self.inner.add_lazy_region(start_addr, size);
    }

    /// Add multiple lazy regions in a single FFI call.
    /// `regions` is a list of (start_addr, size) tuples. Used by
    /// `RustStateSyncMixin._sync_extra_python_pages` to register thousands of
    /// per-page lazy entries (mma_howtouse / Callable workflow) in one
    /// crossing instead of one FFI per page.
    pub fn add_lazy_regions_batch(&mut self, regions: Vec<(u64, u64)>) {
        for (start_addr, size) in regions {
            self.inner.add_lazy_region(start_addr, size);
        }
    }

    /// Number of registered lazy regions (one entry per `add_lazy_region`
    /// call). Exposed so tests can confirm a batch registration actually
    /// landed regions rather than silently no-op'ing.
    pub fn lazy_region_count(&self) -> usize {
        self.inner.memory().lazy_region_count()
    }

    /// Whether a byte address falls inside any registered lazy region.
    pub fn is_in_lazy_region(&self, addr: u64) -> bool {
        self.inner.memory().is_addr_in_lazy_region(addr)
    }

    /// Get dirty page addresses.
    pub fn get_dirty_pages(&self) -> Vec<u64> {
        self.inner.get_dirty_pages()
    }

    /// Clear dirty page tracking.
    pub fn clear_dirty_pages(&mut self) {
        self.inner.clear_dirty_pages();
    }

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        self.inner.add_hook(addr);
    }

    /// Remove a hook address.
    pub fn remove_hook(&mut self, addr: u64) {
        self.inner.remove_hook(addr);
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.inner.is_hooked(addr)
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        self.inner.clear_hooks();
    }

    /// Check if constraints are satisfiable.
    pub fn satisfiable(&self) -> bool {
        self.inner.satisfiable()
    }

    /// Fork the state (O(1) CoW).
    pub fn fork(&self) -> Self {
        PyRustSimState {
            inner: self.inner.fork(),
        }
    }

    /// Configure address concretization (legacy interface).
    #[pyo3(signature = (use_approximate, range_limit=None))]
    pub fn configure_concretization(&mut self, use_approximate: bool, range_limit: Option<u64>) {
        self.inner
            .configure_concretization(use_approximate, range_limit);
    }

    /// Configure address concretization with full strategy configuration.
    #[pyo3(signature = (use_approximate, read_range_limit=None, write_range_limit=None, symbolic_write_addresses=false, avoid_multivalued_reads=false, avoid_multivalued_writes=false))]
    pub fn configure_concretization_strategies(
        &mut self,
        use_approximate: bool,
        read_range_limit: Option<u64>,
        write_range_limit: Option<u64>,
        symbolic_write_addresses: bool,
        avoid_multivalued_reads: bool,
        avoid_multivalued_writes: bool,
    ) {
        self.inner.concretizer.configure_strategies(
            use_approximate,
            read_range_limit,
            write_range_limit,
            symbolic_write_addresses,
            avoid_multivalued_reads,
            avoid_multivalued_writes,
        );
    }

    /// Set whether to track history.
    pub fn set_track_history(&mut self, track: bool) {
        self.inner.set_track_history(track);
    }

    /// Set maximum history length.
    pub fn set_max_history(&mut self, max: usize) {
        self.inner.set_max_history(max);
    }

    /// Get the current maximum history length (0 = unlimited).
    pub fn get_max_history(&self) -> usize {
        self.inner.max_history
    }

    /// Get the detailed execution history as `(addr, jumpkind, jump_target)` tuples.
    pub fn detailed_history(&self) -> Vec<(u64, u8, u64)> {
        self.inner
            .detailed_history()
            .iter()
            .map(|h| (h.addr, h.jumpkind, h.jump_target))
            .collect()
    }

    /// Append a basic-block address to `history` (honors the cap).
    /// Test/debug helper — production paths go through the interpreter.
    pub fn add_history(&mut self, addr: u64) {
        self.inner.add_to_history(addr);
    }

    /// Append a detailed history entry (honors the cap).
    /// Test/debug helper — production paths go through the interpreter.
    pub fn add_detailed_history(&mut self, addr: u64, jumpkind: u8, jump_target: u64) {
        self.inner.add_history_entry(addr, jumpkind, jump_target);
    }

    /// Push a call frame onto the call stack (Ijk_Call simulation).
    /// Test/debug helper — production paths go through the interpreter.
    pub fn push_call_frame(&mut self, call_site: u64, callee: u64, ret_addr: u64, sp: u64) {
        self.inner.push_call(call_site, callee, ret_addr, sp);
    }

    /// Get the call stack as a list of (call_site, callee, ret_addr, sp) tuples,
    /// in push order (outermost first, innermost last). Mirrors
    /// `ExplorationStateSnapshot.get_call_stack()` for direct PyRustSimState
    /// inspection without an export round-trip.
    pub fn get_call_stack(&self) -> Vec<(u64, u64, u64, u64)> {
        self.inner
            .call_stack()
            .iter()
            .map(|e| (e.call_site_addr, e.callee_addr, e.return_addr, e.stack_ptr))
            .collect()
    }

    /// Export the complete state as a snapshot.
    pub fn export_full(&self) -> ExplorationStateSnapshot {
        self.inner.export_full()
    }
}
