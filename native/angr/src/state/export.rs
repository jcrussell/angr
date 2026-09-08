//! Python export snapshot (`ExplorationStateSnapshot`).
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

// =============================================================================
// State Snapshot for Exploration Export
// =============================================================================

/// Internal memory page data (addr, data, permissions, symbolic_offsets).
type PageData = (u64, Vec<u8>, u8, Vec<u16>);

/// Complete state snapshot for exploration export.
///
/// This contains all information needed to reconstruct an angr SimState
/// from a Rust execution state.
#[pyclass(name = "ExplorationStateSnapshot")]
pub struct ExplorationStateSnapshot {
    /// Unique state identifier.
    #[pyo3(get)]
    pub state_id: u64,
    /// Parent state ID (for fork tracking).
    #[pyo3(get)]
    pub parent_id: Option<u64>,
    /// Program counter.
    #[pyo3(get)]
    pub pc: u64,
    /// Architecture name.
    #[pyo3(get)]
    pub arch_name: String,
    /// Raw register bytes.
    registers_raw: Vec<u8>,
    /// Memory pages: (addr, data, permissions, symbolic_offsets).
    memory_pages: Vec<PageData>,
    /// Number of constraints in the solver.
    #[pyo3(get)]
    pub constraint_count: usize,
    /// Basic block history.
    history: Vec<u64>,
    /// Named register values: (name, concrete_value, size_bits).
    /// Pre-computed at export time so Python doesn't need offset tables.
    named_registers: Vec<(String, u128, u32)>,
    /// Names of registers that hold a SYMBOLIC value (and so were skipped
    /// from `named_registers`). Collected for ~free in the same export loop.
    /// Python uses this to attach a lazy register proxy on the plain-export
    /// path so a Rust-computed symbolic register (e.g. a clz/ctz result) is
    /// recovered on demand instead of being silently dropped (angr-4ju9e).
    symbolic_register_names: Vec<String>,
    /// Call stack entries: (call_site_addr, callee_addr, return_addr, stack_ptr).
    call_stack: Vec<(u64, u64, u64, u64)>,
    /// Detailed execution history: (addr, jumpkind, jump_target).
    detailed_history: Vec<(u64, u8, u64)>,
    /// Heap allocations: (addr, size) for active allocations.
    heap_allocated: Vec<(u64, u64)>,
    /// Heap freed addresses.
    heap_freed: Vec<u64>,
    /// Open file descriptors: (fd, name, position, flags, content_len, is_open).
    open_fds: Vec<(u32, String, u64, u32, usize, bool)>,
    /// Inspection event counts per type.
    inspection_counts: Vec<(String, u64)>,
    /// Inspection enabled bitmask.
    #[pyo3(get)]
    pub inspection_enabled: u8,
}

// ExplorationStateSnapshot pymethod surface (angr-hv4lt.11).
//
// Two tiers, kept intentionally so future contributors don't assume the whole
// surface is load-bearing:
//
// * LOAD-BEARING — consumed by the Python bridge in rust_state_export.py's
//   `_snapshot_to_angr` reconstruction path: `get_registers_named` and
//   `get_symbolic_register_names` (register writeback + lazy symbolic-register
//   proxy, angr-4ju9e) and `page_count` / `get_page` (memory-page writeback).
//   Removing or renaming any of these breaks materialization — sweep
//   rust_state_export.py in the same change.
//
// * INTROSPECTION / DEBUG-ONLY — no current Python caller (verified by grep
//   over angr/ at audit time): `get_call_stack`, `get_call_stack_depth`,
//   `get_detailed_history`, `get_detailed_history_str`, `get_heap_allocated`,
//   `get_heap_freed`, `get_heap_alloc_count`, `get_heap_free_count`,
//   `get_open_fds`, `get_fd_count`, `get_inspection_counts`,
//   `get_symbolic_offsets`, `memory_load`, `page_addresses`, `get_history`,
//   `get_registers_raw` (angr-9ke6b.5 re-audit: the Python bridge reconstructs
//   registers from `get_registers_named` + `get_symbolic_register_names`; the
//   flat buffer is only read by the Rust/pytest arch-offset parity tests).
//   The Python sync helpers (`_sync_rust_callstack_to_state`,
//   `_sync_rust_heap_brk_to_state`, `_sync_state_posix_fds_to_rust`, ...) read
//   the LIVE `rust_mgr.get_state_*` accessors instead of a materialized
//   snapshot, so these duplicate that data at snapshot granularity. Retained
//   as a stable Rust-side introspection API (exercised by `state/tests/export.rs`,
//   angr-n0irt.17) — not dead, just not on the materialization hot path.
#[pymethods]
impl ExplorationStateSnapshot {
    /// Get raw register bytes.
    pub fn get_registers_raw(&self) -> Vec<u8> {
        self.registers_raw.clone()
    }

    /// Get named register values as a dict: {name: (value, size_bits)}.
    ///
    /// Pre-computed at export time using Rust's register tables,
    /// so Python doesn't need architecture-specific offset mapping.
    pub fn get_registers_named(&self) -> std::collections::HashMap<String, (u128, u32)> {
        self.named_registers
            .iter()
            .map(|(name, value, bits)| (name.clone(), (*value, *bits)))
            .collect()
    }

    /// Get the names of registers that hold a symbolic value (skipped from
    /// `get_registers_named`). Python attaches a lazy register proxy when this
    /// is non-empty so symbolic registers are recovered on demand (angr-4ju9e).
    pub fn get_symbolic_register_names(&self) -> Vec<String> {
        self.symbolic_register_names.clone()
    }

    /// Get history (basic block addresses visited).
    pub fn get_history(&self) -> Vec<u64> {
        self.history.clone()
    }

    /// Get call stack as list of (call_site_addr, callee_addr, return_addr, stack_ptr) tuples.
    pub fn get_call_stack(&self) -> Vec<(u64, u64, u64, u64)> {
        self.call_stack.clone()
    }

    /// Get call stack depth.
    pub fn get_call_stack_depth(&self) -> usize {
        self.call_stack.len()
    }

    /// Get detailed history as list of (addr, jumpkind, jump_target) tuples.
    ///
    /// jumpkind: 0=Boring, 1=Call, 2=Ret, 3=Syscall, 4=Other
    pub fn get_detailed_history(&self) -> Vec<(u64, u8, u64)> {
        self.detailed_history.clone()
    }

    /// Get detailed history with string jumpkinds.
    ///
    /// Returns list of (addr, jumpkind_str, jump_target) tuples.
    pub fn get_detailed_history_str(&self) -> Vec<(u64, String, u64)> {
        self.detailed_history
            .iter()
            .map(|(addr, jk, target)| (*addr, HistoryEntry::jumpkind_str(*jk).to_string(), *target))
            .collect()
    }

    /// Get the number of memory pages.
    pub fn page_count(&self) -> usize {
        self.memory_pages.len()
    }

    /// Get a memory page by index.
    /// Returns (addr, data, permissions, symbolic_offsets) or None.
    pub fn get_page(&self, index: usize) -> Option<(u64, Vec<u8>, u8, Vec<u16>)> {
        self.memory_pages
            .get(index)
            .map(|p| (p.0, p.1.clone(), p.2, p.3.clone()))
    }

    /// Get all memory page addresses.
    pub fn page_addresses(&self) -> Vec<u64> {
        self.memory_pages.iter().map(|p| p.0).collect()
    }

    /// Load bytes from memory at a given address.
    /// Returns None if the address is not mapped, or if `size` is so large
    /// that `offset + size` cannot be represented (`size` arrives straight
    /// from Python across `#[pymethods]`, so it is untrusted: a bare `+`
    /// would wrap and let an out-of-range request pass the `<= len` bound
    /// check and return a short, wrong slice).
    pub fn memory_load(&self, addr: u64, size: usize) -> Option<Vec<u8>> {
        let page_addr = addr & !0xFFF;
        let offset = (addr & 0xFFF) as usize;
        let end = offset.checked_add(size)?;

        // Find the page
        for page in &self.memory_pages {
            if page.0 == page_addr && end <= page.1.len() {
                return Some(page.1[offset..end].to_vec());
            }
        }
        None
    }

    /// Get heap allocations as list of (addr, size) tuples.
    pub fn get_heap_allocated(&self) -> Vec<(u64, u64)> {
        self.heap_allocated.clone()
    }

    /// Get heap freed addresses.
    pub fn get_heap_freed(&self) -> Vec<u64> {
        self.heap_freed.clone()
    }

    /// Get number of active heap allocations.
    pub fn get_heap_alloc_count(&self) -> usize {
        self.heap_allocated.len()
    }

    /// Get number of distinct freed heap addresses (`HeapMetadata::freed` is a
    /// set — a pointer freed twice counts once).
    pub fn get_heap_free_count(&self) -> usize {
        self.heap_freed.len()
    }

    /// Get open file descriptors as list of (fd, name, position, flags, content_len, is_open).
    pub fn get_open_fds(&self) -> Vec<(u32, String, u64, u32, usize, bool)> {
        self.open_fds.clone()
    }

    /// Get the number of tracked file descriptors.
    pub fn get_fd_count(&self) -> usize {
        self.open_fds.len()
    }

    /// Get inspection event counts as list of (event_name, count) tuples.
    ///
    /// On a state produced by `RustSimState::merge` these are the counts of
    /// the *first* branch only — the other branches' tallies are dropped, not
    /// summed (angr-sqfj8.89; see the `warn_config_divergence("inspection
    /// ...")` block in `fork.rs::merge`).
    pub fn get_inspection_counts(&self) -> Vec<(String, u64)> {
        self.inspection_counts.clone()
    }

    /// Get symbolic byte offsets for a page.
    /// Returns empty vec if page not found.
    pub fn get_symbolic_offsets(&self, page_addr: u64) -> Vec<u16> {
        for page in &self.memory_pages {
            if page.0 == page_addr {
                return page.3.clone();
            }
        }
        Vec::new()
    }
}

impl RustSimState {
    /// Export the complete state as a snapshot.
    ///
    /// This creates a self-contained snapshot that can be used to
    /// reconstruct an angr SimState.
    ///
    /// Does **not** flush deferred symbolic stores — see the note at the
    /// memory-page export below. Prefer [`Self::flush_and_export_full`]
    /// unless the pending queue is known to be empty.
    pub fn export_full(&self) -> ExplorationStateSnapshot {
        // Export registers
        let registers_raw = self.get_registers_raw();

        // Export named registers: read each GP register by name
        let mut named_registers = Vec::new();
        let mut symbolic_register_names = Vec::new();
        let ctx = self.solver.borrow();
        for &name in self.arch.register_names() {
            if let Some(size) = self.arch.register_size(name) {
                let bv = self.registers.get_reg(name, &ctx);
                if let Some(bv) = bv {
                    if let Some(val) = bv.as_u128() {
                        named_registers.push((name.to_string(), val, size * 8));
                    } else {
                        // Symbolic register: record the name so Python can
                        // recover the AST on demand (angr-4ju9e). Recovering
                        // the full AST here for every export would blow up the
                        // hot path, so we only export the cheap name list.
                        symbolic_register_names.push(name.to_string());
                    }
                }
            }
        }

        // Pending writes are NOT flushed before the memory pages below:
        // `SymbolicMemory::flush_pending_writes` needs `&mut`, and this method
        // takes `&self`. Nor does the snapshot advertise the shortfall —
        // `ExplorationStateSnapshot` has no pending-write count, so a deferred
        // store that never got materialized is simply missing from the
        // exported pages with nothing to signal it.
        //
        // The mitigation is therefore at the call sites, not here: every
        // production export path goes through `flush_and_export_full`
        // instead — `state::pymethods::PyRustSimState::export_full`,
        // `exploration::state_api`, `exploration::pending_api` (angr-sqfj8.27)
        // and `exploration::stepping`. Call `export_full` directly only when
        // the pending queue is known to be empty.

        // Export memory pages as tuples: (addr, data, permissions, symbolic_offsets)
        let mut memory_pages: Vec<PageData> = Vec::new();
        for (page_num, page) in self.memory.pages().iter() {
            let page_addr = crate::memory::PageIndex::from_raw(*page_num).base_addr();
            let data = page.load_concrete(0, crate::memory::PAGE_SIZE as u16);
            let permissions = page.permissions().to_bits();
            let symbolic_offsets = page.symbolic_offsets();

            memory_pages.push((page_addr, data, permissions, symbolic_offsets));
        }

        // Get constraint count
        let constraint_count = self.solver.borrow().num_constraints();

        // Export call stack
        let call_stack: Vec<(u64, u64, u64, u64)> = self
            .call_stack
            .iter()
            .map(|e| (e.call_site_addr, e.callee_addr, e.return_addr, e.stack_ptr))
            .collect();

        // Export detailed history
        let detailed_history: Vec<(u64, u8, u64)> = self
            .detailed_history
            .iter()
            .map(|e| (e.addr, e.jumpkind, e.jump_target))
            .collect();

        ExplorationStateSnapshot {
            state_id: self.state_id,
            parent_id: self.parent_id,
            pc: self.pc,
            arch_name: self.arch.name().to_string(),
            registers_raw,
            memory_pages,
            constraint_count,
            history: self.history.iter().copied().collect(),
            named_registers,
            symbolic_register_names,
            call_stack,
            detailed_history,
            heap_allocated: self
                .heap_metadata
                .allocated
                .iter()
                .map(|(&addr, &size)| (addr, size))
                .collect(),
            heap_freed: self.heap_metadata.freed.clone(),
            open_fds: self
                .fs
                .all_fds()
                .iter()
                .filter_map(|&fd| {
                    let info = self.fs.fd_info(fd)?;
                    Some((fd, info.0.to_string(), info.1, info.2, info.3, info.4))
                })
                .collect(),
            inspection_counts: self
                .inspection
                .event_counts()
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 0)
                .filter_map(|(i, &count)| {
                    InspectEvent::from_u8(i as u8).map(|e| (e.name().to_string(), count))
                })
                .collect(),
            inspection_enabled: self.inspection.enabled_mask(),
        }
    }

    /// Flush pending writes and then export.
    /// This materializes any deferred symbolic stores before creating the snapshot.
    pub fn flush_and_export_full(&mut self) -> ExplorationStateSnapshot {
        // Flush pending writes using the current solver context
        {
            let ctx = self.solver.borrow();
            // SILENT(cat-b): export has no way to fetch a lazy page or retry a
            // failed materialization, so a flush error costs the rest of the
            // queue (`flush_pending_writes` already took it). Log it rather
            // than exporting a silently-incomplete snapshot without a trace
            // (angr-9ke6b.100).
            if let Err(e) = self.memory.flush_pending_writes(&ctx, &self.concretizer) {
                log::warn!(
                    "flush_and_export_full: pending-write flush failed ({e:?}); \
                     remaining deferred stores are not reflected in the exported snapshot"
                );
            }
        }
        self.export_full()
    }
}
