//! The load-resolution ladder — `IRExpr` load evaluation and its
//! Python-callback fallbacks.
//!
//! Split out of `expressions.rs` (angr-fs8kb.66). `eval_load` is the entry
//! point; `load_layered` picks between the native `rust_memory` fast path
//! (`try_rust_memory_load`), the layered pending-store buffer
//! (`load_layered_at`, `covering_range`) and the Python callback, depending on
//! how concrete the address is (`load_concrete_addr`,
//! `load_concrete_addr_below_pending`, `load_symbolic_addr`,
//! `symbolic_overlap_load`, `symbolic_store_load`). The multi-address /
//! ITE-chain callback helpers (`dispatch_multi_load`,
//! `build_ite_load_from_callbacks`, `build_ite_store_from_callbacks`,
//! `convert_load_result`, `try_convert_symbolic_value`) and the `LoadG`
//! guarded-load pair (`resolve_loadg_load`, `apply_loadg_conversion`) live here
//! too. The op dispatchers are in `expressions_arith.rs`; the read-side
//! `state.inspect` dispatchers in `expressions_inspect.rs`.

use super::bv_utils::{build_ite_chain, bytes_to_bv, splice_bytes_over_bv};
use super::pending_store::covering_range;
use super::*;
use rustc_hash::FxHashMap;

impl<'a> VEXInterpreter<'a> {
    pub(super) fn eval_load(
        &mut self,
        callbacks: &PythonCallbacks,
        addr: &IRExpr,
        ty: IRType,
        endness: Endness,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        let load_start = profile_start!(self);
        let addr_val = self.eval_expr_with_callbacks(callbacks, addr, tyenv)?;
        let size = ty.bytes() as usize;
        if self.profiling_enabled {
            self.stats.load_stmt_count += 1;
        }

        let value = self.load_layered(callbacks, &addr_val, size, load_start)?;

        if let Some(injected) =
            self.dispatch_mem_read_inspect(callbacks, &addr_val, &value, size, endness)
        {
            return Ok(injected);
        }
        Ok(value)
    }

    /// Find a buffered symbolic store in `map` whose byte range fully covers
    /// `[addr, addr + size)` and extract the covered bytes. Handles offset loads
    /// into a wider symbolic store that the exact-address hash lookups miss
    /// (angr-ofyh). Mirrors the low-bit extraction convention of those lookups:
    /// byte `k` of the stored value is bits `[k*8, k*8+8)`, so a load at offset
    /// `o = addr - s_addr` returns bits `[o*8, (o+size)*8)`. Short-circuits when
    /// no symbolic stores are buffered so the common all-concrete load pays
    /// nothing.
    ///
    /// Coverage is decided by `pending_store::covering_range` — the same
    /// offset-space test the concrete pending-store buffer uses — so a
    /// symbolic store that wraps past the top of the guest address space
    /// still serves the loads in its wrapped tail (angr-fs8kb.64). Forming
    /// either end address instead, as this did with `checked_add` /
    /// `saturating_add`, turned such a store into a silent miss.
    pub(super) fn symbolic_overlap_load(
        &self,
        map: &FxHashMap<u64, RustBV>,
        addr: u64,
        size: usize,
    ) -> Option<RustBV> {
        if map.is_empty() {
            return None;
        }
        for (&s_addr, bv) in map.iter() {
            let Some(range) = covering_range(s_addr, (bv.width() / 8) as usize, addr, size) else {
                continue;
            };
            // overflow-ok: `covering_range` bounds `range.end` by the store's
            // byte width, so both bit positions stay inside `bv.width()`, and
            // a zero-size load cannot reach here (`range.end > range.start`
            // would fail, and callers never pass 0).
            let off_bits = (range.start * 8) as u32;
            let hi_bit = (range.end * 8 - 1) as u32;
            return Some(bv.extract(hi_bit, off_bits, self.ctx));
        }
        None
    }

    /// Exact-address hit (with width extract) then overlap fallback into a wider
    /// covering store, over one symbolic-store map. Pairs with
    /// `symbolic_overlap_load` so the pending and flushed buffers share the whole
    /// dispatch instead of open-coding it twice. Returns `None` (falls through to
    /// the concrete buffer) when an exact key is present but narrower than the
    /// load — matching the original if / else-if structure.
    pub(super) fn symbolic_store_load(
        &self,
        map: &FxHashMap<u64, RustBV>,
        addr: u64,
        size: usize,
    ) -> Option<RustBV> {
        if let Some(sym_val) = map.get(&addr) {
            let want = (size * 8) as u32;
            if sym_val.width() == want {
                return Some(sym_val.clone());
            } else if sym_val.width() > want {
                return Some(sym_val.extract((size * 8 - 1) as u32, 0, self.ctx));
            }
            return None;
        }
        self.symbolic_overlap_load(map, addr, size)
    }

    /// The full layered memory read, shared by `Load` (`eval_load`) and
    /// `LoadG` (`resolve_loadg_load`): Rust-native memory first when enabled,
    /// then — for the callback path — the pending / flushed store buffers and
    /// concrete caches in `load_concrete_addr`, or the concretizing
    /// `load_symbolic_addr` for a symbolic address.
    ///
    /// LoadG used to call `load_from_callback` directly (angr-9ke6b.83), which
    /// skips every one of those layers, so a guarded load reading an address
    /// written earlier in the same block observed stale Python memory instead
    /// of the just-stored value.
    pub(super) fn load_layered(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        size: usize,
        load_start: Option<Instant>,
    ) -> Result<RustBV, CbExecutionError> {
        // Try Rust-native memory first if enabled - mirrors try_rust_memory_store
        if self.use_rust_memory
            && let Some(value) = self.try_rust_memory_load(callbacks, addr_val, size, load_start)?
        {
            // SymbolicMemory::load_concrete already bumped record_mem_load.
            if self.profiling_enabled {
                self.stats.rust_memory_load_count += 1;
            }
            return Ok(value);
        }

        if self.profiling_enabled {
            self.stats.fallback_memory_load_count += 1;
        }

        // angr-obrm: callback-path loads bypass SymbolicMemory, so bump
        // the global mem_load counter here for parity with the
        // Rust-memory path. Catches pending-store buffer hits, prefetch
        // cache hits, concrete_memory cache hits, and Python-callback
        // fallbacks alike.
        record_mem_load(size as u64);

        if let Some(addr_concrete) = addr_val.as_u64() {
            self.load_concrete_addr(callbacks, addr_concrete, size)
        } else {
            self.load_symbolic_addr(callbacks, addr_val, size)
        }
    }

    /// `load_layered` at an address that a caller already concretized out of a
    /// symbolic `addr_val` (LoadG's Single / Multiple shapes). The concrete
    /// address is rebuilt as a BV of the original address width so the
    /// Rust-memory layer sees the same pointer size the block does.
    pub(super) fn load_layered_at(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        addr_concrete: u64,
        size: usize,
    ) -> Result<RustBV, CbExecutionError> {
        let concrete_bv = RustBV::concrete(addr_concrete as u128, addr_val.width());
        self.load_layered(callbacks, &concrete_bv, size, None)
    }

    /// Concrete-address load path: walk pending/flushed store buffers, prefetch
    /// and concrete-memory caches before falling back to the Python callback.
    ///
    /// This is the *upper* half of the load-resolution ladder; the lower half
    /// lives in `mod.rs`, where the `load_from_callback` fallback below first
    /// tries `synthesize_unservable_load` (native filler for a page neither
    /// side has) and only then crosses the GIL. Read the two together — no
    /// rung between `load_layered` and the Python callback lives anywhere else.
    pub(super) fn load_concrete_addr(
        &self,
        callbacks: &PythonCallbacks,
        addr_concrete: u64,
        size: usize,
    ) -> Result<RustBV, CbExecutionError> {
        // FAST PATH 0: Check pending stores buffer
        // Stores within the same block are buffered in pending_stores.
        // We must check this buffer before falling through to Python
        // callbacks, which have stale state.

        // First check symbolic stores (preserves symbolic values).
        // Exact-address hit is O(1); an offset/overlap load into a wider
        // symbolic store (e.g. a 4-byte load at X+4 inside an 8-byte symbolic
        // store at X) is missed by the hash lookup and — since symbolic stores
        // no longer push zero placeholder bytes (angr-ofyh) — by the concrete
        // buffer too, so fall back to an overlap scan that extracts the covered
        // bytes from the covering store.
        if let Some(sym_val) =
            self.symbolic_store_load(&self.pending_symbolic_stores, addr_concrete, size)
        {
            return Ok(sym_val);
        }

        // Then check concrete stores via the indexed buffer.
        // try_load fast-skips when no pending store overlaps the
        // load address; falls back to a reverse scan only when the
        // most recent covering store is smaller than the load.
        if let Some(data) = self.pending_stores.try_load(addr_concrete, size) {
            return Ok(bytes_to_bv(data, (size * 8) as u32));
        }
        // No *single* buffered store covers the whole load (a byte-wise write
        // loop followed by a wider read-back). Assemble it from several of
        // them before falling through to the layers below, which know nothing
        // about the still-buffered stores and would answer with stale
        // pre-write bytes (angr-6cp06.69).
        if let Some(data) = self.pending_stores.try_load_assembled(addr_concrete, size) {
            return Ok(bytes_to_bv(&data, (size * 8) as u32));
        }
        // Still no full coverage, but *some* bytes of the load may be
        // buffered — a load straddling the edge of a pending store. The
        // layers below know nothing about the buffer and answer for the whole
        // range, so taking their result verbatim silently undoes the buffered
        // write (angr-6cp06.88). Resolve them first, then splice the buffered
        // bytes back over their positions; `splice_bytes_over_bv` handles a
        // symbolic lower-layer answer as an Extract/Concat rather than a byte
        // patch.
        let partial = self.pending_stores.try_load_partial(addr_concrete, size);
        let below = self.load_concrete_addr_below_pending(callbacks, addr_concrete, size)?;
        match partial {
            Some(overlay) => Ok(splice_bytes_over_bv(&below, &overlay, self.ctx)),
            None => Ok(below),
        }
    }

    /// The rungs of `load_concrete_addr`'s ladder below the pending-store
    /// buffers: previously flushed stores, the Rust concrete-memory cache and
    /// finally the Python callback.
    ///
    /// Split out so `load_concrete_addr` can resolve them *first* and then
    /// overlay a partially-covering pending store onto the answer. Do not call
    /// this directly from a load path — it deliberately skips the pending
    /// buffers, and a load that misses them reads stale pre-block memory
    /// (the same-block read-back bug `load_layered`'s doc comment describes).
    fn load_concrete_addr_below_pending(
        &self,
        callbacks: &PythonCallbacks,
        addr_concrete: u64,
        size: usize,
    ) -> Result<RustBV, CbExecutionError> {
        // Also check previously flushed symbolic stores (cross-block), with the
        // same exact-then-overlap fallback as the pending map above.
        if let Some(sym_val) =
            self.symbolic_store_load(&self.all_flushed_symbolic_stores, addr_concrete, size)
        {
            return Ok(sym_val);
        }

        // Also check previously flushed concrete stores (cross-block)
        if let Some(store_data) = self.all_flushed_stores.get(&addr_concrete)
            && size <= store_data.len()
        {
            let data = &store_data[..size];
            return Ok(bytes_to_bv(data, (size * 8) as u32));
        }

        // FAST PATH: Check if address is in Rust-cached concrete memory
        if let Some(data) = self.try_read_concrete_memory(addr_concrete, size) {
            return Ok(bytes_to_bv(data, (size * 8) as u32));
        }
        // SLOW PATH: Fall back to Python callback
        self.load_from_callback(callbacks, addr_concrete, size)
    }

    /// Symbolic-address load path: concretize, then dispatch by result shape
    /// (single / multiple / strided / too-large / failed).
    pub(super) fn load_symbolic_addr(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        size: usize,
    ) -> Result<RustBV, CbExecutionError> {
        // AVOID_MULTIVALUED_READS: skip concretization and return an
        // unconstrained value even when use_rust_memory is false.
        if self.concretizer.should_avoid_multivalued_read(addr_val) {
            return Ok(self.fresh_unconstrained_read(size));
        }
        // angr-vfst: address_concretization BP_BEFORE — dispatch before the
        // concretizer runs so a user BP could (in a future iter) intervene.
        // MVP: dispatch only; no override path. Gated on bit 17 inside.
        self.dispatch_address_concretization_inspect(callbacks, addr_val, "load", "before", None);
        let conc = self.concretize_cached_read(addr_val);
        // BP_AFTER carries the list of concrete addresses produced.
        let result_addrs = conc.addresses();
        self.dispatch_address_concretization_inspect(
            callbacks,
            addr_val,
            "load",
            "after",
            result_addrs,
        );
        match &*conc {
            ConcretizationResult::Single(addr_concrete) => {
                let addr_concrete = *addr_concrete;
                if let Some(data) = self.try_read_concrete_memory(addr_concrete, size) {
                    return Ok(bytes_to_bv(data, (size * 8) as u32));
                }
                self.load_from_callback(callbacks, addr_concrete, size)
            }
            ConcretizationResult::Multiple(addrs) => {
                // Build ITE chain in Rust instead of delegating to Python
                // This avoids FFI overhead and keeps symbolic ops in Rust's Z3 context
                self.dispatch_multi_load(callbacks, addrs, addr_val, size)
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                // Strided access pattern - generate addresses and build ITE chain in Rust
                let addrs = strided_addrs(*base, *stride, *count);
                self.dispatch_multi_load(callbacks, &addrs, addr_val, size)
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                let descr = format!("range 0x{min:x}-0x{max:x}");
                self.fallback_load_symbolic_full(callbacks, addr_val, size, "Load", &descr)
            }
            ConcretizationResult::Failed(reason) => {
                // Concretization failed entirely (e.g., timeout, no
                // strategy applies). Try the full symbolic load callback;
                // Python's memory model can still resolve it via its
                // own address concretization strategies.
                let descr = format!("concretize failed: {reason}");
                self.fallback_load_symbolic_full(callbacks, addr_val, size, "Load", &descr)
            }
        }
    }

    /// Convert one batched-load result entry to a RustBV.
    ///
    /// Branches: missing index (fresh symbolic), concrete bytes, or symbolic AST
    /// (delegates to `try_convert_symbolic_value`).
    pub(super) fn convert_load_result(
        &self,
        load_results: &[crate::callbacks::BatchLoadEntry],
        i: usize,
        width: u32,
        fallback_name: impl Fn() -> String,
    ) -> RustBV {
        let Some((data, is_symbolic, symbolic_ast)) = load_results.get(i) else {
            return RustBV::symbolic(self.ctx, fallback_name(), width);
        };
        if !*is_symbolic {
            return bytes_to_bv(data, width);
        }
        self.try_convert_symbolic_value(symbolic_ast.as_ref(), width, fallback_name)
    }

    /// Convert an optional symbolic AST to RustBV, falling back to a fresh symbolic.
    ///
    /// Order: handle-table fast path, then claripy-AST conversion, then fresh symbolic.
    pub(super) fn try_convert_symbolic_value(
        &self,
        ast_obj: Option<&Py<PyAny>>,
        width: u32,
        fallback_name: impl FnOnce() -> String,
    ) -> RustBV {
        let Some(ast_obj) = ast_obj else {
            return RustBV::symbolic(self.ctx, fallback_name(), width);
        };
        // Self-attach: the claripy bridge work below needs the GIL, but this
        // helper no longer threads a caller token (angr-vh834 Phase 4). When
        // the GIL is already held (single-threaded path) this is a cheap
        // re-entrant no-op.
        let converted: Option<RustBV> = Python::attach(|py| {
            let ast = ast_obj.bind(py);

            if let Some(table) = self.symbol_table
                && let Some(bv) = try_handle_to_rustbv(ast, table)
            {
                return Some(bv);
            }

            if is_claripy_ast(ast)
                && let Ok(bv) = claripy_to_rustbv(py, ast, self.ctx)
            {
                return Some(bv);
            }
            None
        });

        converted.unwrap_or_else(|| RustBV::symbolic(self.ctx, fallback_name(), width))
    }

    /// Shared dispatch for Multiple/Strided concretization results on the load
    /// path, the counterpart to `dispatch_multi_store`.
    ///
    /// Builds the ITE chain in Rust when the address count is within
    /// [`MAX_ITE_ADDRS`]; beyond that it hands the address AST to Python's full
    /// symbolic-load callback rather than emitting a 256-wide ITE (see the
    /// constant's doc for the cost argument). The store path can pass its
    /// already-computed address list to Python (`call_memory_store_symbolic`);
    /// there is no load-side equivalent taking an address list, so the wide case
    /// re-concretizes inside Python's memory model.
    ///
    /// When the full-load callback isn't wired up, the in-Rust chain is still the
    /// only way to resolve the load, so the cap yields rather than hard-erroring.
    pub(super) fn dispatch_multi_load(
        &self,
        callbacks: &PythonCallbacks,
        addrs: &[u64],
        addr_val: &RustBV,
        size: usize,
    ) -> Result<RustBV, CbExecutionError> {
        if addrs.len() > MAX_ITE_ADDRS && callbacks.has_memory_load_symbolic_full() {
            let descr = format!("{} candidate addresses", addrs.len());
            return self.fallback_load_symbolic_full(callbacks, addr_val, size, "Load", &descr);
        }
        self.build_ite_load_from_callbacks(callbacks, addrs, addr_val, size)
    }

    /// Build an ITE chain for symbolic memory load by loading each candidate address.
    ///
    /// This builds the ITE chain entirely in Rust instead of delegating to Python.
    /// For each candidate address, we load the value via callback and create an ITE:
    /// `ITE(addr == a1, mem[a1], ITE(addr == a2, mem[a2], ...))`
    ///
    /// This is more efficient than calling Python's symbolic memory handler because:
    /// 1. We avoid FFI overhead for the ITE chain construction
    /// 2. The RustBV ITE nodes stay in Rust's Z3 context
    /// 3. The chain is assembled in one pass by `build_ite_chain`, after a single
    ///    batched `call_memory_load_batch` for every candidate address
    pub(super) fn build_ite_load_from_callbacks(
        &self,
        callbacks: &PythonCallbacks,
        addrs: &[u64],
        addr_expr: &RustBV,
        size: usize,
    ) -> Result<RustBV, CbExecutionError> {
        if addrs.is_empty() {
            return Err(CbExecutionError::Memory(MemoryError::SymbolicAddress {
                description: "no candidate addresses".to_string(),
            }));
        }

        let width = (size * 8) as u32;
        let addr_width = addr_expr.width();

        if addrs.len() == 1 {
            return self.load_from_callback(callbacks, addrs[0], size);
        }

        let load_requests: Vec<(u64, u32)> = addrs.iter().map(|&a| (a, size as u32)).collect();
        let load_results = callbacks
            .call_memory_load_batch(&load_requests)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        let mut pairs: Vec<(RustBV, RustBV)> = Vec::with_capacity(addrs.len());

        for (i, addr) in addrs.iter().enumerate() {
            let value = self.convert_load_result(&load_results, i, width, || {
                format!("ite_load_{addr:x}_{size}")
            });

            let addr_const = RustBV::concrete(*addr as u128, addr_width);
            let cond = addr_expr.eq(&addr_const, self.ctx);

            pairs.push((cond, value));
        }

        let default_value = pairs
            .last()
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| RustBV::symbolic(self.ctx, "ite_default", width));

        Ok(build_ite_chain(
            &pairs[..pairs.len() - 1],
            default_value,
            self.ctx,
        ))
    }

    /// Build ITE chain stores for symbolic memory writes with multiple candidate addresses.
    ///
    /// For each candidate address `a_i`, computes:
    ///   `mem[a_i] = ITE(addr == a_i, new_data, mem[a_i])`
    /// This keeps the ITE construction in Rust's Z3 context, avoiding FFI round-trips
    /// for the ITE chain building that Python would otherwise do.
    pub(super) fn build_ite_store_from_callbacks(
        &mut self,
        callbacks: &PythonCallbacks,
        addrs: &[u64],
        addr_expr: &RustBV,
        data_val: &RustBV,
    ) -> Result<(), CbExecutionError> {
        if addrs.is_empty() {
            return Ok(());
        }

        let size = (data_val.width() / 8) as usize;
        let addr_width = addr_expr.width();
        let val_width = data_val.width();

        let load_requests: Vec<(u64, u32)> = addrs.iter().map(|&a| (a, size as u32)).collect();
        let load_results = callbacks
            .call_memory_load_batch(&load_requests)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        for (i, &addr) in addrs.iter().enumerate() {
            let addr_const = RustBV::concrete(addr as u128, addr_width);
            let cond = addr_expr.eq(&addr_const, self.ctx);

            let current = self.convert_load_result(&load_results, i, val_width, || {
                format!("ite_store_cur_{addr:x}")
            });

            let ite_value = cond.ite(data_val, &current, self.ctx);

            self.store_symbolic_value_buffered(callbacks, addr, &ite_value)?;
        }

        Ok(())
    }

    /// Resolve a LoadG load given its address BV. Concrete addresses and
    /// Single/Multiple concretizations go through `load_layered` — the same
    /// Rust-memory / pending-store / flushed-store / concrete-cache ladder
    /// ordinary `Load` uses (angr-9ke6b.83) — and it falls back to the Python full
    /// symbolic load callback for TooLarge / Strided / Failed shapes (so the
    /// load no longer hard-errors when angr's address strategies could resolve
    /// it). Multiple addresses still take the first solution to preserve the
    /// pre-existing LoadG behavior — broader Multiple handling can be added
    /// later if needed.
    pub(super) fn resolve_loadg_load(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        load_size: usize,
        context: &str,
    ) -> Result<RustBV, CbExecutionError> {
        if addr_val.as_u64().is_some() {
            return self.load_layered(callbacks, addr_val, load_size, None);
        }
        let conc = self.concretize_cached_read(addr_val);
        match &*conc {
            ConcretizationResult::Single(a) => {
                let a = *a;
                self.load_layered_at(callbacks, addr_val, a, load_size)
            }
            ConcretizationResult::Multiple(addrs) => {
                let a = *addrs.first().ok_or_else(|| {
                    CbExecutionError::Unsupported(format!("{context} with empty address set"))
                })?;
                self.load_layered_at(callbacks, addr_val, a, load_size)
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                let descr = format!("strided base=0x{base:x} stride=0x{stride:x} count={count}");
                self.fallback_load_symbolic_full(callbacks, addr_val, load_size, context, &descr)
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                let descr = format!("range 0x{min:x}-0x{max:x}");
                self.fallback_load_symbolic_full(callbacks, addr_val, load_size, context, &descr)
            }
            ConcretizationResult::Failed(reason) => {
                let descr = format!("concretize failed: {reason}");
                self.fallback_load_symbolic_full(callbacks, addr_val, load_size, context, &descr)
            }
        }
    }

    /// Apply LoadG conversion (widening) to loaded value.
    ///
    /// LoadG can widen the loaded value with sign or zero extension.
    pub(super) fn apply_loadg_conversion(
        &self,
        cvt: IRLoadGOp,
        value: RustBV,
        target_bits: u32,
    ) -> RustBV {
        let src_bits = value.width();
        if src_bits >= target_bits {
            // No widening needed, possibly truncate
            if src_bits > target_bits {
                // extract(high, low) takes bits [high:low] inclusive, so the
                // low `target_bits` bits are extract(target_bits - 1, 0).
                value.extract(target_bits - 1, 0, self.ctx)
            } else {
                value
            }
        } else {
            // Widen the value
            match cvt {
                // Identity/Unknown should not reach a widening branch (sizes
                // differ only for the Widen* variants); pass through if they do.
                IRLoadGOp::Identity | IRLoadGOp::Unknown => value,
                IRLoadGOp::WidenS { .. } => value.sign_extend(target_bits, self.ctx),
                IRLoadGOp::WidenZ { .. } => value.zero_extend(target_bits, self.ctx),
            }
        }
    }

    /// Attempt to load via Rust-native memory. Returns `Ok(Some(bv))` if Rust
    /// handled the load, `Ok(None)` if the caller should fall back to the
    /// Python path, or `Err` for unrecoverable errors.
    pub(super) fn try_rust_memory_load(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        size: usize,
        load_start: Option<Instant>,
    ) -> Result<Option<RustBV>, CbExecutionError> {
        // AVOID_MULTIVALUED_READS: bypass `load_symbolic_unified` (and its
        // concretization) for symbolic addresses and return unconstrained.
        // Matches the early `return self._default_value(...)` in Python's
        // `address_concretization_mixin.AddressConcretizationMixin.load`.
        if self.concretizer.should_avoid_multivalued_read(addr_val)
            && let Some(rust_mem) = self.rust_memory.as_ref()
        {
            let value = rust_mem.unconstrained_read_value(size as u32, self.ctx);
            profile_add!(load_start, self.stats.load_stmt_time_ns);
            return Ok(Some(value));
        }
        let first_result = match self.rust_memory.as_mut() {
            Some(rust_mem) => rust_mem.load_symbolic_unified(
                addr_val.clone(),
                size as u32,
                self.ctx,
                &self.concretizer,
            ),
            None => return Ok(None),
        };

        match first_result {
            Ok(value) => {
                profile_add!(load_start, self.stats.load_stmt_time_ns);
                Ok(Some(value))
            }
            Err(MemoryError::UnmappedPageInRegion { page_addr }) => {
                // Page is in a lazy region - fetch it (rust_mem borrow is dropped here)
                let prefetch_count = self.page_prefetch_count;
                let page_fetched =
                    self.fetch_page_with_prefetch(callbacks, page_addr, prefetch_count)?;

                // NOTE: We intentionally do NOT auto-map zero pages when page_fetched is false.
                // Python may have actual data for this page from backers (file contents,
                // initialized data). Speculatively creating zero pages causes state
                // divergence between Rust and Python. Instead, we fall through to
                // the Python callback which handles memory correctly.

                if page_fetched
                    && let Some(ref mut rust_mem) = self.rust_memory
                    && let Ok(value) = rust_mem.load_symbolic_unified(
                        addr_val.clone(),
                        size as u32,
                        self.ctx,
                        &self.concretizer,
                    )
                {
                    return Ok(Some(value));
                }
                Ok(None)
            }
            Err(MemoryError::Unmapped {
                addr,
                size: unmapped_size,
            }) => {
                log::debug!(
                    "Unmapped memory load at 0x{addr:x} (size={unmapped_size}), falling back to Python"
                );
                Ok(None)
            }
            Err(MemoryError::SymbolicAddress { .. }) => {
                // Symbolic bytes not fully tracked - fall through to Python.
                // Happens when per-byte symbolic imports don't cover the full
                // multi-byte load, or imports didn't cover all bytes at the addr.
                Ok(None)
            }
            // SILENT(cat-b): angr-0jh0j.83 — `check_access_size`'s own doc
            // ("refuse instead and let the caller bounce to Python") makes this
            // a fallback, not a terminal fault: the load never touched memory,
            // so Python's model can answer it (or raise) exactly as it does for
            // the unmapped/symbolic-address refusals above. Warn rather than
            // debug — unlike those, an oversized `size` cannot come from guest
            // state, only from a caller that computed a bogus width.
            Err(MemoryError::SizeTooLarge { addr, size }) => {
                log::warn!(
                    "Oversized memory load at 0x{addr:x} (size={size}), falling back to Python"
                );
                Ok(None)
            }
            // Every remaining variant is a genuine fault (permission violation,
            // out-of-bounds, zero/unaligned width, unexpected symbolic) that
            // Python cannot answer any better than Rust did, plus the wildcard
            // `MemoryError` is `#[non_exhaustive]` requires. A *new* variant
            // landing here is terminal by default — check whether it deserves
            // an `Ok(None)` arm above before leaving it, which is precisely how
            // `SizeTooLarge` got mistriaged.
            Err(e) => Err(CbExecutionError::Memory(e)),
        }
    }
}
