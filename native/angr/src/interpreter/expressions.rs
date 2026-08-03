use super::bv_utils::{build_balanced_ite, bytes_to_bv};
use super::*;
use crate::vex::ir::{IRCallee, IRRegArray};
use rustc_hash::FxHashMap;

/// The three binop families that parse to a concrete `IROp` but have no native
/// dispatch arm (`VEXOps::binop` returns `OpError::NotBinary`): `Iop_Perm8x*`
/// (=> `VPerm`), `Iop_Pclmul*`, and `Iop_Crc32C`. All deterministic — Python's
/// VEX engine models them exactly — so `eval_binop` routes them to Python
/// fallback rather than fabricating a wrong fresh symbolic. Keep this in lockstep
/// with `opcode_map.rs` if a new must-fallback family is added.
/// See bd `angr-s6miz` and the "Parse-succeeds / dispatch-fabricates (silent
/// BYPASS)" section of `docs/extending-angr/rust_vex_ops.rst`.
fn is_dispatch_fabricate_family(op: &IROp) -> bool {
    matches!(
        op,
        IROp::VPerm { .. }
            | IROp::PclmulLQLQ
            | IROp::PclmulHQHQ
            | IROp::PclmulLQHQ
            | IROp::PclmulHQLQ
            | IROp::Crc32C
    )
}

/// Opt-in escape hatch (`ANGR_RUST_FABRICATE_UNSUPPORTED_IROP`) for the
/// symbolic-operand arm of [`VEXInterpreter::vex_op_fallback`] and for the
/// condition-flag arm of [`VEXInterpreter::eval_ccall`]. When set (to any
/// non-empty, non-`"0"` value) an unsupported op with a symbolic operand
/// fabricates a fresh unconstrained symbolic (the pre-angr-oyzvj behavior)
/// instead of routing the block to Python. Default (unset): route to Python —
/// the correct behavior, since fabricating an unconstrained value silently
/// explores both branches of any downstream condition. Read once per process
/// (cached), matching the `OnceLock` env pattern in `engine.rs`.
///
/// Named `FABRICATE_*`, deliberately NOT `BYPASS_*`: the existing
/// `BYPASS_UNSUPPORTED_IROP` SimOption means the opposite (route to Python's
/// resilience mixin), so reusing that name would invert its sense.
fn fabricate_unsupported_irop() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("ANGR_RUST_FABRICATE_UNSUPPORTED_IROP")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

impl<'a> VEXInterpreter<'a> {
    /// Evaluate an IR expression using Python callbacks for memory loads.
    pub(super) fn eval_expr_with_callbacks(
        &mut self,
        callbacks: &PythonCallbacks,
        expr: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        let expr_start = profile_start!(self);
        let result = self.eval_expr_with_callbacks_inner(callbacks, expr, tyenv);
        profile_add!(expr_start, self.stats.expr_eval_time_ns);
        if self.profiling_enabled {
            self.stats.expr_eval_count += 1;
        }
        // state.inspect expr event (angr-lge2) — fires `when='after'` after
        // every IRExpr evaluation. Gated on `inspect_event_enabled(16)` so
        // the no-BP case is one `AtomicU32::load + AND` per call. This is
        // the highest-frequency dispatch site in the engine (every binop
        // arg, store data, exit guard, etc. comes through here).
        if let Ok(ref value) = result {
            self.dispatch_expr_inspect(callbacks, value);
        }
        result
    }

    fn eval_expr_with_callbacks_inner(
        &mut self,
        callbacks: &PythonCallbacks,
        expr: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        match expr {
            IRExpr::Const(c) => Ok(self.eval_const(c)),

            IRExpr::RdTmp(tmp) => {
                if let Some(Some(val)) = self.temps.get(*tmp as usize) {
                    let value = val.clone();
                    self.dispatch_tmp_read_inspect(callbacks, *tmp, &value);
                    Ok(value)
                } else {
                    Err(CbExecutionError::UnknownTemp(*tmp))
                }
            }

            IRExpr::Get { offset, ty } => {
                let size = ty.bytes();
                let value = self.registers.get(*offset, size, self.ctx);
                self.dispatch_reg_read_inspect(callbacks, *offset, size, &value);
                Ok(value)
            }

            IRExpr::Load { addr, ty, endness } => {
                self.eval_load(callbacks, addr, *ty, *endness, tyenv)
            }

            IRExpr::Unop { op, arg } => self.eval_unop(callbacks, *op, arg, tyenv),

            IRExpr::Binop { op, left, right } => {
                self.eval_binop(callbacks, *op, left, right, tyenv)
            }

            IRExpr::ITE {
                cond,
                iftrue,
                iffalse,
            } => self.eval_ite(callbacks, cond, iftrue, iffalse, tyenv),

            IRExpr::GetI { descr, ix, bias } => self.eval_geti(callbacks, *descr, ix, *bias, tyenv),

            IRExpr::Triop {
                op,
                arg1,
                arg2,
                arg3,
            } => self.eval_triop(callbacks, *op, arg1, arg2, arg3, tyenv),

            IRExpr::Qop {
                op,
                arg1,
                arg2,
                arg3,
                arg4,
            } => self.eval_qop(callbacks, *op, [arg1, arg2, arg3, arg4], tyenv),

            IRExpr::CCall { cee, retty, args } => {
                self.eval_ccall(callbacks, cee, *retty, args, tyenv)
            }

            IRExpr::VECRET | IRExpr::GSPTR => {
                // P7 fix: Request Python fallback instead of failing
                // These special expressions require Python's VEX handling.
                // Reason string carries `VECRET_GSPTR_REASON` so
                // `exploration::run_loop` can bump a per-category counter
                // (angr-2iow prevalence measurement).
                Err(CbExecutionError::NeedPythonFallback(format!(
                    "{} (special expr {:?} requires Python)",
                    crate::interpreter::VECRET_GSPTR_REASON,
                    expr
                )))
            }
        }
    }

    fn eval_load(
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
    fn symbolic_overlap_load(
        &self,
        map: &FxHashMap<u64, RustBV>,
        addr: u64,
        size: usize,
    ) -> Option<RustBV> {
        if map.is_empty() {
            return None;
        }
        let load_hi = addr.checked_add(size as u64)?;
        for (&s_addr, bv) in map.iter() {
            if s_addr > addr {
                continue;
            }
            let s_hi = s_addr.saturating_add((bv.width() / 8) as u64);
            if load_hi <= s_hi {
                let off_bits = (addr - s_addr) * 8;
                let hi_bit = (off_bits + (size as u64) * 8 - 1) as u32;
                return Some(bv.extract(hi_bit, off_bits as u32, self.ctx));
            }
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
    fn load_layered(
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
            return Ok(value);
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
    fn load_layered_at(
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
    fn load_concrete_addr(
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
    fn load_symbolic_addr(
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
                self.build_ite_load_from_callbacks(callbacks, addrs, addr_val, size)
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                // Strided access pattern - generate addresses and build ITE chain in Rust
                let addrs: Vec<u64> = (0..*count).map(|i| base + i * stride).collect();
                self.build_ite_load_from_callbacks(callbacks, &addrs, addr_val, size)
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

    /// Shared tail for the unsupported-op fallback in
    /// `eval_unop`/`eval_binop`/`eval_triop`/`eval_qop`. Symbolic operands route
    /// the block to Python's VEX engine (the reference implementation for the
    /// float / vector-float conversions that land here), mirroring the
    /// dispatch-fabricate family (angr-s6miz). Fabricating a fresh unconstrained
    /// symbolic instead — the pre-angr-oyzvj default — silently diverges: any
    /// downstream condition over the fabricated value explores both branches
    /// unconstrained. That behavior is retained only as an explicit opt-in via
    /// `ANGR_RUST_FABRICATE_UNSUPPORTED_IROP` (visibility via
    /// `vex_bypass_fabricate_count`). Concrete operands propagate the typed
    /// `OpError` so the failure surfaces a typed `RustUnsupportedVexOpError` on
    /// the test path (and routes the state to the errored stash, op + arch in
    /// the message, in live exploration — see the taxonomy note in `errors.rs`)
    /// rather than fabricating a silently-wrong value (angr-sa3j). The per-op
    /// `python_vex_{unop,binop,triop,qop}_fallback_count` is bumped at the call
    /// site; this bumps the aggregate `python_vex_op_fallback_count`.
    fn vex_op_fallback(
        &mut self,
        e: OpError,
        any_sym: bool,
        width: u32,
        name: String,
    ) -> Result<RustBV, CbExecutionError> {
        self.stats.python_vex_op_fallback_count += 1;
        if any_sym {
            if fabricate_unsupported_irop() {
                self.stats.vex_bypass_fabricate_count += 1;
                Ok(RustBV::symbolic(self.ctx, name, width))
            } else {
                Err(CbExecutionError::NeedPythonFallback(format!(
                    "unsupported symbolic IROp routed to Python ({name}): {e}"
                )))
            }
        } else {
            Err(CbExecutionError::Op(e))
        }
    }

    fn eval_unop(
        &mut self,
        callbacks: &PythonCallbacks,
        op: IROp,
        arg: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        record_vex_unop(iropclass(&op));
        let arg_val = self.eval_expr_with_callbacks(callbacks, arg, tyenv)?;
        let arg_is_sym = arg_val.is_symbolic();
        match VEXOps::unop(op, arg_val, self.ctx) {
            Ok(v) => Ok(v),
            // NEON scaffolding: surface explicitly. See
            // `invariant-neon-scaffolding-panic-not-fallback`.
            Err(e @ OpError::UnsupportedNeon { .. }) => Err(CbExecutionError::Op(e)),
            // angr-tkbr.2: unmapped pyvex opcode — propagate past
            // the silent fresh-symbolic fallback so the failure carries
            // op + arch (typed RustUnsupportedVexOpError on the test
            // path; stringified into the errored stash live).
            Err(e @ OpError::UnsupportedVexOp { .. }) => Err(CbExecutionError::Op(e)),
            // Fallback for unsupported unary ops (e.g., float conversions).
            Err(e) => {
                self.stats.python_vex_unop_fallback_count += 1;
                let width = op.result_type().map(|t| t.bits()).unwrap_or(64);
                self.vex_op_fallback(e, arg_is_sym, width, format!("unsup_unop_{:x}", self.pc))
            }
        }
    }

    fn eval_binop(
        &mut self,
        callbacks: &PythonCallbacks,
        op: IROp,
        left: &IRExpr,
        right: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        record_vex_binop(iropclass(&op));
        let left_val = self.eval_expr_with_callbacks(callbacks, left, tyenv)?;
        let right_val = self.eval_expr_with_callbacks(callbacks, right, tyenv)?;
        let fallback_width = op
            .result_type()
            .map(|t| t.bits())
            .unwrap_or(left_val.width().max(right_val.width()));
        let any_sym = left_val.is_symbolic() || right_val.is_symbolic();
        match VEXOps::binop(op, left_val, right_val, self.ctx) {
            Ok(v) => Ok(v),
            // NEON scaffolding: surface explicitly. See
            // `invariant-neon-scaffolding-panic-not-fallback`.
            Err(e @ OpError::UnsupportedNeon { .. }) => Err(CbExecutionError::Op(e)),
            // angr-tkbr.2: unmapped pyvex opcode — propagate past
            // the silent fresh-symbolic fallback so the failure carries
            // op + arch (typed RustUnsupportedVexOpError on the test
            // path; stringified into the errored stash live).
            Err(e @ OpError::UnsupportedVexOp { .. }) => Err(CbExecutionError::Op(e)),
            // angr-s6miz: the three dispatch-fabricate families
            // (Iop_Perm8x* => VPerm, Iop_Pclmul*, Iop_Crc32C) parse to a
            // concrete IROp but have no native dispatch arm, so `binop` returns
            // `NotBinary`. They are deterministic ops Python models exactly, so
            // route the block to Python's VEX engine rather than fabricating a
            // wrong fresh symbolic (the BYPASS arm below). Routed for BOTH
            // symbolic and concrete args — strictly better than the old
            // fabricate-on-sym / hard-error-on-concrete split.
            Err(_) if is_dispatch_fabricate_family(&op) => {
                // angr-9ke6b.85: this IS a Python-VEX-fallback event, so it
                // bumps the same two counters the ordinary `Err(e)` arm below
                // does (per-op subset + aggregate). It cannot route through
                // `vex_op_fallback`: that helper hard-errors on all-concrete
                // args and honors the fabricate opt-in, both of which this
                // family deliberately bypasses.
                self.stats.python_vex_binop_fallback_count += 1;
                self.stats.python_vex_op_fallback_count += 1;
                Err(CbExecutionError::NeedPythonFallback(format!(
                    "{} ({:?})",
                    crate::interpreter::DISPATCH_FABRICATE_REASON,
                    op
                )))
            }
            // Fallback for unsupported binary ops (e.g., vector float ops).
            Err(e) => {
                self.stats.python_vex_binop_fallback_count += 1;
                self.vex_op_fallback(
                    e,
                    any_sym,
                    fallback_width,
                    format!("unsup_binop_{:x}", self.pc),
                )
            }
        }
    }

    fn eval_ite(
        &mut self,
        callbacks: &PythonCallbacks,
        cond: &IRExpr,
        iftrue: &IRExpr,
        iffalse: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        let cond_val = self.eval_expr_with_callbacks(callbacks, cond, tyenv)?;
        // Short-circuit: skip evaluating the dead branch when condition is concrete
        if let Some(v) = cond_val.as_u128() {
            return if v != 0 {
                self.eval_expr_with_callbacks(callbacks, iftrue, tyenv)
            } else {
                self.eval_expr_with_callbacks(callbacks, iffalse, tyenv)
            };
        }
        let true_val = self.eval_expr_with_callbacks(callbacks, iftrue, tyenv)?;
        let false_val = self.eval_expr_with_callbacks(callbacks, iffalse, tyenv)?;
        Ok(cond_val.ite(&true_val, &false_val, self.ctx))
    }

    /// Eagerly concretize a symbolic value via the solver and pin the choice
    /// with an equality constraint so a later solve cannot pick a different
    /// value (which would make the dependent read/arg inconsistent with the
    /// path constraints — unsound). Returns the pinned concrete as a u64, or
    /// None when the solver cannot produce a model (e.g. an UNSAT path). The
    /// caller maps None to its own error/fallback. This is the shared
    /// soundness primitive behind GetI/PutI index concretization and the
    /// dirty-call arg loops.
    pub(super) fn concretize_and_pin(&self, v: &RustBV) -> Option<u64> {
        let concrete = self.ctx.eval(v)?;
        let conc_bv = RustBV::concrete(concrete, v.width());
        let constraint = v.eq(&conc_bv, self.ctx);
        self.ctx.assume_true(&constraint);
        Some(concrete as u64)
    }

    fn eval_geti(
        &mut self,
        callbacks: &PythonCallbacks,
        descr: IRRegArray,
        ix: &IRExpr,
        bias: u32,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        // Evaluate the index expression
        let ix_val = self.eval_expr_with_callbacks(callbacks, ix, tyenv)?;

        // GetI requires a concrete index to compute the register offset
        let idx = if let Some(idx) = ix_val.as_u64() {
            idx
        } else {
            // Symbolic index - concretize using the solver and pin the choice
            // with an equality constraint so a later solve cannot pick a
            // different index, which would make this register read inconsistent
            // with the path constraints (unsound).
            self.concretize_and_pin(&ix_val).ok_or_else(|| {
                CbExecutionError::Unsupported("GetI index concretization failed".to_string())
            })?
        };

        // Calculate the rotating register offset:
        // offset = base + ((idx + bias) % nElems) * elemTy.bytes()
        let elem_size = descr.elemTy.bytes();
        let index = ((idx as u32).wrapping_add(bias)) % descr.nElems;
        let offset = descr.base + index * elem_size;

        // Read from the register file
        Ok(self.registers.get(offset, elem_size, self.ctx))
    }

    fn eval_triop(
        &mut self,
        callbacks: &PythonCallbacks,
        op: IROp,
        arg1: &IRExpr,
        arg2: &IRExpr,
        arg3: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        record_vex_triop(iropclass(&op));
        // VEX Triops are float arithmetic with a rounding mode:
        // (rm, a, b). For FAdd/FSub/FMul/FDiv we route through
        // `binop_with_rm` which honors the VEX rm bits when non-RNE;
        // RNE keeps the native-f{32,64} fast path. Other Triops
        // ignore rm and fall through to `binop`.
        let rm = self.eval_expr_with_callbacks(callbacks, arg1, tyenv)?;
        let v2 = self.eval_expr_with_callbacks(callbacks, arg2, tyenv)?;
        let v3 = self.eval_expr_with_callbacks(callbacks, arg3, tyenv)?;
        let any_sym = v2.is_symbolic() || v3.is_symbolic() || rm.is_symbolic();
        let width = op.result_type().map(|t| t.bits()).unwrap_or(64);
        match VEXOps::binop_with_rm(op, rm, v2, v3, self.ctx) {
            Ok(v) => Ok(v),
            // NEON scaffolding: surface explicitly. See
            // `invariant-neon-scaffolding-panic-not-fallback`.
            Err(e @ OpError::UnsupportedNeon { .. }) => Err(CbExecutionError::Op(e)),
            // angr-tkbr.2: unmapped pyvex opcode — propagate past
            // the silent fresh-symbolic fallback so the failure carries
            // op + arch (typed RustUnsupportedVexOpError on the test
            // path; stringified into the errored stash live).
            Err(e @ OpError::UnsupportedVexOp { .. }) => Err(CbExecutionError::Op(e)),
            Err(e) => {
                self.stats.python_vex_triop_fallback_count += 1;
                self.vex_op_fallback(e, any_sym, width, format!("unsup_triop_{:x}", self.pc))
            }
        }
    }

    /// `args` is the Qop's four operands in IR order (`arg1..arg4`).
    fn eval_qop(
        &mut self,
        callbacks: &PythonCallbacks,
        op: IROp,
        args: [&IRExpr; 4],
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        record_vex_qop(iropclass(&op));
        // VEX Qops are typically fused multiply-add/sub with a
        // rounding mode: (rm, a, b, c). Drop rm for the same reason
        // as Triop above.
        let _rm = self.eval_expr_with_callbacks(callbacks, args[0], tyenv)?;
        let v2 = self.eval_expr_with_callbacks(callbacks, args[1], tyenv)?;
        let v3 = self.eval_expr_with_callbacks(callbacks, args[2], tyenv)?;
        let v4 = self.eval_expr_with_callbacks(callbacks, args[3], tyenv)?;
        let any_sym = v2.is_symbolic() || v3.is_symbolic() || v4.is_symbolic();
        let width = op.result_type().map(|t| t.bits()).unwrap_or(64);
        match VEXOps::qop(op, v2, v3, v4, self.ctx) {
            Ok(v) => Ok(v),
            // NEON scaffolding: surface explicitly. See
            // `invariant-neon-scaffolding-panic-not-fallback`.
            Err(e @ OpError::UnsupportedNeon { .. }) => Err(CbExecutionError::Op(e)),
            // angr-tkbr.2: unmapped pyvex opcode — propagate past
            // the silent fresh-symbolic fallback so the failure carries
            // op + arch (typed RustUnsupportedVexOpError on the test
            // path; stringified into the errored stash live).
            Err(e @ OpError::UnsupportedVexOp { .. }) => Err(CbExecutionError::Op(e)),
            Err(e) => {
                self.stats.python_vex_qop_fallback_count += 1;
                self.vex_op_fallback(e, any_sym, width, format!("unsup_qop_{:x}", self.pc))
            }
        }
    }

    fn eval_ccall(
        &mut self,
        callbacks: &PythonCallbacks,
        cee: &IRCallee,
        retty: IRType,
        args: &[IRExpr],
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        let mut arg_vals = Vec::with_capacity(args.len());
        for arg in args {
            arg_vals.push(self.eval_expr_with_callbacks(callbacks, arg, tyenv)?);
        }

        if let Some(result) =
            ccall::handle_ccall_with_ctx(&cee.name, &arg_vals, retty.bits(), Some(self.ctx))
        {
            return Ok(result);
        }

        // eflags/rflags condition-calculation CCalls that we couldn't handle
        // symbolically follow the same policy as `vex_op_fallback` (angr-oyzvj,
        // angr-9ke6b.88): route the block to Python by default. Fabricating a
        // fresh unconstrained symbolic here is *worse* than for an ordinary op,
        // because the result of a `calculate_condition` CCall IS a branch
        // guard — an unconstrained one explores both directions regardless of
        // the real flag semantics. Retained only as an explicit opt-in via
        // `ANGR_RUST_FABRICATE_UNSUPPORTED_IROP` (same gate, same
        // `vex_bypass_fabricate_count` visibility).
        let is_cond_ccall = cee.name.contains("calculate_condition")
            || cee.name.contains("calculate_eflags")
            || cee.name.contains("calculate_rflags");
        if is_cond_ccall && fabricate_unsupported_irop() {
            log::debug!(
                "CCall '{}' not handled symbolically at 0x{:x}, fabricating symbolic variable \
                 (ANGR_RUST_FABRICATE_UNSUPPORTED_IROP)",
                cee.name,
                self.pc
            );
            self.stats.vex_bypass_fabricate_count += 1;
            return Ok(RustBV::symbolic(
                self.ctx,
                format!("ccall_unsupported_{:x}", self.pc),
                retty.bits(),
            ));
        }

        // Every other unsupported CCall must defer to Python's VEX engine.
        // Returning concrete(0) would silently corrupt the result and let
        // execution continue with bad data.
        Err(CbExecutionError::NeedPythonFallback(format!(
            "unsupported CCall '{}' at 0x{:x}",
            cee.name, self.pc
        )))
    }

    /// Convert one batched-load result entry to a RustBV.
    ///
    /// Branches: missing index (fresh symbolic), concrete bytes, or symbolic AST
    /// (delegates to `try_convert_symbolic_value`).
    fn convert_load_result(
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

    /// Build an ITE chain for symbolic memory load by loading each candidate address.
    ///
    /// This builds the ITE chain entirely in Rust instead of delegating to Python.
    /// For each candidate address, we load the value via callback and create an ITE:
    /// `ITE(addr == a1, mem[a1], ITE(addr == a2, mem[a2], ...))`
    ///
    /// This is more efficient than calling Python's symbolic memory handler because:
    /// 1. We avoid FFI overhead for the ITE chain construction
    /// 2. The RustBV ITE nodes stay in Rust's Z3 context
    /// 3. We can use balanced ITE trees for better solver performance
    fn build_ite_load_from_callbacks(
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

        Ok(build_balanced_ite(
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
        &self,
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

            callbacks
                .call_memory_store_symbolic_value(addr, &ite_value)
                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
        }

        Ok(())
    }

    /// Evaluate an IR constant.
    fn eval_const(&self, c: &IRConst) -> RustBV {
        match c {
            IRConst::U1(v) => RustBV::concrete(*v as u128, 1),
            IRConst::U8(v) => RustBV::concrete(*v as u128, 8),
            IRConst::U16(v) => RustBV::concrete(*v as u128, 16),
            IRConst::U32(v) => RustBV::concrete(*v as u128, 32),
            IRConst::U64(v) => RustBV::concrete(*v as u128, 64),
            IRConst::U128(v) => RustBV::concrete(*v, 128),
            IRConst::F32(v) => RustBV::concrete(v.to_bits() as u128, 32),
            IRConst::F64(v) => RustBV::concrete(v.to_bits() as u128, 64),
            IRConst::V128(v) => RustBV::concrete(*v, 128),
            IRConst::V256(v) => RustBV::concrete(v[0] as u128 | ((v[1] as u128) << 64), 128),
        }
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
    fn try_rust_memory_load(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        size: usize,
        load_start: Option<Instant>,
    ) -> Result<Option<RustBV>, CbExecutionError> {
        // AVOID_MULTIVALUED_READS: bypass `load_symbolic_unified` (and its
        // concretization) for symbolic addresses and return unconstrained.
        // Matches Python's `address_concretization_mixin._load_one` early
        // `return self._default_value(...)`.
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
            Err(e) => Err(CbExecutionError::Memory(e)),
        }
    }

    pub(super) fn eval_expr_simple(
        &self,
        expr: &IRExpr,
        _tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        match expr {
            IRExpr::Const(c) => Ok(self.eval_const(c)),
            IRExpr::RdTmp(tmp) => {
                if let Some(Some(val)) = self.temps.get(*tmp as usize) {
                    Ok(val.clone())
                } else {
                    Err(CbExecutionError::UnknownTemp(*tmp))
                }
            }
            IRExpr::Get { offset, ty } => {
                let size = ty.bytes();
                Ok(self.registers.get(*offset, size, self.ctx))
            }
            _ => Err(CbExecutionError::Unsupported(
                "complex expr in default exit".to_string(),
            )),
        }
    }

    /// Shared prelude for the `dispatch_*_inspect` methods: gate on
    /// `event_bit`, import claripy, and convert `value` into a claripy AST.
    /// Returns `None` when the breakpoint is disabled or the import/convert
    /// fails (the callers all swallow those failures — a missing claripy or a
    /// conversion error must not halt exploration). Centralizes the
    /// claripy-import-failure swallow policy that was previously open-coded at
    /// every dispatch site.
    pub(super) fn inspect_ast(
        &self,
        callbacks: &PythonCallbacks,
        event_bit: u8,
        value: &RustBV,
    ) -> Option<Py<PyAny>> {
        if !callbacks.inspect_event_enabled(event_bit) {
            return None;
        }
        Python::attach(|py| {
            let claripy_mod = py.import("claripy").ok()?;
            crate::claripy_bridge::rustbv_to_claripy(py, value, &claripy_mod).ok()
        })
    }

    /// Fire a `mem_read` inspect callback into Python for this load.
    ///
    /// Mirrors `dispatch_mem_write_inspect` in `statements.rs`. Gated on
    /// `inspect_event_enabled(MemRead)` so the no-breakpoint case costs a
    /// single bitmask test per Load. Symbolic addresses are skipped for
    /// the MVP — only concrete addresses dispatch. The `when='after'`
    /// event is fired once the value has been computed; the BP receives
    /// the loaded value AST as `mem_read_expr`. Errors from the Python
    /// callback are swallowed and logged on the Python side; a user BP
    /// error must not halt exploration.
    ///
    /// Returns `Some(bv)` when the user's BP action overrode
    /// `state.inspect.mem_read_expr` (value injection — angr-uy32); the
    /// caller substitutes it for the loaded value. Returns `None` when the
    /// value is unchanged, so the original load result stands.
    fn dispatch_mem_read_inspect(
        &self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        value: &RustBV,
        size: usize,
        endness: Endness,
    ) -> Option<RustBV> {
        // MemRead = InspectEvent variant 0 — see crate::state::InspectEvent.
        let value_ast = self.inspect_ast(callbacks, 0, value)?;
        let addr_u64 = addr_val.as_u64()?;
        let endness_str = match endness {
            Endness::Little => "Iend_LE",
            Endness::Big => "Iend_BE",
        };
        let mutated = callbacks
            .call_inspect_mem_read(
                self.current_state_id,
                "after",
                addr_u64,
                size as u32,
                Some(&value_ast),
                endness_str,
            )
            .ok()??;
        // The user injected a new value via state.inspect.mem_read_expr.
        // Convert it back to a RustBV; reject a width mismatch defensively
        // so a bad override can't silently corrupt downstream ops.
        let bv = Python::attach(|py| {
            let bound = mutated.bind(py);
            crate::claripy_bridge::claripy_to_rustbv(py, bound, self.ctx).ok()
        })?;
        if bv.width() == (size * 8) as u32 {
            Some(bv)
        } else {
            None
        }
    }

    /// Fire a `reg_read` inspect callback into Python for a VEX `Get`.
    ///
    /// Gated on `inspect_event_enabled(RegRead)` so the no-breakpoint case
    /// is one bitmask test per `IRExpr::Get`. Dispatches `when='after'`
    /// with the loaded register value as `reg_read_expr`. Errors from the
    /// Python callback are swallowed and logged on the Python side.
    fn dispatch_reg_read_inspect(
        &self,
        callbacks: &PythonCallbacks,
        offset: u32,
        size: u32,
        value: &RustBV,
    ) {
        // RegRead = InspectEvent variant 2.
        let Some(value_ast) = self.inspect_ast(callbacks, 2, value) else {
            return;
        };
        let _ = callbacks.call_inspect_reg_read(
            self.current_state_id,
            "after",
            offset,
            size,
            Some(&value_ast),
        );
    }

    /// Fire a `tmp_read` inspect callback for a VEX `RdTmp` (angr-64pi).
    ///
    /// Gated on `inspect_event_enabled(13)` so the no-breakpoint case is
    /// one bitmask test per `RdTmp` evaluation. Dispatches `when='after'`
    /// with the tmp's stored value as `tmp_read_expr`. RdTmp can fire many
    /// times per IRSB (every binop/load/store args go through it); the
    /// claripy AST round-trip is therefore only done when a BP is set.
    fn dispatch_tmp_read_inspect(&self, callbacks: &PythonCallbacks, tmp_num: u32, value: &RustBV) {
        // TmpRead bit assigned in _INSPECT_EVENT_SPECS.
        let Some(value_ast) = self.inspect_ast(callbacks, 13, value) else {
            return;
        };
        let _ = callbacks.call_inspect_tmp_read(
            self.current_state_id,
            "after",
            tmp_num,
            Some(&value_ast),
        );
    }

    /// Fire an `expr` inspect callback for a VEX IRExpr eval (angr-lge2).
    ///
    /// Gated on `inspect_event_enabled(16)` so the no-BP case is one
    /// bitmask test per `eval_expr_with_callbacks` call (the most frequent
    /// dispatch site in the engine — fires for every constant, RdTmp,
    /// register read, load, unop, binop, ITE, etc.). When fired, the
    /// computed RustBV is reconstructed as a claripy AST and passed as
    /// `expr_result`. The original `IRExpr` is intentionally NOT passed
    /// (Rust IRExpr doesn't round-trip cleanly into a `pyvex.IRExpr`);
    /// the BP receives `expr=None` and only the computed value.
    fn dispatch_expr_inspect(&self, callbacks: &PythonCallbacks, value: &RustBV) {
        let Some(value_ast) = self.inspect_ast(callbacks, 16, value) else {
            return;
        };
        let _ = callbacks.call_inspect_expr(self.current_state_id, "after", Some(&value_ast));
    }

    /// Fire an `address_concretization` inspect callback (angr-vfst).
    ///
    /// Gated on `inspect_event_enabled(17)`. The address AST is round-tripped
    /// into a claripy reconstruction for the BP; `result` carries the list
    /// of concrete addresses produced by the concretizer (`None` on
    /// `when="before"`). Mirrors `address_concretization_mixin.py:156-180`'s
    /// BEFORE/AFTER pattern; the strategy / memory / add_constraints attrs
    /// are passed as None because the Rust engine doesn't expose those
    /// objects to BPs (MVP gap, documented in `rust_engine.rst`).
    pub(super) fn dispatch_address_concretization_inspect(
        &self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        action: &str,
        when: &str,
        result: Option<Vec<u64>>,
    ) {
        let Some(addr_ast) = self.inspect_ast(callbacks, 17, addr_val) else {
            return;
        };
        let _ = callbacks.call_inspect_address_concretization(
            self.current_state_id,
            when,
            action,
            &addr_ast,
            result,
        );
    }

    /// Fire a `symbolic_variable` inspect callback (angr-vfst).
    ///
    /// Gated on `inspect_event_enabled(18)`. Fires `when="after"` when the
    /// Rust engine mints a fresh BVS internally — the most common dispatch
    /// site is `load_from_callback`'s fresh-symbol fallback when Python
    /// returns `is_symbolic=True` with no AST. Mirrors
    /// `solver.py:432-439`'s BP_AFTER signature.
    pub(super) fn dispatch_symbolic_variable_inspect(
        &self,
        callbacks: &PythonCallbacks,
        name: &str,
        size_bits: u32,
        value: &RustBV,
    ) {
        let Some(expr_ast) = self.inspect_ast(callbacks, 18, value) else {
            return;
        };
        let _ = callbacks.call_inspect_symbolic_variable(
            self.current_state_id,
            "after",
            name,
            size_bits,
            &expr_ast,
        );
    }
}

#[cfg(test)]
#[path = "expressions_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod expressions_tests;
