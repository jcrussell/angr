use super::helpers::{build_balanced_ite, bytes_to_bv};
use super::*;

impl<'a> CallbackInterpreter<'a> {
    /// Evaluate an IR expression using Python callbacks for memory loads.
    pub(super) fn eval_expr_with_callbacks(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        expr: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        let expr_start = if self.profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let result = self.eval_expr_with_callbacks_inner(py, callbacks, expr, tyenv);
        if let Some(start) = expr_start {
            self.stats.expr_eval_time_ns += start.elapsed().as_nanos() as u64;
            self.stats.expr_eval_count += 1;
        }
        result
    }

    fn eval_expr_with_callbacks_inner(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        expr: &IRExpr,
        tyenv: &TypeEnv,
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

            IRExpr::Load { addr, ty, .. } => {
                let load_start = if self.profiling_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, tyenv)?;
                let size = ty.bytes() as usize;
                if self.profiling_enabled {
                    self.stats.load_stmt_count += 1;
                }

                // Try Rust-native memory first if enabled - mirrors try_rust_memory_store
                if self.use_rust_memory {
                    if let Some(value) =
                        self.try_rust_memory_load(py, callbacks, &addr_val, size, load_start)?
                    {
                        return Ok(value);
                    }
                }

                if let Some(addr_concrete) = addr_val.as_u64() {
                    // FAST PATH 0: Check pending stores buffer
                    // Stores within the same block are buffered in pending_stores.
                    // We must check this buffer before falling through to Python
                    // callbacks, which have stale state.

                    // First check symbolic stores (preserves symbolic values)
                    if let Some(sym_val) = self.pending_symbolic_stores.get(&addr_concrete) {
                        if sym_val.width() == (size * 8) as u32 {
                            return Ok(sym_val.clone());
                        } else if sym_val.width() > (size * 8) as u32 {
                            return Ok(sym_val.extract((size * 8 - 1) as u32, 0, self.ctx));
                        }
                    }

                    // Then check concrete stores via the indexed buffer.
                    // try_load fast-skips when no pending store overlaps the
                    // load address; falls back to a reverse scan only when the
                    // most recent covering store is smaller than the load.
                    if let Some(data) = self.pending_stores.try_load(addr_concrete, size) {
                        return Ok(bytes_to_bv(data, (size * 8) as u32));
                    }

                    // Also check previously flushed symbolic stores (cross-block)
                    if let Some(sym_val) = self.all_flushed_symbolic_stores.get(&addr_concrete) {
                        if sym_val.width() == (size * 8) as u32 {
                            return Ok(sym_val.clone());
                        } else if sym_val.width() > (size * 8) as u32 {
                            return Ok(sym_val.extract((size * 8 - 1) as u32, 0, self.ctx));
                        }
                    }

                    // Also check previously flushed concrete stores (cross-block)
                    if let Some(store_data) = self.all_flushed_stores.get(&addr_concrete) {
                        if size <= store_data.len() {
                            let data = &store_data[..size];
                            return Ok(bytes_to_bv(data, (size * 8) as u32));
                        }
                    }

                    // FAST PATH 1: Check prefetch cache (batch-loaded values)
                    if let Some(prefetched) = self.load_prefetch_cache.get(&(addr_concrete, size)) {
                        return Ok(prefetched.value.clone());
                    }

                    // FAST PATH 2: Check if address is in Rust-cached concrete memory
                    if let Some(data) = self.try_read_concrete_memory(addr_concrete, size) {
                        return Ok(bytes_to_bv(data, (size * 8) as u32));
                    }
                    // SLOW PATH: Fall back to Python callback
                    self.load_from_callback(py, callbacks, addr_concrete, size)
                } else {
                    // Symbolic address - try to concretize for read
                    match &*self.concretize_cached_read(&addr_val) {
                        ConcretizationResult::Single(addr_concrete) => {
                            let addr_concrete = *addr_concrete;
                            if self.arch.pointer_size() == 32
                                && addr_concrete >= 0x400000
                                && addr_concrete < 0x420000
                            {}
                            self.track_concretization_constraint(&addr_val, addr_concrete);
                            if let Some(data) = self.try_read_concrete_memory(addr_concrete, size) {
                                return Ok(bytes_to_bv(data, (size * 8) as u32));
                            }
                            self.load_from_callback(py, callbacks, addr_concrete, size)
                        }
                        ConcretizationResult::Multiple(addrs) => {
                            // Sync constraints before batch load
                            self.sync_before_callback(py, callbacks)?;
                            // Build ITE chain in Rust instead of delegating to Python
                            // This avoids FFI overhead and keeps symbolic ops in Rust's Z3 context
                            self.build_ite_load_from_callbacks(
                                py, callbacks, addrs, &addr_val, size,
                            )
                        }
                        ConcretizationResult::Strided {
                            base,
                            stride,
                            count,
                        } => {
                            // Sync constraints before batch load
                            self.sync_before_callback(py, callbacks)?;
                            // Strided access pattern - generate addresses and build ITE chain in Rust
                            let addrs: Vec<u64> = (0..*count).map(|i| base + i * stride).collect();
                            self.build_ite_load_from_callbacks(
                                py, callbacks, &addrs, &addr_val, size,
                            )
                        }
                        ConcretizationResult::TooLarge { min, max, .. } => {
                            let descr = format!("range 0x{:x}-0x{:x}", min, max);
                            self.fallback_load_symbolic_full(
                                py, callbacks, &addr_val, size, "Load", &descr,
                            )
                        }
                        ConcretizationResult::Failed(reason) => {
                            // Concretization failed entirely (e.g., timeout, no
                            // strategy applies). Try the full symbolic load callback;
                            // Python's memory model can still resolve it via its
                            // own address concretization strategies.
                            let descr = format!("concretize failed: {}", reason);
                            self.fallback_load_symbolic_full(
                                py, callbacks, &addr_val, size, "Load", &descr,
                            )
                        }
                    }
                }
            }

            IRExpr::Unop { op, arg } => {
                let arg_val = self.eval_expr_with_callbacks(py, callbacks, arg, tyenv)?;
                let arg_is_sym = arg_val.is_symbolic();
                match VEXOps::unop(*op, arg_val, self.ctx) {
                    Ok(v) => Ok(v),
                    Err(_) => {
                        // Fallback for unsupported unary ops (e.g., float conversions).
                        // Return fresh symbolic if input was symbolic, else zero.
                        self.stats.python_vex_op_fallback_count += 1;
                        self.stats.python_vex_unop_fallback_count += 1;
                        let width = op.result_type().map(|t| t.bits()).unwrap_or(64);
                        if arg_is_sym {
                            Ok(RustBV::symbolic(
                                self.ctx,
                                format!("unsup_unop_{:x}", self.pc),
                                width,
                            ))
                        } else {
                            Ok(RustBV::concrete(0, width))
                        }
                    }
                }
            }

            IRExpr::Binop { op, left, right } => {
                let left_val = self.eval_expr_with_callbacks(py, callbacks, left, tyenv)?;
                let right_val = self.eval_expr_with_callbacks(py, callbacks, right, tyenv)?;
                let fallback_width = op
                    .result_type()
                    .map(|t| t.bits())
                    .unwrap_or(left_val.width().max(right_val.width()));
                let any_sym = left_val.is_symbolic() || right_val.is_symbolic();
                match VEXOps::binop(*op, left_val, right_val, self.ctx) {
                    Ok(v) => Ok(v),
                    Err(_) => {
                        // Fallback for unsupported binary ops (e.g., vector float ops).
                        self.stats.python_vex_op_fallback_count += 1;
                        self.stats.python_vex_binop_fallback_count += 1;
                        if any_sym {
                            Ok(RustBV::symbolic(
                                self.ctx,
                                format!("unsup_binop_{:x}", self.pc),
                                fallback_width,
                            ))
                        } else {
                            Ok(RustBV::concrete(0, fallback_width))
                        }
                    }
                }
            }

            IRExpr::ITE {
                cond,
                iftrue,
                iffalse,
            } => {
                let cond_val = self.eval_expr_with_callbacks(py, callbacks, cond, tyenv)?;
                // Short-circuit: skip evaluating the dead branch when condition is concrete
                if let Some(v) = cond_val.as_u128() {
                    return if v != 0 {
                        self.eval_expr_with_callbacks(py, callbacks, iftrue, tyenv)
                    } else {
                        self.eval_expr_with_callbacks(py, callbacks, iffalse, tyenv)
                    };
                }
                let true_val = self.eval_expr_with_callbacks(py, callbacks, iftrue, tyenv)?;
                let false_val = self.eval_expr_with_callbacks(py, callbacks, iffalse, tyenv)?;
                Ok(cond_val.ite(&true_val, &false_val, self.ctx))
            }

            IRExpr::GetI { descr, ix, bias } => {
                // Evaluate the index expression
                let ix_val = self.eval_expr_with_callbacks(py, callbacks, ix, tyenv)?;

                // GetI requires a concrete index to compute the register offset
                let idx = if let Some(idx) = ix_val.as_u64() {
                    idx
                } else {
                    // Symbolic index - concretize using solver
                    if let Some(concrete) = self.ctx.eval(&ix_val) {
                        concrete as u64
                    } else {
                        return Err(CbExecutionError::Unsupported(
                            "GetI index concretization failed".to_string(),
                        ));
                    }
                };

                // Calculate the rotating register offset:
                // offset = base + ((idx + bias) % nElems) * elemTy.bytes()
                let elem_size = descr.elemTy.bytes();
                let index = ((idx as u32).wrapping_add(*bias)) % descr.nElems;
                let offset = descr.base + index * elem_size;

                // Read from the register file
                Ok(self.registers.get(offset, elem_size, self.ctx))
            }

            IRExpr::Triop {
                op,
                arg1,
                arg2,
                arg3,
            } => {
                // VEX Triops are float arithmetic with a rounding mode:
                // (rm, a, b). For FAdd/FSub/FMul/FDiv we route through
                // `binop_with_rm` which honors the VEX rm bits when non-RNE;
                // RNE keeps the native-f{32,64} fast path. Other Triops
                // ignore rm and fall through to `binop`.
                let rm = self.eval_expr_with_callbacks(py, callbacks, arg1, tyenv)?;
                let v2 = self.eval_expr_with_callbacks(py, callbacks, arg2, tyenv)?;
                let v3 = self.eval_expr_with_callbacks(py, callbacks, arg3, tyenv)?;
                let any_sym = v2.is_symbolic() || v3.is_symbolic() || rm.is_symbolic();
                let width = op.result_type().map(|t| t.bits()).unwrap_or(64);
                match VEXOps::binop_with_rm(*op, rm, v2, v3, self.ctx) {
                    Ok(v) => Ok(v),
                    Err(_) => {
                        self.stats.python_vex_op_fallback_count += 1;
                        self.stats.python_vex_triop_fallback_count += 1;
                        if any_sym {
                            Ok(RustBV::symbolic(
                                self.ctx,
                                format!("triop_{:x}", self.pc),
                                width,
                            ))
                        } else {
                            Ok(RustBV::concrete(0, width))
                        }
                    }
                }
            }

            IRExpr::Qop {
                op,
                arg1,
                arg2,
                arg3,
                arg4,
            } => {
                // VEX Qops are typically fused multiply-add/sub with a
                // rounding mode: (rm, a, b, c). Drop rm for the same reason
                // as Triop above.
                let _rm = self.eval_expr_with_callbacks(py, callbacks, arg1, tyenv)?;
                let v2 = self.eval_expr_with_callbacks(py, callbacks, arg2, tyenv)?;
                let v3 = self.eval_expr_with_callbacks(py, callbacks, arg3, tyenv)?;
                let v4 = self.eval_expr_with_callbacks(py, callbacks, arg4, tyenv)?;
                let any_sym = v2.is_symbolic() || v3.is_symbolic() || v4.is_symbolic();
                let width = op.result_type().map(|t| t.bits()).unwrap_or(64);
                match VEXOps::qop(*op, v2, v3, v4, self.ctx) {
                    Ok(v) => Ok(v),
                    Err(_) => {
                        self.stats.python_vex_op_fallback_count += 1;
                        self.stats.python_vex_qop_fallback_count += 1;
                        if any_sym {
                            Ok(RustBV::symbolic(
                                self.ctx,
                                format!("qop_{:x}", self.pc),
                                width,
                            ))
                        } else {
                            Ok(RustBV::concrete(0, width))
                        }
                    }
                }
            }

            IRExpr::CCall { cee, retty, args } => {
                let mut arg_vals = Vec::with_capacity(args.len());
                for arg in args {
                    arg_vals.push(self.eval_expr_with_callbacks(py, callbacks, arg, tyenv)?);
                }

                if let Some(result) =
                    ccall::handle_ccall_with_ctx(&cee.name, &arg_vals, retty.bits(), Some(self.ctx))
                {
                    return Ok(result);
                }

                // For eflags/rflags CCalls that we couldn't handle symbolically,
                // return a fresh symbolic variable rather than concrete 0.
                // Concrete 0 corrupts register values; a symbolic variable is sound
                // (unconstrained) and lets the solver handle it.
                let is_cond_ccall = cee.name.contains("calculate_condition")
                    || cee.name.contains("calculate_eflags")
                    || cee.name.contains("calculate_rflags");
                if is_cond_ccall {
                    log::debug!(
                        "CCall '{}' not handled symbolically at 0x{:x}, returning symbolic variable",
                        cee.name,
                        self.pc
                    );
                    return Ok(RustBV::symbolic(
                        self.ctx,
                        format!("ccall_unsupported_{:x}", self.pc),
                        retty.bits(),
                    ));
                }

                // Any other unsupported CCall must defer to Python's VEX engine.
                // Returning concrete(0) would silently corrupt the result and let
                // execution continue with bad data.
                Err(CbExecutionError::NeedPythonFallback(format!(
                    "unsupported CCall '{}' at 0x{:x}",
                    cee.name, self.pc
                )))
            }

            IRExpr::VECRET | IRExpr::GSPTR => {
                // P7 fix: Request Python fallback instead of failing
                // These special expressions require Python's VEX handling
                Err(CbExecutionError::NeedPythonFallback(format!(
                    "special expr {:?} requires Python",
                    expr
                )))
            }
        }
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
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addrs: &[u64],
        addr_expr: &RustBV,
        size: usize,
    ) -> Result<RustBV, CbExecutionError> {
        if addrs.is_empty() {
            return Err(CbExecutionError::Memory(
                "no candidate addresses".to_string(),
            ));
        }

        let width = (size * 8) as u32;
        let addr_width = addr_expr.width();

        // For a single address, just load it directly
        if addrs.len() == 1 {
            return self.load_from_callback(py, callbacks, addrs[0], size);
        }

        // Batch load all addresses at once for efficiency
        let load_requests: Vec<(u64, u32)> = addrs.iter().map(|&a| (a, size as u32)).collect();
        let load_results = callbacks
            .call_memory_load_batch(py, &load_requests)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        // Build (condition, value) pairs for the ITE chain
        let mut pairs: Vec<(RustBV, RustBV)> = Vec::with_capacity(addrs.len());

        for (i, addr) in addrs.iter().enumerate() {
            // Get the loaded value for this address
            let value = if i < load_results.len() {
                let (data, is_symbolic, symbolic_ast) = &load_results[i];
                if *is_symbolic {
                    // Try to convert to RustBV - check handle first (fast path), then claripy (slow path)
                    if let Some(ast_obj) = symbolic_ast {
                        let ast = ast_obj.bind(py);

                        // Fast path: check for RustBVHandle first
                        if let Some(ref table) = self.symbol_table {
                            if let Some(bv) = try_handle_to_rustbv(&ast, table) {
                                bv
                            } else if is_claripy_ast(&ast) {
                                // Slow path: claripy AST conversion
                                match claripy_to_rustbv(py, &ast, self.ctx) {
                                    Ok(bv) => bv,
                                    Err(_) => {
                                        // Fallback to fresh symbolic
                                        RustBV::symbolic(
                                            self.ctx,
                                            format!("ite_load_{:x}_{}", addr, size),
                                            width,
                                        )
                                    }
                                }
                            } else {
                                RustBV::symbolic(
                                    self.ctx,
                                    format!("ite_load_{:x}_{}", addr, size),
                                    width,
                                )
                            }
                        } else if is_claripy_ast(&ast) {
                            match claripy_to_rustbv(py, &ast, self.ctx) {
                                Ok(bv) => bv,
                                Err(_) => {
                                    // Fallback to fresh symbolic
                                    RustBV::symbolic(
                                        self.ctx,
                                        format!("ite_load_{:x}_{}", addr, size),
                                        width,
                                    )
                                }
                            }
                        } else {
                            RustBV::symbolic(
                                self.ctx,
                                format!("ite_load_{:x}_{}", addr, size),
                                width,
                            )
                        }
                    } else {
                        RustBV::symbolic(self.ctx, format!("ite_load_{:x}_{}", addr, size), width)
                    }
                } else {
                    bytes_to_bv(data, width)
                }
            } else {
                // Missing result - create symbolic placeholder
                RustBV::symbolic(self.ctx, format!("ite_load_{:x}_{}", addr, size), width)
            };

            // Build condition: addr_expr == this address
            let addr_const = RustBV::concrete(*addr as u128, addr_width);
            let cond = addr_expr.eq(&addr_const, self.ctx);

            pairs.push((cond, value));
        }

        // Use the last value as default (for robustness, though one condition should always match)
        let default_value = pairs
            .last()
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| RustBV::symbolic(self.ctx, "ite_default", width));

        // Build balanced ITE tree for better solver performance
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
        &mut self,
        py: Python<'_>,
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

        // Batch load current values at all candidate addresses
        let load_requests: Vec<(u64, u32)> = addrs.iter().map(|&a| (a, size as u32)).collect();
        let load_results = callbacks
            .call_memory_load_batch(py, &load_requests)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        // For each candidate address, build ITE and store back
        for (i, &addr) in addrs.iter().enumerate() {
            // Build condition: addr_expr == this address
            let addr_const = RustBV::concrete(addr as u128, addr_width);
            let cond = addr_expr.eq(&addr_const, self.ctx);

            // Get current value at this address
            let current = if i < load_results.len() {
                let (data, is_symbolic, symbolic_ast) = &load_results[i];
                if *is_symbolic {
                    if let Some(ast_obj) = symbolic_ast {
                        let ast = ast_obj.bind(py);
                        if let Some(ref table) = self.symbol_table {
                            if let Some(bv) = try_handle_to_rustbv(&ast, table) {
                                bv
                            } else if is_claripy_ast(&ast) {
                                claripy_to_rustbv(py, &ast, self.ctx).unwrap_or_else(|_| {
                                    RustBV::symbolic(
                                        self.ctx,
                                        format!("ite_store_cur_{:x}", addr),
                                        data_val.width(),
                                    )
                                })
                            } else {
                                RustBV::symbolic(
                                    self.ctx,
                                    format!("ite_store_cur_{:x}", addr),
                                    data_val.width(),
                                )
                            }
                        } else if is_claripy_ast(&ast) {
                            claripy_to_rustbv(py, &ast, self.ctx).unwrap_or_else(|_| {
                                RustBV::symbolic(
                                    self.ctx,
                                    format!("ite_store_cur_{:x}", addr),
                                    data_val.width(),
                                )
                            })
                        } else {
                            RustBV::symbolic(
                                self.ctx,
                                format!("ite_store_cur_{:x}", addr),
                                data_val.width(),
                            )
                        }
                    } else {
                        RustBV::symbolic(
                            self.ctx,
                            format!("ite_store_cur_{:x}", addr),
                            data_val.width(),
                        )
                    }
                } else {
                    bytes_to_bv(data, data_val.width())
                }
            } else {
                RustBV::symbolic(
                    self.ctx,
                    format!("ite_store_cur_{:x}", addr),
                    data_val.width(),
                )
            };

            // Build ITE: if (addr == candidate) then new_data else current
            let ite_value = cond.ite(data_val, &current, self.ctx);

            // Store via symbolic value callback
            callbacks
                .call_memory_store_symbolic_value(py, addr, &ite_value)
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

    /// Resolve a LoadG load given its address BV. Handles concrete addresses,
    /// Single/Multiple concretizations, and falls back to the Python full
    /// symbolic load callback for TooLarge / Strided / Failed shapes (so the
    /// load no longer hard-errors when angr's address strategies could resolve
    /// it). Multiple addresses still take the first solution to preserve the
    /// pre-existing LoadG behavior — broader Multiple handling can be added
    /// later if needed.
    pub(super) fn resolve_loadg_load(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        load_size: usize,
        context: &str,
    ) -> Result<RustBV, CbExecutionError> {
        if let Some(addr_concrete) = addr_val.as_u64() {
            return self.load_from_callback(py, callbacks, addr_concrete, load_size);
        }
        let conc = self.concretize_cached_read(addr_val);
        match &*conc {
            ConcretizationResult::Single(a) => {
                let a = *a;
                self.track_concretization_constraint(addr_val, a);
                self.load_from_callback(py, callbacks, a, load_size)
            }
            ConcretizationResult::Multiple(addrs) => {
                let a = *addrs.first().ok_or_else(|| {
                    CbExecutionError::Unsupported(format!("{} with empty address set", context))
                })?;
                self.load_from_callback(py, callbacks, a, load_size)
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                let descr = format!(
                    "strided base=0x{:x} stride=0x{:x} count={}",
                    base, stride, count
                );
                self.fallback_load_symbolic_full(
                    py, callbacks, addr_val, load_size, context, &descr,
                )
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                let descr = format!("range 0x{:x}-0x{:x}", min, max);
                self.fallback_load_symbolic_full(
                    py, callbacks, addr_val, load_size, context, &descr,
                )
            }
            ConcretizationResult::Failed(reason) => {
                let descr = format!("concretize failed: {}", reason);
                self.fallback_load_symbolic_full(
                    py, callbacks, addr_val, load_size, context, &descr,
                )
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
                IRLoadGOp::Identity => value, // Should not happen if sizes differ
                IRLoadGOp::WidenS => value.sign_extend(target_bits, self.ctx),
                IRLoadGOp::WidenZ => value.zero_extend(target_bits, self.ctx),
            }
        }
    }

    /// Attempt to load via Rust-native memory. Returns `Ok(Some(bv))` if Rust
    /// handled the load, `Ok(None)` if the caller should fall back to the
    /// Python path, or `Err` for unrecoverable errors.
    fn try_rust_memory_load(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        size: usize,
        load_start: Option<Instant>,
    ) -> Result<Option<RustBV>, CbExecutionError> {
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
                if let Some(start) = load_start {
                    self.stats.load_stmt_time_ns += start.elapsed().as_nanos() as u64;
                }
                Ok(Some(value))
            }
            Err(MemoryError::UnmappedPageInRegion { page_addr }) => {
                // Page is in a lazy region - fetch it (rust_mem borrow is dropped here)
                let prefetch_count = self.page_prefetch_count;
                let page_fetched =
                    self.fetch_page_with_prefetch(py, callbacks, page_addr, prefetch_count)?;

                // NOTE: We intentionally do NOT auto-map zero pages when page_fetched is false.
                // Python may have actual data for this page from backers (file contents,
                // initialized data). Speculatively creating zero pages causes state
                // divergence between Rust and Python. Instead, we fall through to
                // the Python callback which handles memory correctly.

                if page_fetched {
                    if let Some(ref mut rust_mem) = self.rust_memory {
                        if let Ok(value) = rust_mem.load_symbolic_unified(
                            addr_val.clone(),
                            size as u32,
                            self.ctx,
                            &self.concretizer,
                        ) {
                            return Ok(Some(value));
                        }
                    }
                }
                Ok(None)
            }
            Err(MemoryError::Unmapped {
                addr,
                size: unmapped_size,
            }) => {
                log::debug!(
                    "Unmapped memory load at 0x{:x} (size={}), falling back to Python",
                    addr,
                    unmapped_size
                );
                Ok(None)
            }
            Err(MemoryError::SymbolicAddress { .. }) => {
                // Symbolic bytes not fully tracked - fall through to Python.
                // Happens when per-byte symbolic imports don't cover the full
                // multi-byte load, or imports didn't cover all bytes at the addr.
                Ok(None)
            }
            Err(e) => Err(CbExecutionError::Memory(e.to_string())),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vex::ir::{Endness, IRType};

    fn new_interp(ctx: &SymContext) -> CallbackInterpreter<'_> {
        CallbackInterpreter::new(VexArch::AMD64, ctx)
    }

    #[test]
    fn eval_const_u32_matches_width_and_value() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let bv = interp.eval_const(&IRConst::U32(0xdead_beef));
        assert_eq!(bv.width(), 32);
        assert_eq!(bv.as_u64(), Some(0xdead_beef));
    }

    #[test]
    fn eval_const_u1_round_trips() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let t = interp.eval_const(&IRConst::U1(true));
        let f = interp.eval_const(&IRConst::U1(false));
        assert_eq!(t.width(), 1);
        assert_eq!(t.as_u64(), Some(1));
        assert_eq!(f.as_u64(), Some(0));
    }

    #[test]
    fn eval_const_u128_preserves_high_bits() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let value: u128 = (1u128 << 100) | 0xff;
        let bv = interp.eval_const(&IRConst::U128(value));
        assert_eq!(bv.width(), 128);
        assert_eq!(bv.as_u128(), Some(value));
    }

    #[test]
    fn eval_const_f32_packs_to_bits() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let bv = interp.eval_const(&IRConst::F32(1.0));
        assert_eq!(bv.width(), 32);
        assert_eq!(bv.as_u64(), Some(f32::to_bits(1.0) as u64));
    }

    #[test]
    fn eval_expr_simple_const() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let env = TypeEnv::new();
        let bv = interp
            .eval_expr_simple(&IRExpr::Const(IRConst::U64(0x1234)), &env)
            .expect("const eval");
        assert_eq!(bv.as_u64(), Some(0x1234));
    }

    #[test]
    fn eval_expr_simple_rdtmp_returns_stored_value() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.temps.resize(4, None);
        interp.temps[2] = Some(RustBV::concrete(0xabc, 32));
        let env = TypeEnv::new();
        let bv = interp
            .eval_expr_simple(&IRExpr::RdTmp(2), &env)
            .expect("temp eval");
        assert_eq!(bv.as_u64(), Some(0xabc));
    }

    #[test]
    fn eval_expr_simple_unknown_temp_errors() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let env = TypeEnv::new();
        let err = interp
            .eval_expr_simple(&IRExpr::RdTmp(0), &env)
            .expect_err("missing temp should error");
        matches!(err, CbExecutionError::UnknownTemp(0));
    }

    #[test]
    fn eval_expr_simple_get_register_reads_zero_initially() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let env = TypeEnv::new();
        // AMD64 RAX = offset 16, 8 bytes
        let bv = interp
            .eval_expr_simple(
                &IRExpr::Get {
                    offset: 16,
                    ty: IRType::I64,
                },
                &env,
            )
            .expect("get eval");
        assert_eq!(bv.width(), 64);
        assert_eq!(bv.as_u64(), Some(0));
    }

    #[test]
    fn eval_expr_simple_get_reads_register_after_write() {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.registers.put_reg("rax", RustBV::concrete(0xfeed, 64));
        let env = TypeEnv::new();
        let bv = interp
            .eval_expr_simple(
                &IRExpr::Get {
                    offset: 16, // RAX on AMD64
                    ty: IRType::I64,
                },
                &env,
            )
            .expect("get eval");
        assert_eq!(bv.as_u64(), Some(0xfeed));
    }

    #[test]
    fn eval_expr_simple_rejects_complex_load() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let env = TypeEnv::new();
        let load = IRExpr::Load {
            addr: Box::new(IRExpr::Const(IRConst::U64(0x1000))),
            ty: IRType::I64,
            endness: Endness::Little,
        };
        let err = interp
            .eval_expr_simple(&load, &env)
            .expect_err("simple eval should not handle Load");
        matches!(err, CbExecutionError::Unsupported(_));
    }

    #[test]
    fn apply_loadg_conversion_widens_zero() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let val = RustBV::concrete(0xff, 8);
        let widened = interp.apply_loadg_conversion(IRLoadGOp::WidenZ, val, 32);
        assert_eq!(widened.width(), 32);
        assert_eq!(widened.as_u64(), Some(0xff));
    }

    #[test]
    fn apply_loadg_conversion_widens_signed() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        // 0xff as signed i8 is -1; zero-extend says 0xff (255); sign-extend says 0xffffffff.
        let val = RustBV::concrete(0xff, 8);
        let widened = interp.apply_loadg_conversion(IRLoadGOp::WidenS, val, 32);
        assert_eq!(widened.width(), 32);
        assert_eq!(widened.as_u64(), Some(0xffff_ffff));
    }

    #[test]
    fn apply_loadg_conversion_identity_passes_through() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let val = RustBV::concrete(0xab, 8);
        let same = interp.apply_loadg_conversion(IRLoadGOp::Identity, val, 8);
        assert_eq!(same.width(), 8);
        assert_eq!(same.as_u64(), Some(0xab));
    }

    #[test]
    fn apply_loadg_conversion_same_width_is_no_op() {
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let val = RustBV::concrete(0xdead_beef, 32);
        let same = interp.apply_loadg_conversion(IRLoadGOp::WidenZ, val, 32);
        assert_eq!(same.width(), 32);
        assert_eq!(same.as_u64(), Some(0xdead_beef));
    }

    #[test]
    fn apply_loadg_conversion_truncates_when_src_wider() {
        // The truncation branch is currently never exercised in production
        // (LoadG always widens), but guard against future refactors:
        // extract(high, low) requires high >= low and yields high - low + 1
        // bits, so the previous extract(0, target_bits) underflowed in
        // release builds. Confirm we now keep the low target_bits.
        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let val = RustBV::concrete(0xdead_beef, 32);
        let truncated = interp.apply_loadg_conversion(IRLoadGOp::Identity, val, 16);
        assert_eq!(truncated.width(), 16);
        assert_eq!(truncated.as_u64(), Some(0xbeef));
    }
}
