//! Compare-and-swap (`IRStmt::CAS`) execution, single and double-width.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** the CAS
//! operands come straight from guest-lifted IR, so this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` and malformed IR
//! returns `CbExecutionError::InvalidIR`. There are no `unwrap`/`expect` sites
//! left: the all-Some/all-None DCAS validation in `execute_cas_stmt` now binds
//! `old_hi`/`data_hi` in the arm that proves them present, so the DCAS branch
//! needs no second unwrap.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::helpers::{bv_to_bytes, reject_symbolic_byte_store};
use super::*;

/// DCAS-only state bundled together so the single-CAS path can pass `None`
/// and the DCAS path can pass `Some(&DcasState)` through `cas_writeback` and
/// the oldHi temp-assignment in `execute_cas_stmt`.
pub(super) struct DcasState<'a> {
    addr_hi_expr: IRExpr,
    data_hi_expr: &'a IRExpr,
    current_hi: RustBV,
    expd_hi_val: RustBV,
    data_hi_val: RustBV,
    old_hi_idx: u32,
}

/// The `IRStmt::CAS` operands, bundled so the CAS handlers can thread the whole
/// statement instead of eight positional params. The `*_hi` fields are `Some`
/// exactly on the DCAS path.
pub(super) struct CasArgs<'s> {
    pub(super) old_hi: Option<u32>,
    pub(super) old_lo: u32,
    pub(super) addr: &'s IRExpr,
    pub(super) expd_hi: Option<&'s IRExpr>,
    pub(super) expd_lo: &'s IRExpr,
    pub(super) data_hi: Option<&'s IRExpr>,
    pub(super) data_lo: &'s IRExpr,
    pub(super) endness: Endness,
}

/// The evaluated low-half values `cas_writeback` compares and stores, plus the
/// DCAS sidecar. Derived from a [`CasArgs`]; kept separate because these are
/// `RustBV`s the caller already evaluated, not IR.
pub(super) struct CasWriteback<'s> {
    /// Result of `(current_lo == expd_lo) [& (current_hi == expd_hi)]`.
    cmp: &'s RustBV,
    data_lo_val: &'s RustBV,
    current_lo: &'s RustBV,
    dcas: Option<&'s DcasState<'s>>,
}

impl<'a> VEXInterpreter<'a> {
    /// CAS handler — supports both single CAS and DCAS (double compare-and-swap,
    /// e.g. x86-64 cmpxchg16b). DCAS = oldHi/expdHi/dataHi all Some, single = all None.
    ///
    /// DCAS semantics (mirroring `_perform_vex_stmt_CAS` in
    /// angr/engines/vex/light/light.py): load both halves at `addr` and
    /// `addr + sizeof(expd_ty)`; compare both vs `(expd_hi, expd_lo)`; on match,
    /// write `(data_hi, data_lo)` back. Only little-endian is supported — the
    /// only real DCAS users (x86-64, ARM64) are LE; BE is rejected explicitly.
    pub(super) fn execute_cas_stmt(
        &mut self,
        callbacks: &PythonCallbacks,
        cas: &CasArgs<'_>,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        let &CasArgs {
            old_hi,
            old_lo,
            addr,
            expd_hi,
            expd_lo,
            data_hi,
            data_lo,
            endness,
        } = cas;

        // Bind the two Hi operands the DCAS branch actually needs right here,
        // in the arm that proves they are present. The IR is guest-derived, so
        // the all-Some/all-None validation below is a real input check — the
        // point of carrying the values out of it is that the DCAS branch then
        // needs no second `unwrap` to re-derive what this match established.
        let dcas_operands = match (old_hi, expd_hi, data_hi) {
            (Some(old_hi), Some(_), Some(data_hi)) => Some((old_hi, data_hi)),
            (None, None, None) => None,
            _ => {
                return Err(CbExecutionError::InvalidIR(
                    "CAS: oldHi/expdHi/dataHi must be all-Some (DCAS) or all-None (single)"
                        .to_string(),
                ));
            }
        };

        // Half type comes from expdLo — both halves share it.
        let half_ty = expd_lo
            .get_type(&irsb.tyenv)
            .ok_or_else(|| CbExecutionError::InvalidIR("CAS expdLo has no type".to_string()))?;

        if dcas_operands.is_some() && endness == Endness::Big {
            // No real arch (x86-64 cmpxchg16b, ARM64 LDXP) is BE; if BE DCAS
            // ever shows up, defer to Python's full-CAS implementation rather
            // than guessing the address-of-Hi vs address-of-Lo convention.
            return Err(CbExecutionError::Unsupported(
                "DCAS with big-endian memory not supported".to_string(),
            ));
        }

        // Load current_lo at addr.
        let load_lo_expr = IRExpr::Load {
            addr: Box::new(addr.clone()),
            ty: half_ty,
            endness,
        };
        let current_lo = self.eval_expr_with_callbacks(callbacks, &load_lo_expr, &irsb.tyenv)?;
        let expd_lo_val = self.eval_expr_with_callbacks(callbacks, expd_lo, &irsb.tyenv)?;
        let data_lo_val = self.eval_expr_with_callbacks(callbacks, data_lo, &irsb.tyenv)?;

        // For DCAS, also load the high half at addr + sizeof(half).
        let dcas = match dcas_operands {
            Some((old_hi_idx, data_hi_expr)) => {
                let addr_hi_expr = Self::cas_compute_addr_hi(addr, half_ty, irsb)?;
                let (current_hi, expd_hi_val, data_hi_val) =
                    self.cas_load_dcas_high(callbacks, &addr_hi_expr, cas, half_ty, irsb)?;
                Some(DcasState {
                    addr_hi_expr,
                    data_hi_expr,
                    current_hi,
                    expd_hi_val,
                    data_hi_val,
                    old_hi_idx,
                })
            }
            None => None,
        };

        // Combined cmp = (current_lo == expd_lo) & (current_hi == expd_hi)?
        let cmp_lo = current_lo.eq(&expd_lo_val, self.ctx);
        let cmp = if let Some(d) = &dcas {
            let cmp_hi = d.current_hi.eq(&d.expd_hi_val, self.ctx);
            cmp_lo.and(&cmp_hi, self.ctx)
        } else {
            cmp_lo
        };

        self.cas_writeback(
            callbacks,
            cas,
            &CasWriteback {
                cmp: &cmp,
                data_lo_val: &data_lo_val,
                current_lo: &current_lo,
                dcas: dcas.as_ref(),
            },
            irsb,
        )?;

        // Write current values to oldLo (and oldHi for DCAS).
        if (old_lo as usize) >= self.temps.len() {
            return Err(CbExecutionError::UnknownTemp(old_lo));
        }
        self.temps[old_lo as usize] = Some(current_lo);
        if let Some(d) = dcas {
            if (d.old_hi_idx as usize) >= self.temps.len() {
                return Err(CbExecutionError::UnknownTemp(d.old_hi_idx));
            }
            self.temps[d.old_hi_idx as usize] = Some(d.current_hi);
        }

        Ok(StmtResult::Continue)
    }

    /// Compute the address of the DCAS high half: `addr + sizeof(half_ty)`.
    /// Synthesised as a fresh `IRExpr::Binop(Add)` rather than mutating the
    /// per-IRSB tyenv — see the `dcas-irexpr-binop-addr-hi` invariant.
    pub(super) fn cas_compute_addr_hi(
        addr: &IRExpr,
        half_ty: IRType,
        irsb: &IRSB,
    ) -> Result<IRExpr, CbExecutionError> {
        let addr_ty = addr
            .get_type(&irsb.tyenv)
            .ok_or_else(|| CbExecutionError::InvalidIR("CAS addr has no type".to_string()))?;
        let half_bytes = half_ty.bytes() as u64;
        let offset_const = match addr_ty {
            IRType::I32 => IRExpr::Const(IRConst::U32(half_bytes as u32)),
            IRType::I64 => IRExpr::Const(IRConst::U64(half_bytes)),
            _ => {
                return Err(CbExecutionError::InvalidIR(
                    "CAS addr must be I32 or I64".to_string(),
                ));
            }
        };
        Ok(IRExpr::Binop {
            op: IROp::Add(addr_ty),
            left: Box::new(addr.clone()),
            right: Box::new(offset_const),
        })
    }

    /// Load and evaluate the DCAS high half. Returns
    /// `(current_hi, expd_hi_val, data_hi_val)`.
    pub(super) fn cas_load_dcas_high(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_hi_expr: &IRExpr,
        cas: &CasArgs<'_>,
        half_ty: IRType,
        irsb: &IRSB,
    ) -> Result<(RustBV, RustBV, RustBV), CbExecutionError> {
        // DCAS-only path: the all-Some/all-None gate in `execute_cas_stmt` has
        // already established that expdHi/dataHi are present.
        let (Some(expd_hi), Some(data_hi)) = (cas.expd_hi, cas.data_hi) else {
            return Err(CbExecutionError::InvalidIR(
                "CAS: DCAS high half needs expdHi/dataHi".to_string(),
            ));
        };
        let load_hi_expr = IRExpr::Load {
            addr: Box::new(addr_hi_expr.clone()),
            ty: half_ty,
            endness: cas.endness,
        };
        let current_hi = self.eval_expr_with_callbacks(callbacks, &load_hi_expr, &irsb.tyenv)?;
        let expd_hi_val = self.eval_expr_with_callbacks(callbacks, expd_hi, &irsb.tyenv)?;
        let data_hi_val = self.eval_expr_with_callbacks(callbacks, data_hi, &irsb.tyenv)?;
        Ok((current_hi, expd_hi_val, data_hi_val))
    }

    /// Perform the CAS writeback. Three cases on `cmp`:
    ///   - concrete false: no store.
    ///   - concrete true:  store `data` (reusing the original IRExpr).
    ///   - symbolic:       store `ITE(cmp, data, current)` — the deferred-fork
    ///     branch, where both outcomes are encoded into a single
    ///     state via ITE rather than splitting into two states.
    pub(super) fn cas_writeback(
        &mut self,
        callbacks: &PythonCallbacks,
        cas: &CasArgs<'_>,
        wb: &CasWriteback<'_>,
        irsb: &IRSB,
    ) -> Result<(), CbExecutionError> {
        let &CasWriteback {
            cmp,
            data_lo_val,
            current_lo,
            dcas,
        } = wb;
        let (addr, data_lo_expr, endness) = (cas.addr, cas.data_lo, cas.endness);

        match cmp.as_u64() {
            Some(0) => Ok(()),
            Some(_) => {
                self.cas_dispatch_store(callbacks, addr, data_lo_expr, data_lo_val, endness, irsb)?;
                if let Some(d) = dcas {
                    self.cas_dispatch_store(
                        callbacks,
                        &d.addr_hi_expr,
                        d.data_hi_expr,
                        &d.data_hi_val,
                        endness,
                        irsb,
                    )?;
                }
                Ok(())
            }
            None => {
                let store_lo = cmp.ite(data_lo_val, current_lo, self.ctx);
                self.cas_store_symbolic_data(callbacks, addr, &store_lo, irsb)?;
                if let Some(d) = dcas {
                    let store_hi = cmp.ite(&d.data_hi_val, &d.current_hi, self.ctx);
                    self.cas_store_symbolic_data(callbacks, &d.addr_hi_expr, &store_hi, irsb)?;
                }
                Ok(())
            }
        }
    }

    /// Dispatch a CAS store: if the precomputed RustBV is concrete, synthesize
    /// an IRStmt::Store that reuses the full Store path (which re-evaluates the
    /// data IRExpr); if symbolic, route through `cas_store_symbolic_data`.
    pub(super) fn cas_dispatch_store(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_expr: &IRExpr,
        data_expr: &IRExpr,
        data_bv: &RustBV,
        endness: Endness,
        irsb: &IRSB,
    ) -> Result<(), CbExecutionError> {
        if data_bv.is_symbolic() {
            self.cas_store_symbolic_data(callbacks, addr_expr, data_bv, irsb)
        } else {
            let store_stmt = IRStmt::Store {
                addr: addr_expr.clone(),
                data: data_expr.clone(),
                endness,
            };
            self.execute_stmt_with_callbacks(callbacks, &store_stmt, irsb)?;
            Ok(())
        }
    }

    /// Store a symbolic data value at an address. Used by CAS when the value
    /// to store is a computed RustBV (e.g. ITE) that cannot be wrapped back
    /// into an IRExpr — see the `cas-llsc-recursion-limit` invariant.
    /// Routes through `memory_store_symbolic_value` for concrete addresses
    /// and `memory_store_symbolic_full` for symbolic addresses.
    ///
    /// Both branches invalidate any stale cached IRSB at the (concretized)
    /// target address(es) before dispatching the store — angr-slbsd. This
    /// reuses the same helpers the ordinary `IRStmt::Store` path uses
    /// (`statements_store.rs`): `invalidate_code_at_store` for a concrete
    /// address, and `concretize_cached_write` + `invalidate_code_on_store`
    /// (mirroring `handle_symbolic_store`'s `Single` case) when the address
    /// itself is symbolic. The actual store still goes through the Python
    /// callback / pending-store buffer below — this call only protects the
    /// Rust-side `block_cache` against self-modifying `lock cmpxchg` writes.
    pub(super) fn cas_store_symbolic_data(
        &mut self,
        callbacks: &PythonCallbacks,
        addr_expr: &IRExpr,
        data_bv: &RustBV,
        irsb: &IRSB,
    ) -> Result<(), CbExecutionError> {
        let addr_val = self.eval_expr_with_callbacks(callbacks, addr_expr, &irsb.tyenv)?;
        let data_size = (data_bv.width() / 8) as usize;
        if let Some(addr_concrete) = addr_val.as_u64() {
            self.invalidate_code_at_store(addr_concrete, data_size);
            if callbacks.has_memory_store_symbolic_value() {
                self.flush_stores(callbacks)?;
                callbacks
                    .call_memory_store_symbolic_value(addr_concrete, data_bv)
                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
            } else {
                reject_symbolic_byte_store(data_bv, addr_concrete, "CAS store")?;
                self.evict_overlapping_symbolic_stores(addr_concrete, data_size);
                self.pending_symbolic_stores
                    .insert(addr_concrete, data_bv.clone());
                let data_bytes = bv_to_bytes(data_bv);
                self.pending_stores.push(addr_concrete, data_bytes);
                if self.pending_stores.len() >= self.max_pending_stores {
                    self.flush_stores(callbacks)?;
                }
            }
        } else {
            self.flush_stores(callbacks)?;
            let conc_result = self.concretize_cached_write(&addr_val);
            self.invalidate_code_on_store(&conc_result, data_size);
            if callbacks.has_memory_store_symbolic_full() {
                callbacks
                    .call_memory_store_symbolic_full(&addr_val, data_bv)
                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
            } else {
                return Err(CbExecutionError::Unsupported(
                    "CAS with symbolic address: \
                     no memory_store_symbolic_full callback"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }
}
