//! `IRStmt::Dirty` execution — the guarded-side-effecting-call statement.
//!
//! Split out of `statements.rs` (angr-fs8kb.62), mirroring the
//! split-by-statement-kind pattern its siblings `statements_cas.rs`
//! (`Ist_CAS`) and `statements_store.rs` (`Ist_Store`) already follow. The
//! whole file is reachable from exactly one arm of
//! `execute_stmt_with_callbacks`.
//!
//! [`VEXInterpreter::handle_dirty_call`] is the entry point; it is a sequence
//! of four phases, one helper each, because they answer independent questions
//! and each has its own failure policy:
//!
//! 1. [`VEXInterpreter::dirty_guard_phase`] — guard classification, including
//!    the `0x5555…5555` poison VEX promises for a false guard.
//! 2. [`VEXInterpreter::dirty_arg_values`] — eager concretization of the
//!    arguments both remaining dispatch paths need as `u64`.
//! 3. [`VEXInterpreter::dirty_native_dispatch`] — the Rust dirty-helper table.
//! 4. [`VEXInterpreter::dirty_python_dispatch`] — the Python `dirty_call`
//!    callback, plus the two no-handler-anywhere policies (route to Python's
//!    VEX engine by default, fabricate only under the opt-in gate).
//!
//! The guard tristate itself ([`GuardClass`]) stays in `statements.rs`: it is
//! shared with `Exit`, `StoreG` and `LoadG`.

use super::bv_utils::bytes_to_bv;
use super::expressions_arith::fabricate_unsupported_irop;
use super::statements::GuardClass;
use super::*;

/// VEX's payload for a guarded dirty call's result temp when the guard turns
/// out false: the repeating `0x5555…5555` bit pattern libvex_ir.h's `IRDirty`
/// documentation promises. Truncated to the temp's width by
/// [`RustBV::concrete`].
const DIRTY_GUARD_FALSE_PATTERN: u128 = 0x5555_5555_5555_5555_5555_5555_5555_5555;

/// The bit width of `dirty`'s result temp, or `0` when it names none.
///
/// A dirty call that names a result temp must have that temp in the block's
/// tyenv; a missing entry is malformed IR, not a "assume 64" case. The width
/// feeds the native-handler write path (`RustBV::concrete`), the
/// Python-callback path (`bytes_to_bv`) and the guard-false poison below, so
/// defaulting silently produces a wrong-width tmp that propagates instead of
/// erroring. Fail loud, matching the `IRStmt::LLSC` arm of
/// `execute_stmt_with_callbacks`, which does the same lookup for the same
/// condition (angr-03vl4.33).
fn dirty_ret_ty_bits(
    dirty: &crate::vex::ir::IRDirty,
    irsb: &IRSB,
) -> Result<u32, CbExecutionError> {
    let Some(tmp) = dirty.tmp else {
        return Ok(0); // No return value.
    };
    Ok(irsb
        .tyenv
        .get(tmp)
        .ok_or_else(|| {
            CbExecutionError::InvalidIR(format!(
                "dirty call '{}': result temp {tmp} not in tyenv",
                dirty.cee.name
            ))
        })?
        .bits())
}

/// The value a guarded dirty call writes to its result temp when the guard is
/// false — see [`DIRTY_GUARD_FALSE_PATTERN`] and
/// [`VEXInterpreter::dirty_guard_phase`]'s `GuardClass::Never` arm.
///
/// Declines above 128 bits, where `RustBV::Concrete`'s `u128` payload cannot
/// hold the pattern (the same storage ceiling `RustBV::ones` documents): a
/// masked-to-zero-high-half poison would be a silently wrong *defined* value,
/// so route the block to Python instead. Only `Ity_V256` can reach this, and
/// no dirty helper in the supported set returns one.
fn dirty_guard_false_poison(name: &str, bits: u32) -> Result<RustBV, CbExecutionError> {
    if bits > 128 {
        return Err(CbExecutionError::NeedPythonFallback(format!(
            "dirty call '{name}': guard-false poison for a {bits}-bit result \
             temp exceeds the 128-bit concrete payload"
        )));
    }
    Ok(RustBV::concrete(DIRTY_GUARD_FALSE_PATTERN, bits))
}

impl<'a> VEXInterpreter<'a> {
    /// Execute an `IRStmt::Dirty` call: guard handling, native dirty-helper
    /// dispatch, and the Python-callback / fresh-symbolic fallbacks.
    ///
    /// Extracted verbatim from `execute_stmt_with_callbacks` (cudgw.18), then
    /// decomposed into the four phase helpers the module doc lists
    /// (angr-fs8kb.62).
    pub(super) fn handle_dirty_call(
        &mut self,
        callbacks: &PythonCallbacks,
        dirty: &crate::vex::ir::IRDirty,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        if self.dirty_guard_phase(callbacks, dirty, irsb)? == DirtyGuard::Skipped {
            return Ok(StmtResult::Continue);
        }

        let arg_vals = self.dirty_arg_values(callbacks, dirty, &irsb.tyenv)?;
        let ret_ty_bits = dirty_ret_ty_bits(dirty, irsb)?;

        if let Some(args) = arg_vals.as_deref()
            && self.dirty_native_dispatch(dirty, args, ret_ty_bits)?
        {
            return Ok(StmtResult::Continue);
        }

        self.dirty_python_dispatch(callbacks, dirty, arg_vals.as_deref(), ret_ty_bits)
    }

    /// Phase 1: classify `dirty`'s guard and act on the two decided cases.
    ///
    /// Returns [`DirtyGuard::Skipped`] when the call must not run — the caller
    /// continues to the next statement; the result temp (if any) has already
    /// been defined here.
    fn dirty_guard_phase(
        &mut self,
        callbacks: &PythonCallbacks,
        dirty: &crate::vex::ir::IRDirty,
        irsb: &IRSB,
    ) -> Result<DirtyGuard, CbExecutionError> {
        let Some(guard) = &dirty.guard else {
            return Ok(DirtyGuard::Run);
        };
        let guard_val = self.eval_expr_with_callbacks(callbacks, guard, &irsb.tyenv)?;
        match self.classify_guard(&guard_val) {
            // Guard is false - skip the dirty call, but still define the
            // result temp. libvex_ir.h's `IRDirty` doc promises that "if
            // at runtime the guard evaluates to false, .tmp has an
            // 0x555...555 bit pattern written to it", which is precisely
            // why VEX emits conditional calls that assign `.tmp` and why
            // downstream statements read it unconditionally. Our `temps`
            // is a `Vec<Option<RustBV>>` with no lazy default, so leaving
            // the slot `None` made such a read fail with
            // `CbExecutionError::UnknownTemp` (angr-fs8kb.57). Mirrors
            // `handle_loadg`'s `Never` arm, which writes `alt` for the
            // same reason.
            GuardClass::Never => {
                if let Some(tmp) = dirty.tmp {
                    let bits = dirty_ret_ty_bits(dirty, irsb)?;
                    let poison = dirty_guard_false_poison(&dirty.cee.name, bits)?;
                    self.write_tmp(tmp, poison)?;
                }
                Ok(DirtyGuard::Skipped)
            }
            // Guard must be true - run it unconditionally.
            GuardClass::Always => Ok(DirtyGuard::Run),
            GuardClass::Symbolic => {
                // Both feasible: we can't fork mid-block, so pin the guard
                // true and run the call. Loses the not-taken branch but
                // matches angr's existing dirty-helper concretization.
                log::debug!(
                    "dirty call '{}': symbolic guard concretized to taken branch",
                    dirty.cee.name
                );
                self.ctx.assume_true(&guard_val);
                Ok(DirtyGuard::Run)
            }
        }
    }

    /// Phase 2: evaluate the call's arguments down to concrete `u64`s.
    ///
    /// Eager-concretizes symbolic args via the solver so that native dispatch
    /// and the Python callback — which both expect concrete `u64` args — can
    /// run; the equality constraint is pinned so downstream branches stay
    /// consistent. `None` means the solver could not produce a concrete value
    /// for some arg (e.g. an UNSAT path); both dispatch paths have their own
    /// fallback policy for that, so this is not an error here.
    fn dirty_arg_values(
        &mut self,
        callbacks: &PythonCallbacks,
        dirty: &crate::vex::ir::IRDirty,
        tyenv: &crate::vex::ir::TypeEnv,
    ) -> Result<Option<Vec<u64>>, CbExecutionError> {
        let mut arg_vals: Vec<u64> = Vec::with_capacity(dirty.args.len());
        for arg in &dirty.args {
            let val = self.eval_expr_with_callbacks(callbacks, arg, tyenv)?;
            if let Some(concrete) = val.as_u64() {
                arg_vals.push(concrete);
            } else if let Some(concrete) = self.concretize_and_pin(&val) {
                arg_vals.push(concrete);
            } else {
                // SILENT(cat-a): expected control flow — `None` is the
                // documented "solver produced no model for this arg" signal
                // this function's own doc defines, and every caller path
                // handles it explicitly (`dirty_python_dispatch` turns it into
                // a `CbExecutionError::Unsupported` on the arm that needs the
                // args, and native dispatch is skipped).
                return Ok(None);
            }
        }
        Ok(Some(arg_vals))
    }

    /// Phase 3: try the native dirty-helper table. `Ok(true)` means the helper
    /// ran and its result temp / register writes have been applied.
    fn dirty_native_dispatch(
        &mut self,
        dirty: &crate::vex::ir::IRDirty,
        arg_vals: &[u64],
        ret_ty_bits: u32,
    ) -> Result<bool, CbExecutionError> {
        let Some(result) =
            self.dirty_dispatch
                .try_call(&mut self.dirty_helper_state, &dirty.cee.name, arg_vals)
        else {
            return Ok(false);
        };

        log::trace!(
            "Native dirty call: {} (args: {:?})",
            dirty.cee.name,
            arg_vals
        );

        // Store result in temporary if specified
        if let Some(tmp) = dirty.tmp
            && let Some(return_value) = result.return_value
        {
            let value = RustBV::concrete(return_value as u128, ret_ty_bits);
            self.write_tmp(tmp, value)?;
        }

        // Apply any register writes from the helper
        for (offset, value) in result.reg_writes {
            // Convert u64 value to RustBV and store in register
            let bv = RustBV::concrete(value as u128, 64);
            self.registers.put(offset, bv);
        }

        Ok(true)
    }

    /// Phase 4: the Python `dirty_call` callback, and the two policies that
    /// apply when nobody in this process models the helper.
    ///
    /// `arg_vals` is `None` when phase 2 could not concretize — that is fatal
    /// only on the path that actually needs the args (the callback), so the
    /// no-handler-anywhere arms below run first.
    fn dirty_python_dispatch(
        &mut self,
        callbacks: &PythonCallbacks,
        dirty: &crate::vex::ir::IRDirty,
        arg_vals: Option<&[u64]>,
        ret_ty_bits: u32,
    ) -> Result<StmtResult, CbExecutionError> {
        // No native handler matched and Python has no `dirty_call` callback
        // registered either, so nobody in this process models the helper.
        // Follow the same policy as `vex_op_fallback` / `eval_ccall`
        // (angr-c7xno.44): route the block to Python's VEX engine by default,
        // and only fabricate a fresh unconstrained symbolic under the opt-in
        // `ANGR_RUST_FABRICATE_UNSUPPORTED_IROP` gate. Fabricating by default
        // silently diverges — an unconstrained tmp makes every downstream
        // condition over it explore both branches regardless of what the
        // helper really computes.
        if !callbacks.has_dirty_call() {
            if !fabricate_unsupported_irop() {
                return Err(CbExecutionError::NeedPythonFallback(format!(
                    "dirty call '{}': no native handler and no Python callback",
                    dirty.cee.name
                )));
            }
            log::warn!(
                "dirty call '{}': no native handler and no Python callback; \
                         stubbing with a fresh symbolic tmp \
                         (ANGR_RUST_FABRICATE_UNSUPPORTED_IROP)",
                dirty.cee.name
            );
            self.stats.vex_bypass_fabricate_count += 1;
            if let Some(tmp) = dirty.tmp {
                // `ret_ty_bits` is the tyenv width of exactly this temp, and
                // `IRType::bits` never yields 0, so the 0 sentinel means
                // "no result temp" and cannot be reached inside this arm.
                let stub = RustBV::symbolic(
                    self.ctx,
                    format!("dirty_{}_stub", dirty.cee.name),
                    ret_ty_bits,
                );
                self.write_tmp(tmp, stub)?;
            }
            return Ok(StmtResult::Continue);
        }

        let Some(arg_vals) = arg_vals else {
            // `dirty_arg_values` bailed early because the solver could not
            // produce a concrete value for one of the args (UNSAT).
            // `concretize_and_pin` only fails on an empty model, and the pins
            // added for the preceding args merely tighten the constraint set —
            // re-running the loop would fail on the same arg. So there is
            // nothing more aggressive to try; surface a clear error and let
            // Python apply its own fallback.
            return Err(CbExecutionError::Unsupported(format!(
                "dirty call '{}' arg unconcretizable",
                dirty.cee.name
            )));
        };

        // Call Python callback
        self.stats.python_dirty_call_count += 1;
        let (data, is_symbolic, _symbolic_ast) = callbacks
            .call_dirty_call(&dirty.cee.name, arg_vals, ret_ty_bits)
            .map_err(|e| {
                CbExecutionError::Callback(format!("dirty call {} failed: {}", dirty.cee.name, e))
            })?;

        // Store result in temporary if specified
        if let Some(tmp) = dirty.tmp {
            let result = if is_symbolic {
                // Create a symbolic value for the result
                RustBV::symbolic(self.ctx, format!("dirty_{}", dirty.cee.name), ret_ty_bits)
            } else {
                // Convert the little-endian callback bytes to a concrete value.
                bytes_to_bv(&data, ret_ty_bits)
            };

            self.write_tmp(tmp, result)?;
        }

        Ok(StmtResult::Continue)
    }
}

/// Whether [`VEXInterpreter::dirty_guard_phase`] decided the call should run.
///
/// A bool would read as `if self.dirty_guard_phase(..)? { return ... }` at the
/// one call site, where neither polarity is obviously the skip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirtyGuard {
    /// The guard is absent, always-true, or was pinned true — execute the call.
    Run,
    /// The guard cannot be true; the result temp has been poisoned and the
    /// statement is done.
    Skipped,
}
