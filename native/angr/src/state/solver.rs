//! Solver/constraint-access cluster for `RustSimState`.
//!
//! The thin accessors that forward to the shared `Rc<RefCell<SymContext>>`:
//! the `solver()` handle itself, constraint accumulation (`add_constraint`),
//! satisfiability (`satisfiable` / `set_sat_cache`), and the concrete-value
//! queries (`eval` / `min` / `max`). Split out of `mod.rs` per the god-object
//! decomposition (angr-0mqkc.5); mirrors the `construction.rs` / `options.rs`
//! / `registers.rs` / `memory.rs` extension-impl pattern.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// How many exploration prune gates have seen an undecided satisfiability
/// query (see `RustSimState::survives_sat_prune`). Process-wide and
/// diagnostic-only: it exists to rate-limit the warn, nothing branches on it.
static UNDECIDED_SAT_PRUNE_GATES: AtomicU64 = AtomicU64::new(0);

impl RustSimState {
    /// Get a reference to the solver context.
    pub fn solver(&self) -> &Rc<RefCell<SymContext>> {
        &self.solver
    }

    /// Add a constraint.
    pub fn add_constraint(&self, constraint: RustBV) {
        let ctx = self.solver.borrow();
        ctx.assume_true(&constraint);
    }

    /// Check if current constraints are satisfiable.
    ///
    /// Lenient form: a Z3 Unknown (timeout) reads as `false`, so a caller
    /// that *drops* the state on `false` must use `satisfiable_checked` or
    /// `survives_sat_prune` instead (`invariant-z3-unknown-not-unsat`).
    pub fn satisfiable(&self) -> bool {
        let ctx = self.solver.borrow();
        ctx.is_sat()
    }

    /// Check if current constraints are satisfiable, reporting an undecided
    /// (Z3 Unknown / timeout) query as `None` instead of collapsing it into
    /// `false` — see `SymContext::is_sat_checked`.
    pub fn satisfiable_checked(&self) -> Option<bool> {
        let ctx = self.solver.borrow();
        ctx.is_sat_checked()
    }

    /// The exploration loop's keep-or-drop gate: `true` when the state must
    /// be kept, `false` only when the solver *proved* it unsatisfiable.
    ///
    /// `lazy_solves` short-circuits the query entirely (the caller has opted
    /// out of eager satisfiability checking). Otherwise an undecided query
    /// keeps the state and logs — "Z3 gave up" is not "proven contradictory",
    /// and dropping on it silently deletes a very likely feasible successor
    /// (angr-03vl4.86). This replaces the `lazy_solves || state.satisfiable()`
    /// shape at every prune site in `exploration/`; the warn is rate-limited
    /// to powers of two because a timing-out context tends to time out on
    /// every successor it spawns.
    pub fn survives_sat_prune(&self, lazy_solves: bool) -> bool {
        if lazy_solves {
            return true;
        }
        match self.satisfiable_checked() {
            Some(sat) => sat,
            None => {
                let n = UNDECIDED_SAT_PRUNE_GATES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    log::warn!(
                        "state {} at pc 0x{:x}: satisfiability undecided (Z3 timeout) at an \
                         exploration prune gate; keeping the state rather than pruning it \
                         (undecided prune gates so far: {n})",
                        self.state_id(),
                        self.pc()
                    );
                }
                true
            }
        }
    }

    /// Prime the SAT cache (avoids redundant Z3 checks after branch forking).
    pub fn set_sat_cache(&self, value: bool) {
        self.solver.borrow().set_sat_cache(value);
    }

    /// Evaluate an expression to a concrete value.
    pub fn eval(&self, expr: &RustBV) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.eval(expr)
    }

    /// Get minimum value of an expression.
    pub fn min(&self, expr: &RustBV, signed: bool) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.min(expr, signed)
    }

    /// Get maximum value of an expression.
    pub fn max(&self, expr: &RustBV, signed: bool) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.max(expr, signed)
    }
}
