//! Concretize-and-pin for the x87 transcendentals (angr-i5lj.1,
//! angr-i5lj.2): Z3 has no sin/cos/log/exp theory, so a symbolic operand of
//! one of the ops `vex::transcendentals::pins_binop` / `pins_triop` name is
//! sampled from the solver and pinned to that sample with a hard constraint,
//! and the op then runs on the concrete value through `VEXOps` like any
//! other. The strategy and the options it was chosen over are in the
//! `vex::transcendentals` module doc. A failed pin leaves the operands
//! symbolic, so the op takes the same `RawOpcode` fallthrough it took
//! before the pins moved here (angr-mx00p.116.6).

use crate::symbolic::{RustBV, SymContext};

/// `x` pinned to one model value: the value is sampled with `ctx.eval` and
/// `x == value` is assumed. `None` (nothing assumed) when the path is UNSAT.
pub(crate) fn pin_operand(x: &RustBV, ctx: &SymContext) -> Option<RustBV> {
    let pinned = RustBV::concrete(ctx.eval(x)?, x.width());
    ctx.assume_true(&x.eq(&pinned));
    Some(pinned)
}

/// `a` and `b` pinned to one JOINT witness (angr-z8elx). Two independent
/// `ctx.eval` calls are not model-consistent in strict-deterministic mode
/// (`eval` short-circuits to a per-variable `min`, deliberately skipping the
/// model cache), so path-correlated operands could each be individually
/// feasible yet jointly infeasible — and the pins would then turn a SAT
/// context UNSAT, silently killing a feasible path. Only a symbolic operand
/// gets a constraint. `None` (nothing assumed) when the path is UNSAT.
pub(crate) fn pin_operands(a: &RustBV, b: &RustBV, ctx: &SymContext) -> Option<(RustBV, RustBV)> {
    let vals = ctx.eval_many(&[a.clone(), b.clone()])?;
    let pinned_a = RustBV::concrete(vals[0], a.width());
    let pinned_b = RustBV::concrete(vals[1], b.width());
    for (operand, pinned) in [(a, &pinned_a), (b, &pinned_b)] {
        if operand.is_symbolic() {
            ctx.assume_true(&operand.eq(pinned));
        }
    }
    Some((pinned_a, pinned_b))
}

test_submod!("transcendental_pins_tests.rs" => tests);
