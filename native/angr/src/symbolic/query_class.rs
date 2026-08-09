//! Structural classification of the solver queries that actually reach Z3
//! (S1 spike, angr-op0dn.3).
//!
//! The M1 moonshot asks: what fraction of `z3_check_count` could a cheap
//! pre-solver tier decide without Z3? Answering that in *structural counts*
//! (not wall-time) needs per-check attribution, so this module hangs a
//! classification on the query that is currently in flight and reads it back
//! inside [`timed_check`](super::solver_build::timed_check) — the single
//! funnel every `solver.check()` goes through. One check = one bucket bump,
//! so the buckets sum to `z3_check_count` exactly.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** this module
//! classifies guest-derived query ASTs, so it carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`; an unrecognized shape
//! falls into `QueryClass::Hard`, never a panic. The two surviving non-test
//! `expect`s ([`classify_bool`] and [`classify_extrema`]) read the single
//! element of a set whose length was checked on the line above; the
//! `#[cfg(test)]` `tests` child opts out of the deny wholesale, so count
//! non-test sites only when checking that claim.
//!
//! Two protocol rules from the bead, both load-bearing for the >=30% gate:
//!
//! - **No double-counting.** The classifiers run *after* the fast paths that
//!   already ship (`as_u128()` short-circuits in `solving_ops.rs`, the cached
//!   model hit in `check_branch_feasibility`, constructor folds in
//!   `value_ops.rs`). A query only gets classified once it is genuinely about
//!   to hit the solver, so a bucket bump is a check a new tier would have to
//!   save on top of today's engine.
//! - **Forced vs free witness.** Model queries split into [`ForcedValue`]
//!   (the constraint set syntactically pins every symbol the target reads, so
//!   the value is unique and a fast path cannot change concretization) and
//!   [`FreeWitness`] (the witness is solver-chosen; fast-pathing it changes
//!   which value the engine picks and collides with the M2 determinism work).
//!
//! [`ForcedValue`]: QueryClass::ForcedValue
//! [`FreeWitness`]: QueryClass::FreeWitness
//!
//! Classification walks the [`RustBV`] IR, so it is not free. It is off by
//! default and enabled per-process with `ANGR_RUST_QUERY_CLASS=1`; when off,
//! every check lands in [`QueryClass::Unclassified`] and the query entry
//! points do no IR walking at all.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::{BVOp, RustBV};
use std::cell::Cell;
use std::collections::HashSet;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Node budget for a single classification walk. A query whose IR is bigger
/// than this is `Hard` by definition — a "cheap static predicate" that has to
/// visit thousands of nodes is not cheap.
const NODE_BUDGET: usize = 256;

/// How many constraints a single classification will scan before giving up.
/// Long constraint lists are exactly the queries a pre-solver tier cannot
/// decide anyway, and the scan is O(|C|) per query.
const CONSTRAINT_BUDGET: usize = 64;

/// The structural bucket a Z3-bound query falls into.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub(super) enum QueryClass {
    /// Classification disabled, or a check issued outside any classified
    /// query (e.g. an internal re-check). Not addressable.
    Unclassified = 0,
    /// Decidable with no solver and no constraint reasoning: the query
    /// expression has no free symbols left, so constant folding settles it.
    /// A non-zero count here means an existing fold/fast-path is leaking.
    TrivialDecide = 1,
    /// Exactly one free symbol, and every constraint that mentions it is
    /// itself single-symbol and interval-shaped: decidable by interval
    /// arithmetic over that symbol (feeds angr-op0dn.9.2).
    SingleVarRange = 2,
    /// Model query whose target is syntactically pinned by the constraints
    /// (every symbol it reads has an `x == const` constraint, or is already a
    /// `Constrained` leaf). The value is unique — a fast path here cannot
    /// change concretization.
    ForcedValue = 3,
    /// Model query whose witness is solver-chosen. Fast-pathing this changes
    /// which value the engine concretizes to (M2 determinism interaction).
    FreeWitness = 4,
    /// Multi-symbol, non-interval-shaped, or over budget. Not addressable by
    /// a cheap tier.
    Hard = 5,
    /// Boolean query that still has free symbols but whose *verdict* is fixed
    /// by syntax alone: `Eq(a, a)`, `Ult(x, 0)`, `Ule(x, UMAX)`, a mask-bit
    /// contradiction, … (feeds angr-op0dn.9.1). Distinct from
    /// [`TrivialDecide`](QueryClass::TrivialDecide), which means "no free
    /// symbols left at all".
    SyntacticDecide = 6,
}

/// Number of variants in [`QueryClass`].
pub(super) const NUM_QUERY_CLASSES: usize = 7;

/// Per-class count of `solver.check()` calls. Sums to `z3_check_count`.
pub(crate) static Z3_CHECK_CLASS_COUNT: [AtomicU64; NUM_QUERY_CLASSES] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Stats-key suffixes, indexed by `QueryClass as usize`.
pub(crate) const CLASS_NAMES: [&str; NUM_QUERY_CLASSES] = [
    "unclassified",
    "trivial_decide",
    "single_var_range",
    "forced_value",
    "free_witness",
    "hard",
    "syntactic_decide",
];

thread_local! {
    /// The class of the query currently in flight on this thread.
    static CURRENT_CLASS: Cell<QueryClass> = const { Cell::new(QueryClass::Unclassified) };
}

/// Whether `ANGR_RUST_QUERY_CLASS` asked for classification. Read once.
pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ANGR_RUST_QUERY_CLASS")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    })
}

/// Attribute one `solver.check()` to the in-flight query's class.
pub(crate) fn record_check() {
    let idx = CURRENT_CLASS.with(|c| c.get()) as usize;
    Z3_CHECK_CLASS_COUNT[idx].fetch_add(1, Ordering::Relaxed);
}

/// Zero every class counter (called from `reset_solver_stats`).
pub(crate) fn reset() {
    for c in &Z3_CHECK_CLASS_COUNT {
        c.store(0, Ordering::Relaxed);
    }
}

/// RAII binding of the in-flight query class, restoring the previous one on
/// drop so a nested query (min → eval on a cached model, say) cannot leave a
/// stale class behind for its caller's later checks.
pub(crate) struct ClassScope {
    prev: QueryClass,
}

impl ClassScope {
    pub(crate) fn new(class: QueryClass) -> Self {
        let prev = CURRENT_CLASS.with(|c| c.replace(class));
        Self { prev }
    }
}

impl Drop for ClassScope {
    fn drop(&mut self) {
        CURRENT_CLASS.with(|c| c.set(self.prev));
    }
}

/// Bind a query class for the duration of the returned guard, but only when
/// classification is enabled — so a disabled build pays one atomic-free bool
/// load per query and never walks the IR.
pub(crate) fn scope(classify: impl FnOnce() -> QueryClass) -> Option<ClassScope> {
    enabled().then(|| ClassScope::new(classify()))
}

/// Ops a cheap interval/range tier could evaluate over a single symbol.
fn interval_shaped_op(op: &BVOp) -> bool {
    matches!(
        op,
        BVOp::Eq
            | BVOp::Ne
            | BVOp::Ult
            | BVOp::Ule
            | BVOp::Ugt
            | BVOp::Uge
            | BVOp::Slt
            | BVOp::Sle
            | BVOp::Sgt
            | BVOp::Sge
            | BVOp::And
            | BVOp::Or
            | BVOp::Not
            | BVOp::Xor
            | BVOp::Add
            | BVOp::Sub
            | BVOp::ZeroExt(_)
            | BVOp::SignExt(_)
            | BVOp::Extract(_, _)
    )
}

/// Structural summary of one expression tree, computed in a single walk.
#[derive(Default)]
struct Shape {
    /// Free symbol ids reachable from the root (`Constrained` leaves are
    /// pinned, so they do not count as free).
    vars: HashSet<u64>,
    /// Every op on the way is in the interval-decidable whitelist.
    interval_shaped: bool,
    /// The walk stayed inside [`NODE_BUDGET`].
    within_budget: bool,
}

impl Shape {
    /// Cheap to decide with interval arithmetic over at most one symbol.
    fn single_var(&self) -> bool {
        self.within_budget && self.interval_shaped && self.vars.len() == 1
    }
}

fn walk(bv: &RustBV, shape: &mut Shape, budget: &mut usize) {
    if *budget == 0 {
        shape.within_budget = false;
        return;
    }
    *budget -= 1;
    match bv {
        RustBV::Concrete { .. } | RustBV::Constrained { .. } => {}
        RustBV::Symbolic { id, .. } => {
            shape.vars.insert(*id);
        }
        RustBV::Expression { op, operands, .. } => {
            if !interval_shaped_op(op) {
                shape.interval_shaped = false;
            }
            for operand in operands.iter() {
                walk(operand, shape, budget);
            }
        }
    }
}

fn shape_of(bv: &RustBV) -> Shape {
    let mut shape = Shape {
        vars: HashSet::new(),
        interval_shaped: true,
        within_budget: true,
    };
    let mut budget = NODE_BUDGET;
    walk(bv, &mut shape, &mut budget);
    shape
}

/// Symbol ids pinned to a constant by a top-level `x == const` constraint
/// assumed true (or `x != const` assumed false is NOT a pin — only equality
/// pins a value). `None` when the constraint list is over budget.
fn pinned_symbols(constraints: &[(RustBV, bool)]) -> Option<HashSet<u64>> {
    if constraints.len() > CONSTRAINT_BUDGET {
        return None;
    }
    let mut pinned = HashSet::new();
    for (cond, is_true) in constraints {
        if !is_true {
            continue;
        }
        let RustBV::Expression {
            op: BVOp::Eq,
            operands,
            ..
        } = cond
        else {
            continue;
        };
        if operands.len() != 2 {
            continue;
        }
        match (&operands[0], &operands[1]) {
            (RustBV::Symbolic { id, .. }, rhs) if rhs.as_u128().is_some() => {
                pinned.insert(*id);
            }
            (lhs, RustBV::Symbolic { id, .. }) if lhs.as_u128().is_some() => {
                pinned.insert(*id);
            }
            _ => {}
        }
    }
    Some(pinned)
}

/// True when every constraint touching `var` is itself single-symbol over
/// `var` and interval-shaped — the precondition for deciding a query about
/// `var` by interval arithmetic alone.
fn constraints_interval_shaped(constraints: &[(RustBV, bool)], var: u64) -> bool {
    if constraints.len() > CONSTRAINT_BUDGET {
        return false;
    }
    constraints.iter().all(|(cond, _)| {
        let shape = shape_of(cond);
        if !shape.vars.contains(&var) {
            // Irrelevant to `var` — but only if it is a well-formed walk;
            // an over-budget constraint may well mention `var` deeper down.
            return shape.within_budget;
        }
        shape.single_var() && shape.interval_shaped
    })
}

/// Bounded structural equality — the reflexivity test behind `Eq(a, a)`.
/// `RustBV`'s `PartialEq` only compares *concrete* values (it is `false` for
/// any symbolic operand), so it cannot answer this.
fn struct_eq(a: &RustBV, b: &RustBV, budget: &mut usize) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    match (a, b) {
        (
            RustBV::Concrete {
                value: v1,
                width: w1,
            },
            RustBV::Concrete {
                value: v2,
                width: w2,
            },
        ) => v1 == v2 && w1 == w2,
        (RustBV::Symbolic { id: i1, .. }, RustBV::Symbolic { id: i2, .. }) => i1 == i2,
        (RustBV::Constrained { id: i1, .. }, RustBV::Constrained { id: i2, .. }) => i1 == i2,
        (
            RustBV::Expression {
                op: o1,
                operands: p1,
                ..
            },
            RustBV::Expression {
                op: o2,
                operands: p2,
                ..
            },
        ) => {
            o1 == o2
                && a.width() == b.width()
                && p1.len() == p2.len()
                && p1
                    .iter()
                    .zip(p2.iter())
                    .all(|(x, y)| struct_eq(x, y, budget))
        }
        _ => false,
    }
}

/// Largest unsigned value representable at `width` bits.
fn umax(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

/// The syntactic-verdict tier's pattern set (angr-op0dn.9.1): does the
/// *condition's own syntax* fix its truth value, even though free symbols
/// remain? Returns the forced verdict, or `None` when Z3 is genuinely needed.
///
/// Only patterns the constructors in `value_ops.rs` do **not** already fold
/// are listed — `eq_into`/`ne_into` fold concrete-vs-concrete and the
/// `zext`-vs-const cases, so those never reach here.
pub(crate) fn syntactic_decide(cond: &RustBV) -> Option<bool> {
    let RustBV::Expression { op, operands, .. } = cond else {
        return None;
    };
    if operands.len() != 2 {
        return None;
    }
    let (lhs, rhs) = (&operands[0], &operands[1]);
    let mut budget = NODE_BUDGET;
    let reflexive = struct_eq(lhs, rhs, &mut budget);
    let (lc, rc) = (lhs.as_u128(), rhs.as_u128());
    let (lw, rw) = (lhs.width(), rhs.width());
    match op {
        // Reflexivity.
        BVOp::Eq if reflexive => Some(true),
        BVOp::Ne if reflexive => Some(false),
        BVOp::Ult | BVOp::Ugt | BVOp::Slt | BVOp::Sgt if reflexive => Some(false),
        BVOp::Ule | BVOp::Uge | BVOp::Sle | BVOp::Sge if reflexive => Some(true),
        // Unsigned bounds: nothing is below 0, nothing is above UMAX.
        BVOp::Ult if rc == Some(0) || lc == Some(umax(lw)) => Some(false),
        BVOp::Ugt if lc == Some(0) || rc == Some(umax(rw)) => Some(false),
        BVOp::Ule if lc == Some(0) || rc == Some(umax(rw)) => Some(true),
        BVOp::Uge if rc == Some(0) || lc == Some(umax(lw)) => Some(true),
        // Mask-bit contradiction: `And(x, m) == c` (either operand order) is
        // unsatisfiable when `c` sets a bit that `m` clears.
        BVOp::Eq | BVOp::Ne => {
            let (masked, c) = match (lc, rc) {
                (None, Some(c)) => (lhs, c),
                (Some(c), None) => (rhs, c),
                _ => return None,
            };
            let RustBV::Expression {
                op: BVOp::And,
                operands: mask_ops,
                ..
            } = masked
            else {
                return None;
            };
            let m = mask_ops.iter().find_map(RustBV::as_u128)?;
            (c & !m != 0).then_some(matches!(op, BVOp::Ne))
        }
        _ => None,
    }
}

/// Classify a bare satisfiability query (`is_sat`): no query expression, so
/// cheapness is a property of the constraint set alone.
pub(crate) fn classify_sat(constraints: &[(RustBV, bool)]) -> QueryClass {
    if constraints.is_empty() {
        return QueryClass::TrivialDecide;
    }
    if constraints.len() > CONSTRAINT_BUDGET {
        return QueryClass::Hard;
    }
    let mut vars: HashSet<u64> = HashSet::new();
    for (cond, _) in constraints {
        let shape = shape_of(cond);
        if !shape.within_budget || !shape.interval_shaped {
            return QueryClass::Hard;
        }
        vars.extend(shape.vars);
        if vars.len() > 1 {
            return QueryClass::Hard;
        }
    }
    if vars.is_empty() {
        QueryClass::TrivialDecide
    } else {
        QueryClass::SingleVarRange
    }
}

/// Classify a boolean query (`is_sat` with `cond` asserted, branch
/// feasibility, `solution`).
#[allow(
    clippy::expect_used,
    reason = "`shape.vars` was just checked to hold exactly one element (`vars.len() == 1` / `single_var()`), so the first `iter().next()` is always `Some`. `shape` is derived from the query AST, but the length check is what makes this total"
)]
pub(crate) fn classify_bool(cond: &RustBV, constraints: &[(RustBV, bool)]) -> QueryClass {
    let shape = shape_of(cond);
    if shape.within_budget && shape.vars.is_empty() {
        return QueryClass::TrivialDecide;
    }
    if syntactic_decide(cond).is_some() {
        return QueryClass::SyntacticDecide;
    }
    if !shape.within_budget {
        return QueryClass::Hard;
    }
    if shape.vars.len() == 1 && shape.interval_shaped {
        let var = *shape.vars.iter().next().expect("len == 1");
        if constraints_interval_shaped(constraints, var) {
            return QueryClass::SingleVarRange;
        }
    }
    QueryClass::Hard
}

/// Classify a model query (`eval`, `eval_upto`). Splits forced values (unique
/// by construction) from solver-chosen witnesses, per the bead's protocol.
pub(crate) fn classify_eval(target: &RustBV, constraints: &[(RustBV, bool)]) -> QueryClass {
    let shape = shape_of(target);
    if !shape.within_budget {
        return QueryClass::Hard;
    }
    if shape.vars.is_empty() {
        // No free symbols: `Constrained` leaves and folds settle it.
        return QueryClass::TrivialDecide;
    }
    if let Some(pinned) = pinned_symbols(constraints)
        && shape.vars.iter().all(|v| pinned.contains(v))
    {
        return QueryClass::ForcedValue;
    }
    QueryClass::FreeWitness
}

/// Classify a batched model query (`eval_many`): a single check produces one
/// model read by every target, so the batch is only as cheap as its hardest
/// member.
pub(crate) fn classify_eval_many(targets: &[RustBV], constraints: &[(RustBV, bool)]) -> QueryClass {
    /// Higher = less addressable; the batch takes the max.
    fn rank(c: QueryClass) -> u8 {
        match c {
            QueryClass::TrivialDecide | QueryClass::SyntacticDecide => 0,
            QueryClass::SingleVarRange => 1,
            QueryClass::ForcedValue => 2,
            QueryClass::FreeWitness => 3,
            QueryClass::Hard | QueryClass::Unclassified => 4,
        }
    }
    targets
        .iter()
        .map(|t| classify_eval(t, constraints))
        .max_by_key(|c| rank(*c))
        .unwrap_or(QueryClass::TrivialDecide)
}

/// Classify an extrema query (`min`, `max`, `range`). These are the ones a
/// single-variable interval tier would answer directly.
#[allow(
    clippy::expect_used,
    reason = "`shape.vars` was just checked to hold exactly one element (`vars.len() == 1` / `single_var()`), so the first `iter().next()` is always `Some`. `shape` is derived from the query AST, but the length check is what makes this total"
)]
pub(crate) fn classify_extrema(target: &RustBV, constraints: &[(RustBV, bool)]) -> QueryClass {
    let shape = shape_of(target);
    if !shape.within_budget {
        return QueryClass::Hard;
    }
    if shape.vars.is_empty() {
        return QueryClass::TrivialDecide;
    }
    if shape.single_var() {
        let var = *shape.vars.iter().next().expect("len == 1");
        if constraints_interval_shaped(constraints, var) {
            return QueryClass::SingleVarRange;
        }
    }
    QueryClass::Hard
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests {
    use super::*;
    use crate::symbolic::SymContext;

    fn sym(id: u64, width: u32) -> RustBV {
        RustBV::symbolic_with_id(id, format!("v{id}"), width)
    }

    #[test]
    fn concrete_condition_is_trivial() {
        let cond = RustBV::concrete(1, 1);
        assert_eq!(classify_bool(&cond, &[]), QueryClass::TrivialDecide);
    }

    #[test]
    fn reflexive_comparisons_are_syntactically_decided() {
        let ctx = SymContext::new_mock();
        // `x + 1` on both sides: structurally equal, but not concrete, so
        // RustBV's own PartialEq says "not equal" and only `struct_eq` sees it.
        let lhs = sym(1, 32).add(&RustBV::concrete(1, 32), &ctx);
        let rhs = sym(1, 32).add(&RustBV::concrete(1, 32), &ctx);
        assert_eq!(syntactic_decide(&lhs.eq(&rhs, &ctx)), Some(true));
        assert_eq!(syntactic_decide(&lhs.ne(&rhs, &ctx)), Some(false));
        assert_eq!(syntactic_decide(&lhs.ult(&rhs, &ctx)), Some(false));
        assert_eq!(syntactic_decide(&lhs.ule(&rhs, &ctx)), Some(true));
        assert_eq!(
            classify_bool(&lhs.eq(&rhs, &ctx), &[]),
            QueryClass::SyntacticDecide
        );
    }

    #[test]
    fn unsigned_bounds_are_syntactically_decided() {
        let ctx = SymContext::new_mock();
        let x = sym(1, 32);
        // Nothing is below 0, everything is at or below UMAX.
        assert_eq!(
            syntactic_decide(&x.ult(&RustBV::concrete(0, 32), &ctx)),
            Some(false)
        );
        assert_eq!(
            syntactic_decide(&x.ule(&RustBV::concrete(0xffff_ffff, 32), &ctx)),
            Some(true)
        );
        assert_eq!(
            syntactic_decide(&x.uge(&RustBV::concrete(0, 32), &ctx)),
            Some(true)
        );
        // A real bound still needs the solver.
        assert_eq!(
            syntactic_decide(&x.ult(&RustBV::concrete(10, 32), &ctx)),
            None
        );
    }

    #[test]
    fn mask_bit_contradiction_is_syntactically_decided() {
        let ctx = SymContext::new_mock();
        let masked = sym(1, 32).and(&RustBV::concrete(0xff, 32), &ctx);
        // `(x & 0xff) == 0x100` sets a bit the mask clears → unsat.
        assert_eq!(
            syntactic_decide(&masked.eq(&RustBV::concrete(0x100, 32), &ctx)),
            Some(false)
        );
        assert_eq!(
            syntactic_decide(&masked.ne(&RustBV::concrete(0x100, 32), &ctx)),
            Some(true)
        );
        // In-mask constant is a genuine query.
        assert_eq!(
            syntactic_decide(&masked.eq(&RustBV::concrete(0x42, 32), &ctx)),
            None
        );
    }

    #[test]
    fn single_symbol_comparison_is_single_var() {
        let ctx = SymContext::new_mock();
        let cond = sym(1, 32).ult(&RustBV::concrete(10, 32), &ctx);
        let constraints = [(sym(1, 32).uge(&RustBV::concrete(2, 32), &ctx), true)];
        assert_eq!(
            classify_bool(&cond, &constraints),
            QueryClass::SingleVarRange
        );
    }

    #[test]
    fn two_symbols_are_hard() {
        let ctx = SymContext::new_mock();
        let cond = sym(1, 32).ult(&sym(2, 32), &ctx);
        assert_eq!(classify_bool(&cond, &[]), QueryClass::Hard);
    }

    #[test]
    fn single_var_query_with_multi_var_constraint_is_hard() {
        let ctx = SymContext::new_mock();
        let cond = sym(1, 32).ult(&RustBV::concrete(10, 32), &ctx);
        let constraints = [(sym(1, 32).eq(&sym(2, 32), &ctx), true)];
        assert_eq!(classify_bool(&cond, &constraints), QueryClass::Hard);
    }

    #[test]
    fn pinned_target_is_forced_value() {
        let ctx = SymContext::new_mock();
        let target = sym(1, 32).add(&RustBV::concrete(4, 32), &ctx);
        let constraints = [(sym(1, 32).eq(&RustBV::concrete(7, 32), &ctx), true)];
        assert_eq!(
            classify_eval(&target, &constraints),
            QueryClass::ForcedValue
        );
    }

    #[test]
    fn unpinned_target_is_free_witness() {
        let ctx = SymContext::new_mock();
        let target = sym(1, 32).add(&RustBV::concrete(4, 32), &ctx);
        let constraints = [(sym(1, 32).ult(&RustBV::concrete(7, 32), &ctx), true)];
        assert_eq!(
            classify_eval(&target, &constraints),
            QueryClass::FreeWitness
        );
    }

    #[test]
    fn extrema_over_one_symbol_is_single_var() {
        let ctx = SymContext::new_mock();
        let target = sym(1, 32);
        let constraints = [(sym(1, 32).ule(&RustBV::concrete(9, 32), &ctx), true)];
        assert_eq!(
            classify_extrema(&target, &constraints),
            QueryClass::SingleVarRange
        );
    }

    #[test]
    fn class_scope_restores_previous_class() {
        let outer = ClassScope::new(QueryClass::FreeWitness);
        {
            let _inner = ClassScope::new(QueryClass::Hard);
            assert_eq!(CURRENT_CLASS.with(|c| c.get()), QueryClass::Hard);
        }
        assert_eq!(CURRENT_CLASS.with(|c| c.get()), QueryClass::FreeWitness);
        drop(outer);
        assert_eq!(CURRENT_CLASS.with(|c| c.get()), QueryClass::Unclassified);
    }
}
