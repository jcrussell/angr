#![allow(clippy::arc_with_non_send_sync)]
use super::*;
#[cfg(feature = "vex-engine-z3")]
use crate::symbolic::Z3AstPtr;

// angr-9o4n.1: Constraint round-trip spike via Z3_solver_to_string /
// Z3_solver_from_string. Drives whether SMT-LIB2 is the right format for
// angr-9o4n state save/restore. See bead notes for measured numbers.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_smtlib2_constraint_round_trip() {
    use std::time::Instant;

    // Build a non-trivial constraint set: 32-bit BVs + Extract + Concat +
    // multiple assertions. Names are uniquified so we don't collide with
    // any other test in the same Z3 thread-local context.
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "rt_x_9o4n", 32);
    let y = RustBV::symbolic(&ctx, "rt_y_9o4n", 32);

    // 10 < x < 20 (unsigned)
    let ten = RustBV::concrete(10, 32);
    let twenty = RustBV::concrete(20, 32);
    ctx.assume_true(&x.ugt(&ten, &ctx));
    ctx.assume_true(&x.ult(&twenty, &ctx));

    // Extract: high 16 bits of y are zero.
    let y_high = y.extract(31, 16, &ctx);
    let zero16 = RustBV::concrete(0, 16);
    ctx.assume_true(&y_high.eq(&zero16, &ctx));

    // Concat: low(x,16) ++ low(y,16) == 0x000B_0007 (x=11 satisfies low(x,16)=0x000B;
    // y_low=0x0007 satisfies the concat).
    let x_low = x.extract(15, 0, &ctx);
    let y_low = y.extract(15, 0, &ctx);
    let combined = x_low.concat(&y_low, &ctx);
    let target = RustBV::concrete(0x000B_0007, 32);
    ctx.assume_true(&combined.eq(&target, &ctx));

    // Sanity: original is SAT and the witness values fall in expected ranges.
    let original_sat = ctx.is_sat();
    assert!(original_sat, "constraint set should be sat");
    let x_witness = ctx.eval(&x).expect("x evaluable");
    let y_witness = ctx.eval(&y).expect("y evaluable");
    assert_eq!(
        x_witness, 11,
        "x must be 11 (the only value with 10<x<20 whose low 16 bits = 0x000B)"
    );
    assert_eq!(y_witness, 0x0000_0007, "y_high=0, y_low=0x0007");

    // Step 1: Serialize via Solver::to_string (SMT-LIB2 S-expression).
    let serialize_start = Instant::now();
    let serialized = ctx.debug_solver_string();
    let serialize_ns = serialize_start.elapsed().as_nanos() as u64;
    let serialized_bytes = serialized.len();
    assert!(
        !serialized.is_empty(),
        "serialized SMT-LIB2 must be non-empty"
    );

    // Step 2: Parse into a fresh z3::Solver (shares the thread-local Z3
    // context, but is a logically independent solver). Constants declared
    // by name in the SMT-LIB2 string re-resolve to the SAME Z3 ASTs as the
    // originals because Z3 interns named constants in the context.
    let deserialize_start = Instant::now();
    let new_solver = z3::Solver::new();
    new_solver.from_string(serialized);
    let deserialize_ns = deserialize_start.elapsed().as_nanos() as u64;

    // Step 3a: check_sat matches.
    let new_check_start = Instant::now();
    let new_sat = matches!(new_solver.check(), z3::SatResult::Sat);
    let new_check_ns = new_check_start.elapsed().as_nanos() as u64;
    assert_eq!(
        new_sat, original_sat,
        "round-tripped solver sat-result must match"
    );

    // Step 3b: model values for x, y match the original witness (the
    // constraint set is restrictive enough that x=11, y_low=7 are forced).
    let new_model = new_solver
        .get_model()
        .expect("sat solver must produce model");
    let new_x_val = new_model
        .eval(&x.to_z3_ast(), true)
        .and_then(|bv| bv.as_u64())
        .expect("model should evaluate x");
    let new_y_val = new_model
        .eval(&y.to_z3_ast(), true)
        .and_then(|bv| bv.as_u64())
        .expect("model should evaluate y");
    assert_eq!(
        new_x_val as u128, x_witness,
        "round-tripped x model value must match"
    );
    assert_eq!(
        new_y_val as u128, y_witness,
        "round-tripped y model value must match"
    );

    // Step 4: report measurements. Captured by `cargo test -- --nocapture`
    // or `cargo test test_smtlib2_constraint_round_trip -- --nocapture`,
    // and pasted into the bead notes.
    eprintln!(
        "[angr-9o4n.1] SMT-LIB2 round-trip (small): {} bytes; \
         to_string={}us from_string={}us check_sat={}us; \
         5 assertions, 2 32-bit BV vars (Extract+Concat).",
        serialized_bytes,
        serialize_ns / 1000,
        deserialize_ns / 1000,
        new_check_ns / 1000,
    );
}

// angr-9o4n.1: scaling check. Mid-sized constraint set (~100 assertions,
// 32 BV vars) to give a sense of cost as exploration state grows.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_smtlib2_constraint_round_trip_scaled() {
    use std::time::Instant;

    let ctx = SymContext::new();
    const NVARS: usize = 32;
    let vars: Vec<RustBV> = (0..NVARS)
        .map(|i| RustBV::symbolic(&ctx, format!("rt_scaled_x{i}_9o4n"), 32))
        .collect();

    // For each var: low(x) > i, low(x) < i+100 — gives a range constraint.
    // Then chain pairs: vars[i] != vars[i+1] for i in 0..NVARS-1.
    for (i, v) in vars.iter().enumerate() {
        let lo = RustBV::concrete(i as u128, 32);
        let hi = RustBV::concrete((i + 100) as u128, 32);
        ctx.assume_true(&v.ugt(&lo, &ctx));
        ctx.assume_true(&v.ult(&hi, &ctx));
    }
    for w in vars.windows(2) {
        let neq = w[0].eq(&w[1], &ctx);
        ctx.assume_false(&neq);
    }

    let original_sat = ctx.is_sat();
    assert!(original_sat, "scaled constraint set should be sat");

    let serialize_start = Instant::now();
    let serialized = ctx.debug_solver_string();
    let serialize_ns = serialize_start.elapsed().as_nanos() as u64;
    let serialized_bytes = serialized.len();

    let deserialize_start = Instant::now();
    let new_solver = z3::Solver::new();
    new_solver.from_string(serialized);
    let deserialize_ns = deserialize_start.elapsed().as_nanos() as u64;

    let new_check_start = Instant::now();
    let new_sat = matches!(new_solver.check(), z3::SatResult::Sat);
    let new_check_ns = new_check_start.elapsed().as_nanos() as u64;
    assert_eq!(new_sat, original_sat);

    // Spot-check one variable's model value carries across.
    let original_v0 = ctx.eval(&vars[0]).expect("v0 evaluable");
    let new_model = new_solver
        .get_model()
        .expect("sat solver must produce model");
    let new_v0 = new_model
        .eval(&vars[0].to_z3_ast(), true)
        .and_then(|bv| bv.as_u64())
        .expect("model should evaluate v0");
    // Note: models from independent solver checks need not be identical.
    // We assert that the new model also satisfies the constraint (0 < v0 < 100).
    assert!(
        new_v0 > 0 && new_v0 < 100,
        "new model v0={new_v0} must satisfy 0 < v0 < 100; original was {original_v0}",
    );

    let n_assertions = NVARS * 2 + (NVARS - 1);
    eprintln!(
        "[angr-9o4n.1] SMT-LIB2 round-trip (scaled): {} bytes; \
         to_string={}us from_string={}us check_sat={}us; \
         {} assertions, {} 32-bit BV vars.",
        serialized_bytes,
        serialize_ns / 1000,
        deserialize_ns / 1000,
        new_check_ns / 1000,
        n_assertions,
        NVARS,
    );
}

// angr-rwzi: Validate SMT-LIB2 round-trip across a SEPARATE Z3 context.
// Same-thread/same-context worked in angr-9o4n.1 because constants in the
// shared context dedupe by (symbol, sort). The realistic save/restore path
// (different thread or different process) gets a fresh Z3_context, so the
// open question is whether name-based re-resolution (BV::new_const) in the
// new context binds to the same AST that `Z3_solver_from_string` creates.
//
// Discriminator: the constraint set forces x=11, y=7. If name interning
// works cross-context, the new model returns 11/7 via name lookup. If the
// re-declared const is disconnected from the parsed assertions,
// model.eval(.., model_completion=true) returns the Z3 default (0).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_smtlib2_cross_context_round_trip() {
    use std::time::Instant;
    use z3::ast::{Ast, BV};
    use z3::{Config, Context, Solver, with_z3_context};

    // -------- Build constraints in the default (original) context. --------
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "xctx_x_rwzi", 32);
    let y = RustBV::symbolic(&ctx, "xctx_y_rwzi", 32);

    let ten = RustBV::concrete(10, 32);
    let twenty = RustBV::concrete(20, 32);
    ctx.assume_true(&x.ugt(&ten, &ctx));
    ctx.assume_true(&x.ult(&twenty, &ctx));

    let y_high = y.extract(31, 16, &ctx);
    let zero16 = RustBV::concrete(0, 16);
    ctx.assume_true(&y_high.eq(&zero16, &ctx));

    let x_low = x.extract(15, 0, &ctx);
    let y_low = y.extract(15, 0, &ctx);
    let combined = x_low.concat(&y_low, &ctx);
    let target = RustBV::concrete(0x000B_0007, 32);
    ctx.assume_true(&combined.eq(&target, &ctx));

    assert!(ctx.is_sat());
    let x_witness = ctx.eval(&x).expect("x evaluable");
    let y_witness = ctx.eval(&y).expect("y evaluable");
    assert_eq!(x_witness, 11);
    assert_eq!(y_witness, 0x0000_0007);

    // Record the original AST/ctx pointers so we can prove the new
    // context's by-name lookup yields a DIFFERENT AST (i.e. is truly
    // cross-context). Cast to `usize` here so we can move them across the
    // `Send + Sync` bound of `with_z3_context` (Z3 raw pointers wrap
    // `NonNull` which isn't `Send`).
    let original_x_ast_usize = x.to_z3_ast().get_z3_ast().as_ptr() as usize;
    let original_ctx_usize = z3::Context::thread_local().get_z3_context().as_ptr() as usize;

    let serialize_start = Instant::now();
    let serialized = ctx.debug_solver_string();
    let serialize_ns = serialize_start.elapsed().as_nanos() as u64;
    let serialized_bytes = serialized.len();

    // -------- Switch to a freshly-created Z3 context. --------
    // `Context::new` allocates a separate `Z3_context`; `with_z3_context`
    // swaps DEFAULT_CONTEXT for the closure body, so all subsequent
    // `Solver::new`, `BV::new_const`, `from_string`, model eval, etc.
    // resolve against the new context. The `Send + Sync` bound on the
    // closure type prevents accidentally smuggling Z3 ASTs from the old
    // context across the boundary; we only pass in plain `String`.
    let cfg = Config::new();
    let new_ctx = Context::new(&cfg);
    let new_ctx_usize_for_assert = new_ctx.get_z3_context().as_ptr() as usize;

    let (
        new_sat,
        new_x_val,
        new_y_val,
        new_x_ast_usize,
        seen_ctx_usize,
        deserialize_ns,
        new_check_ns,
    ) = with_z3_context(&new_ctx, || -> (bool, u64, u64, usize, usize, u64, u64) {
        // Sanity: confirm we really are in a different context.
        let in_closure_ctx_usize = z3::Context::thread_local().get_z3_context().as_ptr() as usize;

        let solver = Solver::new();
        let deserialize_start = Instant::now();
        solver.from_string(serialized.clone());
        let deserialize_ns = deserialize_start.elapsed().as_nanos() as u64;

        let check_start = Instant::now();
        let sat = matches!(solver.check(), z3::SatResult::Sat);
        let check_ns = check_start.elapsed().as_nanos() as u64;

        // Re-resolve constants by NAME in the new context — this is the
        // realistic save/restore path (consumer holds only names + sorts,
        // not the original ASTs).
        let x_new = BV::new_const("xctx_x_rwzi", 32);
        let y_new = BV::new_const("xctx_y_rwzi", 32);
        let x_new_ast_usize = x_new.get_z3_ast().as_ptr() as usize;

        let model = solver.get_model().expect("sat solver must produce model");
        let x_val = model
            .eval(&x_new, true)
            .and_then(|v| v.as_u64())
            .expect("model must evaluate x_new");
        let y_val = model
            .eval(&y_new, true)
            .and_then(|v| v.as_u64())
            .expect("model must evaluate y_new");

        (
            sat,
            x_val,
            y_val,
            x_new_ast_usize,
            in_closure_ctx_usize,
            deserialize_ns,
            check_ns,
        )
    });

    // -------- Verify we actually used a different context. --------
    assert_ne!(
        original_ctx_usize, new_ctx_usize_for_assert,
        "test bug: new context pointer equals original; not testing cross-context"
    );
    assert_eq!(
        seen_ctx_usize, new_ctx_usize_for_assert,
        "with_z3_context did not actually swap the thread-local context"
    );
    // ASTs are per-context: the same-name BV in the new context must be a
    // different `Z3_ast` pointer than the one in the original context.
    assert_ne!(
        original_x_ast_usize, new_x_ast_usize,
        "test bug: cross-context BV::new_const returned an AST pointer \
         identical to the original-context AST — contexts are not actually \
         distinct"
    );

    // -------- The actual cross-context round-trip claims. --------
    assert!(
        new_sat,
        "cross-context round-tripped solver must remain SAT"
    );
    assert_eq!(
        new_x_val as u128, x_witness,
        "cross-context model must give x=11 via name lookup; got {new_x_val} \
         (=0 would mean the by-name constant in the new context is \
         disconnected from the parsed assertions)"
    );
    assert_eq!(
        new_y_val as u128, y_witness,
        "cross-context model must give y=7 via name lookup; got {new_y_val}"
    );

    eprintln!(
        "[angr-rwzi] SMT-LIB2 cross-context round-trip: {} bytes; \
         to_string={}us from_string={}us check_sat={}us; \
         original_ctx=0x{:x} new_ctx=0x{:x}; \
         original_x_ast=0x{:x} new_x_ast=0x{:x}",
        serialized_bytes,
        serialize_ns / 1000,
        deserialize_ns / 1000,
        new_check_ns / 1000,
        original_ctx_usize,
        new_ctx_usize_for_assert,
        original_x_ast_usize,
        new_x_ast_usize,
    );
}

/// angr-3ms1 step 1a: on the None (no-lineage) branch,
/// `scope_savepoint_push`/`pop` bump `bare_z3_push_depth` in lockstep
/// with the per-context Z3 solver's stack. Nested pushes accumulate;
/// matching pops drain the counter back to 0.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_bare_z3_push_depth_none_branch_balanced() {
    let ctx = SymContext::new();
    assert!(ctx.lineage_arc().is_none());
    assert_eq!(ctx.bare_z3_push_depth(), 0);

    ctx.scope_savepoint_push();
    assert_eq!(ctx.bare_z3_push_depth(), 1);

    ctx.scope_savepoint_push();
    assert_eq!(ctx.bare_z3_push_depth(), 2);

    ctx.scope_savepoint_pop();
    assert_eq!(ctx.bare_z3_push_depth(), 1);

    ctx.scope_savepoint_pop();
    assert_eq!(
        ctx.bare_z3_push_depth(),
        0,
        "counter must drain back to 0 after balanced pops"
    );

    // The Some-branch sibling test lives separately
    // (`test_bare_z3_push_depth_some_branch_inert`); here we also
    // confirm the None branch left `scope_savepoints` untouched, so
    // the two paths don't accidentally double-count.
    assert_eq!(ctx.scope_savepoint_depth(), 0);
}

/// angr-ph300.48: `try_pop()` refuses an unbalanced under-pop on the None
/// lineage branch (which would reach z3-rs's panic) and returns `false`,
/// while a balanced push/pop pair pops cleanly and returns `true`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_try_pop_refuses_bare_underpop() {
    let ctx = SymContext::new();
    assert!(ctx.lineage_arc().is_none());
    assert_eq!(ctx.bare_z3_push_depth(), 0);

    // No matching push: refused, counter stays at 0, no panic.
    assert!(!ctx.try_pop(), "under-pop with empty scope must be refused");
    assert_eq!(ctx.bare_z3_push_depth(), 0);

    // Balanced push then pop: accepted, counter drains.
    ctx.push();
    assert_eq!(ctx.bare_z3_push_depth(), 1);
    assert!(ctx.try_pop(), "balanced pop must succeed");
    assert_eq!(ctx.bare_z3_push_depth(), 0);

    // And a second pop with the stack empty again is refused.
    assert!(!ctx.try_pop(), "second under-pop must be refused");
}

/// angr-ph300.48: on the Some (shared-lineage) branch a mismatched pop is
/// harmlessly ignored, so `try_pop()` is always safe (returns `true`) even
/// with no matching push — it never touches the panic-prone bare Z3 scope.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_try_pop_always_safe_on_some_branch() {
    use super::super::lineage::SharedLineageSolver;

    let ctx = SymContext::new();
    let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
    ctx.set_lineage_for_testing(Arc::clone(&lin));

    // No matching push, but the Some branch silently ignores it.
    assert!(ctx.try_pop(), "Some-branch pop is always safe");
    assert_eq!(ctx.bare_z3_push_depth(), 0);
}

/// angr-3ms1 step 1a: on the Some (shared-lineage) branch,
/// `scope_savepoint_push`/`pop` record on `scope_savepoints` and must
/// NOT touch `bare_z3_push_depth` — the counter only tracks pushes
/// against the per-context Z3 solver.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_bare_z3_push_depth_some_branch_inert() {
    use super::super::lineage::SharedLineageSolver;

    let ctx = SymContext::new();
    let lin = Arc::new(Mutex::new(SharedLineageSolver::new(build_solver(30_000))));
    ctx.set_lineage_for_testing(Arc::clone(&lin));

    assert_eq!(ctx.bare_z3_push_depth(), 0);

    ctx.scope_savepoint_push();
    ctx.scope_savepoint_push();
    assert_eq!(ctx.scope_savepoint_depth(), 2);
    assert_eq!(
        ctx.bare_z3_push_depth(),
        0,
        "Some branch must not touch bare_z3_push_depth"
    );

    ctx.scope_savepoint_pop();
    ctx.scope_savepoint_pop();
    assert_eq!(ctx.scope_savepoint_depth(), 0);
    assert_eq!(ctx.bare_z3_push_depth(), 0);
}

/// angr-3ms1 step 1a: `fork()` copies the parent's
/// `bare_z3_push_depth` into the child. The slice-1c materialization
/// gate inspects the parent's value at fork time, but copying the
/// value into the child keeps the post-fork accounting consistent
/// for any future code path that threads bare pushes across fork.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_bare_z3_push_depth_inherited_on_fork() {
    let parent = SymContext::new();
    assert_eq!(parent.bare_z3_push_depth(), 0);

    // A fork before any push: child inherits the 0.
    let child_zero = parent.fork();
    assert_eq!(
        child_zero.bare_z3_push_depth(),
        0,
        "fork before any push must hand the child a 0 depth"
    );

    // After two bare pushes, the parent's counter is 2; a fork at
    // that point hands the child the same depth.
    parent.scope_savepoint_push();
    parent.scope_savepoint_push();
    assert_eq!(parent.bare_z3_push_depth(), 2);

    let child_two = parent.fork();
    assert_eq!(
        child_two.bare_z3_push_depth(),
        2,
        "child must inherit the parent's bare_z3_push_depth at fork time"
    );

    // Drain the parent's pushes; the child's copy stays at 2 — it's
    // a per-context counter, not a shared cell.
    parent.scope_savepoint_pop();
    parent.scope_savepoint_pop();
    assert_eq!(parent.bare_z3_push_depth(), 0);
    assert_eq!(
        child_two.bare_z3_push_depth(),
        2,
        "child's counter is independent of parent's post-fork mutations"
    );
}

/// angr-3ms1 step 1b: a freshly constructed context has
/// `use_shared_lineage_solver == false`. Setter flips it; the value
/// round-trips through the getter.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_use_shared_lineage_solver_default_and_setter() {
    let ctx = SymContext::new();
    assert!(
        !ctx.use_shared_lineage_solver(),
        "default must be off so the slice-1c gate stays inert on plain RustExplorationManager runs"
    );

    ctx.set_use_shared_lineage_solver(true);
    assert!(ctx.use_shared_lineage_solver());

    ctx.set_use_shared_lineage_solver(false);
    assert!(!ctx.use_shared_lineage_solver());
}

/// angr-3ms1 step 1b: `fork()` copies the parent's
/// `use_shared_lineage_solver` value into the child so a single
/// setter call on the seed state propagates to every descendant via
/// fork — no per-fork plumbing on the Python side. Like
/// `bare_z3_push_depth`, the child carries its own AtomicBool, so
/// post-fork mutations on either side don't bleed into the other.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_use_shared_lineage_solver_inherited_on_fork() {
    let parent = SymContext::new();
    assert!(!parent.use_shared_lineage_solver());

    // Default-off parent forks a default-off child.
    let child_off = parent.fork();
    assert!(
        !child_off.use_shared_lineage_solver(),
        "fork before opt-in must hand the child a false flag"
    );

    // Opt the parent in; subsequent fork hands the child the same
    // value.
    parent.set_use_shared_lineage_solver(true);
    let child_on = parent.fork();
    assert!(
        child_on.use_shared_lineage_solver(),
        "child must inherit the parent's opt-in at fork time"
    );

    // Per-context independence: flipping the parent off does not
    // disturb the child's already-inherited true.
    parent.set_use_shared_lineage_solver(false);
    assert!(!parent.use_shared_lineage_solver());
    assert!(
        child_on.use_shared_lineage_solver(),
        "child's flag is independent of parent's post-fork mutations"
    );
}

/// angr-3ms1 step 1c: when the parent has opted in AND has no bare
/// Z3 pushes outstanding, `fork()` mints a fresh `SharedLineageSolver`
/// and installs it in the child. The parent's own lineage is not
/// touched — staying `None` so the parent keeps querying its
/// per-context solver. The two contexts therefore hold distinct
/// solver instances after the fork.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_fork_mints_lineage_when_gate_passes() {
    let parent = SymContext::new();
    parent.set_use_shared_lineage_solver(true);
    assert_eq!(parent.bare_z3_push_depth(), 0);
    assert!(
        parent.lineage_arc().is_none(),
        "parent starts without a lineage"
    );

    let child = parent.fork();
    assert!(
        child.lineage_arc().is_some(),
        "child must receive a freshly minted lineage when the gate passes"
    );
    assert!(
        parent.lineage_arc().is_none(),
        "parent's lineage must NOT change as a side effect of forking — \
         minting only installs on the child"
    );
}

/// angr-3ms1 step 1c: with the opt-in flag off (the default),
/// `fork()` keeps the pre-1c behavior of Arc::cloning the parent's
/// lineage Arc. Default `None` parent → `None` child.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_fork_skips_mint_when_flag_off() {
    let parent = SymContext::new();
    assert!(!parent.use_shared_lineage_solver());

    let child = parent.fork();
    assert!(
        child.lineage_arc().is_none(),
        "default-off flag must keep the slice-1c gate inert — no mint"
    );
}

/// angr-3ms1 step 1c: condition (b) of the gate refuses to mint
/// while the parent's per-context solver has outstanding bare Z3
/// pushes (`bare_z3_push_depth > 0`). Without this guard the child's
/// new lineage would take over Z3 stack ownership while the parent's
/// unbalanced pushes are still live, leaking the parent's pushed-only
/// constraints into the new lineage base — the failure mode that
/// `test_fork_inside_push_isolation` exposed in earlier slice 4c.3
/// attempts (see `v5a5-bare-z3-push-depth-counter-design`).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_fork_skips_mint_when_bare_push_outstanding() {
    let parent = SymContext::new();
    parent.set_use_shared_lineage_solver(true);

    // A bare push on the None lineage branch bumps bare_z3_push_depth
    // to 1 — the gate must refuse to mint while this is non-zero.
    parent.scope_savepoint_push();
    assert_eq!(parent.bare_z3_push_depth(), 1);

    let child = parent.fork();
    assert!(
        child.lineage_arc().is_none(),
        "gate must refuse to mint while parent has outstanding bare pushes"
    );

    // Clean up the parent's push so the test's per-context solver
    // returns to a balanced state (avoids tripping debug_asserts in
    // later teardown).
    parent.scope_savepoint_pop();
    assert_eq!(parent.bare_z3_push_depth(), 0);
}

/// angr-3ms1 step 1c: a newly minted lineage is seeded with the
/// parent's existing assertions as base assertions (scope 0). The
/// child's first query routes through `with_z3_solver`'s Some branch,
/// running `switch_to(empty)` then `solver.check()` — which respects
/// the base assertions installed at fork time. Verifies the child's
/// solver returns UNSAT when the parent's constraints already entail
/// it, even though the child added no constraints of its own.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_fork_minted_lineage_seeded_with_parent_constraints() {
    let parent = SymContext::new();
    parent.set_use_shared_lineage_solver(true);

    // Parent asserts x == 5 on its per-context solver (None branch).
    let x = RustBV::symbolic(&parent, "fork_mint_seed_x", 8);
    let five = RustBV::concrete(5, 8);
    parent.assume_true(&x.eq(&five, &parent));

    // Fork → child gets a fresh lineage seeded with x == 5.
    let child = parent.fork();
    assert!(child.lineage_arc().is_some());

    // The child's lineage solver knows about x == 5: assume_true(x == 6)
    // through the lineage path produces UNSAT.
    let six = RustBV::concrete(6, 8);
    child.assume_true(&x.eq(&six, &parent));
    assert!(
        !child.is_sat(),
        "child must see parent's x == 5 (base) ∧ self-added x == 6 → UNSAT"
    );

    // The parent's per-context solver is untouched — adding the
    // child's contradictory constraint did NOT leak into the parent.
    assert!(
        parent.is_sat(),
        "parent must remain SAT — its per-context solver only holds x == 5"
    );
}

/// angr-3ms1 step 1c: the opt-in flag inherits parent→child in
/// fork(), so a single setter call on a seed state propagates the
/// minting behavior to every descendant. Each fork along that chain
/// mints its own fresh lineage (the gate keeps passing because the
/// flag stays true and bare_z3_push_depth stays 0).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_fork_chain_mints_fresh_lineage_at_each_level() {
    let parent = SymContext::new();
    parent.set_use_shared_lineage_solver(true);

    let child = parent.fork();
    let grandchild = child.fork();

    let child_lin = child.lineage_arc().expect("child must have a lineage");
    let grandchild_lin = grandchild
        .lineage_arc()
        .expect("grandchild must have a lineage");
    assert!(
        !Arc::ptr_eq(&child_lin, &grandchild_lin),
        "each fork mints its own fresh lineage — Arc identities must differ"
    );
    assert!(
        grandchild.use_shared_lineage_solver(),
        "flag inherits down the chain"
    );
}

/// Round-trip the snapshot through serde JSON and verify the captured
/// assumed_constraints reconstruct equivalent Z3 ASTs (angr-x04s.1.2
/// acceptance check). Uses Z3 `Bool::eq` to confirm that the original
/// and restored constraint ASTs are *structurally* the same expression
/// (after Z3's `simplify()`), not just satisfy the same models.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_snapshot_assumed_constraints_roundtrip_via_z3() {
    use z3::ast::Ast;

    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "snap_x04s_x", 32);
    let y = RustBV::symbolic(&ctx, "snap_x04s_y", 32);
    let zero = RustBV::concrete(0, 32);
    let ten = RustBV::concrete(10, 32);
    let twenty = RustBV::concrete(20, 32);

    ctx.assume_true(&x.ugt(&ten, &ctx));
    ctx.assume_true(&x.ult(&twenty, &ctx));
    ctx.assume_false(&y.eq(&zero, &ctx));

    let pre = ctx.get_assumed_constraints();
    assert_eq!(pre.len(), 3);

    // Capture the original Z3 Bool ASTs (after simplify) so we can
    // compare structurally against the restored ones.
    let original_simplified: Vec<z3::ast::Bool> = pre
        .iter()
        .map(|(cond, is_true)| {
            let b = cond.to_z3_bool();
            let b = if *is_true { b } else { b.not() };
            b.simplify()
        })
        .collect();

    let snap = ctx.to_snapshot();
    let json = serde_json::to_string(&snap).expect("serialize");
    let restored: SymContextSnapshot = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.assumed_constraints.len(), 3);

    let ctx2 = SymContext::new();
    ctx2.restore_from_snapshot(&restored);
    let post = ctx2.get_assumed_constraints();
    assert_eq!(post.len(), 3);

    // is_true flags match.
    for (i, ((_, a), (_, b))) in pre.iter().zip(post.iter()).enumerate() {
        assert_eq!(a, b, "is_true flag mismatch at slot {i}");
    }

    // Each restored constraint produces a Z3 Bool that is
    // structurally identical (after simplify) to the original — proves
    // the AST cache rebuild followed the original tree shape.
    for (i, (cond, is_true)) in post.iter().enumerate() {
        let restored_bool = cond.to_z3_bool();
        let restored_bool = if *is_true {
            restored_bool
        } else {
            restored_bool.not()
        };
        assert_eq!(
            restored_bool.simplify(),
            original_simplified[i],
            "restored constraint slot {i} does not match original Z3 AST"
        );
    }

    // The restored context must still be SAT and concretize x and y
    // to values that honor every constraint.
    assert!(ctx2.is_sat());
    let x_val = ctx2.eval(&x).expect("x evaluable");
    let y_val = ctx2.eval(&y).expect("y evaluable");
    assert!(x_val > 10 && x_val < 20, "x={x_val} must satisfy 10<x<20");
    assert_ne!(y_val, 0, "y must be non-zero");
}

/// angr-82g6: constraints added via `add_constraint_raw` (no RustBV
/// available, e.g. the Python claripy-sync fallback path) must
/// survive snapshot round-trip via the `solver_smtlib2` dump.
/// `assumed_constraint_count` stays unchanged across the trip (the
/// raw entries never touch the BV log), but `num_constraints` and
/// solver SAT state are preserved end-to-end.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_snapshot_add_constraint_raw_roundtrip() {
    use z3::ast::Ast;

    let ctx = SymContext::new();
    // Mix: one assume_true (RustBV-known), one add_constraint_raw
    // (Z3-only, no RustBV form recorded).
    let x = RustBV::symbolic(&ctx, "snap_82g6_x", 32);
    let ten = RustBV::concrete(10, 32);
    let twenty = RustBV::concrete(20, 32);
    ctx.assume_true(&x.ugt(&ten, &ctx));

    // Raw path: build a Z3 Bool directly and feed it through
    // add_constraint_raw — mirrors `_add_constraints_to_state`'s
    // Z3-ptr fast path when claripy_to_rustbv fails to translate.
    let z3_ctx = z3::Context::thread_local();
    let raw_bool = {
        let x_z3 = x.to_z3_ast();
        let twenty_z3 = twenty.to_z3_ast();
        x_z3.bvult(&twenty_z3)
    };
    let raw_ptr = raw_bool.get_z3_ast().as_ptr() as usize;
    let z3_ast_ptr = unsafe { Z3AstPtr::from_borrowed_raw(&z3_ctx, raw_ptr) }
        .expect("raw Bool must yield a Z3AstPtr");
    ctx.add_constraint_raw(z3_ast_ptr);

    // Pre-snapshot bookkeeping. `num_constraints` counts both paths;
    // `assumed_constraint_count` only the assume path.
    let pre_total = ctx.num_constraints();
    let pre_assumed = ctx.assumed_constraint_count();
    assert_eq!(pre_total, 2, "raw + assume = 2 logical constraints");
    assert_eq!(pre_assumed, 1, "only the assume entry hits the BV log");

    // Round-trip through serde.
    let snap = ctx.to_snapshot();
    assert_eq!(snap.assumed_constraints.len(), 1);
    assert!(
        snap.reassert_assumed,
        "a non-merged context must reconstruct its assume class from IR"
    );
    assert!(
        !snap.residual_smtlib2.is_empty(),
        "snapshot must carry an SMT-LIB2 residual dump when raw constraints \
         are present"
    );
    let json = serde_json::to_string(&snap).expect("serialize");
    let restored: SymContextSnapshot = serde_json::from_str(&json).expect("deserialize");

    let ctx2 = SymContext::new();
    ctx2.restore_from_snapshot(&restored);

    // angr-82g6: num_constraints now matches pre-snapshot (was the
    // bug — restored counted only assumed_constraints).
    assert_eq!(
        ctx2.num_constraints(),
        pre_total,
        "num_constraints must round-trip through snapshot"
    );
    assert_eq!(
        ctx2.assumed_constraint_count(),
        pre_assumed,
        "assumed_constraint_count is preserved (raw entries stay raw)"
    );

    // Solver is still SAT and respects BOTH constraints (x > 10
    // AND x < 20).
    assert!(ctx2.is_sat());
    let x_val = ctx2.eval(&x).expect("x evaluable");
    assert!(
        x_val > 10 && x_val < 20,
        "restored x={x_val} must satisfy 10 < x < 20 — including \
         the raw-path x<20 constraint"
    );
}

/// angr-82g6: backward-compat path — a snapshot deserialized from
/// JSON that omits `solver_smtlib2` (older payloads) must still
/// restore via the assumed-replay codepath alone.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_snapshot_missing_solver_smtlib2_deserializes_with_default() {
    // Legacy JSON shape — a snapshot that omits both residual fields. The
    // serde defaults give an empty residual and `reassert_assumed == true`,
    // so restore falls through to the assume-only replay path.
    let legacy_json = r#"{"assumed_constraints":[]}"#;
    let restored: SymContextSnapshot =
        serde_json::from_str(legacy_json).expect("legacy JSON must parse");
    assert!(restored.residual_smtlib2.is_empty());
    assert!(restored.reassert_assumed);

    let ctx = SymContext::new();
    ctx.restore_from_snapshot(&restored);
    assert_eq!(ctx.num_constraints(), 0);
}

/// Mock-backend round-trip: with the Z3 feature off, the snapshot
/// still preserves the `(RustBV, bool)` log and restore replays the
/// pairs into a fresh context.
#[cfg(not(feature = "vex-engine-z3"))]
#[test]
fn test_snapshot_assumed_constraints_roundtrip_mock() {
    let ctx = SymContext::new_mock();
    let cond_a = RustBV::concrete(1, 1);
    let cond_b = RustBV::concrete(0, 1);
    ctx.assume_true(&cond_a);
    ctx.assume_false(&cond_b);
    let snap = ctx.to_snapshot();
    let json = serde_json::to_string(&snap).expect("serialize");
    let restored: SymContextSnapshot = serde_json::from_str(&json).expect("deserialize");
    let ctx2 = SymContext::new_mock();
    ctx2.restore_from_snapshot(&restored);
    let post = ctx2.get_assumed_constraints();
    assert_eq!(post.len(), 2);
    assert_eq!(post[0].1, true);
    assert_eq!(post[1].1, false);
}

// angr-ahypj: SymContext::translate_into cross-context constraint twin. Unlike
// the SMT-LIB2 round-trip above, this moves the path-constraint state into a
// fresh context via Z3_translate (no string serialization). A symbolic var
// pinned by a constraint, plus a derived value sharing that var's leaf, must
// re-evaluate identically after translation — and survive an A->B->A bounce.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_symcontext_translate_into_cross_context() {
    use z3::{Config, Context};

    let src = SymContext::new();
    let x = RustBV::symbolic(&src, "ahypj_tx_x", 32);
    src.assume_true(&x.eq(&RustBV::concrete(0x1234, 32), &src));
    // Derived value references x — eval under the target context only resolves
    // if the translated derived-leaf and the translated constraint-leaf
    // hash-cons to the SAME node (shared-AST identity across translate_into).
    let derived = x.add(&RustBV::concrete(1, 32), &src);

    let original = Context::thread_local();
    let target = Context::new(&Config::new());
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        target.get_z3_context().as_ptr() as usize,
        "test bug: target == source context",
    );

    Context::set_thread_local(&target);
    let new_ctx = src.translate_into(&target);
    let derived_t = derived.translate_into(&target);
    let got = new_ctx.eval(&derived_t);
    let sat = new_ctx.is_sat();
    let assumed_len = new_ctx.get_assumed_constraints().len();

    // A->B->A: bounce the already-translated context into a third context.
    let third = Context::new(&Config::new());
    Context::set_thread_local(&third);
    let back_ctx = new_ctx.translate_into(&third);
    let derived_b = derived_t.translate_into(&third);
    let got_back = back_ctx.eval(&derived_b);
    Context::set_thread_local(&original);

    assert_eq!(
        got,
        Some(0x1235),
        "translated derived(x)=x+1 must eval via the transferred constraint",
    );
    assert!(sat, "translated context must remain SAT");
    assert_eq!(assumed_len, 1, "assumed-constraint BV log must round-trip");
    assert_eq!(
        got_back,
        Some(0x1235),
        "A->B->A round trip must preserve the eval witness",
    );
}

// =============================================================================
// angr-t3l5o Phase 1: two-class (assume-IR + residual-text) round-trip tests.
// =============================================================================

/// THE WIN PATH: a pure-`assume` context emits NO SMT-LIB2 residual text —
/// the assume class round-trips entirely as `RustBV` IR. Proves the old
/// full-solver text dump vanishes on the common (codegate/cmu) migration path.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_snapshot_pure_assume_residual_empty() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "t3l5o_win_x", 32);
    let y = RustBV::symbolic(&ctx, "t3l5o_win_y", 32);
    let ten = RustBV::concrete(10, 32);
    let twenty = RustBV::concrete(20, 32);
    let zero = RustBV::concrete(0, 32);
    ctx.assume_true(&x.ugt(&ten, &ctx));
    ctx.assume_true(&x.ult(&twenty, &ctx));
    ctx.assume_false(&y.eq(&zero, &ctx));

    let pre_sat = ctx.is_sat();
    let pre_count = ctx.num_constraints();
    assert!(pre_sat);

    let snap = ctx.to_snapshot();
    assert!(
        snap.reassert_assumed,
        "pure-assume context is reconstructible"
    );
    assert_eq!(
        snap.residual_smtlib2, "",
        "WIN PATH: a pure-assume context must emit NO SMT-LIB2 residual text"
    );
    assert_eq!(snap.assumed_constraints.len(), 3);

    let json = serde_json::to_string(&snap).expect("serialize");
    let restored: SymContextSnapshot = serde_json::from_str(&json).expect("deserialize");
    let ctx2 = SymContext::new();
    ctx2.restore_from_snapshot(&restored);

    assert_eq!(ctx2.is_sat(), pre_sat, "restored sat must match");
    assert_eq!(
        ctx2.num_constraints(),
        pre_count,
        "restored constraint_count must match (rebuilt from IR)"
    );
    let x_val = ctx2.eval(&x).expect("x evaluable");
    let y_val = ctx2.eval(&y).expect("y evaluable");
    assert!(x_val > 10 && x_val < 20, "x={x_val} must satisfy 10<x<20");
    assert_ne!(y_val, 0, "y must be non-zero");
}

/// An `add_bv_constraint` (address-concretization) constraint has no `RustBV`
/// assume entry, so it round-trips via the residual text dump. The restored
/// solver must pin x to the concretized value.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_snapshot_add_bv_constraint_roundtrip() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "t3l5o_bv_x", 32);
    ctx.add_bv_constraint(&x, 0x1000);

    let pre_sat = ctx.is_sat();
    let pre_count = ctx.num_constraints();
    assert!(pre_sat);
    assert_eq!(pre_count, 1, "one bv-eq constraint");

    let snap = ctx.to_snapshot();
    assert!(snap.reassert_assumed);
    assert_eq!(
        snap.assumed_constraints.len(),
        0,
        "add_bv_constraint records no assume entry"
    );
    assert!(
        !snap.residual_smtlib2.is_empty(),
        "the bv-eq constraint must live in the residual dump"
    );

    let json = serde_json::to_string(&snap).expect("serialize");
    let restored: SymContextSnapshot = serde_json::from_str(&json).expect("deserialize");
    let ctx2 = SymContext::new();
    ctx2.restore_from_snapshot(&restored);

    assert_eq!(ctx2.is_sat(), pre_sat);
    assert_eq!(
        ctx2.num_constraints(),
        pre_count,
        "bv-eq constraint_count round-trips"
    );
    let x_val = ctx2.eval(&x).expect("x evaluable");
    assert_eq!(
        x_val, 0x1000,
        "restored x must equal the concretized address"
    );
}

/// A mixed assume + raw context: the assume class rebuilds from IR, the raw
/// class from the residual text. Both must be present and binding after
/// restore, with the assume entry alone in the BV-export log.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_snapshot_mixed_assume_raw_roundtrip() {
    use z3::ast::Ast;

    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "t3l5o_mix_x", 32);
    let ten = RustBV::concrete(10, 32);
    let twenty = RustBV::concrete(20, 32);
    // assume: x > 10 (RustBV-known)
    ctx.assume_true(&x.ugt(&ten, &ctx));
    // raw: x < 20 (no RustBV form recorded)
    let z3_ctx = z3::Context::thread_local();
    let raw_bool = {
        let xz = x.to_z3_ast();
        let tw = twenty.to_z3_ast();
        xz.bvult(&tw)
    };
    let raw_ptr = raw_bool.get_z3_ast().as_ptr() as usize;
    let z3_ast_ptr =
        unsafe { Z3AstPtr::from_borrowed_raw(&z3_ctx, raw_ptr) }.expect("raw Bool yields ptr");
    ctx.add_constraint_raw(z3_ast_ptr);

    let pre_sat = ctx.is_sat();
    let pre_count = ctx.num_constraints();
    assert_eq!(pre_count, 2, "assume + raw = 2 logical constraints");

    let snap = ctx.to_snapshot();
    assert!(snap.reassert_assumed);
    assert_eq!(
        snap.assumed_constraints.len(),
        1,
        "only the assume entry hits the BV log"
    );
    assert!(
        !snap.residual_smtlib2.is_empty(),
        "the raw entry lives in the residual dump"
    );

    let json = serde_json::to_string(&snap).expect("serialize");
    let restored: SymContextSnapshot = serde_json::from_str(&json).expect("deserialize");
    let ctx2 = SymContext::new();
    ctx2.restore_from_snapshot(&restored);

    assert_eq!(ctx2.is_sat(), pre_sat);
    assert_eq!(
        ctx2.num_constraints(),
        pre_count,
        "both assume and raw classes restore"
    );
    assert_eq!(
        ctx2.assumed_constraint_count(),
        1,
        "raw entry stays out of the BV log"
    );
    let x_val = ctx2.eval(&x).expect("x evaluable");
    assert!(
        x_val > 10 && x_val < 20,
        "restored x={x_val} must satisfy x>10 (assume) AND x<20 (raw)"
    );
}

/// A `merge()`-d context round-trips: the merge guards survive via the
/// residual text dump, and crucially the assume class is NOT re-asserted
/// (which would collapse the guarded disjunctions into unconditional
/// constraints and over-constrain the merged state).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_snapshot_merge_context_roundtrip() {
    let ctx1 = SymContext::new();
    let x = RustBV::symbolic(&ctx1, "t3l5o_merge_x", 32);
    let five = RustBV::concrete(5, 32);
    ctx1.assume_true(&x.ugt(&five, &ctx1)); // path 1: x > 5

    let ctx2 = SymContext::new();
    let three = RustBV::concrete(3, 32);
    ctx2.assume_true(&x.ult(&three, &ctx2)); // path 2: x < 3

    let m1 = RustBV::symbolic(&ctx1, "t3l5o_merge_m1", 1);
    let m2 = RustBV::symbolic(&ctx1, "t3l5o_merge_m2", 1);
    let merged = ctx1.merge(&[&ctx2], &[m1.clone(), m2.clone()]);

    let pre_sat = merged.is_sat();
    let pre_count = merged.num_constraints();
    assert!(pre_sat, "merged disjunction is satisfiable");

    let snap = merged.to_snapshot();
    assert!(
        !snap.reassert_assumed,
        "a merged context must NOT re-assert its export-only assume pairs"
    );
    assert!(
        !snap.residual_smtlib2.is_empty(),
        "merge guards live in the residual dump"
    );

    let json = serde_json::to_string(&snap).expect("serialize");
    let restored: SymContextSnapshot = serde_json::from_str(&json).expect("deserialize");
    let ctx3 = SymContext::new();
    ctx3.restore_from_snapshot(&restored);

    assert_eq!(ctx3.is_sat(), pre_sat, "restored sat must match");
    assert_eq!(
        ctx3.num_constraints(),
        pre_count,
        "restored constraint_count must match the merge guards"
    );

    // Discriminator: select the x<3 branch (m1=false, m2=true). The merge
    // guards must allow x<3 here — a buggy re-assert of the assume class
    // would force x>5 AND x<3 → UNSAT.
    let zero1 = RustBV::concrete(0, 1);
    let one1 = RustBV::concrete(1, 1);
    ctx3.assume_true(&m1.eq(&zero1, &ctx3));
    ctx3.assume_true(&m2.eq(&one1, &ctx3));
    assert!(
        ctx3.is_sat(),
        "x<3 (m2) branch must remain satisfiable — merge guards intact"
    );
    let x_val = ctx3.eval(&x).expect("x evaluable");
    assert!(
        x_val < 3,
        "x={x_val} must fall in the selected x<3 branch (guards not collapsed)"
    );
}
