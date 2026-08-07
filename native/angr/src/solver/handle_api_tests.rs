//! Unit tests for the handle-based (claripy-bypass) solver API.
//!
//! Split from the parent module per `rust-mod-tests-sibling-extraction`.
//!
//! angr-sqfj8.125: before this file the `op_*` wrappers, their error paths
//! (`invalid_handle_id`, `reject_extend_narrowing`) and the handle-based
//! query methods had *no* in-crate coverage — every regression had to be
//! caught by the much slower Python integration suite
//! (`tests/engines/rust/test_solver_ops.py`). Nothing here needs Python: the
//! handle API is pure Rust, so these run in a plain `cargo test`.

use super::*;

/// Intern a concrete value and return just its handle id — the form every
/// `op_*` wrapper takes.
fn c(ctx: &RustSolverContext, value: u128, width: u32) -> u64 {
    ctx.create_concrete(value, width).id()
}

/// A handle id that is provably absent from a fresh table.
const MISSING: u64 = u64::MAX;

/// The error message of a wrapper that must fail.
///
/// Goes through `err()` rather than `unwrap_err()` because `RustBVHandle` is a
/// `#[pyclass]` and carries no `Debug` impl for the latter to print. Rendering
/// a `PyErr` touches the interpreter, so this is the one place in the file
/// that needs it initialized — the ops themselves are pure Rust.
fn err_msg(result: PyResult<RustBVHandle>) -> String {
    let err = result.err().expect("expected an Err");
    pyo3::Python::initialize();
    Python::attach(|_py| err.to_string())
}

// =============================================================================
// Symbol-table lifecycle
// =============================================================================

#[test]
fn test_create_concrete_roundtrips_through_eval_handle() {
    let ctx = RustSolverContext::new();
    let h = ctx.create_concrete(0xdead_beef, 32);
    assert_eq!(h.width(), 32);
    assert_eq!(h.concrete(), Some(0xdead_beef));
    assert_eq!(ctx.eval_handle(h.id()), Some(0xdead_beef));
}

#[test]
fn test_create_symbolic_is_not_concrete_and_counts() {
    let ctx = RustSolverContext::new();
    assert_eq!(ctx.handle_count(), 0);
    let x = ctx.create_symbolic("hapi_x", 64);
    assert_eq!(x.width(), 64);
    assert!(!x.is_concrete());
    assert_eq!(ctx.handle_count(), 1);
    let _ = ctx.create_concrete(1, 64);
    assert_eq!(ctx.handle_count(), 2);
}

/// Every handle-taking query must decline a missing id rather than panic.
#[test]
fn test_missing_handle_queries_decline() {
    let ctx = RustSolverContext::new();
    assert_eq!(ctx.eval_handle(MISSING), None);
    assert_eq!(ctx.min_handle(MISSING, false), None);
    assert_eq!(ctx.max_handle(MISSING, false), None);
    assert!(ctx.eval_upto_handle(MISSING, 4).is_empty());
    assert!(!ctx.solution_handle(MISSING, 0));
    assert!(ctx.add_constraint_handle(MISSING).is_err());
}

// =============================================================================
// op_* wrappers — concrete folding
// =============================================================================

#[test]
fn test_arith_wrappers_fold_concrete() {
    let ctx = RustSolverContext::new();
    let ten = c(&ctx, 10, 8);
    let three = c(&ctx, 3, 8);
    // 246 == -10 as a signed 8-bit value.
    let neg_ten = c(&ctx, 246, 8);

    let cases: [(&str, u64, u128); 7] = [
        ("op_add", ctx.op_add(ten, three).unwrap().id(), 13),
        ("op_sub", ctx.op_sub(ten, three).unwrap().id(), 7),
        ("op_mul", ctx.op_mul(ten, three).unwrap().id(), 30),
        ("op_udiv", ctx.op_udiv(ten, three).unwrap().id(), 3),
        ("op_urem", ctx.op_urem(ten, three).unwrap().id(), 1),
        // Signed division truncates toward zero: -10 / 3 == -3 (0xfd).
        ("op_sdiv", ctx.op_sdiv(neg_ten, three).unwrap().id(), 0xfd),
        // ... and the remainder takes the dividend's sign: -1 (0xff).
        ("op_srem", ctx.op_srem(neg_ten, three).unwrap().id(), 0xff),
    ];
    for (name, handle, expected) in cases {
        assert_eq!(ctx.eval_handle(handle), Some(expected), "{name}");
    }

    assert_eq!(ctx.eval_handle(ctx.op_neg(ten).unwrap().id()), Some(246));
}

#[test]
fn test_bitwise_wrappers_fold_concrete() {
    let ctx = RustSolverContext::new();
    let a = c(&ctx, 0b1100, 8);
    let b = c(&ctx, 0b1010, 8);

    assert_eq!(
        ctx.eval_handle(ctx.op_and(a, b).unwrap().id()),
        Some(0b1000)
    );
    assert_eq!(ctx.eval_handle(ctx.op_or(a, b).unwrap().id()), Some(0b1110));
    assert_eq!(
        ctx.eval_handle(ctx.op_xor(a, b).unwrap().id()),
        Some(0b0110)
    );

    let f0 = c(&ctx, 0x0f, 8);
    assert_eq!(ctx.eval_handle(ctx.op_not(f0).unwrap().id()), Some(0xf0));
}

#[test]
fn test_shift_and_rotate_wrappers_fold_concrete() {
    let ctx = RustSolverContext::new();
    let one = c(&ctx, 1, 8);
    let high = c(&ctx, 0x80, 8);
    let edges = c(&ctx, 0x81, 8);

    assert_eq!(
        ctx.eval_handle(ctx.op_shl(edges, one).unwrap().id()),
        Some(0x02)
    );
    assert_eq!(
        ctx.eval_handle(ctx.op_lshr(high, one).unwrap().id()),
        Some(0x40)
    );
    // Arithmetic shift replicates the sign bit, so 0x80 >>a 1 == 0xc0.
    assert_eq!(
        ctx.eval_handle(ctx.op_ashr(high, one).unwrap().id()),
        Some(0xc0)
    );
    assert_eq!(
        ctx.eval_handle(ctx.op_rotl(edges, one).unwrap().id()),
        Some(0x03)
    );
    assert_eq!(
        ctx.eval_handle(ctx.op_rotr(edges, one).unwrap().id()),
        Some(0xc0)
    );
}

/// Every comparison wrapper must yield a 1-bit handle carrying 1/0, not a
/// full-width value — the interpreter feeds these straight to
/// `add_constraint_handle`, which treats a non-1-bit operand differently.
#[test]
fn test_comparison_wrappers_yield_1bit_truth_values() {
    let ctx = RustSolverContext::new();
    // 200 as u8 is -56 signed, so the signed and unsigned answers differ.
    let a = c(&ctx, 200, 8);
    let b = c(&ctx, 5, 8);

    let cases: [(&str, RustBVHandle, u128); 10] = [
        ("op_eq", ctx.op_eq(a, b).unwrap(), 0),
        ("op_ne", ctx.op_ne(a, b).unwrap(), 1),
        ("op_ult", ctx.op_ult(a, b).unwrap(), 0),
        ("op_ule", ctx.op_ule(a, b).unwrap(), 0),
        ("op_ugt", ctx.op_ugt(a, b).unwrap(), 1),
        ("op_uge", ctx.op_uge(a, b).unwrap(), 1),
        ("op_slt", ctx.op_slt(a, b).unwrap(), 1),
        ("op_sle", ctx.op_sle(a, b).unwrap(), 1),
        ("op_sgt", ctx.op_sgt(a, b).unwrap(), 0),
        ("op_sge", ctx.op_sge(a, b).unwrap(), 0),
    ];
    for (name, handle, expected) in cases {
        assert_eq!(handle.width(), 1, "{name} must return a 1-bit handle");
        assert_eq!(ctx.eval_handle(handle.id()), Some(expected), "{name}");
    }
}

#[test]
fn test_conversion_wrappers_fold_concrete() {
    let ctx = RustSolverContext::new();
    let byte = c(&ctx, 0x81, 8);

    let zx = ctx.op_zero_extend(byte, 16).unwrap();
    assert_eq!(zx.width(), 16);
    assert_eq!(ctx.eval_handle(zx.id()), Some(0x0081));

    let sx = ctx.op_sign_extend(byte, 16).unwrap();
    assert_eq!(sx.width(), 16);
    assert_eq!(ctx.eval_handle(sx.id()), Some(0xff81));

    let word = c(&ctx, 0xabcd, 16);
    let tr = ctx.op_truncate(word, 8).unwrap();
    assert_eq!(tr.width(), 8);
    assert_eq!(ctx.eval_handle(tr.id()), Some(0xcd));

    let hi = ctx.op_extract(word, 15, 8).unwrap();
    assert_eq!(hi.width(), 8);
    assert_eq!(ctx.eval_handle(hi.id()), Some(0xab));

    // `a` becomes the high bits.
    let ab = c(&ctx, 0xab, 8);
    let cd = c(&ctx, 0xcd, 8);
    let cat = ctx.op_concat(ab, cd).unwrap();
    assert_eq!(cat.width(), 16);
    assert_eq!(ctx.eval_handle(cat.id()), Some(0xabcd));
}

#[test]
fn test_op_ite_selects_by_condition() {
    let ctx = RustSolverContext::new();
    let t = c(&ctx, 1, 1);
    let f = c(&ctx, 0, 1);
    let then_v = c(&ctx, 0xaa, 8);
    let else_v = c(&ctx, 0x55, 8);

    let taken = ctx.op_ite(t, then_v, else_v).unwrap();
    assert_eq!(taken.width(), 8);
    assert_eq!(ctx.eval_handle(taken.id()), Some(0xaa));
    assert_eq!(
        ctx.eval_handle(ctx.op_ite(f, then_v, else_v).unwrap().id()),
        Some(0x55)
    );
}

// =============================================================================
// op_* wrappers — error paths
// =============================================================================

/// `binop`'s two failure modes both surface as `ValueError` with a message
/// naming the cause; the interpreter relies on the distinction.
#[test]
fn test_binop_error_paths() {
    let ctx = RustSolverContext::new();
    let a = c(&ctx, 1, 8);
    let wide = c(&ctx, 1, 32);

    let missing = err_msg(ctx.op_add(a, MISSING));
    assert!(missing.contains("invalid handle id"), "{missing}");

    let mismatch = err_msg(ctx.op_add(a, wide));
    assert!(mismatch.contains("width mismatch"), "{mismatch}");
    assert!(mismatch.contains("8-bit vs 32-bit"), "{mismatch}");
}

/// The `opt_op` wrappers have to name every operand they dereference in the
/// error — a bare `None` says nothing about which handle missed.
#[test]
fn test_opt_op_error_paths_name_their_operands() {
    let ctx = RustSolverContext::new();
    let a = c(&ctx, 1, 8);
    let cond = c(&ctx, 1, 1);

    for (name, msg) in [
        ("op_neg", err_msg(ctx.op_neg(MISSING))),
        ("op_not", err_msg(ctx.op_not(MISSING))),
        ("op_truncate", err_msg(ctx.op_truncate(MISSING, 4))),
        ("op_extract", err_msg(ctx.op_extract(MISSING, 3, 0))),
        ("op_zero_extend", err_msg(ctx.op_zero_extend(MISSING, 16))),
        ("op_sign_extend", err_msg(ctx.op_sign_extend(MISSING, 16))),
        ("op_concat", err_msg(ctx.op_concat(a, MISSING))),
        ("op_ite", err_msg(ctx.op_ite(cond, a, MISSING))),
    ] {
        assert!(msg.contains("invalid handle id"), "{name}: {msg}");
        assert!(msg.contains(&MISSING.to_string()), "{name}: {msg}");
    }
}

/// angr-ph300.38: an extend asked to *narrow* must be rejected at the
/// boundary. Left through, it returns a handle wider than requested and only
/// blows up much later as a Z3 sort error.
#[test]
fn test_extend_rejects_narrowing_width() {
    let ctx = RustSolverContext::new();
    let word = c(&ctx, 0xabcd, 16);

    for (name, msg) in [
        ("op_zero_extend", err_msg(ctx.op_zero_extend(word, 8))),
        ("op_sign_extend", err_msg(ctx.op_sign_extend(word, 8))),
    ] {
        assert!(
            msg.contains("to_width 8 < source width 16"),
            "{name}: {msg}"
        );
        assert!(msg.contains("op_truncate/op_extract"), "{name}: {msg}");
    }

    // Same width is not narrowing — it must still be accepted.
    assert!(ctx.op_zero_extend(word, 16).is_ok());
    assert!(ctx.op_sign_extend(word, 16).is_ok());
}

// =============================================================================
// Constraint + query methods (need a real solver)
// =============================================================================

/// A 1-bit constraint handle is asserted as-is; the solver must then agree
/// with it on every query method.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_add_constraint_handle_1bit_pins_symbolic_value() {
    let ctx = RustSolverContext::new();
    let x = ctx.create_symbolic("hapi_pin_x", 32).id();
    let five = c(&ctx, 5, 32);
    let eq = ctx.op_eq(x, five).unwrap();
    assert_eq!(eq.width(), 1);

    ctx.add_constraint_handle(eq.id()).unwrap();
    assert!(ctx.satisfiable());
    assert_eq!(ctx.eval_handle(x), Some(5));
    assert_eq!(ctx.min_handle(x, false), Some(5));
    assert_eq!(ctx.max_handle(x, false), Some(5));
    assert_eq!(ctx.eval_upto_handle(x, 4), vec![5]);
    assert!(ctx.solution_handle(x, 5));
    assert!(!ctx.solution_handle(x, 6));
}

/// A wider-than-1-bit handle is constrained as `!= 0`, not truncated to its
/// low bit — the branch the interpreter hits when it hands over a raw value
/// rather than a comparison.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_add_constraint_handle_wide_asserts_nonzero() {
    let ctx = RustSolverContext::new();
    let x = ctx.create_symbolic("hapi_nz_x", 8).id();
    ctx.add_constraint_handle(x).unwrap();
    assert!(ctx.satisfiable());
    assert!(!ctx.solution_handle(x, 0), "0 must be excluded by != 0");
    assert!(ctx.solution_handle(x, 1));
    // Low bit clear but value non-zero: allowed, so this is not a `& 1` test.
    assert!(ctx.solution_handle(x, 2));
}

/// `min_handle`/`max_handle` read the signed flag; with the sign bit free the
/// signed and unsigned extremes of the same range differ.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_min_max_handle_respect_signedness() {
    let ctx = RustSolverContext::new();
    let x = ctx.create_symbolic("hapi_minmax_x", 8).id();
    // No constraints: the full 8-bit range is open.
    assert_eq!(ctx.min_handle(x, false), Some(0));
    assert_eq!(ctx.max_handle(x, false), Some(0xff));
    // Signed: 0x80 (-128) is the minimum, 0x7f (127) the maximum.
    assert_eq!(ctx.min_handle(x, true), Some(0x80));
    assert_eq!(ctx.max_handle(x, true), Some(0x7f));
}

/// `eval_upto_handle` must stop at the requested count and enumerate every
/// solution when fewer exist than requested.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_eval_upto_handle_enumerates_and_caps() {
    let ctx = RustSolverContext::new();
    let x = ctx.create_symbolic("hapi_upto_x", 8).id();
    let three = c(&ctx, 3, 8);
    ctx.add_constraint_handle(ctx.op_ult(x, three).unwrap().id())
        .unwrap();

    let mut all = ctx.eval_upto_handle(x, 10);
    all.sort_unstable();
    assert_eq!(all, vec![0, 1, 2]);
    assert_eq!(ctx.eval_upto_handle(x, 2).len(), 2);
}

/// Handles minted before a `fork()` stay resolvable in the child, and
/// constraints added to the child do not leak back to the parent.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_fork_keeps_handles_and_isolates_constraints() {
    let parent = RustSolverContext::new();
    let x = parent.create_symbolic("hapi_fork_x", 8).id();
    let seven = c(&parent, 7, 8);

    let child = parent.fork();
    assert_eq!(child.handle_count(), parent.handle_count());

    child
        .add_constraint_handle(child.op_eq(x, seven).unwrap().id())
        .unwrap();
    assert_eq!(child.eval_handle(x), Some(7));
    // The parent never saw that constraint.
    assert!(parent.solution_handle(x, 0));
}
