//! The symbolic-address fallback inside the [`SymbolicMemory::load`] /
//! [`SymbolicMemory::store`] convenience wrappers (angr-mv08h pin-on-path
//! semantics).
//!
//! Both wrappers accept an address `RustBV` that may be symbolic. When
//! `RustBV::as_u64` fails they ask the solver for *one* witness via
//! `SymContext::eval` and then `concretize::pin_fallback_addr` it onto the
//! path, so a later `eval` of the same address cannot pick a different
//! solution and make the access invisible. When the solver has no witness at
//! all they report `MemoryError::SymbolicAddress` rather than guessing.
//!
//! These wrappers have **no production caller** (angr-9ke6b.229) — production
//! traffic enters at `load_symbolic`/`load_symbolic_unified` and the four
//! `store_*` concretizer entry points — and every other test in
//! `memory::tests` hands them a concrete address, so without this file the
//! whole `None` arm of both wrappers is dead code in test as well
//! (angr-6cp06.66).

use super::super::*;

/// Symbolic 64-bit address constrained to exactly `{a, b}` via
/// `(addr == a) | (addr == b)`, the same two-solution idiom
/// `symbolic_cross_page` uses.
#[cfg(feature = "vex-engine-z3")]
fn two_solution_addr(ctx: &SymContext, name: &str, a: u64, b: u64) -> RustBV {
    let addr = RustBV::symbolic(ctx, name, 64);
    let eq_a = addr.eq(&RustBV::concrete(a as u128, 64), ctx);
    let eq_b = addr.eq(&RustBV::concrete(b as u128, 64), ctx);
    ctx.assume_true(&eq_a.or(&eq_b, ctx));
    assert!(ctx.is_sat(), "two-solution constraint must be SAT");
    addr
}

/// An unsatisfiable context, so `SymContext::eval` yields no witness for any
/// symbolic value. Built from two contradictory equalities on a *symbolic*
/// operand rather than a concrete `false`, so nothing folds the contradiction
/// away before it reaches the solver.
#[cfg(feature = "vex-engine-z3")]
fn unsat_ctx_with_symbolic_addr() -> (SymContext, RustBV) {
    let ctx = SymContext::new_mock();
    let addr = RustBV::symbolic(&ctx, "unsat_addr", 64);
    ctx.assume_true(&addr.eq(&RustBV::concrete(0x1000, 64), &ctx));
    ctx.assume_true(&addr.eq(&RustBV::concrete(0x2000, 64), &ctx));
    assert!(!ctx.is_sat(), "contradictory equalities must be UNSAT");
    (ctx, addr)
}

/// `load` with a two-solution address returns the bytes at whichever solution
/// the solver picked, and pins that choice: the address is single-valued
/// afterwards. Without the pin, a later `eval` of the same address could
/// legally answer with the *other* solution, and the caller's loaded value
/// would then belong to an address the path no longer says was read.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_load_symbolic_addr_pins_chosen_solution() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.store_concrete(0x1000, RustBV::concrete(0x1111_1111, 32))
        .expect("store at 0x1000");
    mem.store_concrete(0x1004, RustBV::concrete(0x2222_2222, 32))
        .expect("store at 0x1004");

    let addr = two_solution_addr(&ctx, "load_pin_addr", 0x1000, 0x1004);
    let loaded = mem
        .load(addr.clone(), 4, &ctx)
        .expect("symbolic-address load must resolve");

    let chosen = ctx.eval(&addr).expect("address must stay resolvable");
    assert!(
        chosen == 0x1000 || chosen == 0x1004,
        "witness must be one of the two constrained solutions, got {chosen:#x}"
    );
    let expected = if chosen == 0x1000 {
        0x1111_1111
    } else {
        0x2222_2222
    };
    assert_eq!(
        ctx.eval(&loaded),
        Some(expected),
        "loaded value must be the one stored at the chosen address {chosen:#x}"
    );

    let solutions = ctx.eval_upto(&addr, 4);
    assert_eq!(
        solutions,
        vec![chosen],
        "pin_fallback_addr must leave the address single-valued"
    );
}

/// `store` with a two-solution address writes at whichever solution the
/// solver picked, pins it, and leaves the other candidate untouched. The pin
/// is what makes the write visible: an unpinned path could later concretize
/// the same address to the other candidate and read back the stale value.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_store_symbolic_addr_pins_chosen_solution() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.store_concrete(0x1000, RustBV::concrete(0, 32))
        .expect("zero 0x1000");
    mem.store_concrete(0x1004, RustBV::concrete(0, 32))
        .expect("zero 0x1004");

    let addr = two_solution_addr(&ctx, "store_pin_addr", 0x1000, 0x1004);
    mem.store(&addr, RustBV::concrete(0xDEAD_BEEF, 32), &ctx)
        .expect("symbolic-address store must resolve");

    let chosen = ctx.eval(&addr).expect("address must stay resolvable");
    let other = if chosen == 0x1000 { 0x1004 } else { 0x1000 };
    assert_eq!(
        ctx.eval(&mem.load_concrete(chosen as u64, 4, &ctx).expect("load chosen")),
        Some(0xDEAD_BEEF),
        "the chosen address {chosen:#x} must hold the stored value"
    );
    assert_eq!(
        ctx.eval(&mem.load_concrete(other, 4, &ctx).expect("load other")),
        Some(0),
        "the rejected candidate {other:#x} must be untouched"
    );

    let solutions = ctx.eval_upto(&addr, 4);
    assert_eq!(
        solutions,
        vec![chosen],
        "pin_fallback_addr must leave the address single-valued"
    );
}

/// With no witness available, `load` reports `SymbolicAddress` instead of
/// falling back to some default address. The distinct message text is the
/// only thing separating this from the store-side arm below.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_load_unresolvable_symbolic_addr_errors() {
    let (ctx, addr) = unsat_ctx_with_symbolic_addr();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    match mem.load(addr, 4, &ctx) {
        Err(MemoryError::SymbolicAddress { description }) => {
            assert_eq!(description, "could not resolve address");
        }
        other => panic!("expected SymbolicAddress, got {other:?}"),
    }
}

/// Store-side mirror of `test_load_unresolvable_symbolic_addr_errors`. Also
/// pins that the failed store leaves memory alone — the wrapper returns
/// before `store_concrete` is reached, so neither candidate page is written
/// or auto-mapped.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_store_unresolvable_symbolic_addr_errors() {
    let (ctx, addr) = unsat_ctx_with_symbolic_addr();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.store_concrete(0x1000, RustBV::concrete(0x5555_5555, 32))
        .expect("seed 0x1000");

    match mem.store(&addr, RustBV::concrete(0xDEAD_BEEF, 32), &ctx) {
        Err(MemoryError::SymbolicAddress { description }) => {
            assert_eq!(description, "could not resolve address for store");
        }
        other => panic!("expected SymbolicAddress, got {other:?}"),
    }

    // The seeded byte survives; nothing was written under the failed store.
    // `load_concrete`'s own permission/mapping checks are unaffected by the
    // UNSAT context — only solver *witnesses* are unavailable.
    let seeded = mem.load_concrete(0x1000, 4, &ctx).expect("seed still loads");
    assert_eq!(
        seeded.as_u64(),
        Some(0x5555_5555),
        "a failed store must not modify memory"
    );
}
