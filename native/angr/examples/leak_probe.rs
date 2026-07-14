//! Deterministic allocator-churn probe for the nightly valgrind memcheck gate.
//!
//! Runs a fixed number of self-contained iterations over the core Rust
//! engine primitives (SymContext + solver, RustBV, SymbolicMemory,
//! RustSimState fork) and drops every object it creates. A correct engine
//! therefore leaks a *constant* number of bytes regardless of the iteration
//! count: one-time z3 globals and lazy statics. A per-iteration leak shows up
//! as leaked bytes that scale with `ANGR_LEAK_ITERS`, which is exactly what
//! `tests/benchmarks/run_valgrind_leak_check.py` measures (it runs this binary
//! twice, at N and 10N, and gates on the per-iteration slope).
//!
//! Deliberately Python-free: running valgrind through CPython drowns real
//! findings in interpreter false-positives, and the RSS gate
//! (`run_leak_check.py`) already covers the Python-driven Callable path.
//!
//! Run standalone with:
//!   ANGR_LEAK_ITERS=200 cargo run --release --example leak_probe

use rustylib::memory::{Permission, SymbolicMemory};
use rustylib::state::RustSimState;
use rustylib::symbolic::{RustBV, SymContext};
use rustylib::vex::ir::Endness;

const RW: Permission = Permission {
    read: true,
    write: true,
    execute: false,
};

/// Solver churn: symbolic constraints, a feasibility check, a context fork.
fn exercise_solver(seed: u64) {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "x", 64);
    let lo = RustBV::concrete(seed as u128, 64);
    let hi = RustBV::concrete(seed as u128 + 100, 64);

    ctx.assume_false(&x.ult(&lo, &ctx));
    ctx.assume_true(&x.ult(&hi, &ctx));

    let probe = x.ult(&RustBV::concrete(seed as u128 + 50, 64), &ctx);
    ctx.push();
    let _ = ctx.check_branch_feasibility(&probe);
    ctx.pop();

    // Fork carries the constraint set into a fresh solver — the hot path the
    // exploration loop hits on every branch.
    let child = ctx.fork();
    let y = RustBV::symbolic(&child, "y", 64);
    child.assume_true(&y.ult(&hi, &child));
}

/// Memory churn: concrete + symbolic stores, loads, and a page fork.
fn exercise_memory(seed: u64) {
    let ctx = SymContext::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    let base = 0x400000u64;
    mem.map(base, 0x4000, RW);

    for page in 0..4u64 {
        let addr = base + page * 0x1000;
        mem.store_concrete(addr, RustBV::concrete(seed as u128 + page as u128, 64))
            .unwrap();
        let _ = mem.load_concrete(addr, 8, &ctx).unwrap();
    }

    let sym = RustBV::symbolic(&ctx, "m", 64);
    mem.store_concrete(base + 0x2000, sym).unwrap();
    let _ = mem.load_concrete(base + 0x2000, 8, &ctx).unwrap();

    let _forked = mem.fork();
}

/// State churn: build a state, fork it, drop both.
fn exercise_state(seed: u64) {
    let mut state = RustSimState::new("AMD64").unwrap();
    state.set_register("rax", RustBV::concrete(seed as u128, 64));
    state.set_register("rsp", RustBV::concrete(0x7FFF_FFFF_0000, 64));
    state.set_pc(0x401000 + seed);
    let _forked = state.fork();
}

/// Deliberate 64-byte-per-iteration leak, enabled only by
/// `ANGR_LEAK_PROBE_INJECT=1`. This exists so the gate itself is testable: a
/// leak check that has never been observed to fail is indistinguishable from
/// one that cannot fail. `run_valgrind_leak_check.py --self-test` runs the
/// probe with this on and asserts the slope threshold trips.
fn inject_leak() {
    // `std::mem::forget(Box::new(..))` does NOT work here: in release the
    // allocation is dead (never read, never dropped) and LLVM elides it
    // outright, so memcheck sees nothing and the self-test silently passes.
    // Leaking a raw pointer through `black_box` keeps the malloc opaque and
    // unreachable, which is exactly "definitely lost".
    let ptr = Box::into_raw(vec![0u8; 64].into_boxed_slice());
    std::hint::black_box(ptr);
}

fn main() {
    let iters: u64 = std::env::var("ANGR_LEAK_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let inject = std::env::var("ANGR_LEAK_PROBE_INJECT").as_deref() == Ok("1");

    for i in 0..iters {
        exercise_solver(i);
        exercise_memory(i);
        exercise_state(i);
        if inject {
            inject_leak();
        }
    }

    println!("leak_probe: completed {iters} iterations (inject={inject})");
}
