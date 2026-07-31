//! Behavioral smoke tests run by the Rust feature-flag CI matrix.
//!
//! Each Cargo `--features` combination listed in
//! `.github/workflows/nightly-ci.yml::rust_feature_flags` runs `cargo test`,
//! which picks up the feature-gated tests in this file. A combo that
//! compiles but produces wrong runtime answers should fail at least one of
//! these checks. The checks are deliberately minimal — the unit tests in
//! `src/` cover deeper behavior; this file exists so that *every* shipped
//! feature combo demonstrably executes its core code path end-to-end.
//!
//! See bd issue angr-7c9j.
//!
//! Coverage by feature combo:
//! - `""`            → smoke_concrete_arithmetic, smoke_symbolic_handles
//! - `"automaton"`   → + smoke_automaton_dfa
//! - `"vex-engine"`  → + smoke_vex_lifter_imports
//! - `"vex-engine,vex-engine-z3"` → + smoke_z3_solve_constraint

// Integration-test harness code panicking on unwrap/expect is the desired
// behavior, same rationale as #[cfg(test)] code in lib.rs -- not part of the
// angr-9ke6b.212 production-code debt tracker.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rustylib::symbolic::{RustBV, SymContext};

/// Concrete arithmetic via SymContext is the universal core. Available in
/// every feature combo because `symbolic` is not feature-gated.
#[test]
fn smoke_concrete_arithmetic() {
    let ctx = SymContext::new();
    let a = RustBV::concrete(7, 32);
    let b = RustBV::concrete(35, 32);
    let sum = a.add(&b, &ctx);
    assert!(
        sum.is_concrete(),
        "add of two concrete BVs should be concrete"
    );
    assert_eq!(sum.as_u128(), Some(42));
    assert_eq!(sum.width(), 32);
}

/// Symbolic-handle construction: the no-z3 stub builds a `Symbolic { id, name,
/// width }` variant; with z3 it also attaches a Z3 BV AST. Either way, the
/// public BV API must report it as symbolic.
#[test]
fn smoke_symbolic_handles() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "smoke_x", 32);
    assert!(x.is_symbolic());
    assert!(!x.is_concrete());
    assert_eq!(x.width(), 32);
    assert_eq!(x.as_u128(), None);
}

/// Z3-backed solving: build `x == 0xDEADBEEF`, push as assumption, evaluate
/// `x`. Without z3 this exact path is unavailable (eval would only return
/// concrete inputs), so the test is gated to the z3 feature.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn smoke_z3_solve_constraint() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "smoke_x", 32);
    let target = RustBV::concrete(0xDEAD_BEEF, 32);
    let cond = x.eq(&target, &ctx);
    assert_eq!(cond.width(), 1);
    ctx.assume_true(&cond);
    assert!(ctx.is_sat(), "x == 0xDEADBEEF must be satisfiable");
    assert_eq!(ctx.eval(&x), Some(0xDEAD_BEEF));
}

/// VEX engine module is reachable: lifter and IR types compile and
/// instantiate. We don't lift a real IRSB here — pyvex isn't loaded — but we
/// do touch the public surface so a feature-flag regression that breaks the
/// module's public API (cf. angr-asth) shows up.
#[cfg(feature = "vex-engine")]
#[test]
fn smoke_vex_lifter_imports() {
    use rustylib::vex::ir::Endness;
    let _ = Endness::Little;
    let _ = Endness::Big;
}

/// Automaton smoke: build a 3-state DFA that accepts only the string "ab",
/// then verify the transitions exist and the language is non-empty.
#[cfg(feature = "automaton")]
#[test]
fn smoke_automaton_dfa() {
    use rustylib::automaton::DFA;
    let mut dfa = DFA::new();
    let s0 = dfa.add_state();
    let s1 = dfa.add_state();
    let s2 = dfa.add_state();
    dfa.set_start_state(s0);
    dfa.add_transition(s0, b'a' as u32, s1);
    dfa.add_transition(s1, b'b' as u32, s2);
    dfa.add_final_state(s2);
    assert_eq!(dfa.transition(s0, b'a' as u32), Some(s1));
    assert_eq!(dfa.transition(s1, b'b' as u32), Some(s2));
    assert!(!dfa.is_empty(), "DFA accepting 'ab' is non-empty");
}
