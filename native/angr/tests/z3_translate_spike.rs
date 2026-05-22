//! Spike test for angr-59jk.2 — measure Z3_translate cost between two
//! Z3 contexts. Validates shared-nothing parallel exploration is feasible;
//! feeds the design comparison in companion bead angr-59jk.1.
//!
//! Three checks:
//!   1. `translate_correctness` — translated AST satisfies the same model
//!      as the source AST.
//!   2. `translate_cost_micro` — per-AST-node translate latency for a
//!      synthetic ~4000-node chain of bvadd/bvxor/extract/concat ops.
//!   3. `translate_cost_state_export_shape` — translate a graph closer to
//!      a real exported-state shape (many disjoint BVs + a few wide
//!      constraints over them). Reports total wall, node count, ns/node.

#![cfg(feature = "vex-engine-z3")]

use std::time::Instant;
use z3::ast::BV;
use z3::{Config, Context, SatResult, Solver, Translate};

/// AST built in ctx_a, translated to ctx_b, must remain SAT and give the
/// expected eval for the original symbol.
#[test]
fn translate_correctness() {
    let cfg = Config::new();
    let ctx_a = Context::new(&cfg);
    let ctx_b = Context::new(&cfg);

    // Build everything in ctx_a.
    Context::set_thread_local(&ctx_a);
    let x = BV::new_const("x", 32);
    let y = BV::new_const("y", 32);
    // sum = (x[15:0] :: y[15:0]) — 32-bit BV via Concat of two Extracts
    let xl = x.extract(15, 0);
    let yl = y.extract(15, 0);
    let sum = xl.concat(&yl);
    let target = BV::from_u64(0xDEAD_BEEF, 32);
    let constraint_a = sum.eq(&target);

    // Translate the constraint to ctx_b. The translated Bool carries
    // translated copies of x and y; freshly-named consts in ctx_b are
    // distinct from any consts created here.
    let x_b: BV = x.translate(&ctx_b);
    let y_b: BV = y.translate(&ctx_b);
    let constraint_b = constraint_a.translate(&ctx_b);

    // Solver lives in ctx_b — set thread_local so Solver::new() picks it up.
    Context::set_thread_local(&ctx_b);
    let solver = Solver::new();
    solver.assert(&constraint_b);
    assert_eq!(
        solver.check(),
        SatResult::Sat,
        "translated constraint must be SAT in ctx_b"
    );

    // Recover a model from ctx_b's solver and check sum bits match target.
    let model = solver.get_model().expect("model");
    let xv = model.eval(&x_b, true).and_then(|bv| bv.as_u64()).expect("x eval");
    let yv = model.eval(&y_b, true).and_then(|bv| bv.as_u64()).expect("y eval");
    let recombined = ((xv as u32 & 0xFFFF) << 16) | (yv as u32 & 0xFFFF);
    assert_eq!(recombined, 0xDEAD_BEEF, "x and y low-halves must form 0xDEADBEEF");
}

/// Build a chain of ~4000 BV nodes in ctx_a, translate to ctx_b, report
/// ns/node. Chain shape: each iter adds (acc + i, then xor x), so 2 new
/// node types per iter plus a fresh constant — ~3 nodes per iter, plus
/// the leaves.
#[test]
fn translate_cost_micro() {
    const N: usize = 1000;

    let cfg = Config::new();
    let ctx_a = Context::new(&cfg);
    let ctx_b = Context::new(&cfg);

    Context::set_thread_local(&ctx_a);
    let x = BV::new_const("x", 32);
    let mut acc = x.clone();
    for i in 0..N {
        let k = BV::from_u64(i as u64, 32);
        acc = acc.bvadd(&k).bvxor(&x);
    }
    // approximate raw-node count: x leaf + N constants + N bvadd + N bvxor
    let node_count: u64 = 1 + 3 * (N as u64);

    let start = Instant::now();
    let _translated: BV = acc.translate(&ctx_b);
    let elapsed = start.elapsed();

    let ns_per_node = elapsed.as_nanos() / node_count as u128;
    eprintln!(
        "Z3_translate micro: N={}, ~{} AST nodes, total {:?}, {} ns/node",
        N, node_count, elapsed, ns_per_node
    );

    // Sanity: 1000-iter chain must translate in well under a second.
    assert!(
        elapsed.as_secs() < 5,
        "translate took unreasonably long: {:?}",
        elapsed
    );
}

/// Closer to a real exported-state shape: lots of small independent BVs
/// (representing per-register / per-memory-byte symbols) plus a handful
/// of wider constraints touching subsets of them. Measures the per-AST
/// translate cost when there's no deep tree sharing.
#[test]
fn translate_cost_state_export_shape() {
    const NUM_LEAVES: usize = 256;
    const NUM_CONSTRAINTS: usize = 32;

    let cfg = Config::new();
    let ctx_a = Context::new(&cfg);
    let ctx_b = Context::new(&cfg);

    Context::set_thread_local(&ctx_a);
    let leaves: Vec<BV> = (0..NUM_LEAVES)
        .map(|i| BV::new_const(format!("v{i}"), 32))
        .collect();

    // Build constraints touching ~8 leaves each.
    let mut constraints = Vec::with_capacity(NUM_CONSTRAINTS);
    for c in 0..NUM_CONSTRAINTS {
        let base = c * 8 % NUM_LEAVES;
        let mut acc = leaves[base].clone();
        for j in 1..8 {
            let idx = (base + j) % NUM_LEAVES;
            acc = acc.bvxor(&leaves[idx]);
        }
        let rhs = BV::from_u64((c as u64).wrapping_mul(0x1337), 32);
        constraints.push(acc.eq(&rhs));
    }

    let start = Instant::now();
    let translated_leaves: Vec<BV> = leaves.iter().map(|bv| bv.translate(&ctx_b)).collect();
    let translated_constraints = constraints.translate(&ctx_b);
    let elapsed = start.elapsed();

    // Approximate node count:
    //   leaves: NUM_LEAVES
    //   per-constraint: 7 bvxor + 1 from_u64 const + 1 _eq = 9
    let node_count: u64 = NUM_LEAVES as u64 + (9 * NUM_CONSTRAINTS) as u64;
    let ns_per_node = elapsed.as_nanos() / node_count as u128;

    eprintln!(
        "Z3_translate state-shape: {} leaves + {} constraints (~{} nodes), total {:?}, {} ns/node",
        NUM_LEAVES, NUM_CONSTRAINTS, node_count, elapsed, ns_per_node
    );

    // sanity: translated objects must equal original count
    assert_eq!(translated_leaves.len(), NUM_LEAVES);
    assert_eq!(translated_constraints.len(), NUM_CONSTRAINTS);

    // Confirm translated constraints satisfiable as a sanity proxy: assert
    // them all into ctx_b's solver and expect SAT.
    Context::set_thread_local(&ctx_b);
    let solver = Solver::new();
    for c in &translated_constraints {
        solver.assert(c);
    }
    assert_eq!(
        solver.check(),
        SatResult::Sat,
        "translated state-shape constraints must remain SAT"
    );
}
