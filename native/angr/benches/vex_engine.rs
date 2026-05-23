//! Criterion benchmarks for Rust symbolic execution engine hot paths.
//!
//! Run with: cargo bench --manifest-path native/angr/Cargo.toml

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rustylib::concretize::AddressConcretizer;
use rustylib::memory::{Permission, SymbolicMemory};
use rustylib::symbolic::lineage::{ScopeFrame, ScopePath, SharedLineageSolver};
use rustylib::symbolic::lineage_assumptions::{
    AssumptionFrame, AssumptionPath, SharedLineageSolverAssumptions,
};
use rustylib::symbolic::{RustBV, SymContext};
use rustylib::vex::ir::Endness;
use rustylib::vex::{IROp, IRType, VEXOps};
use z3::ast::BV;

// ---------------------------------------------------------------------------
// RustBV operations
// ---------------------------------------------------------------------------

fn bench_rustbv_concrete_arithmetic(c: &mut Criterion) {
    let ctx = SymContext::new();
    let a = RustBV::concrete(0xDEADBEEF, 64);
    let b = RustBV::concrete(0xCAFEBABE, 64);

    let mut group = c.benchmark_group("rustbv_concrete");
    group.bench_function("add", |bench| bench.iter(|| black_box(a.add(&b, &ctx))));
    group.bench_function("sub", |bench| bench.iter(|| black_box(a.sub(&b, &ctx))));
    group.bench_function("concat", |bench| {
        bench.iter(|| black_box(a.concat(&b, &ctx)))
    });
    group.bench_function("extract_32", |bench| {
        bench.iter(|| black_box(a.extract(31, 0, &ctx)))
    });
    group.bench_function("reverse", |bench| bench.iter(|| black_box(a.reverse(&ctx))));
    group.finish();
}

fn bench_rustbv_symbolic_arithmetic(c: &mut Criterion) {
    let ctx = SymContext::new();
    let a = RustBV::symbolic(&ctx, "x", 64);
    let b = RustBV::symbolic(&ctx, "y", 64);

    let mut group = c.benchmark_group("rustbv_symbolic");
    group.bench_function("add", |bench| bench.iter(|| black_box(a.add(&b, &ctx))));
    group.bench_function("concat", |bench| {
        bench.iter(|| black_box(a.concat(&b, &ctx)))
    });
    group.bench_function("extract_32", |bench| {
        bench.iter(|| black_box(a.extract(31, 0, &ctx)))
    });
    group.bench_function("reverse", |bench| bench.iter(|| black_box(a.reverse(&ctx))));
    group.finish();
}

fn bench_rustbv_build_z3_ast(c: &mut Criterion) {
    let ctx = SymContext::new();
    // Build a moderately complex expression tree:
    // reverse(concat(extract(x, 31, 0), y) + z)
    let x = RustBV::symbolic(&ctx, "x", 64);
    let y = RustBV::symbolic(&ctx, "y", 32);
    let z = RustBV::symbolic(&ctx, "z", 64);
    let extracted = x.extract(31, 0, &ctx);
    let concatenated = extracted.concat(&y, &ctx);
    let sum = concatenated.add(&z, &ctx);
    let expr = sum.reverse(&ctx);

    c.bench_function("rustbv_z3", |bench| {
        bench.iter(|| {
            let mut cache = std::collections::HashMap::new();
            black_box(expr.to_z3_ast_cached(&mut cache))
        })
    });
}

// ---------------------------------------------------------------------------
// SymContext solver operations
// ---------------------------------------------------------------------------

fn bench_symcontext_fork(c: &mut Criterion) {
    let ctx = SymContext::new();
    // Add some constraints to make fork realistic
    let x = RustBV::symbolic(&ctx, "x", 32);
    let ten = RustBV::concrete(10, 32);
    let hundred = RustBV::concrete(100, 32);
    let gt = x.ult(&ten, &ctx); // x < 10 (we'll negate via assume_false)
    let lt = x.ult(&hundred, &ctx); // x < 100
    ctx.assume_false(&gt); // x >= 10
    ctx.assume_true(&lt); // x < 100

    let mut group = c.benchmark_group("symcontext_fork");
    group.bench_function("2_constraints", |bench| {
        bench.iter(|| black_box(ctx.fork()))
    });
    group.finish();
}

fn bench_symcontext_fork_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("symcontext_fork_scaling");

    for n_constraints in [5, 20, 50] {
        let ctx = SymContext::new();
        let x = RustBV::symbolic(&ctx, "x", 32);
        for i in 0..n_constraints {
            let bound = RustBV::concrete(i as u128 + 1000, 32);
            let cmp = x.ult(&bound, &ctx);
            ctx.assume_false(&cmp); // x >= bound
        }
        group.bench_function(format!("{n_constraints}_constraints"), |bench| {
            bench.iter(|| black_box(ctx.fork()))
        });
    }
    group.finish();
}

fn bench_symcontext_check_branch(c: &mut Criterion) {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let bound = RustBV::concrete(50, 32);
    let cmp = x.ult(&bound, &ctx);
    // Constrain x to [10, 100) so both branches are feasible
    let lo = RustBV::concrete(10, 32);
    let hi = RustBV::concrete(100, 32);
    ctx.assume_false(&x.ult(&lo, &ctx));
    ctx.assume_true(&x.ult(&hi, &ctx));

    c.bench_function("symcontext_check_branch", |bench| {
        bench.iter(|| {
            // Use push/pop so state doesn't accumulate
            ctx.push();
            let result = black_box(ctx.check_branch_feasibility(&cmp));
            ctx.pop();
            result
        })
    });
}

fn bench_symcontext_assume_true(c: &mut Criterion) {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let bound = RustBV::concrete(42, 32);
    let cmp = x.ult(&bound, &ctx);

    c.bench_function("symcontext_assume", |bench| {
        bench.iter(|| {
            ctx.push();
            ctx.assume_true(black_box(&cmp));
            ctx.pop();
        })
    });
}

fn bench_symcontext_push_pop(c: &mut Criterion) {
    let ctx = SymContext::new();
    c.bench_function("symcontext_push_pop", |bench| {
        bench.iter(|| {
            ctx.push();
            ctx.pop();
        })
    });
}

// ---------------------------------------------------------------------------
// SymbolicMemory operations
// ---------------------------------------------------------------------------

fn bench_memory_concrete(c: &mut Criterion) {
    let ctx = SymContext::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    let base = 0x400000u64;
    mem.map(
        base,
        0x1000,
        Permission {
            read: true,
            write: true,
            execute: false,
        },
    );

    // Pre-store some data
    let val = RustBV::concrete(0xDEADBEEF_CAFEBABE, 64);
    mem.store_concrete(base, val).unwrap();

    let mut group = c.benchmark_group("memory_concrete");
    group.bench_function("load_8bytes", |bench| {
        bench.iter(|| black_box(mem.load_concrete(base, 8, &ctx).unwrap()))
    });
    group.bench_function("store_8bytes", |bench| {
        let v = RustBV::concrete(0x1234567890ABCDEF, 64);
        bench.iter(|| {
            mem.store_concrete(base + 0x100, black_box(v.clone()))
                .unwrap();
        })
    });
    group.bench_function("load_1byte", |bench| {
        bench.iter(|| black_box(mem.load_concrete(base, 1, &ctx).unwrap()))
    });
    group.finish();
}

fn bench_memory_symbolic_load(c: &mut Criterion) {
    let ctx = SymContext::new();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    let base = 0x400000u64;
    mem.map(
        base,
        0x1000,
        Permission {
            read: true,
            write: true,
            execute: false,
        },
    );

    // Fill with known data
    for i in 0..256u64 {
        let v = RustBV::concrete(i as u128, 8);
        mem.store_concrete(base + i, v).unwrap();
    }

    // Create a symbolic address constrained to a small range
    let addr_sym = RustBV::symbolic(&ctx, "addr", 64);
    let lo = RustBV::concrete(base as u128, 64);
    let hi = RustBV::concrete((base + 16) as u128, 64);
    // addr >= base
    ctx.assume_false(&addr_sym.ult(&lo, &ctx));
    // addr < base + 16
    ctx.assume_true(&addr_sym.ult(&hi, &ctx));

    c.bench_function("memory_symbolic_load", |bench| {
        bench.iter(|| {
            ctx.push();
            let result =
                mem.load_symbolic_unified(black_box(addr_sym.clone()), 1, &ctx, &concretizer);
            ctx.pop();
            black_box(result)
        })
    });
}

fn bench_memory_fork(c: &mut Criterion) {
    let mut mem = SymbolicMemory::new(Endness::Little);
    let base = 0x400000u64;
    mem.map(
        base,
        0x10000,
        Permission {
            read: true,
            write: true,
            execute: false,
        },
    );

    // Write to several pages to make fork non-trivial
    for page in 0..16u64 {
        let v = RustBV::concrete(page as u128, 64);
        mem.store_concrete(base + page * 0x1000, v).unwrap();
    }

    c.bench_function("memory_fork", |bench| bench.iter(|| black_box(mem.fork())));
}

// ---------------------------------------------------------------------------
// State fork
// ---------------------------------------------------------------------------

fn bench_state_fork(c: &mut Criterion) {
    let mut state = rustylib::state::RustSimState::new("AMD64").unwrap();

    // Set up some registers and memory like a real state
    state.set_register("rax", RustBV::concrete(0x1234, 64));
    state.set_register("rbx", RustBV::concrete(0x5678, 64));
    state.set_register("rsp", RustBV::concrete(0x7FFF_FFFF_0000, 64));
    state.set_pc(0x401000);

    c.bench_function("state_fork", |bench| bench.iter(|| black_box(state.fork())));
}

// ---------------------------------------------------------------------------
// NEON SIMD ops (Mul8x16 / VGetElem / VSetElem — landed in angr-bkcs.2)
// ---------------------------------------------------------------------------

/// Microbenches for the NEON ops implemented in commit da0966893
/// (Iop_Mul8x{8,16}, Iop_GetElem{N}x{M}, Iop_SetElem{N}x{M}).
///
/// Each op gets two variants:
///   * `*_concrete` — both operands concrete, exercising the bit-twiddle
///     fast path.
///   * `*_symbolic` — one or both operands symbolic, exercising the
///     Z3 AST / ITE-chain path.
///
/// Mul8x16 is also paired with a scalar `Mul64` baseline so the
/// per-lane multiply cost is comparable to plain 64-bit arithmetic.
fn bench_rustbv_neon_ops(c: &mut Criterion) {
    let ctx = SymContext::new();
    let mut group = c.benchmark_group("rustbv_neon_ops");

    // ---- Iop_Mul8x16: 16 lanes of 8-bit multiply over a V128.
    let mul_lo64 = 0x0303_0303_0303_0303u128;
    let mul_v128_l = RustBV::concrete(mul_lo64 | (mul_lo64 << 64), 128);
    let mul_v128_r_lo = 0x0505_0505_0505_0505u128;
    let mul_v128_r = RustBV::concrete(mul_v128_r_lo | (mul_v128_r_lo << 64), 128);
    group.bench_function("mul8x16_concrete", |bench| {
        bench.iter(|| {
            black_box(
                VEXOps::binop(
                    IROp::VMul {
                        elem: IRType::I8,
                        count: 16,
                    },
                    mul_v128_l.clone(),
                    mul_v128_r.clone(),
                    &ctx,
                )
                .unwrap(),
            )
        })
    });

    let mul_v128_sym_l = RustBV::symbolic(&ctx, "neon_mul_l", 128);
    let mul_v128_sym_r = RustBV::symbolic(&ctx, "neon_mul_r", 128);
    group.bench_function("mul8x16_symbolic", |bench| {
        bench.iter(|| {
            black_box(
                VEXOps::binop(
                    IROp::VMul {
                        elem: IRType::I8,
                        count: 16,
                    },
                    mul_v128_sym_l.clone(),
                    mul_v128_sym_r.clone(),
                    &ctx,
                )
                .unwrap(),
            )
        })
    });

    // Scalar Mul64 baseline so the per-bench cost can be compared against
    // plain 64-bit multiply.
    let scalar_a = RustBV::concrete(0xDEAD_BEEF, 64);
    let scalar_b = RustBV::concrete(0xCAFE_BABE, 64);
    group.bench_function("mul64_concrete_baseline", |bench| {
        bench.iter(|| {
            black_box(
                VEXOps::binop(
                    IROp::Mul(IRType::I64),
                    scalar_a.clone(),
                    scalar_b.clone(),
                    &ctx,
                )
                .unwrap(),
            )
        })
    });

    // ---- Iop_GetElem8x16 (V128 → byte lane): concrete idx hits the
    // bit-slice fast path; symbolic idx walks the ITE chain over 16 lanes.
    let lane_vec = RustBV::concrete(0xFEDC_BA98_7654_3210u128 | (0x0011_2233_4455_6677u128 << 64), 128);
    let lane_idx_concrete = RustBV::concrete(7, 8);
    group.bench_function("get_elem8x16_concrete_idx", |bench| {
        bench.iter(|| {
            black_box(
                VEXOps::binop(
                    IROp::VGetElem {
                        elem: IRType::I8,
                        count: 16,
                    },
                    lane_vec.clone(),
                    lane_idx_concrete.clone(),
                    &ctx,
                )
                .unwrap(),
            )
        })
    });

    let lane_idx_sym = RustBV::symbolic(&ctx, "neon_get_idx", 8);
    group.bench_function("get_elem8x16_symbolic_idx", |bench| {
        bench.iter(|| {
            black_box(
                VEXOps::binop(
                    IROp::VGetElem {
                        elem: IRType::I8,
                        count: 16,
                    },
                    lane_vec.clone(),
                    lane_idx_sym.clone(),
                    &ctx,
                )
                .unwrap(),
            )
        })
    });

    // ---- Iop_SetElem8x16: triop, dispatched through binop_with_rm by
    // reinterpreting (rm, left, right) as (vec, idx, val).
    let set_val = RustBV::concrete(0xAB, 8);
    group.bench_function("set_elem8x16_concrete_idx", |bench| {
        bench.iter(|| {
            black_box(
                VEXOps::binop_with_rm(
                    IROp::VSetElem {
                        elem: IRType::I8,
                        count: 16,
                    },
                    lane_vec.clone(),
                    lane_idx_concrete.clone(),
                    set_val.clone(),
                    &ctx,
                )
                .unwrap(),
            )
        })
    });

    group.bench_function("set_elem8x16_symbolic_idx", |bench| {
        bench.iter(|| {
            black_box(
                VEXOps::binop_with_rm(
                    IROp::VSetElem {
                        elem: IRType::I8,
                        count: 16,
                    },
                    lane_vec.clone(),
                    lane_idx_sym.clone(),
                    set_val.clone(),
                    &ctx,
                )
                .unwrap(),
            )
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Lineage variant head-to-head: push/pop vs check-with-assumptions
// (angr-3ms1 alternative-d spike, 2026-05-23)
// ---------------------------------------------------------------------------

/// Build N synthetic per-state paths sharing a 2-frame ancestor prefix
/// (`x > 0`, `x < BIG`) and diverging at the leaf with a unique
/// `x == k_i` constraint. Returns the parallel push/pop `ScopePath` list
/// and assumption `AssumptionPath` list. Frame ids match up across the
/// two lists for the shared-ancestor frames (not the leaves — separate
/// FrameId mints) so each variant builds its own shared-prefix shape.
///
/// Choosing `x == k_i` per leaf means each per-state path is satisfiable
/// in isolation but pairwise-UNSAT — realistic for symbolic-execution
/// path constraints.
fn build_workload(
    n_states: usize,
    k_per_state: usize,
) -> (Vec<ScopePath>, Vec<AssumptionPath>) {
    let x = BV::new_const("lineage_bench_x", 32);
    let zero = BV::from_u64(0, 32);
    let big = BV::from_u64(1_000_000_000, 32);

    // Push/pop variant: one shared ancestor prefix for all states.
    let pp_ancestor: Vec<ScopeFrame> = vec![
        ScopeFrame::new(true, x.bvugt(&zero)),
        ScopeFrame::new(true, x.bvult(&big)),
    ];
    // Assumption variant: separate ancestor frames (different FrameIds
    // and different tag Bools — they describe the same constraints but
    // are independent of the push/pop ancestor frames).
    let asm_ancestor: Vec<AssumptionFrame> = vec![
        AssumptionFrame::new(x.bvugt(&zero)),
        AssumptionFrame::new(x.bvult(&big)),
    ];

    let mut pp_paths = Vec::with_capacity(n_states);
    let mut asm_paths = Vec::with_capacity(n_states);
    for i in 0..n_states {
        let mut pp = pp_ancestor.clone();
        let mut asm = asm_ancestor.clone();
        for j in 0..k_per_state {
            // Different shape per (state, frame): x + i*100 + j*7 != 0
            // (always SAT in isolation, gives each frame a distinct AST
            // so Z3 doesn't fold them into a single hash-consed assertion).
            let offset = ((i * 100) + (j * 7)) as u64;
            let term = x.bvadd(&BV::from_u64(offset, 32));
            let cstr = term.eq(&BV::from_u64(0, 32)).not();
            pp.push(ScopeFrame::new(true, cstr.clone()));
            asm.push(AssumptionFrame::new(cstr));
        }
        pp_paths.push(pp);
        asm_paths.push(asm);
    }
    (pp_paths, asm_paths)
}

/// Deterministic pseudo-random state ordering — interleaves between
/// states so the workload mimics BFS-style cross-state thrash (vs.
/// per-state batching where all queries for state i happen together).
/// Uses xorshift32 on a fixed seed so each run is identical.
fn bfs_query_order(n_states: usize, n_queries: usize) -> Vec<usize> {
    let mut seed: u32 = 0xCAFE_BABE;
    let mut out = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        out.push((seed as usize) % n_states);
    }
    out
}

/// Build a linear descendant-chain workload (angr-3ms1 LIFO hypothesis,
/// 2026-05-23). `paths[i]` is `ancestor + [chain_frames[0..=i]]` so each
/// successive state is a direct descendant of the previous (one new frame
/// appended at the bottom).
///
/// Visiting `paths[0..n]` in order models the steady-state pop_back stream
/// produced by `use_lifo=True`: every transition is "descend by one frame",
/// which makes `SharedLineageSolver::switch_to` do exactly 0 pops and 1
/// push per query (best case for the lineage push/pop architecture).
fn build_lifo_chain_workload(n_chain: usize) -> Vec<ScopePath> {
    let x = BV::new_const("lineage_bench_chain_x", 32);
    let zero = BV::from_u64(0, 32);
    let big = BV::from_u64(1_000_000_000, 32);

    let ancestor: Vec<ScopeFrame> = vec![
        ScopeFrame::new(true, x.bvugt(&zero)),
        ScopeFrame::new(true, x.bvult(&big)),
    ];

    // Mint one fresh frame per chain link. Clone semantics preserve the
    // FrameId, so every state[j] (j >= i) shares the exact same frame at
    // depth `i + ancestor.len()`.
    let chain_frames: Vec<ScopeFrame> = (0..n_chain)
        .map(|i| {
            let offset = ((i * 7) + 13) as u64;
            let term = x.bvadd(&BV::from_u64(offset, 32));
            let cstr = term.eq(&BV::from_u64(0, 32)).not();
            ScopeFrame::new(true, cstr)
        })
        .collect();

    let mut paths = Vec::with_capacity(n_chain);
    for i in 0..n_chain {
        let mut path = ancestor.clone();
        path.extend(chain_frames[..=i].iter().cloned());
        paths.push(path);
    }
    paths
}

/// Build a depth-`depth` binary-tree DFS-preorder workload (angr-3ms1
/// LIFO hypothesis, 2026-05-23). Each leaf's path is `ancestor +
/// [side_frame_for_each_level]`; sibling leaves share `FrameId`s for the
/// shared-prefix levels. Visiting `paths[0..2^depth]` in numeric order
/// is DFS-preorder of a full binary tree (the rightmost spine is visited
/// first, then we backtrack to a sibling and descend its rightmost spine).
///
/// Models the realistic pop_back stream when `use_lifo=True` and the
/// explorer fans out into 2^depth descendants of a common ancestor: most
/// transitions are descend-by-1 (the last frame flips L→R), but
/// occasionally we backtrack `k` levels (when crossing a higher-level
/// branch), so the amortized transition is O(2) pops + O(2) pushes per
/// query.
fn build_lifo_dfs_tree_workload(depth: usize) -> Vec<ScopePath> {
    let x = BV::new_const("lineage_bench_dfs_x", 32);
    let zero = BV::from_u64(0, 32);
    let big = BV::from_u64(1_000_000_000, 32);

    let ancestor: Vec<ScopeFrame> = vec![
        ScopeFrame::new(true, x.bvugt(&zero)),
        ScopeFrame::new(true, x.bvult(&big)),
    ];

    // Mint exactly two frames per level: left-side and right-side. Cloning
    // into multiple leaf paths preserves the FrameId, so the lineage
    // common-prefix walk sees siblings as sharing the (level, side) prefix.
    let level_frames: Vec<(ScopeFrame, ScopeFrame)> = (0..depth)
        .map(|lvl| {
            let l_off = ((lvl * 11) + 1) as u64;
            let r_off = ((lvl * 11) + 503) as u64;
            let lf = ScopeFrame::new(
                true,
                x.bvadd(&BV::from_u64(l_off, 32))
                    .eq(&BV::from_u64(0, 32))
                    .not(),
            );
            let rf = ScopeFrame::new(
                true,
                x.bvadd(&BV::from_u64(r_off, 32))
                    .eq(&BV::from_u64(0, 32))
                    .not(),
            );
            (lf, rf)
        })
        .collect();

    let n_leaves = 1usize << depth;
    let mut paths = Vec::with_capacity(n_leaves);
    for leaf in 0..n_leaves {
        let mut path = ancestor.clone();
        for lvl in 0..depth {
            let side = (leaf >> (depth - 1 - lvl)) & 1;
            let frame = if side == 0 {
                &level_frames[lvl].0
            } else {
                &level_frames[lvl].1
            };
            path.push(frame.clone());
        }
        paths.push(path);
    }
    paths
}

fn bench_lineage_push_pop_vs_assumptions(c: &mut Criterion) {
    let n_states = 50;
    let k_per_state = 10;
    let n_queries = 200;

    let (pp_paths, asm_paths) = build_workload(n_states, k_per_state);
    let order = bfs_query_order(n_states, n_queries);

    let mut group = c.benchmark_group("lineage_variants");
    group.sample_size(10); // each iter does n_queries Z3 checks — expensive

    group.bench_function("push_pop_bfs_thrash", |bench| {
        bench.iter(|| {
            let solver = z3::Solver::new();
            let mut lin = SharedLineageSolver::new(solver);
            for &idx in &order {
                let path = &pp_paths[idx];
                lin.with_solver(path, |s| black_box(s.check()));
            }
        })
    });

    group.bench_function("assumptions_bfs_thrash", |bench| {
        bench.iter(|| {
            let solver = z3::Solver::new();
            let mut lin = SharedLineageSolverAssumptions::new(solver);
            for &idx in &order {
                let path = &asm_paths[idx];
                lin.with_solver(path, |s, tags| black_box(s.check_assumptions(tags)));
            }
        })
    });

    // Best-case for push/pop: per-state batching (no cross-state thrash).
    // Same workload — every state runs all its queries consecutively
    // before switching. The hot-cache fast path should fire on every
    // query after the first per state.
    let batched_order: Vec<usize> = (0..n_states)
        .flat_map(|i| std::iter::repeat(i).take(n_queries / n_states))
        .collect();

    group.bench_function("push_pop_per_state_batched", |bench| {
        bench.iter(|| {
            let solver = z3::Solver::new();
            let mut lin = SharedLineageSolver::new(solver);
            for &idx in &batched_order {
                let path = &pp_paths[idx];
                lin.with_solver(path, |s| black_box(s.check()));
            }
        })
    });

    group.bench_function("assumptions_per_state_batched", |bench| {
        bench.iter(|| {
            let solver = z3::Solver::new();
            let mut lin = SharedLineageSolverAssumptions::new(solver);
            for &idx in &batched_order {
                let path = &asm_paths[idx];
                lin.with_solver(path, |s, tags| black_box(s.check_assumptions(tags)));
            }
        })
    });

    // Per-state-solvers baseline: each state has its own private Z3
    // solver populated with the path constraints. This is the "no
    // lineage at all" reference — the production path angr is on today
    // (modulo lazy materialization). Any lineage variant that doesn't
    // beat this on the BFS-thrash workload is a regression by
    // construction.
    group.bench_function("per_state_solvers_bfs_thrash", |bench| {
        bench.iter(|| {
            // Build a fresh per-state solver array each iter so the
            // setup cost is comparable across variants (lineage variants
            // also build a fresh solver per iter).
            let solvers: Vec<z3::Solver> = pp_paths
                .iter()
                .map(|path| {
                    let s = z3::Solver::new();
                    for f in path {
                        s.assert(&f.z3_assertion);
                    }
                    s
                })
                .collect();
            for &idx in &order {
                black_box(solvers[idx].check());
            }
        })
    });

    group.bench_function("per_state_solvers_per_state_batched", |bench| {
        bench.iter(|| {
            let solvers: Vec<z3::Solver> = pp_paths
                .iter()
                .map(|path| {
                    let s = z3::Solver::new();
                    for f in path {
                        s.assert(&f.z3_assertion);
                    }
                    s
                })
                .collect();
            for &idx in &batched_order {
                black_box(solvers[idx].check());
            }
        })
    });

    // angr-3ms1 LIFO hypothesis (2026-05-23): if `use_lifo=True` in the
    // explorer, pop_back consistently returns a direct descendant of the
    // last-stepped state. This variant simulates that stream — a 200-deep
    // linear descendant chain — and is the best-case workload for the
    // lineage push/pop architecture. The number of states equals
    // n_queries so each chain link is visited exactly once.
    let chain_paths = build_lifo_chain_workload(n_queries);

    group.bench_function("push_pop_lifo_chain", |bench| {
        bench.iter(|| {
            let solver = z3::Solver::new();
            let mut lin = SharedLineageSolver::new(solver);
            for path in &chain_paths {
                lin.with_solver(path, |s| black_box(s.check()));
            }
        })
    });

    // Per-state-solvers baseline for the chain workload — every state
    // gets a fresh Z3 solver populated with its full path. This is the
    // no-lineage reference for the LIFO variant, same role as
    // `per_state_solvers_bfs_thrash` plays for the BFS variant. Any
    // lineage variant that doesn't beat this on the chain workload is
    // a regression.
    group.bench_function("per_state_solvers_lifo_chain", |bench| {
        bench.iter(|| {
            let solvers: Vec<z3::Solver> = chain_paths
                .iter()
                .map(|path| {
                    let s = z3::Solver::new();
                    for f in path {
                        s.assert(&f.z3_assertion);
                    }
                    s
                })
                .collect();
            for idx in 0..chain_paths.len() {
                black_box(solvers[idx].check());
            }
        })
    });

    // angr-3ms1 LIFO hypothesis, realistic variant (2026-05-23): DFS
    // preorder over a depth-8 full binary tree (256 leaves visited in
    // numeric order). Most transitions are descend-by-1 (last frame
    // flips L→R, switching to a sibling leaf), but every 2^k-th
    // transition backtracks k levels and re-descends. Amortized cost
    // per transition is O(2) pops + O(2) pushes (~1 in 2 transitions
    // walks deeper into the tree). 256 ≥ 200 so this exceeds n_queries
    // by design — to keep per-iter cost in the same ballpark as the
    // other variants, only the first 200 leaves are visited.
    let dfs_paths = build_lifo_dfs_tree_workload(8);
    let dfs_order_n: usize = 200;

    group.bench_function("push_pop_lifo_dfs_tree", |bench| {
        bench.iter(|| {
            let solver = z3::Solver::new();
            let mut lin = SharedLineageSolver::new(solver);
            for idx in 0..dfs_order_n {
                let path = &dfs_paths[idx];
                lin.with_solver(path, |s| black_box(s.check()));
            }
        })
    });

    group.bench_function("per_state_solvers_lifo_dfs_tree", |bench| {
        bench.iter(|| {
            let solvers: Vec<z3::Solver> = dfs_paths[..dfs_order_n]
                .iter()
                .map(|path| {
                    let s = z3::Solver::new();
                    for f in path {
                        s.assert(&f.z3_assertion);
                    }
                    s
                })
                .collect();
            for idx in 0..dfs_order_n {
                black_box(solvers[idx].check());
            }
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_rustbv_concrete_arithmetic,
    bench_rustbv_symbolic_arithmetic,
    bench_rustbv_build_z3_ast,
    bench_symcontext_fork,
    bench_symcontext_fork_scaling,
    bench_symcontext_check_branch,
    bench_symcontext_assume_true,
    bench_symcontext_push_pop,
    bench_memory_concrete,
    bench_memory_symbolic_load,
    bench_memory_fork,
    bench_state_fork,
    bench_rustbv_neon_ops,
    bench_lineage_push_pop_vs_assumptions,
);
criterion_main!(benches);
