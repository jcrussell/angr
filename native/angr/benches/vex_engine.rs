//! Criterion benchmarks for Rust symbolic execution engine hot paths.
//!
//! Run with: cargo bench --manifest-path native/angr/Cargo.toml

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rustylib::concretize::AddressConcretizer;
use rustylib::memory::{Permission, SymbolicMemory};
use rustylib::symbolic::{RustBV, SymContext};
use rustylib::vex::ir::Endness;

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
);
criterion_main!(benches);
