//! Benchmarks for the VEX execution engine.
//!
//! Run with: cargo bench --features vex-engine

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

#[cfg(feature = "vex-engine")]
mod benchmarks {
    use super::*;
    use rustylib::memory::{SymbolicMemory, Permission};
    use rustylib::symbolic::{RustBV, SymContext};
    use rustylib::arch::{RegisterFile, AMD64};
    use rustylib::vex::{VexArch, Endness, IRSB};
    use rustylib::vex::ir::{IRStmt, JumpKind};
    use rustylib::interpreter::VEXInterpreter;

    /// Benchmark memory forking (Copy-on-Write).
    pub fn bench_memory_fork(c: &mut Criterion) {
        let ctx = SymContext::new_mock();
        let mut memory = SymbolicMemory::new(Endness::Little);

        // Set up memory with some data
        memory.map(0x1000, 0x10000, Permission::RWX);
        for i in 0..256u64 {
            let val = RustBV::concrete(i as u128, 64);
            memory.store_concrete(0x1000 + i * 8, val).ok();
        }

        c.bench_function("memory_fork", |b| {
            b.iter(|| {
                let forked = memory.fork();
                black_box(forked)
            })
        });
    }

    /// Benchmark memory fork + modify (typical symbolic execution pattern).
    pub fn bench_memory_fork_modify(c: &mut Criterion) {
        let ctx = SymContext::new_mock();
        let mut memory = SymbolicMemory::new(Endness::Little);

        memory.map(0x1000, 0x10000, Permission::RWX);
        for i in 0..256u64 {
            let val = RustBV::concrete(i as u128, 64);
            memory.store_concrete(0x1000 + i * 8, val).ok();
        }

        c.bench_function("memory_fork_modify", |b| {
            b.iter(|| {
                let mut forked = memory.fork();
                let val = RustBV::concrete(0xDEADBEEF, 32);
                forked.store_concrete(0x1000, val).ok();
                black_box(forked)
            })
        });
    }

    /// Benchmark concrete memory reads.
    pub fn bench_memory_read(c: &mut Criterion) {
        let ctx = SymContext::new_mock();
        let mut memory = SymbolicMemory::new(Endness::Little);

        memory.map(0x1000, 0x10000, Permission::RWX);
        let init_val = RustBV::concrete(0xAAAAAAAAAAAAAAAAu64 as u128, 64);
        memory.store_concrete(0x1000, init_val).ok();

        c.bench_function("memory_read_8bytes", |b| {
            b.iter(|| {
                let data = memory.load_concrete(black_box(0x1000), 8, &ctx);
                black_box(data)
            })
        });
    }

    /// Benchmark concrete memory writes.
    pub fn bench_memory_write(c: &mut Criterion) {
        let ctx = SymContext::new_mock();
        let mut memory = SymbolicMemory::new(Endness::Little);

        memory.map(0x1000, 0x10000, Permission::RWX);

        c.bench_function("memory_write_8bytes", |b| {
            b.iter(|| {
                let data = RustBV::concrete(0xDEADBEEFCAFEBABEu64 as u128, 64);
                memory.store_concrete(black_box(0x1000), data).ok();
            })
        });
    }

    /// Benchmark register file operations.
    pub fn bench_register_access(c: &mut Criterion) {
        let ctx = SymContext::new_mock();
        let mut regs = RegisterFile::new(Box::new(AMD64));

        c.bench_function("register_write_read", |b| {
            b.iter(|| {
                let val = RustBV::concrete(0x12345678DEADBEEF, 64);
                regs.put_reg("rax", val);
                let result = regs.get_reg("rax", &ctx);
                black_box(result)
            })
        });
    }

    /// Benchmark RustBV concrete operations.
    pub fn bench_bv_concrete_ops(c: &mut Criterion) {
        let ctx = SymContext::new_mock();

        c.bench_function("bv_add_concrete", |b| {
            b.iter(|| {
                let a = RustBV::concrete(black_box(12345), 64);
                let b = RustBV::concrete(black_box(67890), 64);
                let result = a.add(&b, &ctx);
                black_box(result)
            })
        });

        c.bench_function("bv_mul_concrete", |b| {
            b.iter(|| {
                let a = RustBV::concrete(black_box(12345), 64);
                let b = RustBV::concrete(black_box(67890), 64);
                let result = a.mul(&b, &ctx);
                black_box(result)
            })
        });

        c.bench_function("bv_and_concrete", |b| {
            b.iter(|| {
                let a = RustBV::concrete(black_box(0xFF00FF00FF00FF00), 64);
                let b = RustBV::concrete(black_box(0x00FF00FF00FF00FF), 64);
                let result = a.and(&b, &ctx);
                black_box(result)
            })
        });
    }

    /// Benchmark VEX interpreter creation.
    pub fn bench_interpreter_create(c: &mut Criterion) {
        let ctx = SymContext::new_mock();

        c.bench_function("interpreter_create", |b| {
            b.iter(|| {
                let interp = VEXInterpreter::new(VexArch::AMD64, &ctx);
                black_box(interp)
            })
        });
    }

    /// Benchmark simple IRSB execution.
    pub fn bench_irsb_execute(c: &mut Criterion) {
        let ctx = SymContext::new_mock();

        // Create a simple IRSB: just an IMark
        let mut irsb = IRSB::new(0x401000, VexArch::AMD64);
        irsb.statements.push(IRStmt::IMark {
            addr: 0x401000,
            len: 4,
            delta: 0,
        });
        irsb.jumpkind = JumpKind::Boring;

        c.bench_function("irsb_execute_simple", |b| {
            b.iter(|| {
                let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);
                let result = interp.execute_block(&irsb);
                black_box(result)
            })
        });
    }

    /// Benchmark creating many forks (path explosion scenario).
    pub fn bench_many_forks(c: &mut Criterion) {
        let ctx = SymContext::new_mock();
        let mut memory = SymbolicMemory::new(Endness::Little);

        memory.map(0x1000, 0x10000, Permission::RWX);
        for i in 0..256u64 {
            let val = RustBV::concrete(i as u128, 64);
            memory.store_concrete(0x1000 + i * 8, val).ok();
        }

        let mut group = c.benchmark_group("many_forks");

        for num_forks in [10, 50, 100, 500].iter() {
            group.bench_with_input(BenchmarkId::from_parameter(num_forks), num_forks, |b, &n| {
                b.iter(|| {
                    let mut forks = Vec::with_capacity(n);
                    let mut current = memory.clone();
                    for i in 0..n {
                        current = current.fork();
                        let val = RustBV::concrete(i as u128, 64);
                        current.store_concrete(0x1000, val).ok();
                        forks.push(current.clone());
                    }
                    black_box(forks)
                })
            });
        }
        group.finish();
    }
}

#[cfg(feature = "vex-engine")]
criterion_group!(
    benches,
    benchmarks::bench_memory_fork,
    benchmarks::bench_memory_fork_modify,
    benchmarks::bench_memory_read,
    benchmarks::bench_memory_write,
    benchmarks::bench_register_access,
    benchmarks::bench_bv_concrete_ops,
    benchmarks::bench_interpreter_create,
    benchmarks::bench_irsb_execute,
    benchmarks::bench_many_forks,
);

#[cfg(feature = "vex-engine")]
criterion_main!(benches);

#[cfg(not(feature = "vex-engine"))]
fn main() {
    eprintln!("VEX engine feature not enabled. Run with --features vex-engine");
}
