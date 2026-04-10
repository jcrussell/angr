//! VEX engine benchmarks (placeholder).
//! Benchmarks will be added as engine components are ported.

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("noop", |b| b.iter(|| {}));
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
