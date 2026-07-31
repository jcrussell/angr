//! Microbenchmark for IRSB JSON deserialization (spike angr-3ijo).
//!
//! Captures sample IRSB JSON via `CAPTURE_IRSB=path` env on a real benchmark,
//! then runs many iterations of `deserialize_irsb` to time JSON parse + convert.
//!
//! Build: cargo build --manifest-path native/angr/Cargo.toml --release --example bench_deserialize
//! Run:   ./target/release/examples/bench_deserialize /tmp/sample_irsbs.json

// Example/benchmark-harness code panicking on unwrap/expect is the desired
// behavior, same rationale as #[cfg(test)] code in lib.rs -- not part of the
// angr-9ke6b.212 production-code debt tracker.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::fs;
use std::time::Instant;

use rustylib::vex::deserialize_irsb;
use rustylib::vex::pyvex_bridge::PyVexIRSB;

fn main() {
    let path = env::args().nth(1).expect("usage: bench_deserialize PATH");
    let raw = fs::read_to_string(&path).expect("read");
    let samples: Vec<String> = serde_json::from_str(&raw).expect("parse outer");
    println!("loaded {} IRSB JSON samples", samples.len());

    // Warm up
    for s in &samples[..50.min(samples.len())] {
        let _ = deserialize_irsb(s).unwrap();
    }

    // Bench: full deserialize_irsb (json parse + convert)
    const ITERS: usize = 100;
    let start = Instant::now();
    let mut total = 0;
    for _ in 0..ITERS {
        for s in &samples {
            let irsb = deserialize_irsb(s).unwrap();
            total += irsb.statements.len();
        }
    }
    let elapsed = start.elapsed();
    let calls = ITERS * samples.len();
    let avg_ns = elapsed.as_nanos() as f64 / calls as f64;
    let total_bytes: usize = samples.iter().map(std::string::String::len).sum();
    println!(
        "full deserialize_irsb: {} calls in {:.3}s = {:.1}us/call ({:.1} MB/s, total stmts={})",
        calls,
        elapsed.as_secs_f64(),
        avg_ns / 1000.0,
        (total_bytes * ITERS) as f64 / elapsed.as_secs_f64() / 1e6,
        total
    );

    // Bench: just serde_json parse to PyVexIRSB
    let start = Instant::now();
    let mut tot2 = 0;
    for _ in 0..ITERS {
        for s in &samples {
            let pyvex: PyVexIRSB = serde_json::from_str(s).unwrap();
            tot2 += pyvex.statements.len();
        }
    }
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() as f64 / calls as f64;
    println!(
        "json parse only: {} calls in {:.3}s = {:.1}us/call (stmts={})",
        calls,
        elapsed.as_secs_f64(),
        avg_ns / 1000.0,
        tot2
    );

    // Diff = convert phase
    println!("(convert phase = full - json parse)");
}
