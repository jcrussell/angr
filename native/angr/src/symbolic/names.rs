//! Fresh-name counters.
//!
//! A named leaf is the clarirs variable of that name (angr-mx00p.12), so two
//! mints of one name are one variable. Every counter that numbers fresh
//! symbol names lives here, keyed by its caller, so a snapshot carries them
//! all ([`watermarks`]) and a restore raises each one past every name the
//! restored state already holds ([`reserve`]). Without that, a state restored
//! from another process (whose counters also started at 0) would see the
//! next local mint reuse one of its names. The symbol-id counter behind
//! `SymContext::new_bv` and `unique_name` is the snapshot's `next_id`
//! instead.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, MutexGuard, OnceLock};

static COUNTERS: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();

#[allow(
    clippy::expect_used,
    reason = "`COUNTERS` poison guard: poison requires a thread to unwind out of a live `MutexGuard`, and the crate ships with `panic = \"abort\"` (workspace `Cargo.toml`, angr-1cue), so no unwind is possible"
)]
fn counters() -> MutexGuard<'static, HashMap<String, u64>> {
    COUNTERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("name counter mutex poisoned")
}

/// The next index for names under `key`: 0, 1, 2, ... for each key on its own.
#[cfg_attr(not(feature = "vex-engine"), allow(dead_code))]
pub(crate) fn next(key: &str) -> u64 {
    let mut counters = counters();
    match counters.get_mut(key) {
        Some(n) => {
            let id = *n;
            // overflow-ok: a u64 name counter does not reach 2^64 mints.
            *n += 1;
            id
        }
        None => {
            counters.insert(key.to_owned(), 1);
            0
        }
    }
}

/// Every counter's next index, as a snapshot records it.
pub(crate) fn watermarks() -> BTreeMap<String, u64> {
    counters().iter().map(|(k, v)| (k.clone(), *v)).collect()
}

/// Raise each counter to at least its recorded next index; a counter already
/// past it is left alone.
pub(crate) fn reserve(watermarks: &BTreeMap<String, u64>) {
    let mut counters = counters();
    for (key, &mark) in watermarks {
        let n = counters.entry(key.clone()).or_insert(0);
        *n = (*n).max(mark);
    }
}

test_submod!("names_tests.rs" => names_tests);
