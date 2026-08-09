// Tests for exploration/stats_api.rs (see rust-mod-tests-sibling-extraction
// for why these live in a sibling file rather than an inline `mod tests`).
//
// `stats()` / `get_fallback_stats()` are two of the most Python-visible
// introspection APIs in the crate — `run_single.py --counters-json`, the
// benchmark gate's `bench_diff`, and the parallel kill-gates all read the key
// names produced here — yet the module had no Rust-level test at all
// (angr-c7xno.22).
use super::*;
use crate::stash::{STASH_ACTIVE, STASH_FOUND};
use crate::state::RustSimState;

/// Read `key` out of `dict`, failing loudly when the exporter never set it.
fn get<'py>(dict: &Bound<'py, PyDict>, key: &str) -> Bound<'py, PyAny> {
    dict.get_item(key)
        .expect("dict lookup")
        .unwrap_or_else(|| panic!("stats dict is missing key {key:?}"))
}

fn get_u64(dict: &Bound<'_, PyDict>, key: &str) -> u64 {
    get(dict, key).extract().expect("u64 counter")
}

/// The whole reason `set_fallback_counter_items` exists: BOTH `_stats` and
/// `_get_fallback_stats` must export the shared counter block under identical
/// key names and values (angr-9ke6b.70). Rather than restate the key list —
/// which would have to be edited in lockstep with the source, i.e. exactly the
/// drift being guarded — the expected set is *derived* by calling the helper
/// into a scratch dict, so a counter added there is automatically demanded of
/// both exporters.
///
/// Every counter is given a distinct non-zero value first, so a missing key
/// cannot pass as `0 == 0`.
#[test]
fn shared_fallback_counter_block_matches_across_both_exporters() {
    Python::initialize();
    Python::attach(|py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        mgr.dcas_unsupported_count = 3;
        mgr.vecret_gsptr_fallback_count = 5;
        mgr.simprocedure_python_fallback_count = 7;
        mgr.simprocedure_fallback_by_name.insert("strlen".into(), 11);
        mgr.syscall_python_fallback_count = 13;
        mgr.syscall_python_fallback_by_num.insert(60, 17);
        mgr.syscall_native_count = 19;
        mgr.syscall_native_by_num.insert(1, 23);

        let scratch = PyDict::new(py);
        mgr.set_fallback_counter_items(py, &scratch)
            .expect("shared block");
        let shared: Vec<String> = scratch
            .keys()
            .iter()
            .map(|k| k.extract().expect("str key"))
            .collect();
        assert!(
            shared.len() >= 8,
            "shared block unexpectedly small: {shared:?}"
        );

        let stats = mgr._stats(py).expect("stats");
        let fallback = mgr._get_fallback_stats(py).expect("fallback stats");
        for key in &shared {
            let from_stats = get(&stats, key);
            let from_fallback = get(&fallback, key);
            assert!(
                from_stats.eq(&from_fallback).expect("py eq"),
                "key {key:?} disagrees between stats() and get_fallback_stats(): \
                 {from_stats:?} vs {from_fallback:?}"
            );
        }
    });
}

/// `stats()` reports live stash populations and the manager-level counters,
/// not a snapshot taken at construction time.
#[test]
fn stats_reports_live_stash_populations_and_counters() {
    Python::initialize();
    Python::attach(|py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        mgr.steps = 42;
        mgr.vex_fallback_count = 9;
        mgr.vex_fallback_addrs
            .insert(0x40_1000, "unsupported op".into());
        mgr.vex_fallback_addrs
            .insert(0x40_2000, "unsupported op".into());
        mgr.sm
            .push(STASH_ACTIVE, RustSimState::new("amd64").expect("state"));
        mgr.sm
            .push(STASH_ACTIVE, RustSimState::new("amd64").expect("state"));
        mgr.sm
            .push(STASH_FOUND, RustSimState::new("amd64").expect("state"));
        mgr.sm.avoided_count = 4;
        mgr.sm.pruned_count = 5;
        mgr.sm.deadended_count = 6;

        let stats = mgr._stats(py).expect("stats");
        assert_eq!(get_u64(&stats, "steps"), 42);
        assert_eq!(get_u64(&stats, "active"), 2);
        assert_eq!(get_u64(&stats, "found"), 1);
        assert_eq!(get_u64(&stats, "vex_fallback_count"), 9);
        assert_eq!(get_u64(&stats, "vex_fallback_unique_addrs"), 2);
        assert_eq!(get_u64(&stats, "avoided_count"), 4);
        assert_eq!(get_u64(&stats, "pruned_count"), 5);
        assert_eq!(get_u64(&stats, "deadended_count"), 6);
    });
}

/// `reconvergence_rate` is a derived ratio, and its denominator is only
/// non-zero once the run loop has sampled at least one step. The zero case
/// must report 0.0 rather than a NaN Python floats propagate silently.
#[test]
fn stats_reconvergence_rate_is_derived_and_zero_guarded() {
    Python::initialize();
    Python::attach(|py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

        let stats = mgr._stats(py).expect("stats");
        let rate: f64 = get(&stats, "reconvergence_rate").extract().expect("f64");
        assert_eq!(rate, 0.0, "no samples must not divide by zero");

        mgr.reconvergence_collision_states = 3;
        mgr.reconvergence_active_observed = 12;
        mgr.reconvergence_samples = 4;
        mgr.reconvergence_max_group = 2;
        let stats = mgr._stats(py).expect("stats");
        let rate: f64 = get(&stats, "reconvergence_rate").extract().expect("f64");
        assert!(
            (rate - 0.25).abs() < 1e-12,
            "rate must be collisions/observed, got {rate}"
        );
        assert_eq!(get_u64(&stats, "reconvergence_samples"), 4);
        assert_eq!(get_u64(&stats, "reconvergence_max_group"), 2);
    });
}

/// `build_count_dict`'s second return value is the summed total, and the three
/// native-proc fallback totals are exported from it. The documented invariant
/// (`stats_api::_stats`: "sum(symbolic + not_implemented + other) ==
/// native_proc_fallbacks") is what downstream attribution relies on.
#[test]
fn stats_native_proc_totals_sum_their_per_name_breakdowns() {
    Python::initialize();
    Python::attach(|py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let native = &mut mgr.profiling.native_proc_stats;
        native.native_calls = 100;
        native.python_fallbacks = 1 + 2 + 4 + 8 + 16;
        native.symbolic_fallbacks_by_name.insert("strlen".into(), 1);
        native.symbolic_fallbacks_by_name.insert("memcpy".into(), 2);
        native
            .not_implemented_fallbacks_by_name
            .insert("strtok".into(), 4);
        native.other_fallbacks_by_name.insert("free".into(), 8);
        native.other_fallbacks_by_name.insert("malloc".into(), 16);

        let stats = mgr._stats(py).expect("stats");
        assert_eq!(get_u64(&stats, "native_proc_calls"), 100);
        assert_eq!(get_u64(&stats, "native_proc_symbolic_fallbacks"), 3);
        assert_eq!(get_u64(&stats, "native_proc_not_implemented_fallbacks"), 4);
        assert_eq!(get_u64(&stats, "native_proc_other_fallbacks"), 24);
        assert_eq!(
            get_u64(&stats, "native_proc_symbolic_fallbacks")
                + get_u64(&stats, "native_proc_not_implemented_fallbacks")
                + get_u64(&stats, "native_proc_other_fallbacks"),
            get_u64(&stats, "native_proc_fallbacks"),
            "per-reason breakdown must partition native_proc_fallbacks"
        );

        // The same three maps are re-exported unprefixed by
        // `_native_procedure_stats`; keep the two views in agreement.
        let per_proc = mgr._native_procedure_stats(py).expect("native proc stats");
        assert_eq!(get_u64(&per_proc, "symbolic_fallbacks"), 3);
        assert_eq!(get_u64(&per_proc, "not_implemented_fallbacks"), 4);
        assert_eq!(get_u64(&per_proc, "other_fallbacks"), 24);
    });
}

/// `get_fallback_stats()` keys its address map by a `0x`-prefixed lowercase hex
/// string, not by an int — the Python side prints these verbatim, so the
/// formatting is part of the API.
#[test]
fn get_fallback_stats_keys_addresses_as_lowercase_hex() {
    Python::initialize();
    Python::attach(|py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        mgr.vex_fallback_count = 2;
        mgr.vex_fallback_addrs
            .insert(0x40_1abc, "unsupported dirty helper".into());

        let fallback = mgr._get_fallback_stats(py).expect("fallback stats");
        assert_eq!(get_u64(&fallback, "count"), 2);
        let addrs = get(&fallback, "addresses");
        let addrs = addrs.cast::<PyDict>().expect("addresses is a dict");
        let reason: String = get(addrs, "0x401abc").extract().expect("reason str");
        assert_eq!(reason, "unsupported dirty helper");
    });
}
