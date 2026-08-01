//! In-module unit tests for `callbacks/config.rs` (angr-c4xcs.7).
//!
//! These cover the pure value types — `DeferredFork` and `ExecutionConfig` —
//! with no Z3 context and no GIL: the pyclass structs are
//! plain Rust structs whose constructors/`__repr__` take no `Python<'_>`, so
//! `condition_ast=None` keeps every case Python-free (mirrors the existing
//! `callbacks_tests::test_loop_execution_event`, which also skips
//! `Python::initialize`).

use super::*;

#[test]
fn test_deferred_fork_new_fields() {
    let fork = DeferredFork::new(0x400123, true, 0x400456, 7, 3, None);
    assert_eq!(fork.branch_addr, 0x400123);
    assert!(fork.path_taken);
    assert_eq!(fork.unexplored_target, 0x400456);
    assert_eq!(fork.condition_id, 7);
    assert_eq!(fork.push_level, 3);
    assert!(fork.condition_ast.is_none());
}

#[test]
fn test_deferred_fork_repr_hex_and_pushlevel() {
    let fork = DeferredFork::new(0x1000, false, 0x2000, 1, 5, None);
    let repr = fork.__repr__();
    // Addresses are hex-formatted; path_taken and push_level are echoed.
    assert!(repr.contains("branch_addr=0x1000"), "{repr}");
    assert!(repr.contains("unexplored=0x2000"), "{repr}");
    assert!(repr.contains("path_taken=false"), "{repr}");
    assert!(repr.contains("push_level=5"), "{repr}");
}

#[test]
fn test_execution_config_py_new_overrides_only_two_args() {
    // py_new starts from Default and overrides ONLY the two Python-tunable
    // args; every other field (notably enable_eager_prefetch=false) comes
    // straight from Default — no separate hardcoded value to drift.
    let cfg = ExecutionConfig::py_new(123, false);
    let def = ExecutionConfig::default();
    assert_eq!(cfg.max_deferred_forks, 123);
    assert!(!cfg.use_deferred_forks);
    // Everything the args did NOT touch matches Default.
    assert_eq!(cfg.enable_eager_prefetch, def.enable_eager_prefetch);
    assert!(!cfg.enable_eager_prefetch, "inherited from Default (off)");
    assert_eq!(cfg.max_prefetch_batch, def.max_prefetch_batch);
    assert_eq!(cfg.enable_stride_detection, def.enable_stride_detection);
    assert_eq!(cfg.max_symbolic_ip_targets, def.max_symbolic_ip_targets);
}

#[test]
fn test_execution_config_py_new_defaults_match_default_trait() {
    // Calling ExecutionConfig() from Python (both args defaulted) must produce
    // exactly the engine's runtime Default — the divergence tracked by
    // angr-ph300.68 is now impossible because py_new is built ..Default.
    let cfg = ExecutionConfig::py_new(500, true);
    let def = ExecutionConfig::default();
    assert_eq!(cfg.max_deferred_forks, def.max_deferred_forks);
    assert_eq!(cfg.use_deferred_forks, def.use_deferred_forks);
    assert!(cfg.use_deferred_forks, "Default enables deferred forks");
    assert_eq!(cfg.enable_eager_prefetch, def.enable_eager_prefetch);
    assert!(
        !cfg.enable_eager_prefetch,
        "Default disables eager prefetch (per-page on demand)"
    );
    assert_eq!(cfg.max_prefetch_batch, def.max_prefetch_batch);
    assert_eq!(cfg.enable_stride_detection, def.enable_stride_detection);
    assert_eq!(cfg.max_symbolic_ip_targets, def.max_symbolic_ip_targets);
}

#[test]
fn test_execution_config_repr_contains_key_knobs() {
    let cfg = ExecutionConfig::py_new(42, true);
    let repr = cfg.__repr__();
    assert!(repr.contains("max_deferred_forks=42"), "{repr}");
    assert!(repr.contains("use_deferred_forks=true"), "{repr}");
    assert!(repr.contains("eager_prefetch=false"), "{repr}");
    assert!(repr.contains("max_prefetch_batch=256"), "{repr}");
}
