//! In-module unit tests for `callbacks/config.rs` (angr-c4xcs.7).
//!
//! These cover the `ExecutionConfig` pyclass with no Z3 context and no GIL:
//! its constructor/`__repr__` take no `Python<'_>`.
//!
//! `DeferredFork` had two tests here covering its `#[new]` constructor and
//! `__repr__`; both went away with that Python surface (angr-03vl4.6). Its
//! behaviour is covered by the consumers instead —
//! `exploration::fork_materialize_tests` and `callback_types_tests`.

use super::*;

#[test]
fn test_execution_config_py_new_overrides_only_two_args() {
    // py_new starts from Default and overrides ONLY the two Python-tunable
    // args; every other field (notably enable_eager_prefetch=false) comes
    // straight from Default — no separate hardcoded value to drift.
    let cfg = PyExecutionConfig::py_new(123, false).inner;
    let def = ExecutionConfig::default();
    assert_eq!(cfg.max_deferred_forks, 123);
    assert!(!cfg.use_deferred_forks);
    // Everything the args did NOT touch matches Default.
    assert_eq!(cfg.enable_eager_prefetch, def.enable_eager_prefetch);
    assert!(!cfg.enable_eager_prefetch, "inherited from Default (off)");
    assert_eq!(cfg.max_prefetch_batch, def.max_prefetch_batch);
    assert_eq!(cfg.max_symbolic_ip_targets, def.max_symbolic_ip_targets);
}

#[test]
fn test_execution_config_py_new_defaults_match_default_trait() {
    // Calling ExecutionConfig() from Python (both args defaulted) must produce
    // exactly the engine's runtime Default. A divergence between the two is
    // impossible because py_new is built ..Default (angr-ph300.68).
    let cfg = PyExecutionConfig::py_new(500, true).inner;
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
    assert_eq!(cfg.max_symbolic_ip_targets, def.max_symbolic_ip_targets);
}

#[test]
fn test_execution_config_repr_contains_key_knobs() {
    let cfg = PyExecutionConfig::py_new(42, true);
    let repr = cfg.__repr__();
    assert!(repr.contains("max_deferred_forks=42"), "{repr}");
    assert!(repr.contains("use_deferred_forks=true"), "{repr}");
    assert!(repr.contains("eager_prefetch=false"), "{repr}");
    assert!(repr.contains("max_prefetch_batch=256"), "{repr}");
}
