//! In-module unit tests for `callbacks/config.rs` (angr-c4xcs.7).
//!
//! These cover the pure value types — `DeferredFork`, `BranchPolicy`, and
//! `ExecutionConfig` — with no Z3 context and no GIL: the pyclass structs are
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
fn test_branch_policy_default_is_take_true() {
    // The engine's documented default branch policy.
    assert_eq!(BranchPolicy::default(), BranchPolicy::TakeTrue);
}

#[test]
fn test_branch_policy_staticmethod_constructors() {
    assert_eq!(BranchPolicy::take_true(), BranchPolicy::TakeTrue);
    assert_eq!(BranchPolicy::take_false(), BranchPolicy::TakeFalse);
    assert_eq!(
        BranchPolicy::take_fallthrough(),
        BranchPolicy::TakeFallthrough
    );
    assert_eq!(BranchPolicy::alternate(), BranchPolicy::Alternate);
    // Distinct variants must not compare equal.
    assert_ne!(BranchPolicy::TakeTrue, BranchPolicy::TakeFalse);
}

#[test]
fn test_execution_config_py_new_defaults() {
    // py_new signature defaults: max_deferred_forks=500, use_deferred_forks=false.
    let cfg = ExecutionConfig::py_new(500, false);
    assert_eq!(cfg.max_deferred_forks, 500);
    assert!(!cfg.use_deferred_forks);
    assert_eq!(cfg.branch_policy, BranchPolicy::TakeTrue);
    // py_new turns eager prefetch ON (distinct from the Default impl below).
    assert!(cfg.enable_eager_prefetch);
    assert_eq!(cfg.max_prefetch_batch, 256);
    assert_eq!(cfg.max_concretization_range, 65536);
    assert!(cfg.enable_stride_detection);
    assert_eq!(cfg.max_symbolic_ip_targets, 257);
}

#[test]
fn test_execution_config_default_trait_differs_from_py_new() {
    // The Default impl is the engine's real runtime config; it enables
    // deferred forks and DISABLES eager prefetch (per-page on demand), the
    // opposite of py_new's prefetch flag. Pin both so a future tweak to one
    // path is a conscious change.
    let cfg = ExecutionConfig::default();
    assert_eq!(cfg.max_deferred_forks, 500);
    assert!(cfg.use_deferred_forks, "Default enables deferred forks");
    assert!(
        !cfg.enable_eager_prefetch,
        "Default disables eager prefetch (per-page on demand)"
    );
    assert_eq!(cfg.max_symbolic_ip_targets, 257);
}

#[test]
fn test_execution_config_repr_contains_key_knobs() {
    let cfg = ExecutionConfig::py_new(42, true);
    let repr = cfg.__repr__();
    assert!(repr.contains("max_deferred_forks=42"), "{repr}");
    assert!(repr.contains("use_deferred_forks=true"), "{repr}");
    assert!(repr.contains("TakeTrue"), "{repr}");
}
