// Unit tests for exploration/mod.rs (RustExplorationManager orchestrator).
// Extracted from the inline `#[cfg(test)] mod tests` block; see rust-mod-tests-sibling-extraction.

use super::*;

#[test]
fn test_exploration_manager_creation() {
    Python::initialize();
    Python::attach(|_py| {
        let mgr = RustExplorationManager::new("amd64", None).unwrap();
        assert_eq!(mgr.arch(), "amd64");
        assert_eq!(mgr.active_count(), 0);
        assert_eq!(mgr.found_count(), 0);
    });
}

#[test]
fn test_stash_management() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();

        // Create state
        let state_id = mgr.create_state("active").unwrap();
        // state_id is assigned from a global atomic counter starting at 0
        assert!(state_id < u64::MAX);
        assert_eq!(mgr.active_count(), 1);

        // Check state IDs
        let ids = mgr.get_state_ids("active");
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], state_id);
    });
}

#[test]
fn test_find_avoid_addresses() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();

        mgr.set_find_addrs(vec![0x1000, 0x2000]);
        mgr.set_avoid_addrs(vec![0x3000]);

        assert!(mgr.find_addrs.contains(&0x1000));
        assert!(mgr.find_addrs.contains(&0x2000));
        assert!(mgr.avoid_addrs.contains(&0x3000));
    });
}

/// Orchestrator semantics: pyclass setters mutate the sub-struct, not a
/// shadow field on the manager. This test asserts the delegation pattern
/// established by the angr-4j5u decomposition — `set_*` and `get_*` round
/// through `constraint_solver`, `memory_config`, `environment`, and
/// `profiling` respectively.
#[test]
fn test_orchestrator_delegation() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();

        // ConstraintSolver: lazy_solves + solver_timeout_ms
        mgr.set_lazy_solves(true);
        assert!(mgr.constraint_solver.lazy_solves);
        mgr.set_solver_timeout(7777);
        assert_eq!(mgr.constraint_solver.solver_timeout_ms, 7777);

        // MemoryConfiguration: zero_fill + vex_opt_level (global + override)
        mgr.set_zero_fill_unconstrained(true);
        assert!(mgr.memory_config.zero_fill_unconstrained);
        mgr.set_vex_opt_level(Some(2));
        assert_eq!(mgr.get_vex_opt_level(), Some(2));
        mgr.set_vex_opt_level_override(0xdead, 0);
        assert_eq!(mgr.resolve_vex_opt_level(0xdead), Some(0));
        assert_eq!(mgr.resolve_vex_opt_level(0xbeef), Some(2));

        // ExecutionEnvironment: max_history applies to existing states
        mgr.set_max_history(42);
        assert_eq!(mgr.environment.max_history, 42);
        assert_eq!(mgr.get_max_history(), 42);

        // ProfilingCollector: enable flag flips
        mgr.set_profiling(true);
        assert!(mgr.profiling.profiling_enabled);

        // ConstraintTracker: uniqueness filter reaches the sub-struct
        mgr.register_uniqueness_filter(vec!["rax".into(), "rbx".into()]);
        assert!(mgr.uniqueness_filter_enabled());
        assert_eq!(mgr.constraint_tracker.uniqueness_registers.len(), 2);
        mgr.disable_uniqueness_filter();
        assert!(!mgr.uniqueness_filter_enabled());
    });
}

/// State lifecycle: create + move + reset_for_stage round-trip through the
/// extracted bodies in `state_lifecycle.rs`. Verifies that the StashManager
/// remains the canonical state-storage and the index is kept in sync.
#[test]
fn test_state_lifecycle_orchestration() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();

        let s1 = mgr.create_state("active").unwrap();
        let _s2 = mgr.create_state("active").unwrap();
        assert_eq!(mgr.active_count(), 2);
        assert_eq!(mgr.state_stash(s1).as_deref(), Some("active"));

        // Move s1 to a custom stash; index follows.
        assert!(mgr.move_state(s1, "active", "found").unwrap());
        assert_eq!(mgr.state_stash(s1).as_deref(), Some("found"));
        assert_eq!(mgr.active_count(), 1);
        assert_eq!(mgr.found_count(), 1);

        // reset_for_stage: keep s1, drop everything else.
        let kept = mgr.reset_for_stage(s1).unwrap();
        assert_eq!(kept, s1);
        assert_eq!(mgr.active_count(), 1);
        assert_eq!(mgr.found_count(), 0);
        let active_ids = mgr.get_state_ids("active");
        assert_eq!(active_ids, vec![s1]);
    });
}
