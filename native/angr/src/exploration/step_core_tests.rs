//! Unit tests for the `Send + Sync` step-config snapshot in `step_core.rs`
//! (angr-ph300.3.1). These assert that [`RustExplorationManager::step_context`]
//! faithfully mirrors the manager's current configuration into the owned,
//! borrow-free [`StepContext`] a worker thread consumes.
//!
//! Snapshot consistency is the contract the work-stealing worker path relies
//! on: a field the manager reads during a step but that `step_context` fails to
//! copy would silently diverge worker behavior from the single-threaded loop.
//! Actually *running* `run_interpreter_step_core` needs a live lifted block +
//! binary regions (covered end-to-end by the Python suite); these pin the
//! cheap, deterministic half — that the bundle equals its source — in
//! `cargo test`.

use super::*;

use pyo3::Python;

#[test]
fn step_context_snapshots_find_avoid_and_stop_addr_union() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_find_addrs(vec![0x1000, 0x2000]);
        mgr.set_avoid_addrs(vec![0x3000]);

        let ctx = mgr.step_context();

        assert_eq!(ctx.find_addrs, mgr.find_addrs);
        assert_eq!(ctx.avoid_addrs, mgr.avoid_addrs);
        // stop_addrs is the interpreter's block-chain stop set: find ∪ avoid.
        assert_eq!(ctx.stop_addrs, mgr.stop_addrs);
        assert!(ctx.stop_addrs.contains(&0x1000));
        assert!(ctx.stop_addrs.contains(&0x2000));
        assert!(ctx.stop_addrs.contains(&0x3000));
        assert_eq!(ctx.stop_addrs.len(), 3);
    });
}

#[test]
fn step_context_snapshots_solver_and_memory_config() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_lazy_solves(true);
        mgr.set_max_steps_per_run(17);
        mgr.set_block_granular(true);
        mgr.set_vex_opt_level(Some(1));
        mgr.set_vex_opt_level_override(0xabc, 0);

        let ctx = mgr.step_context();

        assert!(
            ctx.lazy_solves,
            "lazy_solves off the ConstraintSolver sub-struct"
        );
        assert_eq!(ctx.max_steps_per_run, 17);
        assert!(ctx.block_granular);
        assert_eq!(ctx.vex_opt_level, Some(1));
        assert_eq!(ctx.vex_opt_level_overrides.get(&0xabc), Some(&0));
    });
}

#[test]
fn step_context_cc_snapshot_matches_calling_convention() {
    Python::initialize();
    Python::attach(|_py| {
        let mgr = RustExplorationManager::new("amd64", None).unwrap();
        let cc = &mgr.environment.calling_convention;
        let ctx = mgr.step_context();

        // The CC trait object is not Clone, so step_context snapshots only the
        // scalar/vec fields the post-step native arms consult (angr-vh834).
        assert_eq!(ctx.cc.arg_registers, cc.arg_registers());
        assert_eq!(ctx.cc.syscall_arg_registers, cc.syscall_arg_registers());
        assert_eq!(ctx.cc.return_register, cc.return_register());
        assert_eq!(ctx.cc.link_register, cc.link_register());
        assert_eq!(ctx.cc.pops_return_addr, cc.pops_return_addr());
        assert_eq!(ctx.cc.pointer_size, cc.pointer_size());
        // pointer_size on the bundle top-level is the same snapshot as cc's.
        assert_eq!(ctx.pointer_size, cc.pointer_size());
    });
}

#[test]
fn step_context_mirrors_env_and_deferred_fork_mode() {
    Python::initialize();
    Python::attach(|_py| {
        let mgr = RustExplorationManager::new("amd64", None).unwrap();
        let ctx = mgr.step_context();

        assert_eq!(ctx.vex_arch, mgr.environment.vex_arch);
        assert_eq!(ctx.os_name, mgr.environment.os_name);
        // Manager-level deferred-fork mode is read straight off exec_config.
        assert_eq!(ctx.use_deferred_forks, mgr.exec_config.use_deferred_forks);
        assert_eq!(
            ctx.materialize_unconstrained_forks,
            mgr.materialize_unconstrained_forks
        );
    });
}

/// The reason this bundle exists (angr-1ilq.3 2b-i): a worker must be able to
/// own the step config across threads. `step_core.rs` carries a compile-time
/// `assert_send_sync::<StepContext>()`; this run-time test complements it by
/// proving a *constructed* snapshot actually satisfies the bound (a moved
/// value, not just the type).
#[test]
fn step_context_is_send_and_sync() {
    Python::initialize();
    Python::attach(|_py| {
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        let mgr = RustExplorationManager::new("amd64", None).unwrap();
        let ctx = mgr.step_context();
        assert_send_sync(&ctx);
        assert_eq!(ctx.vex_arch, mgr.environment.vex_arch);
    });
}
