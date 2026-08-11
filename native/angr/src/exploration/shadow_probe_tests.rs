//! Unit tests for the SI-B shadow probe (angr-9ke6b.63 / .64).
//!
//! Included as the `#[cfg(test)] mod tests` body of `shadow_probe.rs` via
//! `test_submod!`, so `super::*` resolves to that module. The macro supplies
//! the `clippy::unwrap_used` / `expect_used` opt-out this file used to
//! hand-write — load-bearing here, because `shadow_probe.rs` carries its own
//! module-level `deny` of both.
//!
//! These drive [`spawn_shadow_probe_thread`] directly rather than through
//! [`RustExplorationManager::shadow_probe_migrate`]: the manager needs a live
//! Python interpreter to construct, while the scratch thread — where the
//! deserialize-failure and channel-teardown branches actually live — is a plain
//! Rust function. The manager-side wiring stays covered end-to-end by
//! `tests/engines/rust/test_manager_core.py::test_shadow_probe_{off,on}_*`.

use super::*;
use std::time::Duration;

/// Generous upper bound — the scratch thread has to build a fresh `z3::Context`
/// on its first message, which dominates. A hang here is a real deadlock, not
/// slowness.
const REPLY_TIMEOUT: Duration = Duration::from_secs(60);

fn valid_envelope() -> Vec<u8> {
    RustSimState::new("amd64").unwrap().to_serialized()
}

#[test]
fn scratch_thread_reports_ok_for_a_valid_envelope() {
    let (tx, rx) = spawn_shadow_probe_thread();
    tx.send(valid_envelope()).expect("scratch thread alive");
    match rx.recv_timeout(REPLY_TIMEOUT) {
        Ok(Ok(_ns)) => {}
        other => panic!("expected a successful round-trip sample, got {other:?}"),
    }
}

#[test]
fn scratch_thread_reports_err_instead_of_a_fake_free_sample() {
    // Two distinct `from_serialized` rejections: an empty envelope and a
    // bumped version byte. Both used to be swallowed by `if let Ok(s)`, which
    // still sent the (near-zero) elapsed time as if it were a real migration.
    let mut bumped = valid_envelope();
    bumped[0] = bumped[0].wrapping_add(1);

    let (tx, rx) = spawn_shadow_probe_thread();
    for (label, bytes) in [("empty", Vec::new()), ("version-mismatch", bumped)] {
        tx.send(bytes).expect("scratch thread alive");
        match rx.recv_timeout(REPLY_TIMEOUT) {
            Ok(Err(_msg)) => {}
            other => panic!("{label}: expected an Err reply, got {other:?}"),
        }
    }
}

#[test]
fn scratch_thread_keeps_serving_after_a_failed_deserialize() {
    // The failure path must not poison the thread: the probe stays on for the
    // rest of the run, so a single bad payload cannot silence every later
    // sample.
    let (tx, rx) = spawn_shadow_probe_thread();
    tx.send(Vec::new()).expect("scratch thread alive");
    assert!(
        matches!(rx.recv_timeout(REPLY_TIMEOUT), Ok(Err(_))),
        "empty envelope must be reported as a failure"
    );
    tx.send(valid_envelope())
        .expect("scratch thread still alive");
    assert!(
        matches!(rx.recv_timeout(REPLY_TIMEOUT), Ok(Ok(_))),
        "a valid envelope after a failure must still produce a sample"
    );
}

#[test]
fn scratch_thread_exits_when_the_reply_receiver_is_dropped() {
    // The `tx_ns.send(..).is_err() => break` teardown branch. Once the manager
    // side stops listening, the thread must exit rather than spin; we observe
    // the exit through `rx_bytes` dropping, which makes our own sends fail.
    let (tx, rx) = spawn_shadow_probe_thread();
    drop(rx);
    tx.send(valid_envelope()).expect("scratch thread alive");

    let mut exited = false;
    for _ in 0..600 {
        if tx.send(valid_envelope()).is_err() {
            exited = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        exited,
        "scratch thread must exit once its reply receiver is gone"
    );
}

/// Regression: `shadow_probe_migrate` used to take `&mut RustSimState` and
/// serialize the LIVE, still-stepping state directly — since `to_serialized`
/// now flushes memory first (angr-sqfj8.71), and `flush_pending_writes`'s
/// error path silently DISCARDS whatever pending writes it hadn't yet
/// materialized rather than restoring them, a bad flush during the probe
/// could permanently drop a still-active state's pending writes. The fix
/// forks before serializing, so the probe can only ever cost the disposable
/// fork. This pins that the live state's `multi_cell_count`/pending-write
/// state survives the probe unchanged regardless of whether the probe itself
/// succeeds.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn shadow_probe_migrate_does_not_mutate_the_live_state() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        mgr.shadow_probe = true;

        let mut state = RustSimState::new("amd64").expect("amd64 state");
        state.map_memory(0x1000, 0x4000, crate::memory::Permission::RWX);
        let (addr_var, val_bv) = {
            let ctx = state.solver().borrow();
            let addr_var = RustBV::symbolic(&ctx, "shadow_probe_multi_addr", 64);
            ctx.assume_true(
                &addr_var
                    .eq(&RustBV::concrete(0x1000, 64), &ctx)
                    .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
            );
            (addr_var, RustBV::concrete(0xCAFE_BABEu128, 32))
        };
        let concretizer = crate::concretize::AddressConcretizer {
            symbolic_write_addresses: true,
            ..crate::concretize::AddressConcretizer::new()
        };
        let ctx = state.solver().clone();
        let ctx = ctx.borrow();
        state
            .memory_mut()
            .store_symbolic_unified(addr_var, val_bv, &ctx, &concretizer)
            .expect("Multi store must succeed");
        drop(ctx);
        let multi_before = state.memory().multi_cell_count();
        assert_ne!(multi_before, 0, "precondition: Multi cells installed");

        mgr.shadow_probe_migrate(&state);

        assert_eq!(
            state.memory().multi_cell_count(),
            multi_before,
            "the probe must not flush/mutate the live state's Multi cells"
        );
    });
}
