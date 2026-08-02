//! Unit tests for the SI-B shadow probe (angr-9ke6b.63 / .64).
//!
//! Included as the `#[cfg(test)] mod tests` body of `shadow_probe.rs` via
//! `#[path]`, so `super::*` resolves to that module.
//!
//! These drive [`spawn_shadow_probe_thread`] directly rather than through
//! [`RustExplorationManager::shadow_probe_migrate`]: the manager needs a live
//! Python interpreter to construct, while the scratch thread — where the
//! deserialize-failure and channel-teardown branches actually live — is a plain
//! Rust function. The manager-side wiring stays covered end-to-end by
//! `tests/engines/rust/test_manager_core.py::test_shadow_probe_{off,on}_*`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test module: a failed unwrap is the assertion"
)]

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
