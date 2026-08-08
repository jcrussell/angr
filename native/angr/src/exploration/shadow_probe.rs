//! SI-B (angr-1ilq.3 increment 2b'): opt-in "shadow probe" measuring the REAL
//! state-migration serde tax on the live benchmark workload.
//!
//! When `RUST_PARALLEL_SHADOW_PROBE` is set (read once at manager construction
//! into [`RustExplorationManager::shadow_probe`]), each dispatched state is
//! round-tripped through `to_serialized` (on the main thread) +
//! `from_serialized` (on a persistent scratch thread that owns a DIFFERENT Z3
//! context) and the rebuilt state is discarded. This faithfully models the
//! cross-context AST minting that [`crate::state::StateMigrationPayload::reattach`]
//! performs on a steal, without changing exploration behaviour: the probe only
//! MEASURES the migration cost, it never routes the rebuilt state.
//!
//! The probe is synchronous (one outstanding round-trip at a time) — acceptable
//! because it is a measurement-only mode, never on in real runs or CI. When the
//! flag is off the hook returns immediately and exploration is byte-identical.
//!
//! The scratch thread is detached: it exits cleanly when the manager-side
//! `Sender` drops at teardown (its `recv` loop ends on channel close), so no
//! `JoinHandle` and no `Drop` impl on the manager are required.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** this module
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]`. The one
//! surviving non-test `expect` reads the channel the same function lazily
//! created three statements earlier — see its `#[allow]` reason. (The deny
//! reaches the `#[cfg(test)]` `shadow_probe_tests.rs` child too, which opts
//! out wholesale, so count non-test sites only when checking that claim.)
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::state::RustSimState;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

/// Manager-side endpoints of the scratch-thread channel pair: send serialized
/// state bytes, receive either the `from_serialized` elapsed ns or the error
/// text when the payload was rejected. Stored in
/// [`RustExplorationManager::shadow_probe_chan`].
pub(crate) type ShadowProbeChan = (Sender<Vec<u8>>, Receiver<Result<u64, String>>);

impl RustExplorationManager {
    /// Measure the real migration round-trip cost of `state` (detach/serialize
    /// on the main thread + deserialize/reattach in a foreign Z3 context on the
    /// scratch thread) and accumulate the shadow-probe counters. No-op when the
    /// probe is off. The rebuilt state is discarded — behaviour is unchanged.
    #[allow(
        clippy::expect_used,
        reason = "`shadow_probe_chan` is assigned `Some(spawn_shadow_probe_thread())` by the `is_none()` guard at the top of this same function and nothing between there and here clears it; the intervening fork+serialize borrows `state` and the local `probe_state`, not `self`"
    )]
    pub(crate) fn shadow_probe_migrate(&mut self, state: &RustSimState) {
        if !self.shadow_probe {
            return;
        }
        // Lazily spawn the persistent scratch thread on the first probe step.
        if self.shadow_probe_chan.is_none() {
            self.shadow_probe_chan = Some(spawn_shadow_probe_thread());
        }

        // `to_serialized` flushes memory first (angr-sqfj8.71): a
        // representation-only normalization (Multi cells -> symbolic_objects)
        // on the success path, but `flush_pending_writes`'s error path
        // silently DISCARDS whatever pending writes it hadn't yet materialized
        // (the SILENT(cat-b) log::warn! at that call site) rather than
        // restoring them — fine for the existing export-only callers, which
        // only ever touch a state that's leaving exploration anyway, but not
        // for this probe, which used to run on the live, still-stepping
        // state. Serializing a disposable fork instead keeps the "discarded,
        // byte-identical behaviour" contract this module's doc promises
        // literally true even on that error path — a bad flush can only cost
        // the fork, never the real state. The fork happens outside the timed
        // region so only the real detach/serialize cost is measured; channel
        // transit time is also NOT counted.
        let mut probe_state = state.fork();
        let t0 = Instant::now();
        let bytes = probe_state.to_serialized();
        let ser_ns = t0.elapsed().as_nanos() as u64;
        let nbytes = bytes.len() as u64;

        let (tx, rx) = self
            .shadow_probe_chan
            .as_ref()
            .expect("shadow_probe_chan spawned above");
        // A closed channel during teardown must not panic — just skip the
        // sample. (The thread only exits when our Sender drops, so in practice
        // this never fires mid-run.)
        if tx.send(bytes).is_err() {
            return;
        }
        let deser_ns = match rx.recv() {
            Ok(Ok(ns)) => ns,
            // A failed `from_serialized` costs almost nothing, so folding its
            // elapsed time into the totals would report a spurious near-free
            // migration and bias the SI-C overhead gate downward. Count it
            // separately instead, and warn once so the skew is visible.
            Ok(Err(err)) => {
                if self.parallel_shadow_migration_failures == 0 {
                    log::warn!(
                        "shadow probe: from_serialized failed on the scratch thread ({err}); \
                         this sample is excluded from parallel_shadow_migration_ns/_states/_bytes \
                         (see parallel_shadow_migration_failures)"
                    );
                }
                self.parallel_shadow_migration_failures += 1;
                return;
            }
            Err(_) => return,
        };

        self.parallel_shadow_migration_bytes += nbytes;
        self.parallel_shadow_migration_ns += ser_ns + deser_ns;
        self.parallel_shadow_migration_states += 1;
    }
}

/// Spawn the persistent scratch thread that deserializes shadow-probe payloads
/// in its OWN Z3 context. Returns `(send bytes, receive deserialize-ns)`.
///
/// The thread creates a fresh `z3::Context` and installs it as the thread-local
/// (mirroring `worker_thread` in `exploration/scheduler_pool.rs`): the context lives on
/// this thread's stack for its whole life, created and destroyed on the same
/// thread, so every AST `from_serialized` mints is in a context this thread
/// alone touches — exactly the cross-context reattach the real scheduler does.
///
/// Each reply is `Ok(elapsed_ns)` for a successful round-trip or
/// `Err(<error text>)` when `from_serialized` rejected the payload — the caller
/// must not treat the latter as a (near-free) migration sample.
fn spawn_shadow_probe_thread() -> ShadowProbeChan {
    let (tx_bytes, rx_bytes) = std::sync::mpsc::channel::<Vec<u8>>();
    let (tx_ns, rx_ns) = std::sync::mpsc::channel::<Result<u64, String>>();
    std::thread::spawn(move || {
        let ctx = z3::Context::new(&z3::Config::new());
        z3::Context::set_thread_local(&ctx);
        // The loop ends (and the thread exits cleanly) when `tx_bytes` drops at
        // manager teardown.
        for bytes in rx_bytes {
            let t0 = Instant::now();
            let outcome = match RustSimState::from_serialized(&bytes) {
                Ok(s) => {
                    drop(s);
                    Ok(t0.elapsed().as_nanos() as u64)
                }
                Err(e) => Err(format!("{e:?}")),
            };
            // Ignore a send error: the manager-side Receiver is gone (teardown).
            if tx_ns.send(outcome).is_err() {
                break;
            }
        }
    });
    (tx_bytes, rx_ns)
}

#[cfg(test)]
#[path = "shadow_probe_tests.rs"]
mod tests;
