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
// Grandfathered clippy::unwrap_used/expect_used debt -- angr-9ke6b.212 tracks
// burning this down file by file. Do not add new unwrap()/expect() calls here;
// new files/callers must handle the None/Err case explicitly instead.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::state::RustSimState;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

impl RustExplorationManager {
    /// Measure the real migration round-trip cost of `state` (detach/serialize
    /// on the main thread + deserialize/reattach in a foreign Z3 context on the
    /// scratch thread) and accumulate the shadow-probe counters. No-op when the
    /// probe is off. The rebuilt state is discarded — behaviour is unchanged.
    pub(crate) fn shadow_probe_migrate(&mut self, state: &RustSimState) {
        if !self.shadow_probe {
            return;
        }
        // Lazily spawn the persistent scratch thread on the first probe step.
        if self.shadow_probe_chan.is_none() {
            self.shadow_probe_chan = Some(spawn_shadow_probe_thread());
        }

        // Detach/serialize cost, measured on the main thread (read-only borrow:
        // `to_serialized` is `&self`). Channel transit time is NOT counted.
        let t0 = Instant::now();
        let bytes = state.to_serialized();
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
            Ok(ns) => ns,
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
/// (mirroring the work-stealing worker in `scheduler.rs`): the context lives on
/// this thread's stack for its whole life, created and destroyed on the same
/// thread, so every AST `from_serialized` mints is in a context this thread
/// alone touches — exactly the cross-context reattach the real scheduler does.
fn spawn_shadow_probe_thread() -> (Sender<Vec<u8>>, Receiver<u64>) {
    let (tx_bytes, rx_bytes) = std::sync::mpsc::channel::<Vec<u8>>();
    let (tx_ns, rx_ns) = std::sync::mpsc::channel::<u64>();
    std::thread::spawn(move || {
        let ctx = z3::Context::new(&z3::Config::new());
        z3::Context::set_thread_local(&ctx);
        // The loop ends (and the thread exits cleanly) when `tx_bytes` drops at
        // manager teardown.
        for bytes in rx_bytes {
            let t0 = Instant::now();
            if let Ok(s) = RustSimState::from_serialized(&bytes) {
                drop(s);
            }
            let ns = t0.elapsed().as_nanos() as u64;
            // Ignore a send error: the manager-side Receiver is gone (teardown).
            if tx_ns.send(ns).is_err() {
                break;
            }
        }
    });
    (tx_bytes, rx_ns)
}
