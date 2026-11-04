//! The PyO3-free core of angr's native engine. Today it holds [`symbolic`]:
//! the Z3 solver context (`SymContext`), constraint solving, the shared
//! lineage solver and address concretization (`symbolic::concretize`);
//! [`memory`], the lazy copy-on-write `SymbolicMemory`; [`arch`], the
//! `Arch` trait, the per-arch register tables, `RegisterFile` and the
//! calling conventions; [`state`], `RustSimState` and its filesystem,
//! snapshots and migration payload; and the crate-root helpers every engine
//! module shares ([`arc_shared`], [`duration_ns`], [`elapsed_ns`]).
//! `interpreter` joins it next (angr-mx00p.51).
//!
//! **Dependencies.** `angr-bv`, `angr-vex`, `angr-symex-macros`,
//! `clarirs-core`, `z3` and the plain-data crates: no `pyo3` (check with
//! `cargo tree -p angr-symex-core -e normal`).
//!
//! The engine (`native/angr`) re-exports [`symbolic`], [`memory`], [`arch`]
//! and [`state`], so `crate::symbolic::` / `crate::memory::` /
//! `crate::arch::` / `crate::state::` (and the `rustylib::` spellings of all
//! but `arch`) name these items (angr-mx00p.51.11, angr-mx00p.51.12,
//! angr-mx00p.51.13, angr-mx00p.51.14).
//!
//! # File index
//!
//! | File | Holds |
//! |------|-------|
//! | `arch/` | The `Arch` trait and `AddrOrSymbolic`, the per-arch register tables and offsets, `RegisterFile`, the arch registry and the calling conventions; `arch/mod.rs` indexes its files. |
//! | `memory/` | `SymbolicMemory`, its pages, loads, stores, merge and the multi-value overlay; `memory/mod.rs` indexes its files. |
//! | `state/` | `RustSimState` (construction, fork/merge, registers, memory, history, inspection, export), its filesystem, snapshots and `StateMigrationPayload`; `state/mod.rs` indexes its files. |
//! | `symbolic/` | `SymContext`, `ToZ3`, the Z3 plumbing, the `RustBV` re-export of `angr_bv::SymBV`, and `concretize`; `symbolic/mod.rs` indexes its files. |
//! | `test_boundary_values.rs` | The boundary-value tables for the integer-overflow sweep; test-only, widened by feature `testing`. |
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![warn(unreachable_pub)]

// `test_submod!` and `silent_default!` are `#[macro_export]`ed by angr_vex
// (their docs live in native/angr-vex/src/lib.rs); `#[macro_use]` ahead of
// the `mod` declarations keeps every invocation unqualified.
#[macro_use]
extern crate angr_vex;

/// Nanoseconds in `d`, saturating at `u64::MAX` instead of wrapping.
///
/// Every timing counter in the engine is a `u64` (they accumulate in
/// `AtomicU64`/`u64` fields and are exported to Python through `stats()`), but
/// `Duration::as_nanos` returns `u128`. The obvious `as u64` narrowing is
/// silent even under `[profile.release-checked]` — `as` never panics — so a
/// span past `u64::MAX` ns would report a near-zero duration rather than a huge
/// one. Saturating keeps the impossible case implausible-looking instead of
/// wrong, and gives the narrowing one documented home rather than one
/// undocumented cast per timer (angr-0jh0j.80). Per the overflow rules
/// (`invariant-overflow-fix-refuse-not-saturate-identities`) a *size*-like
/// quantity must refuse rather than
/// saturate; a monotonic timer has no such identity to corrupt, and a pinned
/// `u64::MAX` is the loudest thing a counter can say.
///
/// The bound is ~584 years, so no real process reaches it; this is
/// defensiveness and documentation, not a live bug fix.
pub fn duration_ns(d: std::time::Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// Nanoseconds elapsed since `start`, saturating at `u64::MAX`.
///
/// The `Instant` spelling of [`duration_ns`], which carries the rationale.
/// Timing sites should call this rather than narrowing `as_nanos()`'s `u128`
/// with a bare `as u64`.
pub fn elapsed_ns(start: std::time::Instant) -> u64 {
    duration_ns(start.elapsed())
}

/// Wrap a value in an `Arc` whose inner type is deliberately not `Send`/`Sync`.
///
/// The engine's core shared types (`RustBV`, Z3 AST handles, `FileDescriptor`)
/// are `!Send` by design, and that is what makes the suppression sound — NOT
/// single-threadedness. The engine does step states on real OS worker threads
/// (`exploration::scheduler_pool::worker_thread` in the engine shell, spawned
/// as `angr-worker-N`); what keeps these `Arc`s confined to one thread is that
/// the only cross-thread transport for a state is `StateMigrationPayload`
/// (`state/migration.rs`), which is `Send`-by-construction and guarded at
/// compile time by an `assert_send::<..>()` check. There is no
/// `unsafe impl Send`/`Sync` in production code, so an `arc_shared` `Arc`
/// cannot escape its worker except by round-tripping through
/// `detach_for_migration`/`reattach`, which rebuilds the shared state on the
/// destination thread. Contributors adding a new cross-thread share must go
/// through that payload — do not widen this suppression to cover an `Arc` that
/// is genuinely handed between threads.
///
/// The `Arc`s over them are load-bearing: they give O(1) copy-on-write
/// sharing across `fork()`/snapshot siblings, which `Rc` could not without
/// leaking `!Send` through the public fork/snapshot API surface. Clippy's
/// `arc_with_non_send_sync` flags every such `Arc::new` as a design smell;
/// routing them through this generic helper documents the decision in ONE place
/// instead of a per-call-site `#[allow]`. The lone `#[allow]` below is that
/// single, deliberate suppression (angr-inieg.1).
#[allow(clippy::arc_with_non_send_sync)]
#[inline]
pub fn arc_shared<T>(value: T) -> std::sync::Arc<T> {
    std::sync::Arc::new(value)
}

pub mod arch;
pub mod memory;
pub mod state;
pub mod symbolic;

/// Shared boundary-value tables for the integer-overflow/wraparound proactive
/// test sweep (Harness 6, see the module doc for the full rationale). Not
/// tied to any one subsystem — call sites span `memory`, `state::filesystem`,
/// `syscalls`, `procedures`, `interpreter`, `symbolic` and `segmentlist` — so
/// it is declared once at a crate root rather than nested under any of them.
/// Feature `testing` widens it for the engine shell's tests.
#[cfg(any(test, feature = "testing"))]
pub mod test_boundary_values;
