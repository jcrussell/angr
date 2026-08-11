// Tests for exploration/callback_types.rs (angr-03vl4.28).
//
// `jumpkind_or_boring` is the single seam both Python export paths in
// `exploration::pending_api` (`_get_pending_history_and_jumpkind` and
// `_export_callback_bundle`) go through, so the `Some` / `None` split is
// pinned here rather than duplicated per call site.
//
// Gated on `vex-engine-z3` like `constraints_tests.rs`: building a
// `RustSimState` needs the solver backend.

use super::*;
use crate::state::RustSimState;

/// A callback carrying an exit reports that exit verbatim — no normalization,
/// no default substitution for an unrecognized-but-present jumpkind.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn jumpkind_or_boring_reports_the_recorded_jumpkind() {
    let state = RustSimState::new("amd64").expect("state");
    let pending = PendingCallback::with_context(
        state,
        None,
        CallbackReason::FindPredicate { addr: 0x400000 },
        "Ijk_Call",
        None,
        ForkBundle::empty(),
    );

    assert_eq!(pending.jumpkind_or_boring(), "Ijk_Call");
}

/// A lightweight (predicate-evaluation) callback has no exit of its own, and
/// must fall back to `Ijk_Boring` — the value Python's state-export path
/// expects when the field is absent.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn jumpkind_or_boring_defaults_lightweight_callbacks_to_boring() {
    let state = RustSimState::new("amd64").expect("state");
    let pending =
        PendingCallback::lightweight(state, CallbackReason::FindPredicate { addr: 0x400000 });

    assert!(pending.jumpkind.is_none(), "fixture precondition");
    assert_eq!(
        pending.jumpkind_or_boring(),
        JumpKind::Boring.ijk_name(),
        "the default must stay spelled as the VEX tag, not a hand-written literal"
    );
}
