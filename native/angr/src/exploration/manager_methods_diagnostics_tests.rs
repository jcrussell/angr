// Tests for exploration/manager_methods_diagnostics.rs (see
// rust-mod-tests-sibling-extraction for why these live in a sibling file
// rather than an inline `mod tests`).
//
// `ConstraintSharingWalk` itself is unit-tested at the symbolic level; what is
// pinned here is the *manager*-level fold on top of it (angr-03vl4.25, which
// split this module out of the former `manager_methods_stats`): which states
// are counted, that ALL stashes are walked rather than just the active one,
// and that the derived `structural_duplicates` key really is
// pointers-minus-shapes.
//
// `get_solver_stats` / `reset_solver_stats` are deliberately NOT tested here:
// they front process-global Z3 counters that every other test in this binary
// mutates concurrently, so any assertion on them would be racy by
// construction.
use super::*;
use crate::stash::{STASH_ACTIVE, STASH_DEADENDED, STASH_FOUND};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Push `bv` onto `state`'s assumed-constraint export log — the log
/// `SymContext::fold_sharing_walk` reads. Goes in via
/// `assumed_constraints_push` rather than `assume_true` so the tests below do
/// no solver work at all.
#[cfg(feature = "vex-engine-z3")]
fn assume_bv(state: &RustSimState, bv: RustBV) {
    state.solver().borrow().assumed_constraints_push(bv, true);
}

/// The all-zeroes baseline. Worth pinning because every other assertion here
/// is a delta against it: if a fresh manager reported nonzero sharing, the
/// counts below would say nothing about the constraints the test pushed.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn analyze_constraint_sharing_is_all_zeroes_on_an_empty_manager() {
    let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let out = mgr.analyze_constraint_sharing();
    for key in [
        "total_visits",
        "unique_pointers",
        "unique_shapes",
        "structural_duplicates",
        "states_analyzed",
        "constraints_analyzed",
    ] {
        assert_eq!(
            out.get(key).copied(),
            Some(0),
            "empty manager must report 0 for {key}"
        );
    }
}

/// Two separately-built but structurally identical constraints on ONE state:
/// the walk must see two distinct allocations collapsing to a single shape,
/// which is exactly the `structural_duplicates` signal the analysis exists to
/// report (angr-zdho: what a construction-time hash-cons would save).
///
/// Both constraints live on the same state here; the cross-state case is
/// pinned separately below, since keeping per-state addresses comparable took
/// a fix of its own (angr-gkcxh).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn analyze_constraint_sharing_reports_structurally_duplicate_allocations() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let state = RustSimState::new("amd64").expect("state");
    assume_bv(&state, RustBV::concrete(0x1234, 64));
    assume_bv(&state, RustBV::concrete(0x1234, 64));
    mgr.sm.push(STASH_ACTIVE, state);

    let out = mgr.analyze_constraint_sharing();
    assert_eq!(out.get("states_analyzed").copied(), Some(1));
    assert_eq!(out.get("constraints_analyzed").copied(), Some(2));
    assert_eq!(out.get("total_visits").copied(), Some(2));
    assert_eq!(
        out.get("unique_pointers").copied(),
        Some(2),
        "the two constraints are distinct allocations"
    );
    assert_eq!(
        out.get("unique_shapes").copied(),
        Some(1),
        "...that share one structural shape"
    );
    assert_eq!(
        out.get("structural_duplicates").copied(),
        Some(1),
        "structural_duplicates is unique_pointers - unique_shapes"
    );
}

/// The cross-state counterpart of the test above (angr-gkcxh): each state is
/// folded from its own short-lived `get_assumed_constraints` clone, so a walk
/// that let those temporaries die would score later states' nodes as
/// already-seen and report far fewer than `n` of each. Enough states to make
/// address reuse near-certain if it can happen at all.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn analyze_constraint_sharing_counts_every_state_not_just_the_first() {
    const N: u128 = 32;
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    for value in 0..N {
        let state = RustSimState::new("amd64").expect("state");
        assume_bv(&state, RustBV::concrete(value, 64));
        mgr.sm.push(STASH_ACTIVE, state);
    }

    let out = mgr.analyze_constraint_sharing();
    assert_eq!(out.get("states_analyzed").copied(), Some(N as u64));
    assert_eq!(out.get("constraints_analyzed").copied(), Some(N as u64));
    assert_eq!(
        out.get("unique_pointers").copied(),
        Some(N as u64),
        "one live node per state, none reusing a dropped predecessor's address"
    );
    assert_eq!(
        out.get("unique_shapes").copied(),
        Some(N as u64),
        "the constraint values are all distinct"
    );
    assert_eq!(out.get("structural_duplicates").copied(), Some(0));
}

/// The two population rules in the fold: a constraint-free state is skipped
/// entirely (it contributes to neither counter), and EVERY stash is walked —
/// not just `active`. The latter is what makes the number deterministic
/// across exploration outcomes, so a state parked in `found`/`deadended` must
/// count exactly like an active one.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn analyze_constraint_sharing_walks_all_stashes_and_skips_empty_states() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let active = RustSimState::new("amd64").expect("state");
    assume_bv(&active, RustBV::concrete(1, 64));
    mgr.sm.push(STASH_ACTIVE, active);

    let found = RustSimState::new("amd64").expect("state");
    assume_bv(&found, RustBV::concrete(2, 64));
    assume_bv(&found, RustBV::concrete(3, 64));
    mgr.sm.push(STASH_FOUND, found);

    // No constraints at all: must not be counted as an analyzed state.
    mgr.sm
        .push(STASH_DEADENDED, RustSimState::new("amd64").expect("state"));

    let out = mgr.analyze_constraint_sharing();
    assert_eq!(
        out.get("states_analyzed").copied(),
        Some(2),
        "the constraint-free state is skipped"
    );
    assert_eq!(
        out.get("constraints_analyzed").copied(),
        Some(3),
        "the non-active stash's two constraints are folded in too"
    );
}
