//! angr-1i5h7: native `SimProcedure` dispatch must not swallow a hook that is
//! itself a find/avoid target — plus the parked-bounce flush the parallel
//! found-early-return path uses to recover such a bounce into STASH_ACTIVE.
//!
//! The shared hook fixtures ([`dispatch_hook_with`], [`FindTargetProc`]) live
//! in the parent module; the sibling `native_return` module drives the same
//! dispatch to check the return path's SP adjustment.

use super::*;

/// Control: with no find/avoid targets the hook still dispatches natively.
#[test]
fn native_hook_outside_binary_dispatches_natively() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let outcome = dispatch_hook_with(|_| {});
        assert_eq!(outcome.counters.native_calls, 1, "native proc ran");
        match outcome.ret {
            CoreReturn::Continue(succ) => {
                assert_eq!(succ.len(), 1);
                assert_eq!(
                    succ[0].0.pc(),
                    HOOK_RET,
                    "native dispatch lands at the return address"
                );
            }
            _ => panic!("expected Continue"),
        }
    });
}

/// A hook that IS a find target must bounce to Python instead of running
/// natively — otherwise the state lands at `return_addr` and the run loop's
/// find check never sees the target address (angr-1i5h7).
#[test]
fn native_hook_at_find_addr_bounces_to_python() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let outcome = dispatch_hook_with(|mgr| mgr.set_find_addrs(vec![HOOK_ADDR]));
        assert_eq!(
            outcome.counters.native_calls, 0,
            "native proc must NOT run at a find target"
        );
        match outcome.ret {
            CoreReturn::NeedsPython(bounce) => match bounce.kind {
                BounceKind::SimProcedurePython { addr, .. } => assert_eq!(addr, HOOK_ADDR),
                other => panic!("expected SimProcedurePython bounce, got {other:?}"),
            },
            _ => panic!("expected NeedsPython bounce so the find check can fire"),
        }
    });
}

/// Same guard for avoid targets.
#[test]
fn native_hook_at_avoid_addr_bounces_to_python() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let outcome = dispatch_hook_with(|mgr| mgr.set_avoid_addrs(vec![HOOK_ADDR]));
        assert_eq!(outcome.counters.native_calls, 0);
        assert!(matches!(
            outcome.ret,
            CoreReturn::NeedsPython(PendingBounce {
                kind: BounceKind::SimProcedurePython {
                    addr: HOOK_ADDR,
                    ..
                },
                ..
            })
        ));
    });
}

/// A re-enterable bounce parked in `pending_parallel_bounces` (a state living
/// in NO stash) must be recoverable into STASH_ACTIVE via
/// `flush_parked_bounces_to_active` — restoring the pc to the bounce entry so a
/// later step re-lifts the hook. The parallel found-early-return path calls
/// this so `active_count` reaches parity with the serial loop instead of
/// stranding the queue when `num_find` is hit mid-wave (angr-ph300.8).
#[test]
fn flush_parked_bounce_recovers_reenterable_state_to_active() {
    let mut mgr = RustExplorationManager::new("amd64", None).unwrap();

    // Park a re-enterable SimProcedurePython bounce whose entry addr differs
    // from the state's current pc, so we can prove the flush restored it.
    const BOUNCE_ADDR: u64 = 0x4000;
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(0xdead);
    let id = state.state_id();
    mgr.pending_parallel_bounces.push((
        state,
        BounceKind::SimProcedurePython {
            addr: BOUNCE_ADDR,
            name: "sp".to_string(),
            num_args: 0,
            return_addr: 0x4010,
        },
        id, // lineage root = self
    ));

    // Stash-only view: `active_count` deliberately includes the parked bounce
    // (angr-03vl4.15), so it is `stash_count` that shows the state is in no
    // stash yet.
    assert_eq!(
        mgr.stash_count(STASH_ACTIVE),
        0,
        "parked bounce is in NO stash"
    );

    mgr.flush_parked_bounces_to_active();

    assert!(
        mgr.pending_parallel_bounces.is_empty(),
        "re-enterable bounce drained from the parked queue"
    );
    assert_eq!(mgr.active_count(), 1, "flushed back to STASH_ACTIVE");
    let active = mgr.sm.get(STASH_ACTIVE).expect("active stash exists");
    assert_eq!(active[0].state_id(), id);
    assert_eq!(
        active[0].pc(),
        BOUNCE_ADDR,
        "pc restored to the bounce entry for faithful replay"
    );
}
