//! Unit tests for [`super::brk`] — split out of brk.rs (see angr-syic vein).

use super::*;
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

const DEFAULT_BRK: u64 = 0x1B0_0000;

fn fresh_state() -> RustSimState {
    RustSimState::new("amd64").expect("amd64 state")
}

#[test]
fn default_brk_is_0x1b00000() {
    let state = fresh_state();
    assert_eq!(state.posix_brk(), DEFAULT_BRK);
}

#[test]
fn brk_zero_returns_current_brk() {
    let h = NativeBrkSyscall;
    let mut state = fresh_state();
    let outcome = h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_BRK),
        _ => panic!("expected Continue"),
    }
    assert_eq!(
        state.posix_brk(),
        DEFAULT_BRK,
        "posix_brk unchanged on query"
    );
}

#[test]
fn brk_below_current_is_noop() {
    let h = NativeBrkSyscall;
    let mut state = fresh_state();
    // Set brk to something larger first, then ask for a smaller value.
    state.set_posix_brk(0x1B0_4000);
    let outcome = h
        .call(&mut state, &[RustBV::concrete(0x1B0_2000, 64)])
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0x1B0_4000),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.posix_brk(), 0x1B0_4000, "posix_brk unchanged");
}

#[test]
fn brk_grow_within_same_page_does_not_map() {
    let h = NativeBrkSyscall;
    let mut state = fresh_state();
    // Default is 0x1B00000 (page-aligned). Grow within the same
    // page (offset 0..0x800).
    state.set_posix_brk(0x1B0_0010);
    let outcome = h
        .call(&mut state, &[RustBV::concrete(0x1B0_0800, 64)])
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0x1B0_0800),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.posix_brk(), 0x1B0_0800);
    // No new pages should have been mapped (since both bumps live in
    // page 0x1B00 and we never touched it before).
    assert!(state.memory().page_permissions(DEFAULT_BRK >> 12).is_none());
}

#[test]
fn brk_grow_across_page_boundary_maps_pages() {
    let h = NativeBrkSyscall;
    let mut state = fresh_state();
    // current = default (0x1B00000); grow to 0x1B03000 → maps pages
    // 0x1B00, 0x1B01, 0x1B02 (3 new pages).
    let outcome = h
        .call(&mut state, &[RustBV::concrete(0x1B0_3000, 64)])
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0x1B0_3000),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.posix_brk(), 0x1B0_3000);
    for pn in [0x1B00, 0x1B01, 0x1B02] {
        assert_eq!(
            state.memory().page_permissions(pn),
            Some(Permission::RWX),
            "page_num {:#x} should be mapped RWX",
            pn,
        );
    }
    // Page after the new break must remain unmapped.
    assert!(state.memory().page_permissions(0x1B03).is_none());
}

#[test]
fn brk_grow_then_grow_only_maps_new_pages() {
    let h = NativeBrkSyscall;
    let mut state = fresh_state();

    // First grow: 0x1B00000 → 0x1B01000 (maps page 0x1B00).
    h.call(&mut state, &[RustBV::concrete(0x1B0_1000, 64)])
        .expect("ok");
    // Mutate the freshly-mapped page perms to a sentinel so we can
    // detect if the second grow accidentally re-maps over it.
    state
        .memory_mut()
        .set_page_permissions(0x1B00, Permission::R);

    // Second grow: 0x1B01000 → 0x1B02000 (maps page 0x1B01).
    h.call(&mut state, &[RustBV::concrete(0x1B0_2000, 64)])
        .expect("ok");

    // Page 0x1B00 should still have its sentinel R perms (not re-mapped).
    assert_eq!(
        state.memory().page_permissions(0x1B00),
        Some(Permission::R),
        "first page must not be re-mapped",
    );
    assert_eq!(
        state.memory().page_permissions(0x1B01),
        Some(Permission::RWX),
        "second page should be newly mapped RWX",
    );
}

#[test]
fn brk_collision_falls_back_to_python() {
    let h = NativeBrkSyscall;
    let mut state = fresh_state();
    // Pre-map a page in the path of the brk grow so the handler
    // detects the collision and returns Err.
    state.map_memory(0x1B0_1000, 0x1000, Permission::RW);

    let err = h
        .call(&mut state, &[RustBV::concrete(0x1B0_2000, 64)])
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
    // posix_brk MUST NOT have changed on fallback (Python will run).
    assert_eq!(state.posix_brk(), DEFAULT_BRK);
    // Pre-existing mapping must be intact.
    assert_eq!(
        state.memory().page_permissions(0x1B01),
        Some(Permission::RW),
    );
}

#[test]
fn symbolic_arg_falls_back() {
    let h = NativeBrkSyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let sym = RustBV::symbolic(&ctx, "new_brk", 64);
    let err = h.call(&mut state, &[sym]).expect_err("must fall back");
    match err {
        SyscallError::SymbolicArgument(msg) => assert!(
            msg.contains("new_brk"),
            "message should name the symbolic arg, got {msg:?}",
        ),
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
    assert_eq!(state.posix_brk(), DEFAULT_BRK);
}

#[test]
fn handler_metadata() {
    let h = NativeBrkSyscall;
    assert_eq!(h.name(), "brk");
    assert_eq!(h.num_args(), 1);
}

#[test]
fn fork_preserves_posix_brk() {
    let mut state = fresh_state();
    state.set_posix_brk(0x1B0_5000);
    let forked = state.fork();
    assert_eq!(forked.posix_brk(), 0x1B0_5000);
    // Mutating the parent must not affect the fork.
    state.set_posix_brk(0x1B0_9000);
    assert_eq!(forked.posix_brk(), 0x1B0_5000);
}
