//! Register-access surface of `RustSimState` (`state/registers.rs`).
//!
//! The focus is `set_register`'s IP-detection branch. That branch exists to fix
//! a shipped regression (angr-4rq7): a write that lands on the architecture's
//! IP offset must also refresh the cached `self.pc`, or the register file holds
//! the real target while `pc()` keeps a stale value — usually `0` on a
//! freshly-forked state, which surfaced as `Lift error at 0x0` even though
//! `get_register("rip")` read back correctly.
//!
//! Until angr-5mnx3.37 that branch had no direct test: every `set_register`
//! call site in the crate's tests wrote `rax`/`rbx`/`fs_const`/`gs_const`, and
//! the IP sync was only ever reached indirectly through `set_ip`/`set_pc`.

use super::super::*;

/// The canonical angr-4rq7 shape: `state.regs.ip = target` arrives here as
/// `set_register("rip", ...)` on a forked state whose `pc` is still the
/// parent's, and must move `pc` with it.
#[test]
fn test_set_register_rip_syncs_pc() {
    let mut parent = RustSimState::new("amd64").unwrap();
    parent.set_pc(0x1000);
    let mut child = parent.fork();

    assert!(child.set_register("rip", RustBV::concrete(0x40_1000, 64)));

    assert_eq!(child.pc(), 0x40_1000);
    assert_eq!(child.get_register("rip").unwrap().as_u64(), Some(0x40_1000));
    // The parent is untouched — the sync is a plain field write, not shared.
    assert_eq!(parent.pc(), 0x1000);
}

/// `pc` is an alias name for the same offset on amd64, so it takes the same
/// branch: detection is by offset, not by spelling.
#[test]
fn test_set_register_pc_alias_syncs_pc() {
    let mut state = RustSimState::new("amd64").unwrap();

    assert!(state.set_register("pc", RustBV::concrete(0x40_2000, 64)));

    assert_eq!(state.pc(), 0x40_2000);
}

/// 32-bit arch: `eip` is x86's IP register, so the same branch must fire there.
#[test]
fn test_set_register_x86_eip_syncs_pc() {
    let mut state = RustSimState::new("x86").unwrap();

    assert!(state.set_register("eip", RustBV::concrete(0x0804_8000, 32)));

    assert_eq!(state.pc(), 0x0804_8000);
}

/// A non-IP register must leave `pc` alone — the branch is guarded, not
/// unconditional.
#[test]
fn test_set_register_non_ip_leaves_pc() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(0x1000);

    assert!(state.set_register("rax", RustBV::concrete(0x40_3000, 64)));

    assert_eq!(state.pc(), 0x1000);
}

/// On amd64 `eip` is a 4-byte alias of `rip`'s offset, so writing it takes the
/// IP branch — and the resync re-reads the *full* 8-byte IP rather than reusing
/// the 32-bit value written, so the surviving high half stays in `pc`.
#[test]
fn test_set_register_amd64_eip_alias_resyncs_full_width_pc() {
    let mut state = RustSimState::new("amd64").unwrap();
    assert!(state.set_register("rip", RustBV::concrete(0x1_0000_0000, 64)));
    assert_eq!(state.pc(), 0x1_0000_0000);

    assert!(state.set_register("eip", RustBV::concrete(0x2000, 32)));

    assert_eq!(state.pc(), 0x1_0000_2000);
}

/// A refused write (wrong width, so `put_reg` returns false) must not move
/// `pc`: the sync is gated on the write having actually happened.
#[test]
fn test_set_register_width_mismatch_refuses_and_leaves_pc() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(0x1000);

    // `rip` is 8 bytes on amd64; a 32-bit value is rejected outright.
    assert!(!state.set_register("rip", RustBV::concrete(0x40_4000, 32)));

    assert_eq!(state.pc(), 0x1000);
    assert_eq!(state.get_register("rip").unwrap().as_u64(), Some(0x1000));
}

/// An unknown name has no offset at all, so `register_offset` returning `None`
/// must read as "not the IP" rather than panicking or defaulting to offset 0.
#[test]
fn test_set_register_unknown_name_refuses_and_leaves_pc() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(0x1000);

    assert!(!state.set_register("not_a_register", RustBV::concrete(7, 64)));

    assert_eq!(state.pc(), 0x1000);
}

/// A symbolic IP write succeeds, but there is no concrete value to cache, so
/// `pc` deliberately keeps its previous value — `get_ip()` stays the source of
/// truth for the symbolic case.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_set_register_symbolic_ip_leaves_pc_unchanged() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(0x1000);
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n0irt37_rip", 64)
    };

    assert!(state.set_register("rip", sym));

    assert_eq!(state.pc(), 0x1000);
    assert_eq!(state.get_register("rip").unwrap().as_u64(), None);
}
