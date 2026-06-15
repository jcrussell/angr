//! Unit tests for `arch_prctl.rs` — extracted as a `#[path]` sibling to keep
//! the handler module lean (see `rust-mod-tests-sibling-extraction`).

use super::*;
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

fn fresh_state() -> RustSimState {
    RustSimState::new("amd64").expect("amd64 state")
}

#[test]
fn arch_set_fs_writes_fs_const() {
    let h = NativeArchPrctlSyscall;
    let mut state = fresh_state();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(ARCH_SET_FS as u128, 64),
                RustBV::concrete(0xDEAD_BEEF, 64),
            ],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
    let fs = state.get_register("fs_const").expect("fs_const readable");
    assert_eq!(fs.as_u64(), Some(0xDEAD_BEEF));
}

#[test]
fn arch_set_gs_writes_gs_const() {
    let h = NativeArchPrctlSyscall;
    let mut state = fresh_state();
    h.call(
        &mut state,
        &[
            RustBV::concrete(ARCH_SET_GS as u128, 64),
            RustBV::concrete(0x1234_5678, 64),
        ],
    )
    .expect("ok");
    let gs = state.get_register("gs_const").expect("gs_const readable");
    assert_eq!(gs.as_u64(), Some(0x1234_5678));
}

#[test]
fn arch_get_fs_stores_fs_const_at_addr() {
    let h = NativeArchPrctlSyscall;
    let mut state = fresh_state();
    // Pre-set fs_const so we have a known value to read back.
    state.set_register("fs_const", RustBV::concrete(0xCAFE_BABE, 64));
    // Map the destination page so memory_store succeeds.
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(ARCH_GET_FS as u128, 64),
                RustBV::concrete(0x4000, 64),
            ],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
    let stored = state.memory_load(0x4000, 8).expect("loadable");
    assert_eq!(stored.as_u64(), Some(0xCAFE_BABE));
}

#[test]
fn arch_get_gs_stores_gs_const_at_addr() {
    let h = NativeArchPrctlSyscall;
    let mut state = fresh_state();
    state.set_register("gs_const", RustBV::concrete(0xFEED_FACE, 64));
    state.map_memory(0x5000, 0x1000, Permission::RW);

    h.call(
        &mut state,
        &[
            RustBV::concrete(ARCH_GET_GS as u128, 64),
            RustBV::concrete(0x5000, 64),
        ],
    )
    .expect("ok");
    let stored = state.memory_load(0x5000, 8).expect("loadable");
    assert_eq!(stored.as_u64(), Some(0xFEED_FACE));
}

#[test]
fn unknown_code_returns_einval() {
    let h = NativeArchPrctlSyscall;
    let mut state = fresh_state();
    let outcome = h
        .call(
            &mut state,
            &[RustBV::concrete(0x9999, 64), RustBV::concrete(0, 64)],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, EINVAL),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn symbolic_code_falls_back() {
    let h = NativeArchPrctlSyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let sym = RustBV::symbolic(&ctx, "code", 64);
    let err = h
        .call(&mut state, &[sym, RustBV::concrete(0, 64)])
        .expect_err("must fall back");
    match err {
        SyscallError::SymbolicArgument(msg) => assert!(
            msg.contains("code"),
            "should name the symbolic arg, got {msg:?}"
        ),
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
}

#[test]
fn get_with_symbolic_addr_falls_back() {
    let h = NativeArchPrctlSyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let sym_addr = RustBV::symbolic(&ctx, "addr", 64);
    let err = h
        .call(
            &mut state,
            &[RustBV::concrete(ARCH_GET_FS as u128, 64), sym_addr],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

#[test]
fn get_unmapped_addr_falls_back() {
    let h = NativeArchPrctlSyscall;
    let mut state = fresh_state();
    // No page mapped at 0x9000_0000 → store fails.
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(ARCH_GET_FS as u128, 64),
                RustBV::concrete(0x9000_0000, 64),
            ],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Memory(_)));
}

#[test]
fn handler_metadata() {
    let h = NativeArchPrctlSyscall;
    assert_eq!(h.name(), "arch_prctl");
    assert_eq!(h.num_args(), 2);
}
