//! Tests for syscalls/rlimit.rs (extracted from inline mod tests).

use super::*;
use crate::memory::Permission;
use crate::symbolic::SymContext;

fn fresh_state() -> RustSimState {
    RustSimState::new("amd64").expect("amd64 state")
}

// ---- getrlimit ---------------------------------------------------

#[test]
fn getrlimit_rlimit_stack_writes_rlim_struct_and_returns_zero() {
    let h = NativeGetrlimitSyscall;
    let mut state = fresh_state();
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(RLIMIT_STACK as u128, 64),
                RustBV::concrete(0x4000, 64),
            ],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        other => panic!("expected Continue, got {other:?}"),
    }

    // rlim_cur = 8388608 (concrete)
    let cur = state.memory_load(0x4000, 8).expect("loadable");
    assert_eq!(
        cur.as_u64(),
        Some(RLIMIT_STACK_CUR as u64),
        "rlim_cur should be the concrete 8 MiB constant"
    );

    // rlim_max = fresh symbolic
    let max = state.memory_load(0x4008, 8).expect("loadable");
    assert!(max.is_symbolic(), "rlim_max should be symbolic (fresh BVS)");
}

#[test]
fn getrlimit_non_stack_returns_fresh_symbolic() {
    let h = NativeGetrlimitSyscall;
    let mut state = fresh_state();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 64), // RLIMIT_FSIZE
                RustBV::concrete(0, 64),
            ],
        )
        .expect("ok");
    let ret = match outcome {
        SyscallOutcome::ContinueSymbolic { ret } => ret,
        other => panic!("expected ContinueSymbolic, got {other:?}"),
    };
    assert_eq!(ret.width(), state.arch().bits());
    assert!(ret.as_u64().is_none(), "non-STACK rlimit must be symbolic");
}

#[test]
fn getrlimit_symbolic_resource_returns_error() {
    let h = NativeGetrlimitSyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let outcome = h.call(
        &mut state,
        &[
            RustBV::symbolic(&ctx, "resource_sym", 64),
            RustBV::concrete(0x4000, 64),
        ],
    );
    let err = outcome.expect_err("must error on symbolic resource");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

#[test]
fn getrlimit_symbolic_rlim_returns_error() {
    let h = NativeGetrlimitSyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let outcome = h.call(
        &mut state,
        &[
            RustBV::concrete(RLIMIT_STACK as u128, 64),
            RustBV::symbolic(&ctx, "rlim_sym", 64),
        ],
    );
    let err = outcome.expect_err("must error on symbolic rlim");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

/// Round-trip: write the rlim struct, load it back, confirm
/// rlim_cur is the documented constant and rlim_max stays
/// symbolic. Satisfies the bd acceptance criterion "Integration
/// test confirms getrlimit/setrlimit round-trip".
#[test]
fn getrlimit_setrlimit_roundtrip() {
    let getrlimit = NativeGetrlimitSyscall;
    let setrlimit = NativeSetrlimitSyscall;
    let mut state = fresh_state();
    state.map_memory(0x5000, 0x1000, Permission::RW);

    // 1. getrlimit(RLIMIT_STACK, 0x5000) populates the struct.
    let outcome = getrlimit
        .call(
            &mut state,
            &[
                RustBV::concrete(RLIMIT_STACK as u128, 64),
                RustBV::concrete(0x5000, 64),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    let cur = state
        .memory_load(0x5000, 8)
        .expect("rlim_cur readable")
        .as_u64()
        .expect("concrete");
    assert_eq!(cur, RLIMIT_STACK_CUR as u64);

    // 2. setrlimit(RLIMIT_STACK, 0x5000) returns a fresh symbolic
    //    int (stub semantics). Round-trip in this case = the call
    //    succeeds and yields a symbolic in the return register.
    let outcome = setrlimit
        .call(
            &mut state,
            &[
                RustBV::concrete(RLIMIT_STACK as u128, 64),
                RustBV::concrete(0x5000, 64),
            ],
        )
        .expect("ok");
    let ret = match outcome {
        SyscallOutcome::ContinueSymbolic { ret } => ret,
        other => panic!("expected ContinueSymbolic, got {other:?}"),
    };
    assert!(ret.as_u64().is_none(), "setrlimit stub return is symbolic");

    // 3. Re-load — concrete bytes stick around even after the
    //    "no-op" setrlimit (Python's stub does not touch memory).
    let cur2 = state
        .memory_load(0x5000, 8)
        .expect("rlim_cur still there")
        .as_u64()
        .expect("concrete");
    assert_eq!(cur2, RLIMIT_STACK_CUR as u64);
}

// ---- setrlimit / prlimit64 stubs ---------------------------------

#[test]
fn setrlimit_returns_fresh_symbolic_on_all_arches() {
    let h = NativeSetrlimitSyscall;
    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        let args = [RustBV::concrete(0, bits), RustBV::concrete(0, bits)];
        let outcome = h.call(&mut state, &args).unwrap();
        let ret = match outcome {
            SyscallOutcome::ContinueSymbolic { ret } => ret,
            _ => unreachable!(),
        };
        assert_eq!(ret.width(), bits, "{arch}: width must match arch");
        assert!(ret.as_u64().is_none(), "{arch}: must be symbolic");
    }
}

#[test]
fn prlimit64_returns_fresh_symbolic_on_all_arches() {
    let h = NativePrlimit64Syscall;
    assert_eq!(h.num_args(), 4);
    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        let args: Vec<RustBV> = (0..4).map(|_| RustBV::concrete(0, bits)).collect();
        let outcome = h.call(&mut state, &args).unwrap();
        let ret = match outcome {
            SyscallOutcome::ContinueSymbolic { ret } => ret,
            _ => unreachable!(),
        };
        assert_eq!(ret.width(), bits);
        assert!(ret.as_u64().is_none(), "{arch}: must be symbolic");
    }
}
