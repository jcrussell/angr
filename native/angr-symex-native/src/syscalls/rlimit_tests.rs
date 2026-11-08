//! Tests for syscalls/rlimit.rs (extracted from inline mod tests).

use super::*;
use crate::memory::Permission;

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
    let outcome = h.call(
        &mut state,
        &[
            RustBV::symbol("resource_sym", 64),
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
    let outcome = h.call(
        &mut state,
        &[
            RustBV::concrete(RLIMIT_STACK as u128, 64),
            RustBV::symbol("rlim_sym", 64),
        ],
    );
    let err = outcome.expect_err("must error on symbolic rlim");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

/// Round-trip: write the rlim struct, load it back, confirm
/// rlim_cur is the documented constant and rlim_max stays
/// symbolic.
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

/// `rlim` is unchecked `extract_concrete_arg` output, so `rlim + 8` can
/// overflow. Both the top page and page 0 are mapped so the handler reaches
/// the `rlim_max` store; it must wrap to 0, not panic under CI's
/// `release-checked` (overflow-checks = true) profile (angr-03vl4.66).
#[test]
fn getrlimit_rlim_near_u64_max_wraps_max_field() {
    let h = NativeGetrlimitSyscall;
    let mut state = fresh_state();
    state.map_memory(0xFFFF_FFFF_FFFF_F000, 0x1000, Permission::RW);
    state.map_memory(0, 0x1000, Permission::RW);
    let rlim = u64::MAX - 7; // rlim + 8 == 0
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(RLIMIT_STACK as u128, 64),
                RustBV::concrete(rlim as u128, 64),
            ],
        )
        .expect("must not panic on a wrapping rlim");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
    assert_eq!(
        state.memory_load(rlim, 8).expect("loadable").as_u64(),
        Some(RLIMIT_STACK_CUR as u64)
    );
    assert!(state.memory_load(0, 8).expect("loadable").is_symbolic());
}

/// Harness 6 boundary sweep (angr-03vl4.66): every value in the shared
/// `test_boundary_values` table, not just the single `u64::MAX - 7` pivot
/// above, must complete without panicking and — whenever both touched pages
/// are mapped — must land `rlim_max` at the exact wrapped address
/// `rlim.wrapping_add(8)`.
#[test]
fn getrlimit_rlim_boundary_sweep_never_panics_and_wraps_correctly() {
    let stride = 8u64;
    // Set when some sweep value's rlim_max store genuinely wrapped past
    // u64::MAX (max_addr < rlim), not merely landed unwrapped near the top
    // or was rejected outright — see the boundary_addresses doc comment for
    // why the table needs dedicated entries to ever hit this.
    let mut saw_genuine_wrap = false;
    for &rlim in &crate::test_boundary_values::boundary_addresses() {
        let mut state = fresh_state();
        for a in [
            rlim,
            rlim.wrapping_add(7),
            rlim.wrapping_add(stride),
            rlim.wrapping_add(stride + 7),
        ] {
            let page = a & !0xFFFu64;
            if state.memory().page_permissions(page >> 12).is_none() {
                state.map_memory(page, 0x1000, Permission::RW);
            }
        }

        let outcome = NativeGetrlimitSyscall.call(
            &mut state,
            &[
                RustBV::concrete(RLIMIT_STACK as u128, 64),
                RustBV::concrete(rlim as u128, 64),
            ],
        );

        match outcome {
            Ok(SyscallOutcome::Continue { ret }) => {
                assert_eq!(ret, 0, "rlim={rlim:#x} getrlimit must succeed");
                let cur = state
                    .memory_load(rlim, 8)
                    .unwrap_or_else(|e| panic!("rlim={rlim:#x} rlim_cur unreadable: {e:?}"));
                assert_eq!(
                    cur.as_u64(),
                    Some(RLIMIT_STACK_CUR as u64),
                    "rlim={rlim:#x} rlim_cur must be the concrete 8 MiB constant"
                );
                let max_addr = rlim.wrapping_add(stride);
                let max = state.memory_load(max_addr, 8).unwrap_or_else(|e| {
                    panic!(
                        "rlim={rlim:#x} rlim_max at wrapped addr {max_addr:#x} unreadable: {e:?}"
                    )
                });
                assert!(
                    max.is_symbolic(),
                    "rlim={rlim:#x} rlim_max must be the fresh symbolic write at the \
                     wrapped address {max_addr:#x}, not silently skipped"
                );
                if max_addr < rlim {
                    saw_genuine_wrap = true;
                }
            }
            Ok(other) => panic!("rlim={rlim:#x} unexpected outcome {other:?}"),
            // A legitimately unreachable neighbor page is fine — the
            // property under test is "never panics", not "always succeeds".
            Err(_) => {}
        }
    }
    assert!(
        saw_genuine_wrap,
        "boundary sweep never exercised a genuine rlim_max wraparound (max_addr < rlim) — \
         the table may have regressed to only far-from-top (no field wraps) or \
         right-at-top (first field already overflows) values"
    );
}

/// `struct rlimit` is two `unsigned long`s: 8 bytes on a 32-bit ABI
/// (angr-vmi2l.152). Both fields are 32 bits wide at a 4-byte stride,
/// in the state's endianness, and nothing past `rlim + 8` is written.
#[test]
fn getrlimit_rlimit_stack_uses_32bit_layout_on_32bit_abis() {
    let states = [
        RustSimState::new("x86").expect("x86 state"),
        RustSimState::new("armel").expect("armel state"),
        RustSimState::new_with_endian("mips32", Some(false)).expect("mips32 BE state"),
    ];
    for mut state in states {
        let arch = state.arch().name();
        state.map_memory(0x4000, 0x1000, Permission::RW);
        // Sentinel the 16 bytes a 64-bit-wide write would cover.
        state
            .memory_store(0x4000, RustBV::concrete(u128::MAX, 128))
            .expect("prefill");

        let outcome = NativeGetrlimitSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(RLIMIT_STACK as u128, 32),
                    RustBV::concrete(0x4000, 32),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));

        // `memory_load` decodes in the state's endianness.
        let cur = state.memory_load(0x4000, 4).expect("rlim_cur");
        assert_eq!(cur.as_u64(), Some(RLIMIT_STACK_CUR as u64), "rlim_cur on {arch}");
        let max = state.memory_load(0x4004, 4).expect("rlim_max");
        assert!(max.is_symbolic(), "rlim_max symbolic on {arch}");
        let past = state.memory_load(0x4008, 8).expect("past the struct");
        assert_eq!(past.as_u64(), Some(u64::MAX), "nothing written past rlim+8 on {arch}");
    }
}

// angr-vmi2l.126: each symbol is recorded under the key Python's proc mints
// it with, so the solver proxy's `get_variables` finds it.
#[test]
fn getrlimit_and_setrlimit_record_python_variable_keys() {
    let mut state = fresh_state();
    state.map_memory(0x4000, 0x1000, Permission::RW);
    let ptr = RustBV::concrete(0x4000, 64);
    let stack = RustBV::concrete(u128::from(RLIMIT_STACK), 64);
    let other = RustBV::concrete(7, 64);
    NativeGetrlimitSyscall
        .call(&mut state, &[stack, ptr.clone()])
        .unwrap();
    NativeGetrlimitSyscall
        .call(&mut state, &[other.clone(), ptr.clone()])
        .unwrap();
    NativeSetrlimitSyscall
        .call(&mut state, &[other, ptr])
        .unwrap();
    let keys: Vec<Vec<String>> = state
        .tracked_variables()
        .iter()
        .map(|v| v.key.clone())
        .collect();
    assert_eq!(
        keys,
        [
            vec!["api", "rlimit", "stack"],
            vec!["api", "rlimit", "other"],
            vec!["syscall", "?", "setrlimit"],
        ]
    );
    assert!(state.tracked_variables().iter().all(|v| v.bits == 64));
}
