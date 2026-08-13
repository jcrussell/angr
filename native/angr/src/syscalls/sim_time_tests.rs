//! Tests for `sim_time.rs` — gettimeofday / clock_gettime / time syscall handlers.

use super::*;
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

fn fresh_state() -> RustSimState {
    RustSimState::new("amd64").expect("amd64 state")
}

// ---- gettimeofday ------------------------------------------------

#[test]
fn gettimeofday_null_tv_returns_neg_one() {
    let h = NativeGettimeofdaySyscall;
    let mut state = fresh_state();
    let outcome = h
        .call(
            &mut state,
            &[RustBV::concrete(0, 64), RustBV::concrete(0, 64)],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_ONE),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn gettimeofday_writes_symbolic_struct_and_returns_zero() {
    let h = NativeGettimeofdaySyscall;
    let mut state = fresh_state();
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let outcome = h
        .call(
            &mut state,
            &[RustBV::concrete(0x4000, 64), RustBV::concrete(0, 64)],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
    // Both 8-byte slots must be symbolic (a fresh BVS per call), not
    // the zero-fill of a freshly mapped page.
    let tv_sec = state.memory_load(0x4000, 8).expect("loadable");
    let tv_usec = state.memory_load(0x4008, 8).expect("loadable");
    assert!(tv_sec.is_symbolic(), "tv_sec must be symbolic");
    assert!(tv_usec.is_symbolic(), "tv_usec must be symbolic");
}

#[test]
fn gettimeofday_tz_may_be_symbolic() {
    let h = NativeGettimeofdaySyscall;
    let mut state = fresh_state();
    state.map_memory(0x4000, 0x1000, Permission::RW);
    let ctx = SymContext::new();
    let sym_tz = RustBV::symbolic(&ctx, "tz", 64);
    let outcome = h
        .call(&mut state, &[RustBV::concrete(0x4000, 64), sym_tz])
        .expect("symbolic tz must not block fast path");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
}

#[test]
fn gettimeofday_symbolic_tv_falls_back() {
    let h = NativeGettimeofdaySyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let sym_tv = RustBV::symbolic(&ctx, "tv", 64);
    let err = h
        .call(&mut state, &[sym_tv, RustBV::concrete(0, 64)])
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

#[test]
fn gettimeofday_unmapped_dest_falls_back() {
    let h = NativeGettimeofdaySyscall;
    let mut state = fresh_state();
    // No page mapped at 0xDEAD_0000.
    let err = h
        .call(
            &mut state,
            &[RustBV::concrete(0xDEAD_0000, 64), RustBV::concrete(0, 64)],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Memory(_)));
}

// ---- clock_gettime ----------------------------------------------

#[test]
fn clock_gettime_null_ts_returns_neg_one() {
    let h = NativeClockGettimeSyscall;
    let mut state = fresh_state();
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(CLOCK_REALTIME as u128, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_ONE),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn clock_gettime_writes_symbolic_struct_and_returns_zero() {
    let h = NativeClockGettimeSyscall;
    let mut state = fresh_state();
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(CLOCK_REALTIME as u128, 64),
                RustBV::concrete(0x4000, 64),
            ],
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    let tv_sec = state.memory_load(0x4000, 8).expect("loadable");
    let tv_nsec = state.memory_load(0x4008, 8).expect("loadable");
    assert!(tv_sec.is_symbolic());
    assert!(tv_nsec.is_symbolic());
}

#[test]
fn clock_gettime_non_realtime_falls_back() {
    let h = NativeClockGettimeSyscall;
    let mut state = fresh_state();
    // CLOCK_MONOTONIC = 1; native handler must defer to Python so its
    // SimProcedureError fires (matching Python behavior).
    let err = h
        .call(
            &mut state,
            &[RustBV::concrete(1, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn clock_gettime_symbolic_clock_falls_back() {
    let h = NativeClockGettimeSyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let sym = RustBV::symbolic(&ctx, "which_clock", 64);
    let err = h
        .call(&mut state, &[sym, RustBV::concrete(0x4000, 64)])
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

#[test]
fn handler_metadata() {
    let g = NativeGettimeofdaySyscall;
    assert_eq!(g.name(), "gettimeofday");
    assert_eq!(g.num_args(), 2);
    let c = NativeClockGettimeSyscall;
    assert_eq!(c.name(), "clock_gettime");
    assert_eq!(c.num_args(), 2);
    let t = NativeTimeSyscall;
    assert_eq!(t.name(), "time");
    assert_eq!(t.num_args(), 1);
}

// ---- time --------------------------------------------------------

#[test]
fn time_null_pointer_returns_symbolic_and_does_not_store() {
    let h = NativeTimeSyscall;
    let mut state = fresh_state();
    let outcome = h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok");
    match outcome {
        SyscallOutcome::ContinueSymbolic { ret } => {
            assert!(ret.is_symbolic(), "time return must be symbolic");
            assert_eq!(ret.width(), 64);
        }
        _ => panic!("expected ContinueSymbolic"),
    }
    // last_time updated.
    assert!(state.last_time().is_some());
}

#[test]
fn time_two_calls_are_solver_distinct() {
    // Regression guard for angr-8o7w: two time() returns on one path must
    // be satisfiably unequal. A fixed Z3 name would alias both to the same
    // `new_const`, making `ret1 != ret2` unsatisfiable — fresh_symbolic
    // appends symbol_counter so each return is a distinct Z3 term.
    let h = NativeTimeSyscall;
    let mut state = fresh_state();
    let r1 = match h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok") {
        SyscallOutcome::ContinueSymbolic { ret } => ret,
        _ => panic!("expected ContinueSymbolic"),
    };
    let r2 = match h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok") {
        SyscallOutcome::ContinueSymbolic { ret } => ret,
        _ => panic!("expected ContinueSymbolic"),
    };

    let solver = z3::Solver::new();
    solver.assert(r1.to_z3_ast().eq(r2.to_z3_ast()).not());
    assert_eq!(
        solver.check(),
        z3::SatResult::Sat,
        "two time() returns must be solver-distinct"
    );
}

#[test]
fn time_writes_symbolic_to_pointer() {
    let h = NativeTimeSyscall;
    let mut state = fresh_state();
    state.map_memory(0x4000, 0x1000, Permission::RW);
    let outcome = h
        .call(&mut state, &[RustBV::concrete(0x4000, 64)])
        .expect("ok");
    let ret = match outcome {
        SyscallOutcome::ContinueSymbolic { ret } => ret,
        _ => panic!("expected ContinueSymbolic"),
    };
    let stored = state.memory_load(0x4000, 8).expect("loadable");
    assert!(stored.is_symbolic(), "stored value must be symbolic");
    // The stored value is the same BV that was returned.
    assert_eq!(ret.width(), stored.width());
}

#[test]
fn time_first_call_constrains_nonnegative() {
    let h = NativeTimeSyscall;
    let mut state = fresh_state();
    let outcome = h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok");
    let ret = match outcome {
        SyscallOutcome::ContinueSymbolic { ret } => ret,
        _ => panic!("expected ContinueSymbolic"),
    };
    // Min should be >= 0 (signed).
    let min = state.min(&ret, true).expect("min computable");
    assert!(min as i64 >= 0, "first time() must be SGE 0; got min={min}");
}

#[test]
fn time_consecutive_calls_are_monotonic() {
    let h = NativeTimeSyscall;
    let mut state = fresh_state();
    let first = match h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok") {
        SyscallOutcome::ContinueSymbolic { ret } => ret,
        _ => panic!("expected ContinueSymbolic"),
    };
    // Pin the first call to a concrete value to make the monotonicity
    // constraint testable: first == 100.
    let pin = {
        let ctx = state.solver().borrow();
        first.eq(&RustBV::concrete(100, 64), &ctx)
    };
    state.add_constraint(pin);

    let second = match h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok") {
        SyscallOutcome::ContinueSymbolic { ret } => ret,
        _ => panic!("expected ContinueSymbolic"),
    };
    // second >= first, and first is pinned to 100, so second >= 100.
    let min = state.min(&second, true).expect("min computable");
    assert!(
        min as i64 >= 100,
        "second time() must be >= first; got min={min}"
    );
}

#[test]
fn time_symbolic_pointer_falls_back() {
    let h = NativeTimeSyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let sym = RustBV::symbolic(&ctx, "ptr", 64);
    let err = h.call(&mut state, &[sym]).expect_err("must fall back");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

#[test]
fn time_unmapped_pointer_falls_back() {
    let h = NativeTimeSyscall;
    let mut state = fresh_state();
    let err = h
        .call(&mut state, &[RustBV::concrete(0xDEAD_0000, 64)])
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Memory(_)));
}

// ---- near-u64::MAX pointer regression (angr-03vl4.67) ------------
//
// `tv` / `ts` are unchecked `extract_concrete_arg` output, so the second
// field's address (`ptr + stride`) can overflow. These tests pin the wrap:
// the top page and page 0 are both mapped, so the handler reaches the
// second store and the write must land at 0 rather than panic under CI's
// `release-checked` (overflow-checks = true) profile.

/// Map the highest page and page 0 so a `u64::MAX`-adjacent write and its
/// wrapped-around sibling are both storable.
fn map_wrap_pages(state: &mut RustSimState) {
    state.map_memory(0xFFFF_FFFF_FFFF_F000, 0x1000, Permission::RW);
    state.map_memory(0, 0x1000, Permission::RW);
}

#[test]
fn gettimeofday_tv_near_u64_max_wraps_second_field() {
    let h = NativeGettimeofdaySyscall;
    let mut state = fresh_state();
    map_wrap_pages(&mut state);
    let tv = u64::MAX - 7; // tv + 8 == 0
    let outcome = h
        .call(
            &mut state,
            &[RustBV::concrete(tv as u128, 64), RustBV::concrete(0, 64)],
        )
        .expect("must not panic on a wrapping tv");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
    assert!(state.memory_load(tv, 8).expect("loadable").is_symbolic());
    assert!(state.memory_load(0, 8).expect("loadable").is_symbolic());
}

#[test]
fn clock_gettime_ts_near_u64_max_wraps_second_field() {
    let h = NativeClockGettimeSyscall;
    let mut state = fresh_state();
    map_wrap_pages(&mut state);
    let ts = u64::MAX - 7; // ts + 8 == 0
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(CLOCK_REALTIME as u128, 64),
                RustBV::concrete(ts as u128, 64),
            ],
        )
        .expect("must not panic on a wrapping ts");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
    assert!(state.memory_load(ts, 8).expect("loadable").is_symbolic());
    assert!(state.memory_load(0, 8).expect("loadable").is_symbolic());
}

/// Harness 6 boundary sweep for the fixes above: every value in the shared
/// `test_boundary_values` table — not just the single `u64::MAX - 7` pivot
/// the two tests above pin — must complete without panicking, and (with
/// every touched page mapped) must write the second field's fresh symbolic
/// BV at exactly the wrapped address `tv.wrapping_add(8)`, never silently
/// skip it or land somewhere else.
#[test]
fn gettimeofday_and_clock_gettime_tv_boundary_sweep_never_panics_and_wraps_correctly() {
    let stride = 8u64;
    // Set when some sweep value's second-field store genuinely wrapped past
    // u64::MAX (second_addr < tv), not merely landed unwrapped near the top
    // or was rejected outright — see the boundary_addresses doc comment for
    // why the table needs dedicated entries to ever hit this.
    let mut saw_genuine_wrap = false;
    for &tv in &crate::test_boundary_values::boundary_addresses() {
        for handler in ["gettimeofday", "clock_gettime"] {
            let mut state = fresh_state();
            // Map every page either 8-byte store could touch, including
            // whatever page a wraparound of the second store lands on.
            for a in [
                tv,
                tv.wrapping_add(7),
                tv.wrapping_add(stride),
                tv.wrapping_add(stride + 7),
            ] {
                let page = a & !0xFFFu64;
                if state.memory().page_permissions(page >> 12).is_none() {
                    state.map_memory(page, 0x1000, Permission::RW);
                }
            }

            let outcome = if handler == "gettimeofday" {
                NativeGettimeofdaySyscall.call(
                    &mut state,
                    &[RustBV::concrete(tv as u128, 64), RustBV::concrete(0, 64)],
                )
            } else {
                NativeClockGettimeSyscall.call(
                    &mut state,
                    &[
                        RustBV::concrete(CLOCK_REALTIME as u128, 64),
                        RustBV::concrete(tv as u128, 64),
                    ],
                )
            };

            match outcome {
                Ok(SyscallOutcome::Continue { ret }) if tv == 0 => {
                    assert_eq!(ret, NEG_ONE, "{handler}: tv=0 must return -1");
                }
                Ok(SyscallOutcome::Continue { ret }) => {
                    assert_eq!(ret, 0, "{handler}: tv={tv:#x} must succeed");
                    let second_addr = tv.wrapping_add(stride);
                    let second = state.memory_load(second_addr, 8).unwrap_or_else(|e| {
                        panic!(
                            "{handler}: tv={tv:#x} second field at wrapped addr \
                             {second_addr:#x} unreadable: {e:?}"
                        )
                    });
                    assert!(
                        second.is_symbolic(),
                        "{handler}: tv={tv:#x} second field must be the fresh symbolic \
                         write at the wrapped address {second_addr:#x}, not silently skipped"
                    );
                    if second_addr < tv {
                        saw_genuine_wrap = true;
                    }
                }
                Ok(other) => panic!("{handler}: tv={tv:#x} unexpected outcome {other:?}"),
                // A legitimately unreachable address (still-unmapped
                // neighbor page, etc.) is fine — the property under test is
                // "never panics", not "always succeeds".
                Err(_) => {}
            }
        }
    }
    assert!(
        saw_genuine_wrap,
        "boundary sweep never exercised a genuine second-field wraparound \
         (second_addr < tv) — the table may have regressed to only far-from-top \
         (no field wraps) or right-at-top (first field already overflows) values"
    );
}
