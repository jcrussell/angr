//! amd64 time (201), gettimeofday (96), and clock_gettime (228) syscall handlers.
//!
//! Mirror `procedures/posix/sim_time.py`. Each writes a fresh symbolic
//! `struct timeval` / `struct timespec` to the user-supplied pointer:
//!
//! * `gettimeofday(tv, tz)`:
//!   - `tv == 0` → return -1.
//!   - else write `tv_sec` (8B) + `tv_usec` (8B) at `*tv`, return 0.
//!   - `tz` is intentionally ignored (matches the Python procedure).
//!
//! * `clock_gettime(which_clock, ts)`:
//!   - `which_clock != 0` (CLOCK_REALTIME) → fall back to Python (its
//!     procedure raises `SimProcedureError`).
//!   - `ts == 0` → return -1.
//!   - else write `tv_sec` (8B) + `tv_nsec` (8B) at `*ts`, return 0.
//!
//! Notes:
//! * Python uses `state.solver.BVS(name, bits, key=("api", ...))`. The
//!   `key` enables Python's fresh-bvs deduplication when called from the
//!   *same* SimProcedure instance — angr generates a fresh symbol per
//!   call site otherwise. Native handlers don't have that machinery, so
//!   we always create a fresh `RustBV::symbolic(...)` per invocation.
//!   This mildly differs from Python's intra-procedure deduplication but
//!   matches its inter-call behavior, which is what binaries observe.
//! * `USE_SYSTEM_TIMES`: Python's path that returns `int(time.time())` is
//!   not implemented natively; the SimOption set is currently held on
//!   the Python proxy (see angr-t3l3) and not visible here. We always
//!   use the symbolic path — same default as `auto_load_libs=False`
//!   exploration today. The option is therefore warn-once *rejected* on
//!   the Python side (`_REJECTED_OPTION_NAMES` in
//!   `angr/exploration/rust_manager.py`, policy (b)) so a user who opts
//!   into concrete host times learns these handlers are ignoring it
//!   rather than silently exploring a symbolic-time path (angr-0y0v).
//! * Symbolic args (`tv`, `ts`, `which_clock`) fall back to Python.
//! * Unmapped destination pages cause `state.memory_store` to error,
//!   which we propagate as `SyscallError::Other` so Python (which
//!   auto-faults pages via the default plugin) can handle the store.

use super::require_syscall_args;
use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg, fresh_symbolic};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

use super::errno::NEG_ONE;

const CLOCK_REALTIME: u64 = 0;

pub(crate) struct NativeGettimeofdaySyscall;

impl NativeSyscall for NativeGettimeofdaySyscall {
    fn name(&self) -> &'static str {
        "gettimeofday"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        require_syscall_args!(self, args);
        let tv = extract_concrete_arg(&args[0], "gettimeofday tv")?;
        // tz is intentionally not extracted; Python ignores it too.

        if tv == 0 {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }

        let bits = state.arch().bits();
        // Bind the borrow to a local so it ends before we hit memory_store
        // (which takes a mutable borrow of the solver internally).
        let (tv_sec, tv_usec) = {
            let ctx = state.solver().borrow();
            (
                fresh_symbolic(&ctx, "tv_sec", bits),
                fresh_symbolic(&ctx, "tv_usec", bits),
            )
        };
        let stride = (bits / 8) as u64;
        state.memory_store(tv, tv_sec)?;
        // `tv` is unchecked `extract_concrete_arg` output; wrap explicitly so
        // the near-`u64::MAX` case behaves the same in the shipped
        // (overflow-checks-off) build and under CI's `release-checked` profile
        // (`invariant-proc-address-arith-wrapping`).
        state.memory_store(tv.wrapping_add(stride), tv_usec)?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

/// time(pointer) — return a fresh symbolic time_t in rax, optionally
/// store it at *pointer. Mirrors `procedures/linux_kernel/time.py`:
///
/// * `result := BVS("sys_time", arch.bits)`
/// * if `state.last_time` is `Some(prev)`: constrain `result.SGE(prev)`
/// * else: constrain `result.SGE(0)`
/// * `state.last_time := result`
/// * if `pointer != 0` (concrete): store `result` at `*pointer`
/// * return `result` via rax (ContinueSymbolic)
///
/// Symbolic `pointer`: fall back to Python so its `condition=(pointer != 0)`
/// store logic runs. (Most binaries pass NULL or a concrete stack address.)
/// Build a fresh symbolic `time_t`, apply the monotonic constraint
/// (`>= last_time`, or `>= 0` on the first call), and record it as the new
/// `last_time`. Shared by the `time` *syscall* (`NativeTimeSyscall`) and the
/// libc `time` *procedure* (`procedures::time::NativeTime`), which forward to
/// the same model — Python's `procedures/libc/time.py` `inline_call`s
/// `linux_kernel/time`, so both engines must produce identical behavior.
///
/// The caller performs the optional `*pointer` store (its error type differs
/// between the two callers — `SyscallError` vs `ProcedureError`) and wraps the
/// returned BV in the appropriate outcome.
pub(crate) fn fresh_monotonic_time(state: &mut RustSimState) -> RustBV {
    let bits = state.arch().bits();
    let (sys_time, monotonic_constraint) = {
        let ctx = state.solver().borrow();
        let sys_time = fresh_symbolic(&ctx, "sys_time", bits);
        let zero = RustBV::concrete(0, bits);
        // Monotonic constraint: sys_time >= last_time (or >= 0 first call).
        let lower = state.last_time().cloned().unwrap_or(zero);
        let cmp = sys_time.sge(&lower, &ctx);
        (sys_time, cmp)
    };
    state.add_constraint(monotonic_constraint);
    state.set_last_time(sys_time.clone());
    sys_time
}

pub(crate) struct NativeTimeSyscall;

impl NativeSyscall for NativeTimeSyscall {
    fn name(&self) -> &'static str {
        "time"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        require_syscall_args!(self, args);
        let pointer = extract_concrete_arg(&args[0], "time pointer")?;

        let sys_time = fresh_monotonic_time(state);

        if pointer != 0 {
            state.memory_store(pointer, sys_time.clone())?;
        }

        Ok(SyscallOutcome::ContinueSymbolic { ret: sys_time })
    }
}

pub(crate) struct NativeClockGettimeSyscall;

impl NativeSyscall for NativeClockGettimeSyscall {
    fn name(&self) -> &'static str {
        "clock_gettime"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        require_syscall_args!(self, args);
        let which_clock = extract_concrete_arg(&args[0], "clock_gettime which_clock")?;
        if which_clock != CLOCK_REALTIME {
            // Python raises SimProcedureError for non-REALTIME clocks; let
            // it run so the same diagnostic path fires.
            return Err(SyscallError::Other(format!(
                "clock_gettime: unsupported clock {which_clock}"
            )));
        }
        let ts = extract_concrete_arg(&args[1], "clock_gettime ts")?;
        if ts == 0 {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }

        let bits = state.arch().bits();
        let (tv_sec, tv_nsec) = {
            let ctx = state.solver().borrow();
            (
                fresh_symbolic(&ctx, "tv_sec", bits),
                fresh_symbolic(&ctx, "tv_nsec", bits),
            )
        };
        let stride = (bits / 8) as u64;
        state.memory_store(ts, tv_sec)?;
        // Same unchecked-pointer wrapping rule as `gettimeofday` above.
        state.memory_store(ts.wrapping_add(stride), tv_nsec)?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

test_submod!(z3 "sim_time_tests.rs" => sim_time_tests);
