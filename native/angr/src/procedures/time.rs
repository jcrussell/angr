//! Native libc `time()` (angr-ae54t.21).
//!
//! Angr's Python SimProcedure (`angr/procedures/libc/time.py`) merely
//! `inline_call`s the `linux_kernel/time` syscall and returns its result. The
//! native engine already models that syscall (`syscalls::sim_time::
//! NativeTimeSyscall`), but had no native libc `time` *procedure*, so a PLT
//! call to `time()` round-tripped to Python.
//!
//! This proc forwards to the exact same model via the shared
//! `syscalls::sim_time::fresh_monotonic_time` helper:
//!
//! * `result := BVS("sys_time", arch.bits)` constrained `>= last_time`
//!   (or `>= 0` on the first call); `last_time := result`.
//! * if `pointer != 0` (concrete): store `result` at `*pointer`.
//! * return `result` as the symbolic return value (rax).
//!
//! `pointer` is taken `concrete`, so a symbolic pointer falls back to Python —
//! matching the syscall's `extract_concrete_arg` gate, where Python's
//! `condition=(pointer != 0)` store logic runs.

use crate::syscalls::sim_time::fresh_monotonic_time;

crate::declare_proc! {
    /// ```c
    /// time_t time(time_t *tloc);
    /// ```
    /// Returns a fresh symbolic, monotonically-increasing `time_t`; stores it
    /// at `*tloc` when `tloc` is a concrete non-NULL pointer.
    name = "time",
    struct = NativeTime,
    args = [pointer: concrete],
    call |state| {
        let sys_time = fresh_monotonic_time(state);
        if pointer != 0 {
            state.memory_store(pointer, sys_time.clone())?;
        }
        Ok(Some(sys_time))
    }
}

test_submod!(z3 "time_tests.rs" => tests);
