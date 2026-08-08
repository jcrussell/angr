//! Native (Rust) SimProcedure dispatch classification.
//!
//! Split out of `exploration::helpers` (angr-9ke6b.76). [`dispatch_native_proc`]
//! decides whether a hooked address is served by a native procedure or bounced
//! to Python, and returns a [`NativeProcDisposition`] so the caller can do the
//! register/PC landing after the procedure-registry borrow is released. Shared
//! by the serial (`run_loop.rs`) and parallel (`core_outcome_handlers.rs`) arms
//! so the two cannot drift.
//!
//! **Panic policy (angr-qwyti.11):** carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

/// The address a state should report to Python, with the angr-4rq7 pc==0
/// fallback applied.
///
/// `self.pc` is stale at 0 for states produced by register-file-replace paths
/// that never round-trip through `set_pc` — notably forked successors under the
/// register-proxy write-through gate, whose proxy is bound to the parent
/// state_id — while the IP register already holds the real branch target. The
/// full-export path (`_snapshot_to_angr`) derives `state.addr` from the IP
/// register, so every accessor Python treats as "the state's address" must
/// agree with it or the find/avoid predicate cache keys on addr 0 and misses
/// the genuine find until a later step refreshes `pc` (angr-ph300.23).
///
/// A nonzero `self.pc` is always authoritative (the gate-off path keeps the two
/// in sync); only the `pc == 0` case falls back, and a genuinely-zero IP
/// register still reports 0.
pub(crate) fn effective_pc(state: &RustSimState) -> u64 {
    let pc = state.pc();
    if pc != 0 {
        pc
    } else {
        state.get_ip().as_u64().unwrap_or(0)
    }
}

/// What the native fast path decided, so the follow-up (register/PC/SP
/// landing, sub-call setup, Python bounce) runs after the borrow of the
/// procedure registry is released.
///
/// Both proc dispatch sites consume this: `step_one`'s serial arm
/// (`run_loop_single.rs`) and `handle_simprocedure_core`'s post-step arm
/// (`core_outcome_handlers.rs`). The landing itself is deliberately NOT shared
/// — the serial path resolves the return address through the calling
/// convention (`get_return_addr` + `pops_return_addr`, so LR/X30/$ra arches
/// work) while the core path is handed `return_addr` by the interpreter — but
/// the *decision* is, so the two cannot drift (angr-ph300.73).
pub(crate) enum NativeProcDisposition {
    Returned {
        no_return: bool,
        ret_val: Option<RustBV>,
    },
    SubCall {
        proc_name: String,
        saved_args: Vec<RustBV>,
        target: u64,
        sub_args: Vec<RustBV>,
        resume_tag: u32,
    },
    Fallback,
    /// The native proc touched an unmapped page while STRICT_PAGE_ACCESS is on:
    /// Python's `PrivilegedPagingMixin._initialize_page` would raise
    /// `SimSegfaultException`, so the state is terminal-errored natively instead
    /// of bouncing to Python just to raise. Carries the message Python formats
    /// (`"{page_addr:#x} (unmapped)"`). Only produced when the caller passes
    /// `mirror_segfault`.
    Segfault(String),
}

/// Map a native procedure error onto the `SimSegfaultException` Python would
/// raise for the same access, or `None` when Python would service it (and we
/// must therefore fall back).
///
/// Only the unmapped-page case is mirrored: with STRICT_PAGE_ACCESS off Python
/// lazily initializes the page and keeps going, so the state must still bounce.
pub(crate) fn segfault_message(state: &RustSimState, err: &ProcedureError) -> Option<String> {
    if !state.enforce_permissions() {
        return None;
    }
    match err {
        ProcedureError::Memory(crate::memory::MemoryError::Unmapped { addr, .. }) => {
            let page_addr = addr & !(crate::memory::PAGE_SIZE - 1);
            Some(format!("{page_addr:#x} (unmapped)"))
        }
        _ => None,
    }
}

/// Mutable views onto whichever counter home the caller owns — `CoreCounters`
/// for the parallel path, `profiling.native_proc_stats` for the serial one.
/// The two structs carry the same six fields under two different names
/// (`native_python_fallbacks` vs `python_fallbacks`), so a bundle of `&mut`
/// borrows is the cheapest way to let one dispatcher bump either.
pub(crate) struct NativeProcCounters<'a> {
    pub(crate) native_calls: &'a mut u64,
    pub(crate) python_fallbacks: &'a mut u64,
    pub(crate) call_counts: &'a mut HashMap<String, u64>,
    pub(crate) symbolic_fallbacks_by_name: &'a mut HashMap<String, u64>,
    pub(crate) not_implemented_fallbacks_by_name: &'a mut HashMap<String, u64>,
    pub(crate) other_fallbacks_by_name: &'a mut HashMap<String, u64>,
}

impl NativeProcCounters<'_> {
    fn fallback(&mut self, name: &str, bucket: FallbackBucket) {
        *self.python_fallbacks += 1;
        let map = match bucket {
            FallbackBucket::Symbolic => &mut *self.symbolic_fallbacks_by_name,
            FallbackBucket::NotImplemented => &mut *self.not_implemented_fallbacks_by_name,
            FallbackBucket::Other => &mut *self.other_fallbacks_by_name,
        };
        *map.entry(name.to_string()).or_insert(0) += 1;
    }

    fn native_call(&mut self, name: &str) {
        *self.native_calls += 1;
        *self.call_counts.entry(name.to_string()).or_insert(0) += 1;
    }
}

enum FallbackBucket {
    Symbolic,
    NotImplemented,
    Other,
}

/// Run one native SimProcedure and classify the result, bumping the per-proc
/// counters the same way on both dispatch paths.
///
/// Callers own the parts that genuinely differ:
/// * argument extraction (`args`) — the serial path reads the manager's
///   calling convention, the core path a `CcSnapshot`;
/// * `no_return` — the serial path trusts the Python registration tuple, the
///   core path `native_proc.no_return()`;
/// * the [`NativeProcDisposition::SubCall`] counter bump, which is deliberately
///   NOT done here: both dispatch sites count a sub-call as a native call only
///   once `setup_native_subcall` succeeds, and book an `other` Python fallback
///   when setup fails. Leaving the bump to the caller keeps that shared
///   semantics tied to the setup outcome each caller owns;
/// * `mirror_segfault` — only the core path terminal-errors natively today;
///   `step_one` passes `false` so a strict-page-access fault still bounces to
///   Python via the ordinary fallback counters.
pub(crate) fn dispatch_native_proc(
    native_proc: &dyn crate::procedures::NativeSimProcedure,
    state: &mut RustSimState,
    name: &str,
    no_return: bool,
    args: Result<Vec<RustBV>, crate::arch::ExtractionError>,
    mirror_segfault: bool,
    counters: &mut NativeProcCounters<'_>,
) -> NativeProcDisposition {
    let args = match args {
        Err(e) => {
            log::debug!("Skipping native procedure {name} (arg extraction failed: {e:?})");
            counters.fallback(name, FallbackBucket::Other);
            return NativeProcDisposition::Fallback;
        }
        Ok(args) => args,
    };

    match native_proc.call_ex(state, &args) {
        Ok(ProcOutcome::Return(ret_val)) => {
            counters.native_call(name);
            NativeProcDisposition::Returned { no_return, ret_val }
        }
        Ok(ProcOutcome::CallAndResume {
            target,
            args: sub_args,
            resume_tag,
        }) => NativeProcDisposition::SubCall {
            proc_name: name.to_string(),
            saved_args: args,
            target,
            sub_args,
            resume_tag,
        },
        Err(e) => {
            if mirror_segfault && let Some(msg) = segfault_message(state, &e) {
                log::debug!("Native procedure {name} segfaulted: {msg}");
                counters.native_call(name);
                return NativeProcDisposition::Segfault(msg);
            }
            log::debug!("Native procedure {name} returned error, falling back to Python: {e:?}");
            counters.fallback(
                name,
                match e {
                    ProcedureError::SymbolicArgument(_) => FallbackBucket::Symbolic,
                    ProcedureError::NotImplemented => FallbackBucket::NotImplemented,
                    _ => FallbackBucket::Other,
                },
            );
            NativeProcDisposition::Fallback
        }
    }
}

test_submod!("native_proc_dispatch_tests.rs" => tests);
