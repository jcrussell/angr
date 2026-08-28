//! Unit tests for the `&mut self`-free post-step core (angr-vh834).
//!
//! These drive a single state through [`run_post_step_core`] and assert the
//! structural outcome (successors / forks / pruned / fork_ids / routing) the
//! legacy `step_state_with_skip` match arms produced inline. The full
//! byte-identical proof is the Python `tests/engines/rust/` suite, which now
//! flows entirely through this core.
//!
//! This module holds only the fixtures more than one submodule shares; the
//! tests are split by the `RunResult` arm (or the outcome concern) they drive,
//! mirroring the angr-nbim4.3 split of `core_outcome.rs` itself into
//! `core_outcome_cc.rs` + `core_outcome_handlers.rs`:
//!
//! - `block_end` — `BlockEnd` successors, deferred-fork materialization, and
//!   the natively-forked `SymbolicBranch`.
//! - `errors` — `Error` routing per [`RunErrorKind`], plus `segfault_message`.
//! - `hooks` — angr-1i5h7: native `SimProcedure` dispatch vs find/avoid target
//!   hooks, and the parked-bounce flush.
//! - `native_return` — angr-sqfj8.37 / angr-c7xno.29: the return path's SP
//!   adjustment, per calling convention and for a symbolic SP.
//! - `syscall_dispatch` — angr-c7xno.33: `handle_syscall_core`'s four dispatch
//!   outcomes, plus `CcSnapshot::write_syscall_return`.
//! - `symbolic_jump` — angr-c7xno.33: `handle_symbolic_jump_target_core`.
//! - `native_resume` — angr-c7xno.32: the production `handle_native_resume_core`
//!   path.
//! - `stats` — [`ParallelProfiling`]'s `accumulate_step` / `drain_into`.

use super::*;

use crate::callbacks::{DeferredFork, RunErrorKind, RunResult};
use crate::exploration::RustExplorationManager;
use crate::procedures::NativeProcedureRegistry;
use crate::stash::STASH_ACTIVE;
use crate::state::RustSimState;
use crate::syscalls::NativeSyscallRegistry;

mod block_end;
mod errors;
mod hooks;
mod native_resume;
mod native_return;
mod stats;
mod symbolic_jump;
mod syscall_dispatch;

fn fresh_ctx() -> super::StepContext {
    let mgr = RustExplorationManager::new("amd64", None).unwrap();
    mgr.step_context()
}

// --- angr-1i5h7 hook fixtures, shared by `hooks` and `native_return` ---

const HOOK_ADDR: u64 = 0x50_0000;
const HOOK_RET: u64 = 0x40_2000;

/// Stands in for any zero-arg native libc proc (getenv/time/...) hooked in the
/// extern object, i.e. outside `binary_regions`, where native dispatch fires.
struct FindTargetProc;
impl crate::procedures::NativeSimProcedure for FindTargetProc {
    fn name(&self) -> &'static str {
        "find_target_test"
    }
    fn num_args(&self) -> usize {
        0
    }
    fn call(
        &self,
        state: &mut RustSimState,
        _args: &[crate::symbolic::RustBV],
    ) -> Result<Option<crate::symbolic::RustBV>, crate::procedures::ProcedureError> {
        Ok(Some(crate::symbolic::RustBV::concrete(
            0,
            state.arch().bits(),
        )))
    }
}

/// Drive one `SimProcedure` result for a hook at [`HOOK_ADDR`] with
/// `FindTargetProc` registered, under a manager configured by `cfg` (which
/// seeds find/avoid addresses). Returns the raw outcome.
fn dispatch_hook_with(cfg: impl FnOnce(&mut RustExplorationManager)) -> CoreOutcome {
    dispatch_hook_on_arch("amd64", SpSeed::Default, cfg)
}

/// How [`dispatch_hook_on_arch`] seeds the state's stack pointer.
enum SpSeed {
    /// Leave SP at whatever `RustSimState::new` produced.
    Default,
    /// Seed a concrete SP.
    Concrete(u64),
    /// Seed a fresh symbolic SP — `extract_args_with_abi` never touches SP for
    /// a zero-arg proc, so it stays symbolic through the return (angr-c7xno.29).
    Symbolic,
}

/// [`dispatch_hook_with`] on an arbitrary architecture, optionally seeding SP,
/// so the return path's SP adjustment can be checked per calling convention.
fn dispatch_hook_on_arch(
    arch: &str,
    sp: SpSeed,
    cfg: impl FnOnce(&mut RustExplorationManager),
) -> CoreOutcome {
    let mut mgr = RustExplorationManager::new(arch, None).unwrap();
    cfg(&mut mgr);
    let ctx = mgr.step_context();
    let prof = ParallelProfiling::default();
    let mut procs = NativeProcedureRegistry::new();
    procs.register(std::sync::Arc::new(FindTargetProc));
    let syscalls = NativeSyscallRegistry::new();

    let mut state = RustSimState::new(arch).unwrap();
    state.set_pc(HOOK_ADDR);
    let bits = state.arch().bits();
    match sp {
        SpSeed::Default => {}
        SpSeed::Concrete(sp) => state.set_sp(crate::symbolic::RustBV::concrete(sp as u128, bits)),
        SpSeed::Symbolic => {
            let sym = crate::symbolic::RustBV::symbolic(&state.solver().borrow(), "sym_sp", bits);
            state.set_sp(sym);
        }
    }
    let sid = state.state_id();

    run_post_step_core(
        &CoreCtx {
            ctx: &ctx,
            prof: &prof,
            native_procs: &procs,
            native_syscalls: &syscalls,
            callbacks: None,
        },
        state,
        PostStepInputs {
            result: RunResult::SimProcedure {
                addr: HOOK_ADDR,
                name: "find_target_test".to_string(),
                num_args: 0,
                return_addr: HOOK_RET,
            },
            deferred_forks: Vec::new(),
            last_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        },
        sid,
    )
}
