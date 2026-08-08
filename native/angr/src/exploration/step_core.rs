//! Send + Sync interpreter-step driver (angr-1ilq.3, sub-increment 2b-i).
//!
//! This module isolates the read-only configuration the VEX interpreter step
//! consumes into a self-contained [`StepContext`] bundle, plus a free function
//! [`run_interpreter_step_core`] parameterized over it. The existing
//! single-threaded path
//! (`RustExplorationManager::step_state_inner` in `stepping.rs`) becomes a
//! thin wrapper that builds a `StepContext` from `&self` and calls this
//! function, so the run-loop behavior is unchanged.
//!
//! The point of the extraction is to PROVE — via the compile-time
//! `Send + Sync` assertion below and the full test suite staying green — that a
//! single owned/`Arc`-shared config bundle can drive one interpreter step
//! without borrowing `&mut self`. That de-risks the future work-stealing worker
//! path, where each worker must own (or `Arc`-share) the config rather than
//! reach back into the manager.
//!
//! Scope note: this carries only the fields `run_interpreter_step_core`
//! actually reads (see the inventory in the bead). The post-interpreter handlers
//! (`handle_block_end`, simproc / syscall / symbolic-jump dispatch) and the
//! native registries / `PythonCallbacks` they consume are intentionally NOT
//! bundled yet — that is sub-increment 2b-ii. `PythonCallbacks` stays an
//! explicit parameter (it is already `Send + Sync` and already threaded through
//! the call site), and the mutable block cache is passed by `&mut` because the
//! step swaps it in and out rather than reading it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lru::LruCache;
use rustc_hash::FxHashMap;

use crate::callbacks::{ExecutionConfig, PythonCallbacks};
use crate::concretize::AddressConcretizer;
use crate::interpreter::VEXInterpreter;
use crate::state::RustSimState;
use crate::vex::{IRSB, VexArch};

use super::RustExplorationManager;
use super::core_outcome::CcSnapshot;
use super::stepping::InterpreterStepResult;

/// Self-contained, `Send + Sync` bundle of the read-only configuration one
/// interpreter step reads off the manager.
///
/// Every field is owned or `Arc`-shared so the bundle carries no borrow of the
/// manager — a worker thread can hold it and step a state without touching
/// `&mut self`. Built per step by [`RustExplorationManager::step_context`].
pub(crate) struct StepContext {
    /// VEX architecture (interpreter construction).
    pub(crate) vex_arch: VexArch,
    /// Pointer width in bytes (for the native-resume sentinel address). Snapshot
    /// of `calling_convention.pointer_size()` — the CC trait object itself is
    /// not `Clone`, and `pointer_size` is all the step needs from it.
    pub(crate) pointer_size: u32,
    /// Concrete code regions for native lifting (O(1) `Arc` clones per region).
    pub(crate) binary_regions: Vec<(u64, Arc<Vec<u8>>)>,
    /// `[start, end)` of the main object's code regions (native-dispatch gate).
    pub(crate) main_object_range: Option<(u64, u64)>,
    /// Prefer native procs for non-main-object library hooks (native-dispatch gate).
    pub(crate) prefer_native_library_hooks: bool,
    /// Deferred-fork / branch-policy execution config.
    pub(crate) exec_config: ExecutionConfig,
    /// LAZY_SOLVES: skip Z3 feasibility checks on forks.
    pub(crate) lazy_solves: bool,
    /// Whether Rust-side profiling is enabled (propagated to the interpreter).
    pub(crate) profiling_enabled: bool,
    /// Address concretization strategy configuration.
    pub(crate) concretizer_config: AddressConcretizer,
    /// Global VEX optimization level (None = pyvex default).
    pub(crate) vex_opt_level: Option<i32>,
    /// Per-address VEX optimization level overrides. Shared with the manager's
    /// `MemoryConfiguration` and handed straight to the interpreter, so the
    /// snapshot is an `Arc` bump rather than two map clones per step.
    pub(crate) vex_opt_level_overrides: Arc<FxHashMap<u64, i32>>,
    /// Enable native (in-process) libVEX cold-block lifting (z087y Stage-2).
    /// Applied via `interp.set_native_lift_enabled`; a no-op on the default
    /// (non-`libvex-ffi`) build.
    pub(crate) native_lift_enabled: bool,
    /// Hook addresses.
    pub(crate) hooks: HashSet<u64>,
    /// SimProcedures: address -> (name, num_args, no_return).
    pub(crate) simprocedures: HashMap<u64, (String, usize, bool)>,
    /// Find addresses (registered as interpreter hooks so it stops there).
    pub(crate) find_addrs: HashSet<u64>,
    /// Avoid addresses (registered as interpreter hooks so it stops there).
    pub(crate) avoid_addrs: HashSet<u64>,
    /// Union of find + avoid addresses (interpreter block-chain stop set).
    pub(crate) stop_addrs: HashSet<u64>,
    /// Maximum blocks per interpreter run.
    pub(crate) max_steps_per_run: u32,
    /// Block-granular stepping (break the block chain at every boundary).
    pub(crate) block_granular: bool,
    /// Find condition has callable predicates (limits the run to 1 block).
    pub(crate) find_needs_python: bool,
    /// Avoid condition has callable predicates (limits the run to 1 block).
    pub(crate) avoid_needs_python: bool,
    /// Calling-convention scalar/vec snapshot the post-step native arms read
    /// (angr-vh834). The CC trait object is not `Clone`, so we snapshot the
    /// fields `write_syscall_return` / `extract_*_args` / `setup_native_subcall`
    /// consult.
    pub(crate) cc: CcSnapshot,
    /// OS / SimOS name (lowercase). Drives syscall-table dispatch ("cgc" routes
    /// to the DECREE table). Snapshot of `environment.os_name`.
    pub(crate) os_name: String,
    /// Manager-level deferred-fork mode (drives the `UnconstrainedJump` arm's
    /// drop-vs-materialize decision). Distinct from the per-state
    /// `force_eager_forks` override applied during interpreter stepping.
    pub(crate) use_deferred_forks: bool,
    /// Opt-in: materialize loop-exit forks at an `UnconstrainedJump` even in
    /// deferred mode (angr-ckdy). Snapshot of `materialize_unconstrained_forks`.
    pub(crate) materialize_unconstrained_forks: bool,
}

// Compile-time proof that the config bundle is `Send + Sync` — the property the
// future work-stealing worker path depends on. If a non-`Send`/`Sync` field is
// ever added, this fails to compile.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<StepContext>();
};

impl RustExplorationManager {
    /// Build a [`StepContext`] snapshot of the manager's current step
    /// configuration. Clones/`Arc`-shares so the result borrows nothing from
    /// `self`. Called once per `run_interpreter_step`.
    pub(crate) fn step_context(&self) -> StepContext {
        StepContext {
            vex_arch: self.environment.vex_arch,
            pointer_size: self.environment.calling_convention.pointer_size(),
            binary_regions: self.environment.binary_regions.clone(),
            main_object_range: self.environment.main_object_range,
            prefer_native_library_hooks: self.environment.prefer_native_library_hooks,
            exec_config: self.exec_config.clone(),
            lazy_solves: self.constraint_solver.lazy_solves,
            profiling_enabled: self.profiling.profiling_enabled,
            concretizer_config: self.memory_config.concretizer_config.clone(),
            vex_opt_level: self.memory_config.vex_opt_level,
            vex_opt_level_overrides: Arc::clone(&self.memory_config.vex_opt_level_overrides),
            native_lift_enabled: self.memory_config.native_lift_enabled,
            hooks: self.hooks.clone(),
            simprocedures: self.simprocedures.clone(),
            find_addrs: self.find_addrs.clone(),
            avoid_addrs: self.avoid_addrs.clone(),
            stop_addrs: self.stop_addrs.clone(),
            max_steps_per_run: self.max_steps_per_run,
            block_granular: self.block_granular,
            find_needs_python: self.find_needs_python,
            avoid_needs_python: self.avoid_needs_python,
            cc: {
                let cc = &self.environment.calling_convention;
                CcSnapshot {
                    arg_registers: cc.arg_registers().to_vec(),
                    syscall_arg_registers: cc.syscall_arg_registers().to_vec(),
                    return_register: cc.return_register(),
                    link_register: cc.link_register(),
                    pops_return_addr: cc.pops_return_addr(),
                    pointer_size: cc.pointer_size(),
                    stack_arg_offset: cc.stack_arg_offset(),
                    syscall_stack_arg_offset: cc.syscall_stack_arg_offset(),
                    syscall_error_register: cc.syscall_error_register(),
                }
            },
            os_name: self.environment.os_name.clone(),
            use_deferred_forks: self.exec_config.use_deferred_forks,
            materialize_unconstrained_forks: self.materialize_unconstrained_forks,
        }
    }
}

/// Run the VEX interpreter for one step and recover all owned state from it.
///
/// Behavior-identical extraction of `RustExplorationManager::run_interpreter_step`:
/// it constructs a `VEXInterpreter` against the borrowed solver, runs it until
/// its next event, then fully drains (registers, memory, history, block cache,
/// profiling stats) before the interpreter is dropped at scope end.
///
/// All config comes from `ctx`; the mutable `block_cache` is swapped through by
/// `&mut` (the manager left a placeholder in `self.environment.block_cache`
/// during the call and restores the returned one afterwards), and
/// `callbacks` stays an explicit parameter.
// The two extra params over the original method (`ctx` and `block_cache`)
// stand in for the `&mut self` the method used to carry — that is exactly the
// decoupling this extraction exists to demonstrate, so the count is inherent.
pub(crate) fn run_interpreter_step_core(
    ctx: &StepContext,
    callbacks: &PythonCallbacks,
    state: &mut RustSimState,
    initial_pc: u64,
    skip_addr: Option<u64>,
    setup_start: Option<std::time::Instant>,
    block_cache: &mut LruCache<u64, Arc<IRSB>>,
) -> InterpreterStepResult {
    let solver_rc = state.solver().clone();
    let solver_ref = solver_rc.borrow();

    // Create interpreter with the state's solver.
    //
    // angr-027h: a state carrying `force_eager_forks` (a loop-exit fork
    // resumed at an UnconstrainedJump) overrides the manager-level
    // `use_deferred_forks` so it materializes successors eagerly and BFSes
    // to the find target instead of recursively re-deferring.
    let mut exec_config = ctx.exec_config.clone();
    if state.force_eager_forks() {
        exec_config.use_deferred_forks = false;
    }
    let mut interp = VEXInterpreter::with_config(ctx.vex_arch, &solver_ref, exec_config);

    // Propagate lazy_solves to skip Z3 feasibility checks
    interp.lazy_solves = ctx.lazy_solves;
    // Propagate NO_IP_CONCRETIZATION from the state. Unlike lazy_solves
    // which is a manager-level flag, this is a per-state SimOption.
    interp.no_ip_concretization = state.no_ip_concretization();
    // NO_SYMBOLIC_JUMP_RESOLUTION sibling — same routing, applied at the
    // same short-circuit in eval_next_addr_concretized.
    interp.no_symbolic_jump_resolution = state.no_symbolic_jump_resolution();
    // KEEP_IP_SYMBOLIC: per-state SimOption that tells eval_next_addr_concretized
    // to stash the original symbolic next-pc expression (for restore via
    // set_ip after the manager's set_pc) and to skip the
    // `assume_true(next_val == addr)` narrowing constraint.
    interp.keep_ip_symbolic = state.keep_ip_symbolic();
    interp.set_profiling(ctx.profiling_enabled);
    // Propagate concretization strategy config
    interp.set_concretizer(ctx.concretizer_config.clone());
    // Propagate VEX optimization level settings
    interp.vex_opt_level = ctx.vex_opt_level;
    // z087y Stage-2: opt-in native cold-block lifting (no-op on default build).
    interp.set_native_lift_enabled(ctx.native_lift_enabled);
    // Share the manager's override map; the setters go through `Arc::make_mut`,
    // so a config change while this interpreter holds the Arc copies once
    // instead of mutating it underneath (none do during step execution anyway).
    interp.vex_opt_level_overrides = Arc::clone(&ctx.vex_opt_level_overrides);

    // Copy state registers to interpreter (including symbolic values)
    interp.registers = state.registers().fork();
    interp.set_pc(initial_pc);
    // Forward state_id so inspect dispatch sites can identify which
    // state owns the firing event (angr-uq4n.3/.4).
    interp.current_state_id = state.state_id() as i64;
    // Transfer call stack and detailed history to interpreter
    interp.call_stack = state.call_stack().to_vec();
    interp.detailed_history = state.detailed_history().iter().cloned().collect();
    // Seed the per-state simulated TSC so RDTSC continues this state's own
    // timeline instead of a process-wide counter (angr-9ke6b.173).
    interp.dirty_helper_state.tsc = state.tsc_counter();

    // Set up hooks, skipping the one we just processed (for zero-length hooks)
    for &addr in &ctx.hooks {
        if Some(addr) != skip_addr {
            interp.add_hook(addr);
        }
    }

    // Register SimProcedures, also skipping the one we just processed.
    // The tuple's `no_return` stays behind on purpose: it is read at dispatch
    // time out of `ctx.simprocedures`, not through the interpreter (see
    // `SimProcedureInfo`).
    for (addr, (name, num_args, _no_return)) in &ctx.simprocedures {
        if Some(*addr) != skip_addr {
            interp.register_simprocedure(*addr, name.clone(), *num_args);
        }
    }

    // Register the native sub-call resume sentinel (S2, bead angr-5gf0s):
    // a reserved hook address that a proc returning ProcOutcome::CallAndResume
    // makes the guest routine return to. Recognized by name in
    // `handle_simprocedure_core`; never lifted (is_hooked fires first).
    let resume_sentinel = crate::procedures::native_resume_sentinel(ctx.pointer_size);
    interp.register_simprocedure(
        resume_sentinel,
        crate::procedures::NATIVE_RESUME_SENTINEL_NAME.to_string(),
        0,
    );

    // Add find/avoid addresses as hooks so the interpreter stops there
    for &addr in &ctx.find_addrs {
        interp.add_hook(addr);
    }
    for &addr in &ctx.avoid_addrs {
        interp.add_hook(addr);
    }

    // Copy binary regions for code lifting (O(1) Arc clone per region)
    for (base, data) in &ctx.binary_regions {
        interp.add_concrete_memory_shared(*base, Arc::clone(data));
    }

    // Transfer state's SymbolicMemory into the interpreter.
    // This makes Rust the source of truth for all memory during
    // VEX execution. Loads/stores go to SymbolicMemory directly
    // instead of calling back to Python.
    interp.set_rust_memory(state.take_memory());

    // Share the exploration-level block cache with the interpreter
    // so lifted blocks persist across steps (avoids re-lifting).
    // Swap exploration's populated cache into interp, stash interp's empty one.
    // The placeholder left in the caller's `*block_cache` slot is a
    // zero-preallocation `unbounded()` cache (angr-4xaga.1): nothing reads it
    // before both call sites overwrite the slot with `updated_block_cache`, so
    // its capacity is irrelevant and `new(BLOCK_CACHE_CAPACITY)`'s eager
    // 4096-bucket HashMap alloc would be pure waste on this per-step hot path.
    let interp_empty_cache =
        interp.swap_block_cache(std::mem::replace(block_cache, LruCache::unbounded()));
    // interp now has the exploration's cache; block_cache is a temporary empty placeholder
    let _ = interp_empty_cache; // drop the empty cache

    // Record setup time before execution
    if let Some(start) = setup_start {
        interp.stats_mut().step_setup_time_ns += start.elapsed().as_nanos() as u64;
    }

    // Run until event.
    // When callable predicates are active (find_needs_python), limit to
    // 1 block so the run loop can check the predicate at each PC.
    // Otherwise the interpreter would execute many blocks, skipping past
    // the target address without the predicate ever seeing it.
    let steps_limit = if ctx.find_needs_python || ctx.avoid_needs_python {
        1
    } else {
        ctx.max_steps_per_run
    };
    let (result, _blocks_executed, deferred_forks) =
        interp.run_until_event(callbacks, steps_limit, &ctx.stop_addrs, ctx.block_granular);

    // Drain interpreter state into owned values before drop.
    let last_condition = interp.take_last_branch_condition();
    let stored_conditions = interp.take_stored_conditions();
    let fork_snapshots = interp.take_fork_snapshots();
    let symbolic_ip_at_exit = interp.take_symbolic_ip_at_exit();
    let new_registers = interp.registers.fork();
    let new_pc = interp.get_pc();
    let new_call_stack = std::mem::take(&mut interp.call_stack);
    let new_detailed_history = std::mem::take(&mut interp.detailed_history);
    let new_tsc_counter = interp.dirty_helper_state.tsc;

    // Flush any remaining pending stores to rust_memory before recovery.
    interp.flush_stores_to_rust_memory();
    let recovered_memory = interp.take_rust_memory();

    // Return shared block cache to exploration before interpreter is dropped.
    // The placeholder left behind in `interp.block_cache` is only dropped a few
    // lines later, so a zero-preallocation `unbounded()` cache avoids a wasted
    // 4096-bucket HashMap alloc here too (angr-4xaga.1).
    let updated_block_cache = interp.swap_block_cache(LruCache::unbounded());

    let step_stats = interp.take_stats();

    InterpreterStepResult {
        result,
        deferred_forks,
        last_condition,
        stored_conditions,
        fork_snapshots,
        new_registers,
        new_pc,
        new_call_stack,
        new_detailed_history,
        new_tsc_counter,
        recovered_memory,
        step_stats,
        updated_block_cache,
        symbolic_ip_at_exit,
    }
}

#[cfg(test)]
#[path = "step_core_tests.rs"]
mod tests;
