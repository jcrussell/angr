//! Rust-native simulation state for symbolic execution.
//!
//! `RustSimState` provides a Rust-first state representation that:
//! - Owns registers, memory, and solver context
//! - Supports O(1) forking via copy-on-write
//! - Minimizes Python-Rust state transfer overhead
//! - Enables Rust-native exploration loops
//!
//! # Cross-mixin invariants (I1–I8)
//!
//! The Python `RustExplorationManager` composes several mixins
//! (RustStateCacheMixin, RustStateExportMixin, RustStateSyncMixin,
//! RustCallbackDispatchMixin); the cross-mixin invariants are documented as
//! the source-of-truth header in the `Cross-mixin invariants` section of
//! the `angr/exploration/rust_manager.py` module docstring. Most of those
//! concerns are
//! Python-orchestration only, but the ones that cross the FFI boundary
//! manifest on the Rust side. The list below mirrors I1–I8 with the Rust
//! enforcement site (or "Python-only" when no Rust code participates).
//! Keep this section in sync with the Python header — divergence between
//! the two has produced silent correctness bugs (cache poisoning, lost
//! states, infinite loops) historically; see angr-a2br for the docs-before-
//! split rationale.
//!
//! - **I1. Disk-cache key axes** — Python-only. The cache key mixes
//!   `(binary_path, _RUST_CACHE_VERSION, _PYTHON_METADATA_VERSION, arch)`
//!   in `rust_manager.py::_disk_cache_key`. The Rust side only consumes
//!   the deserialized state via the bulk-setters in this module; it never
//!   inspects the cache key. Bumps of `_RUST_CACHE_VERSION` must accompany
//!   any change to fields pickled here (registers, memory, heap metadata).
//! - **I2. Init pipeline phases** — Python-only. Phases `_load_init_pickle`
//!   → `_deserialize_init_state` → `_apply_init_side_effects` are the
//!   single orchestrator. Rust receives the result; no Rust-side state
//!   machine participates.
//! - **I3. Init-cache user-symbolic gate** — Python-only. `blank_state`
//!   round-trips lose user-created BVS identity, so caching is suppressed
//!   on user-symbolic states. The Rust side has no way to detect a user
//!   symbolic AST after the fact; the gate must hold on the Python side
//!   before any FFI call happens.
//! - **I4. `_apply_state_metadata` allow-list** — Python-only. Only
//!   `LAZY_SOLVES` + `STRICT_PAGE_ACCESS` SimOptions transfer across the
//!   cache-hit path. The Rust engine reads SimOptions through the
//!   `engine.options` snapshot taken at `__init__`; later option changes
//!   on the cached state do NOT propagate. To check user-set options
//!   reliably, inspect the user-supplied state in Python's `__init__`
//!   BEFORE `_run_python_init_if_needed` runs.
//! - **I5. Register filter at the FFI boundary** — enforced by
//!   `set_registers_bulk` (this module). The disk init cache pickles
//!   `arch.register_names.values()` including registers the Rust engine
//!   does not model (cr0..8, ymm0..15, fs_seg, ds_seg, cmstart, cmlen,
//!   fpreg, ...). Python's `rust_state_sync.py` filters to
//!   `_supported_register_names` BEFORE calling into Rust; the Rust setter
//!   returns `PyValueError("unknown register: ...")` if an unsupported
//!   name leaks through. Do not "fix" by extending `arch/amd64.rs` etc.;
//!   the interpreter does not consume those registers.
//! - **I6. State-cache pinning + manager-vs-mixin override** — Python
//!   orchestrates pinning of `_state_roots ∪ {_current_callback_state_id,
//!   _current_stepping_state_id}` in `_cleanup_state_cache`. Rust side
//!   participates by owning the per-state `RustSimState` (Drop runs on
//!   eviction) and by maintaining the `state_roots` table inside
//!   `stash.rs::StashManager`. When `StashManager::remove_state` runs,
//!   `state_roots` and `state_index` move together — see stash.rs for
//!   the localized invariant.
//! - **I7. Rust ↔ Python field sync uses `max()`, not overwrite** — Rust
//!   owns `mmap_base` and `posix_brk` (this module). The Python export
//!   path computes `max(rust_value, python_value)` so a Python-side
//!   advance (e.g. user-set `state.heap.mmap_base` or a fallback
//!   SimProcedure mutation) is never clobbered by a stale Rust value.
//!   Rust setters here accept any value; the directionality is enforced
//!   at the Python export site. Regression tests:
//!   `TestMmapBaseSync.test_export_path_does_not_clobber_higher_python_mmap_base`,
//!   `TestPosixBrkSync.test_export_path_syncs_rust_posix_brk_into_state_posix`.
//! - **I8. Exploration-loop termination** — enforced in
//!   `exploration/run_loop.rs`. The Rust loop terminates on EITHER (a)
//!   `found_count() >= num_find`, OR (b) the active stash returning
//!   `None` from `pop_front`/`pop_back` (→ `active_empty` event). Both
//!   paths are covered by `found_count()`, which includes both
//!   Rust-native finds and Python-predicate-derived finds added via the
//!   need_callback resume path. Earlier code only checked Python
//!   predicate flags and infinite-looped when `find=int` was combined
//!   with a non-predicate technique like DFS.
//!
//! # State metadata + fork invariants
//!
//! The invariants below cross the FFI boundary in addition to (or instead
//! of) I1–I8 above. Each bullet is stated in full here and its backticked
//! label is an anchor for this file only — not a bd memory key — except
//! where the bullet says otherwise. Enforcement-site comments below
//! cross-reference back to this header rather than duplicating the prose.
//!
//! - **`state-id-never-reused`** — `NEXT_STATE_ID` is a monotonic atomic
//!   counter. Once a state ID is absent from every Rust stash
//!   (active|found|avoid|deadended|errored|unconstrained|pruned), it is
//!   unreachable forever. This is the contract that makes it safe for the
//!   Python `_cleanup_state_cache` to drop shadow mappings keyed by
//!   `state_id` (`_state_roots`, `_predicate_matched_ids`,
//!   `_py_state_options`, `_py_state_globals`). Future shadow structures
//!   keyed by `state_id` must prune against `any_stash`, not invent
//!   per-structure LRU caps. Enforced at `next_state_id()` below; the
//!   always-on `assert!` in `fork_with()` confirms the child ID is fresh
//!   (promoted from `debug_assert!` in angr-9ke6b.220, alongside the
//!   sibling ID-minting guards in `exploration/state_lifecycle.rs`).
//!   Snapshot restore imports IDs minted under a *foreign* counter, so
//!   `from_snapshot` calls `reserve_state_id()` to lift the local counter
//!   above every restored ID — otherwise a resumed manager re-mints live
//!   IDs and the collisions silently clobber the stash index.
//! - **`state-metadata-dataclass`** — Per-state Python AST metadata
//!   (`symbolic_pages`, `hook_symbolic_memory`, `addr_to_ast`) is owned by
//!   `RustSimState` on the Rust side; the Python `RustStateCacheMixin`
//!   keeps a parallel `StateMetadata` dataclass (replacing three earlier
//!   per-state dicts) at `angr/exploration/_state_metadata.py`. Eviction
//!   on the Python side drops the dataclass entry; eviction on the Rust
//!   side runs `RustSimState::Drop`, which decrements the Py-refcounts in
//!   the maps below. The two sides do not have to agree on contents at
//!   every callback boundary — Python may have a stale dataclass entry
//!   while Rust has already mutated, and vice versa; what they MUST agree
//!   on is the set of *live* state IDs (`state-id-never-reused`).
//! - **`arc-make-mut-cow`** — Fields that are read on every fork but
//!   mutated rarely are wrapped in `Arc<T>` and use `Arc::make_mut` for
//!   copy-on-write. Mutators must peek the read path first to skip the
//!   CoW clone when the operation would be a no-op (e.g. closing an
//!   already-closed fd, clearing an empty hook set). `Arc`-wrapping a
//!   field mutated on every fork (e.g. `RegisterFile.symbolic`) is a net
//!   loss — the `make_mut` churn offsets the savings. See `FileSystem`
//!   methods, `clear_hooks`, and `set_env_var` for the pattern.
//! - **`arc-collection-iter`** — `Arc<HashSet<u64>>` and `Arc<Vec<T>>` do
//!   NOT implement `IntoIterator` for `&Self`. After `Arc`-wrapping
//!   `hooks` / `environment` / `fs.fds`, `for x in &self.field` becomes
//!   `for x in self.field.iter()`. Compiler errors are obvious (E0277
//!   "is not an iterator") but easy to miss in review.
//! - **`invariant-apply-state-metadata-option-allowlist`** (also a live bd
//!   memory key) — `RustExplorationManager.
//!   _apply_state_metadata` (Python) copies ONLY `LAZY_SOLVES` and
//!   `STRICT_PAGE_ACCESS` from the source state to a cached/disk-loaded
//!   init state. The boolean SimOption mirrors held below
//!   (`no_ip_concretization`, `no_symbolic_jump_resolution`,
//!   `keep_ip_symbolic`) are NOT in the allow-list, so any code that
//!   reads `angr_state.options` from inside `_add_rust_state` on a
//!   cached-init path may not see options the user originally set. Set
//!   these flags from `__init__` on the user-supplied state BEFORE
//!   `_run_python_init_if_needed` runs — not in `_add_rust_state` on the
//!   post-init state.
//! - **`arc-make-mut-fresh-context`** — On a freshly constructed
//!   `SymContext` (no other Arc refs yet), `Arc::make_mut(&mut field)`
//!   returns a unique `&mut` without cloning (refcount==1). `merge()`
//!   exploits this to write into the merged context's `Arc<HashMap>`
//!   while keeping Arc-wrapping for `fork()`. Site lives in
//!   `symbolic/context.rs` (Z3 and mock paths); cited here because
//!   `RustSimState::merge` builds the merged solver before any other
//!   handle is taken.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use pyo3::prelude::*;

use crate::arch::{Arch, RegisterFile, arch_from_name};
use crate::concretize::AddressConcretizer;
use crate::memory::{MemoryError, Permission, SymbolicMemory};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::{Endness, VexArch};

mod construction;
mod export;
mod filesystem;
mod fork;
mod history;
mod hooks;
mod inspection;
mod memory;
/// Cross-worker state migration transport (angr-1ilq.1). Z3-only: migration
/// between distinct per-worker Z3 contexts is meaningless without the solver.
#[cfg(feature = "vex-engine-z3")]
mod migration;
mod options;
mod process;
mod pymethods;
mod registers;
mod snapshot;
mod solver;
mod types;

pub use export::*;
pub use filesystem::*;
pub use inspection::*;
#[cfg(feature = "vex-engine-z3")]
pub use migration::StateMigrationPayload;
pub use snapshot::*;
pub use types::*;

/// Unique identifier for states.
///
/// Monotonic atomic counter — see the module-level `state-id-never-reused`
/// invariant. The Python shadow maps (`_state_roots`,
/// `_predicate_matched_ids`, `_py_state_options`, `_py_state_globals`) all
/// depend on the no-reuse contract for safe eviction.
static NEXT_STATE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Allocate a fresh, never-before-issued state ID.
///
/// Returns a `u64` strictly greater than every previously returned value
/// (modulo wraparound at 2^64, which is unreachable in practice). See the
/// module-level `state-id-never-reused` invariant for why every Python-
/// side shadow structure depends on this property.
fn next_state_id() -> u64 {
    NEXT_STATE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

/// Raise `NEXT_STATE_ID` so the next minted ID is strictly greater than `id`.
///
/// Restoring a snapshot re-materializes states whose IDs were minted by a
/// *different* process (or a different counter epoch), so the local counter
/// knows nothing about them. Without this, a resumed manager mints IDs that
/// collide with its own restored states, and the collisions silently
/// overwrite entries in the `StashManager` state index — see the
/// `state-id-never-reused` invariant above.
pub(crate) fn reserve_state_id(id: u64) {
    NEXT_STATE_ID.fetch_max(id.saturating_add(1), std::sync::atomic::Ordering::SeqCst);
}

/// Pointers to the three glibc locale ctype lookup tables, exposed via the
/// `__ctype_b_loc` / `__ctype_tolower_loc` / `__ctype_toupper_loc` accessors.
///
/// These are returned verbatim by the corresponding native procs. The tables
/// themselves are malloc'd and populated by Python's `__libc_start_main` init
/// pass before the Rust engine takes over; this struct only carries the
/// already-built table pointers. `None` means the init pass never ran (e.g. a
/// blank_state entry), in which case the native proc defers to Python.
#[derive(Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CtypeLocPtrs {
    pub b: Option<u64>,
    pub tolower: Option<u64>,
    pub toupper: Option<u64>,
}

/// Guest-memory addresses of the glibc getopt(3) extern globals `optind`,
/// `optarg`, and `optopt`, resolved Python-side once via
/// `proj.loader.find_symbol(name).rebased_addr` and pushed into Rust at
/// seed-state creation (mirrors the [`CtypeLocPtrs`] init-push channel).
///
/// The native getopt proc (bead angr-bhk0a.2) writes the updated cursor /
/// optarg / optopt back to these guest addresses so the program reads them
/// like real getopt would. `None` means the symbol was absent (statically
/// linked away, or a blank_state with no loader pass), in which case the
/// native proc defers to Python.
#[derive(Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetoptExternAddrs {
    pub optind: Option<u64>,
    pub optarg: Option<u64>,
    pub optopt: Option<u64>,
}

/// One suspended-continuation frame for the native sub-call (ADDS_EXITS)
/// dispatcher (design: `tools/decisions/native_subcall_dispatcher_design.md`,
/// spike angr-5gf0s). When a native proc needs to call a guest function and
/// resume afterwards (the analogue of Python's `SimProcedure.call(...,
/// continue_at="retsite")`), the continuation cannot be a Rust closure — it
/// must be *data* on the state so it survives fork/snapshot. This frame is that
/// data: `proc_name` re-finds the proc in the registry on return, `resume_tag`
/// selects which continuation arm to run, and `saved_args` carries the original
/// proc arguments the continuation needs.
///
/// This is the S1 foundation slice (bead angr-pn3w8): the per-state `Vec` field
/// (a LIFO stack) plus fork/snapshot/proxy plumbing only. The dispatcher that
/// pushes/pops frames is S2 (bead angr-xxukz chain); an empty stack is the
/// default and means no behaviour change.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct NativeResumeFrame {
    /// Registry name of the native proc whose continuation should run.
    pub proc_name: String,
    /// Which continuation arm of that proc to dispatch on resume.
    pub resume_tag: u32,
    /// Original proc arguments the continuation needs after the sub-call.
    pub saved_args: Vec<RustBV>,
    /// Address the *original* caller of this proc should resume at once the
    /// continuation finishes. Captured at sub-call time and used as the PC on
    /// the final `crate::procedures::ProcOutcome::Return` instead of reading
    /// the stack: a stack-return ABI clobbers the slot when the guest routine
    /// returns to the resume sentinel, and a link-register ABI loses the
    /// original return address when the dispatcher overwrites LR with the
    /// sentinel. Storing it on the frame makes resume correct on both (S2,
    /// bead angr-5gf0s).
    pub caller_return_addr: u64,
}

#[cfg(feature = "vex-engine-z3")]
impl NativeResumeFrame {
    /// Cross-context twin of this frame (angr-1ilq.2): `Z3_translate` every
    /// context-bound `RustBV` in `saved_args` into `target_ctx`; all other
    /// fields are context-independent and cloned. Mirrors
    /// [`crate::state::RustSimState::translate_state`] (state/fork.rs) — a
    /// concrete `saved_arg` clones verbatim (see [`RustBV::translate_into`] in
    /// symbolic/value_z3.rs), a symbolic one is re-homed so a stolen state's
    /// resume stack is valid in the worker's context rather than a dangling
    /// foreign-context AST.
    ///
    /// `target_ctx` must be a *different* context from the one these ASTs live
    /// in (the same panic-on-same-context contract as `RustBV::translate_into`).
    pub fn translate_into(&self, target_ctx: &z3::Context) -> NativeResumeFrame {
        NativeResumeFrame {
            proc_name: self.proc_name.clone(),
            resume_tag: self.resume_tag,
            saved_args: self
                .saved_args
                .iter()
                .map(|bv| bv.translate_into(target_ctx))
                .collect(),
            caller_return_addr: self.caller_return_addr,
        }
    }
}

/// A claripy AST held in per-state metadata, shared by `Arc` rather than stored
/// as a bare `Py<PyAny>`.
///
/// The `Arc` exists purely to keep state forking off the GIL (angr-gorvf.4.2).
/// `Py::clone_ref` needs a `Python` token, so cloning a map of bare `Py` handles
/// on an exploration worker forces a `Python::attach` on **every fork** of any
/// state that carries a non-empty overlay — measured as the *sole* GIL holder on
/// four otherwise Python-free corpus benches. `Arc::clone` is an atomic bump and
/// needs no token, so the fork path never touches Python. Dropping the last
/// `Arc` still decrements the Py-refcount correctly: pyo3 defers the decref when
/// the GIL is not held.
pub type SharedPyAst = Arc<Py<PyAny>>;

/// Rust-native simulation state.
///
/// This struct owns all state components and provides O(1) forking
/// through copy-on-write semantics. The solver context is shared
/// via Rc<RefCell<>> to allow constraint accumulation across forks.
///
/// # Design
///
/// - **Registers**: Stored in `RegisterFile` with concrete bytes and symbolic overlays
/// - **Memory**: `SymbolicMemory` with O(1) CoW via `im::OrdMap`
/// - **Solver**: Shared `Rc<RefCell<SymContext>>` for constraint accumulation
/// - **History**: Basic block trace for debugging/analysis
///
/// # Fork Semantics
///
/// Forking a state is O(1) because:
/// - Memory uses persistent data structures (im::OrdMap)
/// - Registers are cloned (small, ~700 bytes for AMD64)
/// - Solver context is cloned with constraint state preserved
/// - History is optionally shared or copied based on config
pub struct RustSimState {
    /// Architecture information.
    arch: Box<dyn Arch>,
    /// VEX architecture enum (cached for quick lookup).
    vex_arch: VexArch,
    /// Register file with concrete and symbolic values.
    registers: RegisterFile,
    /// Symbolic memory with O(1) CoW forking.
    memory: SymbolicMemory,
    /// Shared solver context for constraints.
    /// Using Rc<RefCell<>> to allow mutation during stepping
    /// while maintaining shared ownership for forking.
    solver: Rc<RefCell<SymContext>>,
    /// Program counter.
    pc: u64,
    /// Unique state identifier.
    state_id: u64,
    /// Parent state ID (for tracking fork tree).
    parent_id: Option<u64>,
    /// Basic block history (addresses visited). `VecDeque` so FIFO
    /// cap eviction (`pop_front`) is O(1) amortized rather than the
    /// O(n) buffer shift a `Vec::remove(0)` incurs per block (angr-ph300.57).
    history: VecDeque<u64>,
    /// Detailed execution history with jumpkind and target info.
    detailed_history: VecDeque<HistoryEntry>,
    /// Maximum history length (0 = unlimited).
    max_history: usize,
    /// Hook addresses. Wrapped in Arc for cheap fork — copy-on-write
    /// via Arc::make_mut on add/remove/clear. Mutated only at config time
    /// in typical workloads, so most forks pay no clone cost here.
    ///
    /// See module-level `arc-make-mut-cow` (the read-first peek pattern
    /// in `clear_hooks`) and `arc-collection-iter` (iterate via
    /// `self.hooks.iter()`, not `for x in &self.hooks` — `Arc<HashSet>`
    /// does not implement `IntoIterator` for `&Self`).
    hooks: Arc<HashSet<u64>>,
    /// Hook addresses explicitly removed (via `remove_hook`/`clear_hooks`)
    /// since this state's last fork point. Lets `merge` (angr-9ke6b.121,
    /// bug fix follow-up) distinguish "never touched" from "explicitly
    /// removed" when unioning `hooks` across merge arms — a plain set union
    /// can't tell the two apart and would silently resurrect a hook this
    /// branch removed if a sibling branch still has it. Reset to empty by
    /// `fork_with` (a fresh divergence point); carried over unchanged by
    /// `translate_state` (same logical state, different Z3 context).
    removed_hooks: Arc<HashSet<u64>>,
    /// Address concretization config.
    concretizer: AddressConcretizer,
    /// Whether to track detailed history.
    track_history: bool,
    /// File system state: tracks all file descriptors with metadata.
    /// Cloned on fork so each path gets its own file system state.
    fs: FileSystem,
    /// Heap brk pointer — simple bump allocator for malloc/calloc.
    /// Default: 0xC0000000 (matching angr's DEFAULT_HEAP_LOCATION).
    heap_brk: u64,
    /// POSIX brk pointer — separately tracks `state.posix.brk` from Python
    /// for the brk(2) syscall. Default 0x1B00000 (matches Python default).
    /// Distinct from `heap_brk`, which is the malloc bump allocator.
    posix_brk: u64,
    /// mmap base pointer — mirrors `state.heap.mmap_base` from Python for
    /// the mmap(2) syscall when addr=0 (kernel chooses the mapping). Default
    /// 0xC1000000 (heap_base 0xC0000000 + heap_size 0x00800000 * 2 — matches
    /// `SimHeapBase.mmap_base`). NOT pushed back to Python's
    /// `state.heap.mmap_base` on syscall fallback today; same drift risk as
    /// `posix_brk`. Future cross-engine sync work should address both fields
    /// at the syscall callback boundary.
    mmap_base: u64,
    /// Simulated timestamp counter backing the `RDTSC` dirty helper. Per-state
    /// (was a process-wide `AtomicU64` — angr-9ke6b.173) so the Nth RDTSC along
    /// a path is reproducible across runs, worker counts, and unrelated states.
    /// Default [`crate::vex::dirty::TSC_INITIAL`]; advances by
    /// [`crate::vex::dirty::TSC_STEP`] per RDTSC. Carried across fork +
    /// snapshot; merged as `max` (time is a monotonic watermark, like
    /// `heap_brk`/`mmap_base`).
    tsc_counter: u64,
    /// getopt(3) cursor — index into `argv` (POSIX `optind`). Per-state,
    /// mirrors Python's `state.libc.getopt_optind`. Default 1. Carried across
    /// fork + snapshot so each path resumes option scanning correctly.
    /// Consumed by the native getopt proc (bead angr-bhk0a.3); the
    /// loader-resolved extern-symbol address push is bead angr-bhk0a.2.
    getopt_optind: u32,
    /// getopt(3) cursor — index into the current `argv` element (POSIX
    /// `optchar`, for bundled short options like `-abc`). Mirrors
    /// `state.libc.getopt_optchar`. Default 0. See `getopt_optind`.
    getopt_optchar: u32,
    /// Guest-memory addresses of the glibc getopt(3) extern globals
    /// (`optind`/`optarg`/`optopt`), pushed Python-side at init. See
    /// [`GetoptExternAddrs`]. Carried across fork + snapshot. Consumed by the
    /// native getopt proc (bead angr-bhk0a.2).
    getopt_extern: GetoptExternAddrs,
    /// Suspended native sub-call continuations (LIFO). Empty by default; a
    /// native proc that calls a guest function and resumes pushes a
    /// [`NativeResumeFrame`] here, and the dispatcher pops it on return. Carried
    /// across fork + snapshot so each path resumes its own pending sub-calls.
    /// Foundation slice (bead angr-pn3w8); the dispatcher is S2.
    native_resume_stack: Vec<NativeResumeFrame>,
    /// Pointers to the three glibc locale ctype lookup tables. Built once by
    /// Python's `__libc_start_main` init pass (mallocs + fills them in shared
    /// memory, see `__ctype_b_loc.py` et al.) and pushed into Rust at
    /// seed-state creation. Native `__ctype_*_loc` procs return these verbatim;
    /// `None` (init pass skipped) falls back to Python.
    ctype_loc: CtypeLocPtrs,
    /// Symbolic variable names read from stdin (for posix.dumps(0) export).
    /// Each entry is (name, bit_width) for a symbolic BVS created by native
    /// fgets/fgetc/getchar. On export, Python recreates matching claripy BVS
    /// and writes them to the posix stdin plugin.
    stdin_symbols: Vec<(String, u32)>,
    /// Function call stack. Pushed on Ijk_Call, popped on Ijk_Ret.
    /// Cloned on fork so each path has its own call stack.
    call_stack: Vec<CallStackEntry>,
    /// Heap metadata tracking: allocated regions and freed addresses.
    /// Cloned on fork so each path has its own heap state.
    heap_metadata: HeapMetadata,
    /// Inspection/breakpoint system for tracking memory and register access.
    /// Only records events when enabled (single bitmask check per operation).
    inspection: InspectionManager,
    /// Environment variables map for native getenv/setenv.
    /// Keys and values are byte vectors (no NUL terminator in storage).
    /// Wrapped in Arc for cheap fork — copy-on-write via Arc::make_mut on setenv.
    /// Most paths only read env vars, so the deep clone is rare.
    ///
    /// See module-level `arc-make-mut-cow` and `arc-collection-iter`.
    environment: Arc<HashMap<Vec<u8>, Vec<u8>>>,
    /// Environment keys explicitly removed (`unsetenv`/`clearenv`) since this
    /// state's last fork point. Same rationale and reset/carry rules as
    /// [`Self::removed_hooks`].
    removed_env_keys: Arc<HashSet<Vec<u8>>>,
    /// Per-state symbolic page metadata: `addr -> claripy AST`. Holds whole-page
    /// symbolic ASTs preserved across Python fallback so Rust can re-establish
    /// symbolic memory. Migrated out of Python `_state_metadata` so storage is
    /// owned alongside the rest of the state. Each entry is a strong ref to a
    /// claripy AST; cleared automatically when the state is dropped.
    /// Cloned on fork — see [`SharedPyAst`] for why the `Arc` is load-bearing.
    ///
    /// See module-level `state-metadata-dataclass`: the Python side keeps a
    /// parallel `StateMetadata` dataclass; Rust drops decrement Py-refcounts.
    symbolic_pages: HashMap<u64, SharedPyAst>,
    /// Per-state hook symbolic memory: `addr -> (claripy AST, byte size)`.
    /// Tracks symbolic writes performed inside Python hooks so Rust can replay
    /// them on resume. Cloned on fork.
    ///
    /// See module-level `state-metadata-dataclass`.
    hook_symbolic_memory: HashMap<u64, (SharedPyAst, u32)>,
    /// Per-state addr -> (AST, byte size) recorded by handle registration so
    /// state export can recover the original symbol instead of a fresh BVS.
    /// Cloned on fork.
    ///
    /// See module-level `state-metadata-dataclass`.
    addr_to_ast: HashMap<u64, (SharedPyAst, u32)>,
    /// Most recent symbolic value returned by the time(2) syscall — mirrors
    /// `state.globals['sys_last_time']` in Python's
    /// `procedures/linux_kernel/time.py`. Used to constrain consecutive calls
    /// to be monotonic (`new >= prev`). Cloned on fork; not synced across the
    /// Python boundary today (same drift class as `posix_brk` / `mmap_base`).
    last_time: Option<RustBV>,
    /// Mirrors angr's NO_IP_CONCRETIZATION SimOption. When true, symbolic
    /// jump targets are NOT enumerated via solver — the state routes to the
    /// unconstrained stash without warning. See engines/successors.py:292-296.
    /// Cloned on fork.
    ///
    /// See module-level `invariant-apply-state-metadata-option-allowlist`: this field is
    /// NOT in the `_apply_state_metadata` allow-list, so it must be set on
    /// the user-supplied state in Python `__init__` BEFORE the init pipeline
    /// hits the disk cache — otherwise the cached-init path will silently
    /// reset it to the default on a cache hit.
    no_ip_concretization: bool,
    /// Mirrors angr's NO_SYMBOLIC_JUMP_RESOLUTION SimOption. When true, any
    /// symbolic jump target routes the state to the unconstrained stash
    /// before enumeration is attempted. See engines/successors.py:234-239.
    /// Behaviourally identical to `no_ip_concretization` for the Rust engine
    /// (both short-circuit `eval_next_addr_concretized` for symbolic IPs);
    /// they are separate flags to preserve Python option semantics. Cloned
    /// on fork.
    ///
    /// See module-level `invariant-apply-state-metadata-option-allowlist` — same caveat
    /// as `no_ip_concretization`.
    no_symbolic_jump_resolution: bool,
    /// Mirrors angr's KEEP_IP_SYMBOLIC SimOption. When true, after a symbolic
    /// jump target is concretized to one-or-more concrete pc values, the IP
    /// register is left set to the original symbolic expression (not the
    /// concretized constant) and no `target == addr` narrowing constraint is
    /// added per fork. The engine still uses the concrete `pc` value to drive
    /// the next block lift. See engines/successors.py:297-307,326-331.
    /// Cloned on fork.
    ///
    /// See module-level `invariant-apply-state-metadata-option-allowlist` — same caveat
    /// as `no_ip_concretization`.
    keep_ip_symbolic: bool,
    /// angr-027h: per-state override that forces EAGER forking (immediate
    /// successor materialization) regardless of the manager-level
    /// `ExecutionConfig::use_deferred_forks`. Set on the loop-exit forks that
    /// are resumed at an UnconstrainedJump (where the deferred main chain
    /// overflowed the saved return address and went unconstrained). Without
    /// this, resumed forks continue in deferred mode, re-dive the symbolic
    /// loop nest, re-overflow, and recursively diverge (iter63). Eager
    /// resumption lets them BFS cleanly to a find target. Cloned on fork so
    /// the whole resumed subtree stays eager. Measured on CADET_00001: with
    /// this set, the easter-egg target is reachable; without it the resumed
    /// subtree diverges instead.
    force_eager_forks: bool,
    /// CGC `state.cgc.allocation_base` — high-water bump pointer for the
    /// CGC `allocate(2)` syscall. Pages grow downward from this address.
    /// Default 0xB800_0000 (matches `state_plugins/cgc.py::allocation_base`).
    /// Inert outside DECREE binaries. Cloned on fork.
    cgc_allocation_base: u64,
    /// CGC `state.cgc.sinkholes` — list of freed (addr, length) regions that
    /// `allocate` re-uses via first-fit before bumping `allocation_base`.
    /// Mirrors `state_plugins/cgc.py::sinkholes` (which is a `set`, but we
    /// store an ordered `Vec` to match the Python "sorted by address
    /// descending, first fit" semantics in `get_max_sinkhole`).
    /// Cloned on fork.
    cgc_sinkholes: Vec<(u64, u64)>,
    /// Symex-relevant SimOption names (e.g. `"SHORT_READS"`) mirrored from the
    /// Python SimState option set so native SimProcedures can branch on them
    /// via [`RustSimState::has_option`]. Only the symex-relevant subset is
    /// threaded across the FFI (see `_add_rust_state` in `rust_manager.py`),
    /// not the full option set — the full set stays Python-side
    /// (`rust_state_proxy.options`). Wrapped in `Arc` for cheap fork —
    /// copy-on-write via `Arc::make_mut` on `set_option`; option sets are
    /// configured once at construction in typical workloads, so most forks pay
    /// only an `Arc` refcount bump.
    ///
    /// See module-level `invariant-apply-state-metadata-option-allowlist`: like the other
    /// option mirrors, this is NOT in the `_apply_state_metadata` allow-list,
    /// so it must be set on the Rust state during `_add_rust_state` rather than
    /// relying on the cached-init path to preserve it.
    sim_options: Arc<HashSet<String>>,
    /// SimOption names explicitly disabled (`set_option(name, false)`) since
    /// this state's last fork point. Same rationale and reset/carry rules as
    /// [`Self::removed_hooks`].
    removed_sim_options: Arc<HashSet<String>>,
}

impl RustSimState {
    // =========================================================================
    // Basic Accessors
    // =========================================================================

    /// Get the state ID.
    pub fn state_id(&self) -> u64 {
        self.state_id
    }

    /// Get the parent state ID.
    pub fn parent_id(&self) -> Option<u64> {
        self.parent_id
    }

    /// Get the program counter.
    pub fn pc(&self) -> u64 {
        self.pc
    }

    /// Set the program counter.
    /// Also updates the IP register in the register file so that
    /// get_register("rip"/"eip") returns the current PC.
    pub fn set_pc(&mut self, pc: u64) {
        self.pc = pc;
        let width = self.arch.bits();
        self.registers.set_ip(RustBV::concrete(pc as u128, width));
    }

    /// Get the VEX architecture.
    pub fn vex_arch(&self) -> VexArch {
        self.vex_arch
    }

    /// Get the architecture.
    pub fn arch(&self) -> &dyn Arch {
        self.arch.as_ref()
    }

    // Process / OS-environment state (fd output buffers, stdin symbols, the
    // environment map, heap_brk / posix_brk / mmap_base break pointers, the
    // getopt(3) cursor and extern-global addresses, ctype_loc, the native
    // sub-call resume stack, the CGC allocator/sinkhole state, last_time, and
    // the heap bump-allocator) lives in `process.rs` (extension impl). The
    // inspection manager accessor + inspect_mem_read/write/fork/exit event
    // recorders live in `inspection.rs` alongside the InspectionManager type.

    // History (basic-block visit log, plain + detailed) and call-stack
    // tracking (push_call / pop_call / set_call_stack, honoring max_history)
    // live in `history.rs` (extension impl).

    // Register access (get/set by name & offset, IP/SP, raw bulk bytes, and the
    // RegisterFile ref/replace accessors) lives in `registers.rs`.

    // =========================================================================
    // Memory Access
    // =========================================================================
    //
    // The memory reference/replace accessors, region mapping, dirty-page
    // tracking, lazy regions, and the concrete/symbolic load/store bridge live
    // in `memory.rs` (extension impl). The options/flags cluster
    // (STRICT_PAGE_ACCESS / ENABLE_NX / NO_IP_CONCRETIZATION /
    // NO_SYMBOLIC_JUMP_RESOLUTION / KEEP_IP_SYMBOLIC, `set_option`/`has_option`,
    // eager-fork forcing, and the `SharedLineageSolver` opt-in) lives in
    // `options.rs`. The hook-address set and per-state claripy-AST metadata
    // maps (hook-symbolic-memory, addr-to-AST, symbolic-pages) live in
    // `hooks.rs`.

    // =========================================================================
    // Incremental State Changes
    // =========================================================================

    /// Apply incremental changes to the state.
    ///
    /// This is used when syncing from Python after a SimProcedure runs.
    pub fn apply_changes(&mut self, changes: &StateChanges) {
        // Apply register writes
        for (offset, size, bytes) in &changes.register_writes {
            let mut value: u128 = 0;
            for (i, &b) in bytes.iter().enumerate() {
                value |= (b as u128) << (i * 8);
            }
            let bv = RustBV::concrete(value, *size * 8);
            self.set_register_by_offset(*offset, bv);
        }

        // Apply memory writes — split into 16-byte chunks since RustBV
        // uses u128 internally (max 128 bits per concrete value)
        for (addr, bytes) in &changes.memory_writes {
            let mut offset = 0usize;
            while offset < bytes.len() {
                let remaining = bytes.len() - offset;
                let chunk_size = remaining.min(16);
                let chunk = &bytes[offset..offset + chunk_size];
                let width = (chunk_size * 8) as u32;
                let mut value: u128 = 0;
                for (i, &b) in chunk.iter().enumerate() {
                    value |= (b as u128) << (i * 8);
                }
                let bv = RustBV::concrete(value, width);
                let _ = self.memory.store_concrete(*addr + offset as u64, bv);
                offset += chunk_size;
            }
        }

        // Apply PC change
        if let Some(new_pc) = changes.new_pc {
            self.pc = new_pc;
        }
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Set the address concretization strategy.
    pub fn set_concretizer(&mut self, concretizer: AddressConcretizer) {
        self.concretizer = concretizer;
    }

    /// Configure address concretization.
    pub fn configure_concretization(&mut self, use_approximate: bool, range_limit: Option<u64>) {
        self.concretizer.configure(use_approximate, range_limit);
    }

    /// Set whether to track history.
    pub fn set_track_history(&mut self, track: bool) {
        self.track_history = track;
    }

    /// Set maximum history length.
    ///
    /// Applied retroactively: if the existing `history` or `detailed_history`
    /// buffers already exceed the new cap, oldest entries are evicted (FIFO)
    /// down to `max`. Without this trim, lowering the cap on a state with a
    /// long buffer would leave it stuck — `add_to_history` removes only one
    /// entry per push, so the buffer never converges to the new cap.
    /// `max = 0` disables the cap (legacy unlimited behavior).
    pub fn set_max_history(&mut self, max: usize) {
        self.max_history = max;
        if max > 0 {
            if self.history.len() > max {
                let drop = self.history.len() - max;
                self.history.drain(0..drop);
            }
            if self.detailed_history.len() > max {
                let drop = self.detailed_history.len() - max;
                self.detailed_history.drain(0..drop);
            }
        }
    }

    /// Current maximum history length (`0` = unlimited).
    ///
    /// Read counterpart to `set_max_history`. Exists so the Python-facing
    /// `PyRustSimState::get_max_history` reads the cap through an accessor
    /// like every other field it exposes, rather than reaching into the
    /// private `max_history` field directly (angr-9ke6b.124).
    pub fn max_history(&self) -> usize {
        self.max_history
    }
}

/// Python-facing wrapper for RustSimState.
///
/// This provides the PyO3 interface for creating and manipulating
/// Rust-native simulation states from Python.
#[pyclass(name = "RustSimState", unsendable)]
pub struct PyRustSimState {
    inner: RustSimState,
}

impl PyRustSimState {
    /// Get access to the inner state (for Rust-side use).
    pub fn inner(&self) -> &RustSimState {
        &self.inner
    }
}

#[cfg(test)]
#[path = "../state_tests.rs"]
mod state_tests;
