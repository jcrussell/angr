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
//! the source-of-truth header in
//! `angr/exploration/rust_manager.py:10-100`. Most of those concerns are
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
//! of) I1–I8 above. Each refers to a bd memory key with the full rationale
//! and history; enforcement-site comments below cross-reference back to
//! this header rather than duplicating the prose.
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
//!   `debug_assert!` in `fork()` confirms the child ID is fresh.
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
//! - **`apply-state-metadata-strips-options`** — `RustExplorationManager.
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
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::arch::{Arch, RegisterFile, arch_from_name, arch_from_vex};
use crate::concretize::AddressConcretizer;
use crate::memory::{MemoryError, Permission, SymbolicMemory};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::{Endness, VexArch};

mod export;
mod filesystem;
mod fork;
mod inspection;
mod snapshot;
mod types;

pub use export::*;
pub use filesystem::*;
pub use inspection::*;
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
    /// Basic block history (addresses visited).
    history: Vec<u64>,
    /// Detailed execution history with jumpkind and target info.
    detailed_history: Vec<HistoryEntry>,
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
    /// Per-state symbolic page metadata: `addr -> claripy AST`. Holds whole-page
    /// symbolic ASTs preserved across Python fallback so Rust can re-establish
    /// symbolic memory. Migrated out of Python `_state_metadata` so storage is
    /// owned alongside the rest of the state. Each PyObject is a strong ref to
    /// a claripy AST; cleared automatically when the state is dropped.
    /// Cloned on fork (Py refcounts incremented; cheap for a few entries).
    ///
    /// See module-level `state-metadata-dataclass`: the Python side keeps a
    /// parallel `StateMetadata` dataclass; Rust drops decrement Py-refcounts.
    symbolic_pages: HashMap<u64, Py<PyAny>>,
    /// Per-state hook symbolic memory: `addr -> (claripy AST, byte size)`.
    /// Tracks symbolic writes performed inside Python hooks so Rust can replay
    /// them on resume. Cloned on fork.
    ///
    /// See module-level `state-metadata-dataclass`.
    hook_symbolic_memory: HashMap<u64, (Py<PyAny>, u32)>,
    /// Per-state addr -> (AST, byte size) recorded by handle registration so
    /// state export can recover the original symbol instead of a fresh BVS.
    /// Cloned on fork.
    ///
    /// See module-level `state-metadata-dataclass`.
    addr_to_ast: HashMap<u64, (Py<PyAny>, u32)>,
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
    /// See module-level `apply-state-metadata-strips-options`: this field is
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
    /// See module-level `apply-state-metadata-strips-options` — same caveat
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
    /// See module-level `apply-state-metadata-strips-options` — same caveat
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
    /// the whole resumed subtree stays eager. See bd memory
    /// `benchmark-cadet-eager-reaches-egg`.
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
}

impl RustSimState {
    /// Create a new state for the given architecture.
    ///
    /// # Arguments
    /// * `arch_name` - Architecture name (e.g., "amd64", "x86", "arm")
    /// * `little_endian` - Override endianness (None = use arch default)
    ///
    /// # Returns
    /// New state with default initialization.
    pub fn new(arch_name: &str) -> Result<Self, String> {
        Self::new_with_endian(arch_name, None)
    }

    /// Create a new state with explicit endianness override.
    pub fn new_with_endian(arch_name: &str, little_endian: Option<bool>) -> Result<Self, String> {
        let arch = arch_from_name(arch_name)
            .ok_or_else(|| format!("unknown architecture: {}", arch_name))?;
        let vex_arch = arch.vex_arch();
        let is_le = little_endian.unwrap_or_else(|| arch.is_little_endian());
        let endness = if is_le { Endness::Little } else { Endness::Big };

        Ok(RustSimState {
            vex_arch,
            registers: RegisterFile::new(arch.clone()),
            memory: SymbolicMemory::new(endness),
            solver: Rc::new(RefCell::new(SymContext::new())),
            pc: 0,
            state_id: next_state_id(),
            parent_id: None,
            history: Vec::new(),
            detailed_history: Vec::new(),
            max_history: 1000,
            hooks: Arc::new(HashSet::new()),
            concretizer: AddressConcretizer::default(),
            track_history: true,
            arch,
            fs: FileSystem::default(),
            heap_brk: 0xC000_0000,
            posix_brk: 0x1B0_0000,
            mmap_base: 0xC100_0000,
            stdin_symbols: Vec::new(),
            call_stack: Vec::new(),
            heap_metadata: HeapMetadata::default(),
            inspection: InspectionManager::default(),
            environment: Arc::new(HashMap::new()),
            symbolic_pages: HashMap::new(),
            hook_symbolic_memory: HashMap::new(),
            addr_to_ast: HashMap::new(),
            last_time: None,
            no_ip_concretization: false,
            no_symbolic_jump_resolution: false,
            keep_ip_symbolic: false,
            force_eager_forks: false,
            cgc_allocation_base: 0xB800_0000,
            cgc_sinkholes: Vec::new(),
        })
    }

    /// Create a state from VexArch.
    pub fn from_vex_arch(vex_arch: VexArch) -> Self {
        let arch = arch_from_vex(vex_arch);
        let endness = if arch.is_little_endian() {
            Endness::Little
        } else {
            Endness::Big
        };

        RustSimState {
            vex_arch,
            registers: RegisterFile::new(arch.clone()),
            memory: SymbolicMemory::new(endness),
            solver: Rc::new(RefCell::new(SymContext::new())),
            pc: 0,
            state_id: next_state_id(),
            parent_id: None,
            history: Vec::new(),
            detailed_history: Vec::new(),
            max_history: 1000,
            hooks: Arc::new(HashSet::new()),
            concretizer: AddressConcretizer::default(),
            track_history: true,
            arch,
            fs: FileSystem::default(),
            heap_brk: 0xC000_0000,
            posix_brk: 0x1B0_0000,
            mmap_base: 0xC100_0000,
            stdin_symbols: Vec::new(),
            call_stack: Vec::new(),
            heap_metadata: HeapMetadata::default(),
            inspection: InspectionManager::default(),
            environment: Arc::new(HashMap::new()),
            symbolic_pages: HashMap::new(),
            hook_symbolic_memory: HashMap::new(),
            addr_to_ast: HashMap::new(),
            last_time: None,
            no_ip_concretization: false,
            no_symbolic_jump_resolution: false,
            keep_ip_symbolic: false,
            force_eager_forks: false,
            cgc_allocation_base: 0xB800_0000,
            cgc_sinkholes: Vec::new(),
        }
    }

    /// Create a state with a shared solver context.
    ///
    /// This is used when forking to share constraints across states.
    pub fn with_solver(arch_name: &str, solver: Rc<RefCell<SymContext>>) -> Result<Self, String> {
        Self::with_solver_endian(arch_name, solver, None)
    }

    /// Create a state with a shared solver context and explicit endianness.
    pub fn with_solver_endian(
        arch_name: &str,
        solver: Rc<RefCell<SymContext>>,
        little_endian: Option<bool>,
    ) -> Result<Self, String> {
        let arch = arch_from_name(arch_name)
            .ok_or_else(|| format!("unknown architecture: {}", arch_name))?;
        let vex_arch = arch.vex_arch();
        let is_le = little_endian.unwrap_or_else(|| arch.is_little_endian());
        let endness = if is_le { Endness::Little } else { Endness::Big };

        Ok(RustSimState {
            vex_arch,
            registers: RegisterFile::new(arch.clone()),
            memory: SymbolicMemory::new(endness),
            solver,
            pc: 0,
            state_id: next_state_id(),
            parent_id: None,
            history: Vec::new(),
            detailed_history: Vec::new(),
            max_history: 1000,
            hooks: Arc::new(HashSet::new()),
            concretizer: AddressConcretizer::default(),
            track_history: true,
            arch,
            fs: FileSystem::default(),
            heap_brk: 0xC000_0000,
            posix_brk: 0x1B0_0000,
            mmap_base: 0xC100_0000,
            stdin_symbols: Vec::new(),
            call_stack: Vec::new(),
            heap_metadata: HeapMetadata::default(),
            inspection: InspectionManager::default(),
            environment: Arc::new(HashMap::new()),
            symbolic_pages: HashMap::new(),
            hook_symbolic_memory: HashMap::new(),
            addr_to_ast: HashMap::new(),
            last_time: None,
            no_ip_concretization: false,
            no_symbolic_jump_resolution: false,
            keep_ip_symbolic: false,
            force_eager_forks: false,
            cgc_allocation_base: 0xB800_0000,
            cgc_sinkholes: Vec::new(),
        })
    }

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

    /// Get the stdout buffer (fd=1).
    pub fn stdout_buffer(&self) -> &[u8] {
        self.fd_buffer(1)
    }

    /// Append bytes to the stdout buffer (fd=1).
    pub fn write_stdout(&mut self, data: &[u8]) {
        self.write_fd(1, data);
    }

    /// Check if stdout has been written to.
    pub fn has_stdout(&self) -> bool {
        !self.fs.fd_content(1).is_empty()
    }

    /// Get the output buffer for a file descriptor.
    pub fn fd_buffer(&self, fd: u32) -> &[u8] {
        self.fs.fd_content(fd)
    }

    /// Append bytes to a file descriptor's output buffer.
    pub fn write_fd(&mut self, fd: u32, data: &[u8]) {
        self.fs.write(fd, data);
    }

    /// Get a mutable reference to the file system state.
    pub fn file_system(&mut self) -> &mut FileSystem {
        &mut self.fs
    }

    /// Get a read-only reference to the file system state.
    pub fn file_system_ref(&self) -> &FileSystem {
        &self.fs
    }

    /// Record a symbolic variable that was read from stdin.
    /// Used by native fgets/fgetc/getchar to track stdin reads for posix.dumps(0).
    pub fn record_stdin_symbol(&mut self, name: String, bits: u32) {
        self.stdin_symbols.push((name, bits));
    }

    /// Get the list of symbolic variables read from stdin.
    /// Returns (name, bit_width) tuples in read order.
    pub fn stdin_symbols(&self) -> &[(String, u32)] {
        &self.stdin_symbols
    }

    /// Check if any stdin symbols have been recorded.
    pub fn has_stdin_symbols(&self) -> bool {
        !self.stdin_symbols.is_empty()
    }

    /// Get an environment variable value by key.
    pub fn getenv(&self, key: &[u8]) -> Option<&[u8]> {
        self.environment.get(key).map(|v| v.as_slice())
    }

    /// Set an environment variable.
    pub fn setenv(&mut self, key: Vec<u8>, value: Vec<u8>) {
        Arc::make_mut(&mut self.environment).insert(key, value);
    }

    /// Remove an environment variable. Returns true if the key was present.
    pub fn unsetenv(&mut self, key: &[u8]) -> bool {
        let env = Arc::make_mut(&mut self.environment);
        env.remove(key).is_some()
    }

    /// Clear all environment variables.
    pub fn clearenv(&mut self) {
        Arc::make_mut(&mut self.environment).clear();
    }

    /// Get the environment map (for export).
    pub fn environment(&self) -> &HashMap<Vec<u8>, Vec<u8>> {
        &self.environment
    }

    /// Get the current heap brk pointer.
    pub fn heap_brk(&self) -> u64 {
        self.heap_brk
    }

    /// Get the POSIX brk pointer (mirrors `state.posix.brk` for the brk(2)
    /// syscall). Distinct from `heap_brk`, which is the malloc bump allocator.
    ///
    /// **Invariant I7 (cross-mixin sync):** see `set_posix_brk`.
    pub fn posix_brk(&self) -> u64 {
        self.posix_brk
    }

    /// Set the POSIX brk pointer.
    ///
    /// **Invariant I7 (cross-mixin sync):** Python and Rust both mutate
    /// `posix_brk` independently — Rust on native brk(2), Python on
    /// fallback syscall handlers and user mutation of `state.posix.brk`.
    /// The Python export path computes `max(rust_value, python_value)` so
    /// neither side is silently rewound by a stale value. This Rust setter
    /// is the commanded path: it takes whatever value the caller (native
    /// syscall handler or the FFI cross-sync) supplies, without enforcing
    /// monotonicity locally. Directionality lives at the Python export
    /// site. Regression test:
    /// `TestPosixBrkSync.test_export_path_syncs_rust_posix_brk_into_state_posix`.
    pub fn set_posix_brk(&mut self, addr: u64) {
        self.posix_brk = addr;
    }

    /// Get the mmap base pointer (mirrors `state.heap.mmap_base` for the
    /// mmap(2) syscall; advances when addr=0 native mmap allocates).
    ///
    /// **Invariant I7 (cross-mixin sync):** see `set_mmap_base`.
    pub fn mmap_base(&self) -> u64 {
        self.mmap_base
    }

    /// Set the mmap base pointer.
    ///
    /// **Invariant I7 (cross-mixin sync):** Python and Rust both mutate
    /// `mmap_base` independently — Rust on native mmap(2) with addr=0,
    /// Python on fallback syscall handlers and user mutation of
    /// `state.heap.mmap_base`. The Python export path computes
    /// `max(rust_value, python_value)` so a Python-side advance survives
    /// even if Rust still holds a stale lower value. This Rust setter is
    /// the commanded path: it takes whatever value the caller supplies,
    /// without enforcing monotonicity locally. Directionality lives at
    /// the Python export site. Regression test:
    /// `TestMmapBaseSync.test_export_path_does_not_clobber_higher_python_mmap_base`.
    pub fn set_mmap_base(&mut self, addr: u64) {
        self.mmap_base = addr;
    }

    /// CGC `state.cgc.allocation_base` — current high-water bump pointer
    /// used by the native CGC `allocate(5)` syscall. Inert for non-CGC
    /// binaries. Mirror of `state_plugins/cgc.py::allocation_base`.
    pub fn cgc_allocation_base(&self) -> u64 {
        self.cgc_allocation_base
    }

    /// Update the CGC bump-pointer high-water. Called from the native
    /// `allocate` handler after a fresh bump and (in principle) from the
    /// Python→Rust state sync path when a Python-side allocate ran.
    pub fn set_cgc_allocation_base(&mut self, base: u64) {
        self.cgc_allocation_base = base;
    }

    /// CGC `state.cgc.sinkholes` — `(addr, length)` freelist of
    /// previously-deallocated regions, candidates for reuse by `allocate`.
    pub fn cgc_sinkholes(&self) -> &[(u64, u64)] {
        &self.cgc_sinkholes
    }

    /// Add a region to the CGC sinkhole freelist. Mirrors
    /// `SimStateCGC.add_sinkhole`. Duplicate `(addr, length)` pairs are
    /// silently merged into a single entry (the Python plugin uses a `set`).
    pub fn cgc_add_sinkhole(&mut self, addr: u64, length: u64) {
        if !self
            .cgc_sinkholes
            .iter()
            .any(|&(a, l)| a == addr && l == length)
        {
            self.cgc_sinkholes.push((addr, length));
        }
    }

    /// CGC first-fit allocator over the sinkhole freelist. Walks sinkholes
    /// in descending-address order (matching `SimStateCGC.get_max_sinkhole`)
    /// and returns the first one big enough to fit `length` bytes. The
    /// chosen sinkhole is split if larger than `length`: the leftover at
    /// the LOW end stays in the freelist, the HIGH end is returned.
    /// Returns `None` if no sinkhole fits — caller bumps `allocation_base`.
    pub fn cgc_take_max_sinkhole(&mut self, length: u64) -> Option<u64> {
        // Find index of highest-address sinkhole that fits.
        let mut best: Option<usize> = None;
        let mut best_addr: u64 = 0;
        for (i, &(addr, sz)) in self.cgc_sinkholes.iter().enumerate() {
            if sz >= length && (best.is_none() || addr > best_addr) {
                best = Some(i);
                best_addr = addr;
            }
        }
        let idx = best?;
        let (addr, sz) = self.cgc_sinkholes.swap_remove(idx);
        let remaining = sz - length;
        let chosen = addr + remaining;
        if remaining > 0 {
            self.cgc_sinkholes.push((addr, remaining));
        }
        Some(chosen)
    }

    /// Most recent symbolic value returned by the time(2) syscall, used to
    /// chain monotonic constraints across consecutive calls. Mirrors
    /// `state.globals['sys_last_time']` in Python's
    /// `procedures/linux_kernel/time.py`.
    pub fn last_time(&self) -> Option<&RustBV> {
        self.last_time.as_ref()
    }

    /// Record the most recent time(2) return value (called by the native time
    /// syscall handler).
    pub fn set_last_time(&mut self, bv: RustBV) {
        self.last_time = Some(bv);
    }

    /// Bump-allocate from the heap. Returns the address of the allocation.
    /// Aligns size up to 16 bytes (matching angr's SimHeapBrk).
    pub fn heap_alloc(&mut self, size: u64) -> u64 {
        let aligned = (size + 15) & !15; // round up to 16
        let addr = self.heap_brk;
        self.heap_brk = addr.wrapping_add(aligned);
        self.heap_metadata.record_alloc(addr, size);
        addr
    }

    /// Bump-allocate `size` bytes with the returned address aligned to
    /// `alignment` (must be a power of 2). The size bump is rounded up to 16,
    /// matching `heap_alloc`, so consecutive allocations remain aligned.
    /// Falls back to `heap_alloc` semantics when `alignment` is 0 or 1.
    pub fn heap_alloc_aligned(&mut self, size: u64, alignment: u64) -> u64 {
        if alignment <= 1 {
            return self.heap_alloc(size);
        }
        let mask = alignment - 1;
        let addr = self.heap_brk.wrapping_add(mask) & !mask;
        let aligned = (size + 15) & !15;
        self.heap_brk = addr.wrapping_add(aligned);
        self.heap_metadata.record_alloc(addr, size);
        addr
    }

    /// Record a heap free. Returns the original allocation size if tracked.
    pub fn heap_free(&mut self, addr: u64) -> Option<u64> {
        self.heap_metadata.record_free(addr)
    }

    /// Get heap metadata (for analysis/export).
    pub fn heap_metadata(&self) -> &HeapMetadata {
        &self.heap_metadata
    }

    /// Get the inspection manager (read-only).
    pub fn inspection(&self) -> &InspectionManager {
        &self.inspection
    }

    /// Get the inspection manager (mutable).
    pub fn inspection_mut(&mut self) -> &mut InspectionManager {
        &mut self.inspection
    }

    /// Record a memory read event (if mem_read inspection is enabled).
    #[inline(always)]
    pub fn inspect_mem_read(&mut self, addr: u64, size: u32) {
        if self.inspection.is_enabled(InspectEvent::MemRead) {
            self.inspection
                .record(InspectEvent::MemRead, addr, size, self.pc);
        }
    }

    /// Record a memory write event (if mem_write inspection is enabled).
    #[inline(always)]
    pub fn inspect_mem_write(&mut self, addr: u64, size: u32) {
        if self.inspection.is_enabled(InspectEvent::MemWrite) {
            self.inspection
                .record(InspectEvent::MemWrite, addr, size, self.pc);
        }
    }

    /// Record a fork event (if fork inspection is enabled).
    #[inline(always)]
    pub fn inspect_fork(&mut self) {
        if self.inspection.is_enabled(InspectEvent::Fork) {
            self.inspection.record(InspectEvent::Fork, 0, 0, self.pc);
        }
    }

    /// Record an exit event (if exit inspection is enabled).
    #[inline(always)]
    pub fn inspect_exit(&mut self) {
        if self.inspection.is_enabled(InspectEvent::Exit) {
            self.inspection.record(InspectEvent::Exit, 0, 0, self.pc);
        }
    }

    /// Get the history (basic block addresses visited).
    pub fn history(&self) -> &[u64] {
        &self.history
    }

    /// Add an address to history.
    pub fn add_to_history(&mut self, addr: u64) {
        if self.track_history {
            self.history.push(addr);
            if self.max_history > 0 && self.history.len() > self.max_history {
                self.history.remove(0);
            }
        }
    }

    /// Get the detailed execution history.
    pub fn detailed_history(&self) -> &[HistoryEntry] {
        &self.detailed_history
    }

    /// Add a detailed history entry.
    pub fn add_history_entry(&mut self, addr: u64, jumpkind: u8, jump_target: u64) {
        if self.track_history {
            self.detailed_history.push(HistoryEntry {
                addr,
                jumpkind,
                jump_target,
            });
            if self.max_history > 0 && self.detailed_history.len() > self.max_history {
                self.detailed_history.remove(0);
            }
        }
    }

    /// Replace the detailed history (used when restoring from interpreter).
    /// Honors `max_history` — if the incoming buffer is larger than the cap,
    /// only the most-recent `max_history` entries are kept (FIFO eviction).
    pub fn set_detailed_history(&mut self, mut history: Vec<HistoryEntry>) {
        if self.max_history > 0 && history.len() > self.max_history {
            let drop = history.len() - self.max_history;
            history.drain(0..drop);
        }
        self.detailed_history = history;
    }

    // =========================================================================
    // Call Stack Tracking
    // =========================================================================

    /// Get the current call stack.
    pub fn call_stack(&self) -> &[CallStackEntry] {
        &self.call_stack
    }

    /// Get the call stack depth.
    pub fn call_stack_depth(&self) -> usize {
        self.call_stack.len()
    }

    /// Push a call onto the call stack (on Ijk_Call).
    pub fn push_call(
        &mut self,
        call_site_addr: u64,
        callee_addr: u64,
        return_addr: u64,
        stack_ptr: u64,
    ) {
        self.call_stack.push(CallStackEntry {
            call_site_addr,
            callee_addr,
            return_addr,
            stack_ptr,
        });
    }

    /// Pop a call from the call stack (on Ijk_Ret).
    /// Returns the popped entry, or None if the stack is empty.
    pub fn pop_call(&mut self) -> Option<CallStackEntry> {
        self.call_stack.pop()
    }

    /// Get the current function address (top of call stack), if any.
    pub fn current_function_addr(&self) -> Option<u64> {
        self.call_stack.last().map(|e| e.callee_addr)
    }

    /// Replace the call stack (used when restoring from interpreter).
    pub fn set_call_stack(&mut self, call_stack: Vec<CallStackEntry>) {
        self.call_stack = call_stack;
    }

    // =========================================================================
    // Register Access
    // =========================================================================

    /// Get a register by name.
    pub fn get_register(&self, name: &str) -> Option<RustBV> {
        let ctx = self.solver.borrow();
        self.registers.get_reg(name, &ctx)
    }

    /// Set a register by name.
    pub fn set_register(&mut self, name: &str, value: RustBV) -> bool {
        // angr-4rq7: writing the IP register (e.g. via the RustRegisterProxy
        // write-through gate, which routes `state.regs.ip = target` to
        // set_state_register_symbolic_ast -> set_register("rip", ...)) must
        // keep self.pc in sync, exactly as set_ip/set_pc do. Without this,
        // the register file holds the real address but self.pc stays stale
        // (often 0 on a freshly-forked state): get_state_pc_by_id() and the
        // next block fetch then read 0x0 ("Lift error at 0x0") while
        // get_register("rip") reads correctly — the divergence behind the
        // register-proxy gate corruption.
        let is_ip = self
            .arch
            .register_offset(name)
            .map(|off| off == self.arch.ip_offset())
            .unwrap_or(false);
        let ok = self.registers.put_reg(name, value);
        if ok
            && is_ip
            && let Some(v) = self.registers.get_ip(&self.solver.borrow()).as_u64()
        {
            self.pc = v;
        }
        ok
    }

    /// Get a register by offset.
    pub fn get_register_by_offset(&self, offset: u32, size: u32) -> RustBV {
        let ctx = self.solver.borrow();
        self.registers.get(offset, size, &ctx)
    }

    /// Set a register by offset.
    pub fn set_register_by_offset(&mut self, offset: u32, value: RustBV) {
        self.registers.put(offset, value);
    }

    /// Get the instruction pointer register.
    pub fn get_ip(&self) -> RustBV {
        let ctx = self.solver.borrow();
        self.registers.get_ip(&ctx)
    }

    /// Set the instruction pointer register.
    pub fn set_ip(&mut self, value: RustBV) {
        self.registers.set_ip(value);
        if let Some(v) = self.registers.get_ip(&self.solver.borrow()).as_u64() {
            self.pc = v;
        }
    }

    /// Get the stack pointer register.
    pub fn get_sp(&self) -> RustBV {
        let ctx = self.solver.borrow();
        self.registers.get_sp(&ctx)
    }

    /// Set the stack pointer register.
    pub fn set_sp(&mut self, value: RustBV) {
        self.registers.set_sp(value);
    }

    /// Get all register bytes (for bulk sync to Python).
    pub fn get_registers_raw(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.arch.state_size()];
        self.registers.copy_to_bytes(&mut bytes);
        bytes
    }

    /// Set all register bytes (for bulk sync from Python).
    pub fn set_registers_raw(&mut self, bytes: &[u8]) {
        self.registers.copy_from_bytes(bytes);
    }

    // =========================================================================
    // Memory Access
    // =========================================================================

    /// Get a reference to the memory.
    pub fn memory(&self) -> &SymbolicMemory {
        &self.memory
    }

    /// Get a mutable reference to the memory.
    pub fn memory_mut(&mut self) -> &mut SymbolicMemory {
        &mut self.memory
    }

    /// Take ownership of the memory, replacing it with an empty SymbolicMemory.
    pub fn take_memory(&mut self) -> SymbolicMemory {
        let endness = self.memory.endness();
        std::mem::replace(&mut self.memory, SymbolicMemory::new(endness))
    }

    /// Replace the memory with the given SymbolicMemory.
    pub fn replace_memory(&mut self, memory: SymbolicMemory) {
        self.memory = memory;
    }

    /// Get a reference to the register file.
    pub fn registers(&self) -> &RegisterFile {
        &self.registers
    }

    /// Replace the register file (including symbolic entries).
    pub fn set_registers(&mut self, registers: RegisterFile) {
        self.registers = registers;
    }

    /// Get dirty page numbers (page_num = addr >> 12) from the memory.
    pub fn get_dirty_page_nums(&self) -> Vec<u64> {
        self.memory.get_dirty_pages()
    }

    /// Map a memory region.
    pub fn map_memory(&mut self, addr: u64, size: u64, permissions: Permission) {
        self.memory.map(addr, size, permissions);
    }

    /// Enable or disable strict memory permission enforcement.
    /// Mirrors angr's STRICT_PAGE_ACCESS option.
    pub fn set_enforce_permissions(&mut self, enabled: bool) {
        self.memory.set_enforce_permissions(enabled);
    }

    /// Whether strict memory permission enforcement is enabled.
    pub fn enforce_permissions(&self) -> bool {
        self.memory.enforce_permissions()
    }

    /// Enable or disable non-executable page enforcement on instruction fetch.
    /// Mirrors angr's ENABLE_NX option. The X check fires only when this AND
    /// `enforce_permissions` (STRICT_PAGE_ACCESS) are both on, matching
    /// Python's heavy VEX engine.
    pub fn set_enforce_nx(&mut self, enabled: bool) {
        self.memory.set_enforce_nx(enabled);
    }

    /// Whether non-executable page enforcement is enabled.
    pub fn enforce_nx(&self) -> bool {
        self.memory.enforce_nx()
    }

    /// Enable or disable IP concretization at block boundaries.
    /// Mirrors angr's NO_IP_CONCRETIZATION option. When true, a symbolic jump
    /// target routes the state to the unconstrained stash without warning
    /// (matches engines/successors.py:292-296). Default off.
    pub fn set_no_ip_concretization(&mut self, enabled: bool) {
        self.no_ip_concretization = enabled;
    }

    /// Whether IP concretization is suppressed for symbolic jump targets.
    pub fn no_ip_concretization(&self) -> bool {
        self.no_ip_concretization
    }

    /// Enable or disable resolution of symbolic jump targets.
    /// Mirrors angr's NO_SYMBOLIC_JUMP_RESOLUTION option. When true, any
    /// symbolic jump target routes the state to the unconstrained stash
    /// instead of enumerating concretizations (matches
    /// engines/successors.py:234-239). Default off.
    pub fn set_no_symbolic_jump_resolution(&mut self, enabled: bool) {
        self.no_symbolic_jump_resolution = enabled;
    }

    /// Whether symbolic jump targets are routed to unconstrained without
    /// enumeration.
    pub fn no_symbolic_jump_resolution(&self) -> bool {
        self.no_symbolic_jump_resolution
    }

    /// Enable or disable preservation of the symbolic IP after concretization.
    /// Mirrors angr's KEEP_IP_SYMBOLIC option. When true, the engine still
    /// concretizes the next pc, but the IP register on each successor is left
    /// holding the original symbolic expression and no narrowing constraint is
    /// added (matches engines/successors.py:297-307,326-331). Default off.
    pub fn set_keep_ip_symbolic(&mut self, enabled: bool) {
        self.keep_ip_symbolic = enabled;
    }

    /// Whether the IP register should be kept symbolic across block boundaries.
    pub fn keep_ip_symbolic(&self) -> bool {
        self.keep_ip_symbolic
    }

    /// angr-027h: force eager (immediate) forking for this state regardless of
    /// the manager's `use_deferred_forks` setting. Set on loop-exit forks
    /// resumed at an UnconstrainedJump so they BFS to the find target instead
    /// of recursively re-deferring. Cloned on fork.
    pub fn set_force_eager_forks(&mut self, enabled: bool) {
        self.force_eager_forks = enabled;
    }

    /// Whether this state forces eager forking. See [`Self::set_force_eager_forks`].
    pub fn force_eager_forks(&self) -> bool {
        self.force_eager_forks
    }

    /// Set the fork-time `SharedLineageSolver` materialization opt-in on
    /// this state's solver context (angr-3ms1 step 1b).
    ///
    /// Forwards to [`SymContext::set_use_shared_lineage_solver`]. Setting
    /// on a seed state propagates to every descendant via `fork()` (the
    /// child SymContext inherits the parent's flag), so a single call at
    /// state-creation time is sufficient.
    ///
    /// Inert in this slice — the materialization gate (step 1c) is the
    /// first consumer.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_use_shared_lineage_solver(&self, enabled: bool) {
        self.solver.borrow().set_use_shared_lineage_solver(enabled);
    }

    /// Whether fork-time `SharedLineageSolver` materialization is opted
    /// in on this state's solver context (angr-3ms1 step 1b).
    #[cfg(feature = "vex-engine-z3")]
    pub fn use_shared_lineage_solver(&self) -> bool {
        self.solver.borrow().use_shared_lineage_solver()
    }

    /// Map memory with initial data.
    pub fn map_memory_data(&mut self, addr: u64, data: &[u8], permissions: Permission) {
        self.memory.map_data(addr, data, permissions);
    }

    /// Load from memory.
    pub fn memory_load(&self, addr: u64, size: u32) -> Result<RustBV, MemoryError> {
        let ctx = self.solver.borrow();
        self.memory.load_concrete(addr, size, &ctx)
    }

    /// Store to memory.
    pub fn memory_store(&mut self, addr: u64, value: RustBV) -> Result<(), MemoryError> {
        self.memory.store_concrete(addr, value)
    }

    /// Load from a symbolic address.
    pub fn memory_load_symbolic(&mut self, addr: RustBV, size: u32) -> Result<RustBV, MemoryError> {
        let ctx = self.solver.borrow();
        self.memory
            .load_symbolic_unified(addr, size, &ctx, &self.concretizer)
    }

    /// Store to a symbolic address.
    pub fn memory_store_symbolic(
        &mut self,
        addr: RustBV,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let ctx = self.solver.borrow();
        self.memory
            .store_symbolic_unified(addr, value, &ctx, &self.concretizer)
            .map(|_| ())
    }

    /// Store to a symbolic address using the lazy Multi-cell path
    /// (Phase 1.3/1.4 of angr-czph). Mirrors `memory_store_symbolic` but
    /// routes Multiple/Strided concretization results to per-byte Multi
    /// alternatives instead of eager ITE chains. Single addresses still
    /// short-circuit to the eager concrete store; TooLarge / Failed surface
    /// the same errors so callers can fall back identically.
    pub fn memory_store_symbolic_multi(
        &mut self,
        addr: RustBV,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let ctx = self.solver.borrow();
        self.memory
            .store_symbolic_unified_multi(addr, value, &ctx, &self.concretizer)
            .map(|_| ())
    }

    /// Add a lazy region for on-demand page fetching.
    pub fn add_lazy_region(&mut self, start_addr: u64, size: u64) {
        self.memory.add_lazy_region(start_addr, size);
    }

    /// Get dirty page addresses.
    pub fn get_dirty_pages(&self) -> Vec<u64> {
        self.memory.get_dirty_page_addrs()
    }

    /// Clear dirty page tracking.
    pub fn clear_dirty_pages(&mut self) {
        self.memory.clear_dirty_pages();
    }

    // =========================================================================
    // Solver/Constraint Access
    // =========================================================================

    /// Get a reference to the solver context.
    pub fn solver(&self) -> &Rc<RefCell<SymContext>> {
        &self.solver
    }

    /// Add a constraint.
    pub fn add_constraint(&self, constraint: RustBV) {
        let ctx = self.solver.borrow();
        ctx.assume_true(&constraint);
    }

    /// Check if current constraints are satisfiable.
    pub fn satisfiable(&self) -> bool {
        let ctx = self.solver.borrow();
        ctx.is_sat()
    }

    /// Prime the SAT cache (avoids redundant Z3 checks after branch forking).
    pub fn set_sat_cache(&self, value: bool) {
        self.solver.borrow().set_sat_cache(value);
    }

    /// Evaluate an expression to a concrete value.
    pub fn eval(&self, expr: &RustBV) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.eval(expr)
    }

    /// Get minimum value of an expression.
    pub fn min(&self, expr: &RustBV, signed: bool) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.min(expr, signed)
    }

    /// Get maximum value of an expression.
    pub fn max(&self, expr: &RustBV, signed: bool) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.max(expr, signed)
    }

    // =========================================================================
    // Hooks
    // =========================================================================

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        Arc::make_mut(&mut self.hooks).insert(addr);
    }

    /// Remove a hook address.
    pub fn remove_hook(&mut self, addr: u64) {
        Arc::make_mut(&mut self.hooks).remove(&addr);
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.hooks.contains(&addr)
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        // Avoid CoW clone if already empty.
        if !self.hooks.is_empty() {
            Arc::make_mut(&mut self.hooks).clear();
        }
    }

    // =========================================================================
    // Per-state metadata (claripy AST refs) — see field docs above.
    // =========================================================================

    /// Insert/replace a hook-symbolic-memory entry.
    pub fn set_hook_symbolic_memory(&mut self, addr: u64, ast: Py<PyAny>, size: u32) {
        self.hook_symbolic_memory.insert(addr, (ast, size));
    }

    /// Insert/replace an addr-to-AST entry.
    pub fn set_addr_to_ast(&mut self, addr: u64, ast: Py<PyAny>, size: u32) {
        self.addr_to_ast.insert(addr, (ast, size));
    }

    /// Read-only access to the symbolic-pages map.
    pub fn symbolic_pages(&self) -> &HashMap<u64, Py<PyAny>> {
        &self.symbolic_pages
    }

    /// Read-only access to the hook-symbolic-memory map.
    pub fn hook_symbolic_memory(&self) -> &HashMap<u64, (Py<PyAny>, u32)> {
        &self.hook_symbolic_memory
    }

    /// Read-only access to the addr-to-AST map.
    pub fn addr_to_ast(&self) -> &HashMap<u64, (Py<PyAny>, u32)> {
        &self.addr_to_ast
    }

    /// Replace the entire symbolic-pages map (used by full-page recovery flow).
    pub fn replace_symbolic_pages(&mut self, pages: HashMap<u64, Py<PyAny>>) {
        self.symbolic_pages = pages;
    }

    /// Drop all per-state metadata (called when a state is no longer needed).
    pub fn clear_state_metadata(&mut self) {
        self.symbolic_pages.clear();
        self.hook_symbolic_memory.clear();
        self.addr_to_ast.clear();
    }

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
}

/// Python-facing wrapper for RustSimState.
///
/// This provides the PyO3 interface for creating and manipulating
/// Rust-native simulation states from Python.
#[pyclass(name = "RustSimState", unsendable)]
pub struct PyRustSimState {
    inner: RustSimState,
}

#[pymethods]
impl PyRustSimState {
    /// Create a new state for the given architecture.
    #[new]
    #[pyo3(signature = (arch="amd64", little_endian=None))]
    pub fn new(arch: &str, little_endian: Option<bool>) -> PyResult<Self> {
        let inner =
            RustSimState::new_with_endian(arch, little_endian).map_err(PyValueError::new_err)?;
        Ok(PyRustSimState { inner })
    }

    /// Get the state ID.
    #[getter]
    pub fn state_id(&self) -> u64 {
        self.inner.state_id()
    }

    /// Get the parent state ID.
    #[getter]
    pub fn parent_id(&self) -> Option<u64> {
        self.inner.parent_id()
    }

    /// Get the program counter.
    #[getter]
    pub fn pc(&self) -> u64 {
        self.inner.pc()
    }

    /// Set the program counter.
    #[setter]
    pub fn set_pc(&mut self, pc: u64) {
        self.inner.set_pc(pc);
    }

    /// Get the architecture name.
    #[getter]
    pub fn arch_name(&self) -> &str {
        self.inner.arch().name()
    }

    /// Get the POSIX brk pointer (mirrors Python's `state.posix.brk`).
    #[getter]
    pub fn posix_brk(&self) -> u64 {
        self.inner.posix_brk()
    }

    /// Set the POSIX brk pointer. Used by the Python wrapper at state-creation
    /// time to push `state.posix.brk` (which the angr loader sets based on the
    /// binary's last address) into Rust so subsequent native brk syscalls
    /// start from the correct base.
    #[setter]
    pub fn set_posix_brk(&mut self, addr: u64) {
        self.inner.set_posix_brk(addr);
    }

    /// Get the history (basic block addresses).
    pub fn history(&self) -> Vec<u64> {
        self.inner.history().to_vec()
    }

    /// Get a register value by name.
    pub fn get_register(&self, name: &str) -> PyResult<u128> {
        self.inner
            .get_register(name)
            .and_then(|bv| bv.as_u128())
            .ok_or_else(|| PyValueError::new_err(format!("cannot read register {}", name)))
    }

    /// Set a register value by name.
    pub fn set_register(&mut self, name: &str, value: u128) -> PyResult<()> {
        let size = self
            .inner
            .arch()
            .register_size(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
        let bv = RustBV::concrete(value, size * 8);
        if self.inner.set_register(name, bv) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "failed to set register: {}",
                name
            )))
        }
    }

    /// Set multiple registers in a single FFI call.
    /// Takes a dict of {name: value} pairs.
    ///
    /// **Invariant I5 (cross-mixin):** the caller — Python's
    /// `rust_state_sync.py` — must pre-filter the dict to
    /// `_supported_register_names` for the active architecture. The disk
    /// init cache pickles ALL `arch.register_names.values()` (cr0..8,
    /// ymm0..15, fs_seg, ds_seg, cmstart, cmlen, fpreg, ...), but only a
    /// subset has a slot in `RegisterFile` for amd64/x86/arm/etc. If an
    /// unsupported name leaks through, this method returns
    /// `PyValueError("unknown register: <name>")` rather than silently
    /// dropping the write. Do NOT "fix" by extending `arch/amd64.rs`
    /// unless the interpreter actually consumes the new register. See
    /// module-level invariant I5 in this file.
    pub fn set_registers_bulk(&mut self, registers: &Bound<'_, PyDict>) -> PyResult<()> {
        for (key, val) in registers.iter() {
            let name: String = key.extract()?;
            let value: u128 = val.extract()?;
            let size = self
                .inner
                .arch()
                .register_size(&name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
            // I5 cross-check: register_size returning Some implies the
            // register has a RegisterFile slot. This debug assert documents
            // intent and would catch a regression where arch lookup and
            // RegisterFile membership drift apart.
            #[cfg(debug_assertions)]
            debug_assert!(
                size > 0,
                "I5: register {} has zero size — arch table is malformed",
                name
            );
            let bv = RustBV::concrete(value, size * 8);
            self.inner.set_register(&name, bv);
        }
        Ok(())
    }

    /// Get all register bytes.
    pub fn get_registers_raw(&self) -> Vec<u8> {
        self.inner.get_registers_raw()
    }

    /// Set all register bytes.
    pub fn set_registers_raw(&mut self, bytes: &[u8]) {
        self.inner.set_registers_raw(bytes);
    }

    /// Set a register to a symbolic value from a raw Z3 AST pointer.
    ///
    /// The Z3 AST must be a BitVec in the shared Z3 context.
    /// Used to import symbolic register values (e.g., BVS in rax) from Python.
    /// Set a register to a symbolic value from a raw Z3 AST pointer.
    ///
    /// The Z3 AST must be a BitVec in the shared Z3 context.
    /// Used to import symbolic register values (e.g., BVS in rax) from Python.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_register_symbolic(
        &mut self,
        name: &str,
        z3_ast_ptr: usize,
        width: u32,
    ) -> PyResult<()> {
        use z3::ast::Ast;
        let size = self
            .inner
            .arch()
            .register_size(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
        if width != size * 8 {
            return Err(PyValueError::new_err(format!(
                "width mismatch: register {} is {} bits, got {} bits",
                name,
                size * 8,
                width
            )));
        }
        // Reconstruct z3::ast::BV from raw pointer.
        // SAFETY: caller guarantees `z3_ast_ptr` is a non-null, BV-sorted
        // Z3_ast in the active thread-local context (z3-rs 0.19+ shares the
        // process-global context with claripy's z3 backend). `BV::wrap` takes
        // its own ref. `width` was validated above to match the register's
        // bit width — i.e. the AST's BV sort width.
        let z3_bv = unsafe {
            let raw = std::ptr::NonNull::new_unchecked(z3_ast_ptr as *mut _);
            let ctx = z3::Context::thread_local();
            z3::ast::BV::wrap(&ctx, raw)
        };
        let bv = RustBV::Symbolic {
            id: 0,
            ast: z3_bv,
            width,
            name: Arc::from(name),
        };
        if self.inner.set_register(name, bv) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "failed to set register: {}",
                name
            )))
        }
    }

    /// Set a register to a symbolic value from a full claripy AST.
    ///
    /// Unlike [`set_register_symbolic`] (which wraps a raw Z3 pointer as an
    /// opaque `RustBV::Symbolic` with `id: 0` and so loses leaf-symbol
    /// identity on export), this routes the claripy AST through
    /// `claripy_to_rustbv`. That interns every leaf BVS into the shared
    /// claripy<->Rust symbol cache, so a subregister set like
    /// `state.regs.ecx = BVS('ecx', 32)` round-trips back to the user's
    /// original symbol on export instead of minting a fresh `rcx_N`. This is
    /// the Layer 2 fix for angr-4ju9e / angr-21vi5 — it mirrors the
    /// identity-preserving import path that memory-sourced symbols already use
    /// (`import_symbolic_to_state`).
    pub fn set_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let size = self
            .inner
            .arch()
            .register_size(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
        let solver = self.inner.solver().clone();
        let bv = {
            let ctx = solver.borrow();
            crate::claripy_bridge::claripy_to_rustbv(py, ast, &ctx)
                .map_err(|e| PyValueError::new_err(format!("AST conversion: {}", e)))?
        };
        if bv.width() != size * 8 {
            return Err(PyValueError::new_err(format!(
                "width mismatch: register {} is {} bits, got {} bits",
                name,
                size * 8,
                bv.width()
            )));
        }
        if self.inner.set_register(name, bv) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "failed to set register: {}",
                name
            )))
        }
    }

    /// Map a memory region.
    #[pyo3(signature = (addr, size, permissions=7))]
    pub fn map_memory(&mut self, addr: u64, size: u64, permissions: u8) {
        self.inner
            .map_memory(addr, size, Permission::from_bits(permissions));
    }

    /// Map memory with initial data.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn map_memory_data(&mut self, addr: u64, data: &[u8], permissions: u8) {
        self.inner
            .map_memory_data(addr, data, Permission::from_bits(permissions));
    }

    /// Map multiple memory pages in a single FFI call.
    /// pages is a list of (addr, data, permissions) tuples.
    pub fn map_memory_batch(&mut self, pages: Vec<(u64, Vec<u8>, u8)>) {
        for (addr, data, permissions) in pages {
            self.inner
                .map_memory_data(addr, &data, Permission::from_bits(permissions));
        }
    }

    /// Enable or disable strict memory permission enforcement on load/store.
    /// Mirrors angr's STRICT_PAGE_ACCESS option. Default off.
    #[pyo3(name = "set_enforce_permissions")]
    pub fn py_set_enforce_permissions(&mut self, enabled: bool) {
        self.inner.set_enforce_permissions(enabled);
    }

    /// Whether strict memory permission enforcement is enabled.
    #[pyo3(name = "enforce_permissions")]
    pub fn py_enforce_permissions(&self) -> bool {
        self.inner.enforce_permissions()
    }

    /// Enable or disable non-executable page enforcement on instruction fetch.
    /// Mirrors angr's ENABLE_NX option. The X check fires only when this and
    /// `enforce_permissions` are both on. Default off.
    #[pyo3(name = "set_enforce_nx")]
    pub fn py_set_enforce_nx(&mut self, enabled: bool) {
        self.inner.set_enforce_nx(enabled);
    }

    /// Whether non-executable page enforcement is enabled.
    #[pyo3(name = "enforce_nx")]
    pub fn py_enforce_nx(&self) -> bool {
        self.inner.enforce_nx()
    }

    /// Suppress IP concretization for symbolic jump targets.
    /// Mirrors angr's NO_IP_CONCRETIZATION option. When set, a symbolic IP
    /// at block boundary routes the state to the unconstrained stash
    /// silently instead of being enumerated. Default off.
    #[pyo3(name = "set_no_ip_concretization")]
    pub fn py_set_no_ip_concretization(&mut self, enabled: bool) {
        self.inner.set_no_ip_concretization(enabled);
    }

    /// Whether NO_IP_CONCRETIZATION is active on this state.
    #[pyo3(name = "no_ip_concretization")]
    pub fn py_no_ip_concretization(&self) -> bool {
        self.inner.no_ip_concretization()
    }

    /// Suppress resolution of symbolic jump targets.
    /// Mirrors angr's NO_SYMBOLIC_JUMP_RESOLUTION option. When set, a symbolic
    /// jump target routes the state to the unconstrained stash before
    /// AddressConcretizer enumeration is attempted. Default off.
    #[pyo3(name = "set_no_symbolic_jump_resolution")]
    pub fn py_set_no_symbolic_jump_resolution(&mut self, enabled: bool) {
        self.inner.set_no_symbolic_jump_resolution(enabled);
    }

    /// Whether NO_SYMBOLIC_JUMP_RESOLUTION is active on this state.
    #[pyo3(name = "no_symbolic_jump_resolution")]
    pub fn py_no_symbolic_jump_resolution(&self) -> bool {
        self.inner.no_symbolic_jump_resolution()
    }

    /// Preserve the symbolic IP across block boundaries.
    /// Mirrors angr's KEEP_IP_SYMBOLIC option. When set, the engine still
    /// concretizes the next pc, but each fork's IP register is left holding
    /// the original symbolic expression and no `target == addr` narrowing
    /// constraint is added. Default off.
    #[pyo3(name = "set_keep_ip_symbolic")]
    pub fn py_set_keep_ip_symbolic(&mut self, enabled: bool) {
        self.inner.set_keep_ip_symbolic(enabled);
    }

    /// Whether KEEP_IP_SYMBOLIC is active on this state.
    #[pyo3(name = "keep_ip_symbolic")]
    pub fn py_keep_ip_symbolic(&self) -> bool {
        self.inner.keep_ip_symbolic()
    }

    /// Opt this state's solver context in to fork-time
    /// `SharedLineageSolver` materialization (angr-3ms1 step 1b).
    ///
    /// Inherited by every descendant via `fork()` so a single call on a
    /// seed state suffices. Default off — the slice-1c materialization
    /// gate stays inert on plain `RustExplorationManager` runs to keep
    /// the v5a5-slice-4c.3-retry-failed-bfs-thrash-fundamental
    /// regression (defcon2016quals_baby-re ~10x slowdown under default
    /// BFS) out of CI.
    #[cfg(feature = "vex-engine-z3")]
    #[pyo3(name = "set_use_shared_lineage_solver")]
    pub fn py_set_use_shared_lineage_solver(&self, enabled: bool) {
        self.inner.set_use_shared_lineage_solver(enabled);
    }

    /// Whether fork-time `SharedLineageSolver` materialization is opted
    /// in on this state (angr-3ms1 step 1b).
    #[cfg(feature = "vex-engine-z3")]
    #[pyo3(name = "use_shared_lineage_solver")]
    pub fn py_use_shared_lineage_solver(&self) -> bool {
        self.inner.use_shared_lineage_solver()
    }

    /// Load from memory.
    pub fn memory_load(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        let bv = self
            .inner
            .memory_load(addr, size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let value = bv.to_u128();
        let bytes: Vec<u8> = (0..size as usize)
            .map(|i| (value >> (i * 8)) as u8)
            .collect();
        Ok(bytes)
    }

    /// Store to memory.
    ///
    /// angr-5aj8: split into 16-byte chunks because RustBV::Concrete is
    /// backed by a u128. Packing >16 bytes into a single concrete BV would
    /// shift-overflow during construction and then make store_concrete emit
    /// a 16-byte-cycle pattern over the full claimed width.
    pub fn memory_store(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        let mut offset = 0usize;
        while offset < data.len() {
            let remaining = data.len() - offset;
            let chunk_size = remaining.min(16);
            let chunk = &data[offset..offset + chunk_size];
            let width = (chunk_size * 8) as u32;
            let mut value: u128 = 0;
            for (i, &b) in chunk.iter().enumerate() {
                value |= (b as u128) << (i * 8);
            }
            let bv = RustBV::concrete(value, width);
            self.inner
                .memory_store(addr + offset as u64, bv)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            offset += chunk_size;
        }
        Ok(())
    }

    /// Add a lazy region.
    pub fn add_lazy_region(&mut self, start_addr: u64, size: u64) {
        self.inner.add_lazy_region(start_addr, size);
    }

    /// Add multiple lazy regions in a single FFI call.
    /// `regions` is a list of (start_addr, size) tuples. Used by
    /// `RustStateSyncMixin._sync_extra_python_pages` to register thousands of
    /// per-page lazy entries (mma_howtouse / Callable workflow) in one
    /// crossing instead of one FFI per page.
    pub fn add_lazy_regions_batch(&mut self, regions: Vec<(u64, u64)>) {
        for (start_addr, size) in regions {
            self.inner.add_lazy_region(start_addr, size);
        }
    }

    /// Number of registered lazy regions (one entry per `add_lazy_region`
    /// call). Exposed so tests can confirm a batch registration actually
    /// landed regions rather than silently no-op'ing.
    pub fn lazy_region_count(&self) -> usize {
        self.inner.memory().lazy_region_count()
    }

    /// Whether a byte address falls inside any registered lazy region.
    pub fn is_in_lazy_region(&self, addr: u64) -> bool {
        self.inner.memory().is_addr_in_lazy_region(addr)
    }

    /// Get dirty page addresses.
    pub fn get_dirty_pages(&self) -> Vec<u64> {
        self.inner.get_dirty_pages()
    }

    /// Clear dirty page tracking.
    pub fn clear_dirty_pages(&mut self) {
        self.inner.clear_dirty_pages();
    }

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        self.inner.add_hook(addr);
    }

    /// Remove a hook address.
    pub fn remove_hook(&mut self, addr: u64) {
        self.inner.remove_hook(addr);
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.inner.is_hooked(addr)
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        self.inner.clear_hooks();
    }

    /// Check if constraints are satisfiable.
    pub fn satisfiable(&self) -> bool {
        self.inner.satisfiable()
    }

    /// Fork the state (O(1) CoW).
    pub fn fork(&self) -> Self {
        PyRustSimState {
            inner: self.inner.fork(),
        }
    }

    /// Configure address concretization (legacy interface).
    #[pyo3(signature = (use_approximate, range_limit=None))]
    pub fn configure_concretization(&mut self, use_approximate: bool, range_limit: Option<u64>) {
        self.inner
            .configure_concretization(use_approximate, range_limit);
    }

    /// Configure address concretization with full strategy configuration.
    #[pyo3(signature = (use_approximate, read_range_limit=None, write_range_limit=None, symbolic_write_addresses=false, avoid_multivalued_reads=false, avoid_multivalued_writes=false))]
    pub fn configure_concretization_strategies(
        &mut self,
        use_approximate: bool,
        read_range_limit: Option<u64>,
        write_range_limit: Option<u64>,
        symbolic_write_addresses: bool,
        avoid_multivalued_reads: bool,
        avoid_multivalued_writes: bool,
    ) {
        self.inner.concretizer.configure_strategies(
            use_approximate,
            read_range_limit,
            write_range_limit,
            symbolic_write_addresses,
            avoid_multivalued_reads,
            avoid_multivalued_writes,
        );
    }

    /// Set whether to track history.
    pub fn set_track_history(&mut self, track: bool) {
        self.inner.set_track_history(track);
    }

    /// Set maximum history length.
    pub fn set_max_history(&mut self, max: usize) {
        self.inner.set_max_history(max);
    }

    /// Get the current maximum history length (0 = unlimited).
    pub fn get_max_history(&self) -> usize {
        self.inner.max_history
    }

    /// Get the detailed execution history as `(addr, jumpkind, jump_target)` tuples.
    pub fn detailed_history(&self) -> Vec<(u64, u8, u64)> {
        self.inner
            .detailed_history()
            .iter()
            .map(|h| (h.addr, h.jumpkind, h.jump_target))
            .collect()
    }

    /// Append a basic-block address to `history` (honors the cap).
    /// Test/debug helper — production paths go through the interpreter.
    pub fn add_history(&mut self, addr: u64) {
        self.inner.add_to_history(addr);
    }

    /// Append a detailed history entry (honors the cap).
    /// Test/debug helper — production paths go through the interpreter.
    pub fn add_detailed_history(&mut self, addr: u64, jumpkind: u8, jump_target: u64) {
        self.inner.add_history_entry(addr, jumpkind, jump_target);
    }

    /// Push a call frame onto the call stack (Ijk_Call simulation).
    /// Test/debug helper — production paths go through the interpreter.
    pub fn push_call_frame(&mut self, call_site: u64, callee: u64, ret_addr: u64, sp: u64) {
        self.inner.push_call(call_site, callee, ret_addr, sp);
    }

    /// Get the call stack as a list of (call_site, callee, ret_addr, sp) tuples,
    /// in push order (outermost first, innermost last). Mirrors
    /// `ExplorationStateSnapshot.get_call_stack()` for direct PyRustSimState
    /// inspection without an export round-trip.
    pub fn get_call_stack(&self) -> Vec<(u64, u64, u64, u64)> {
        self.inner
            .call_stack()
            .iter()
            .map(|e| (e.call_site_addr, e.callee_addr, e.return_addr, e.stack_ptr))
            .collect()
    }

    /// Export the complete state as a snapshot.
    pub fn export_full(&self) -> ExplorationStateSnapshot {
        self.inner.export_full()
    }
}

impl PyRustSimState {
    /// Get access to the inner state (for Rust-side use).
    pub fn inner(&self) -> &RustSimState {
        &self.inner
    }

    /// Get mutable access to the inner state.
    pub fn inner_mut(&mut self) -> &mut RustSimState {
        &mut self.inner
    }
}

#[cfg(test)]
#[path = "../state_tests.rs"]
mod state_tests;
