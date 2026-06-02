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

/// Entry in the function call stack.
///
/// Tracks call/return pairs during symbolic execution. Pushed on `Ijk_Call`,
/// popped on `Ijk_Ret`. Cloned on state fork.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CallStackEntry {
    /// Address of the call instruction (caller site).
    pub call_site_addr: u64,
    /// Address of the callee function entry.
    pub callee_addr: u64,
    /// Expected return address (instruction after the call).
    pub return_addr: u64,
    /// Stack pointer value at call time.
    pub stack_ptr: u64,
}

/// Heap metadata tracking for malloc/free/calloc/realloc.
///
/// Tracks allocated regions and freed addresses for heap exploitation
/// analysis. Cloned on fork so each exploration path has its own heap state.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct HeapMetadata {
    /// Currently allocated regions: address -> size in bytes.
    pub allocated: FxHashMap<u64, u64>,
    /// Freed addresses (in order of free calls).
    pub freed: Vec<u64>,
}

impl HeapMetadata {
    /// Record a new allocation.
    pub fn record_alloc(&mut self, addr: u64, size: u64) {
        self.allocated.insert(addr, size);
    }

    /// Record a free. Returns the size of the freed region, or None if not tracked.
    pub fn record_free(&mut self, addr: u64) -> Option<u64> {
        let size = self.allocated.remove(&addr);
        if addr != 0 {
            self.freed.push(addr);
        }
        size
    }

    /// Check if an address was allocated.
    pub fn is_allocated(&self, addr: u64) -> bool {
        self.allocated.contains_key(&addr)
    }

    /// Get the size of an allocation.
    pub fn alloc_size(&self, addr: u64) -> Option<u64> {
        self.allocated.get(&addr).copied()
    }

    /// Get the number of active allocations.
    pub fn alloc_count(&self) -> usize {
        self.allocated.len()
    }

    /// Get the number of free calls.
    pub fn free_count(&self) -> usize {
        self.freed.len()
    }
}

/// File descriptor flags (matching POSIX O_ constants).
#[derive(Clone, Debug, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FdFlags {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

impl FdFlags {
    /// Convert from POSIX O_RDONLY/O_WRONLY/O_RDWR integer flags.
    pub fn from_posix(flags: u32) -> Self {
        match flags & 3 {
            0 => FdFlags::ReadOnly,
            1 => FdFlags::WriteOnly,
            _ => FdFlags::ReadWrite,
        }
    }

    /// Convert to POSIX integer representation.
    pub fn to_posix(&self) -> u32 {
        match self {
            FdFlags::ReadOnly => 0,
            FdFlags::WriteOnly => 1,
            FdFlags::ReadWrite => 2,
        }
    }
}

/// A tracked file descriptor with metadata.
///
/// Represents an open file descriptor with its name, position, flags,
/// and content buffer. Cloned on state fork.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FileDescriptor {
    /// File path/name (e.g. "/dev/stdin", "flag.txt"). Empty for unnamed fds.
    pub name: String,
    /// Current read/write position (seek offset).
    pub position: u64,
    /// Open mode flags.
    pub flags: FdFlags,
    /// Accumulated content buffer (output for write fds, input data for read fds).
    pub content: Vec<u8>,
    /// Whether the fd is currently open.
    pub is_open: bool,
}

impl FileDescriptor {
    /// Create a new open file descriptor.
    pub fn new(name: String, flags: FdFlags) -> Self {
        FileDescriptor {
            name,
            position: 0,
            flags,
            content: Vec::new(),
            is_open: true,
        }
    }

    /// Create a new file descriptor with initial content (e.g. for readable files).
    pub fn with_content(name: String, flags: FdFlags, content: Vec<u8>) -> Self {
        FileDescriptor {
            name,
            position: 0,
            flags,
            content,
            is_open: true,
        }
    }
}

/// File system state tracking.
///
/// Manages file descriptors beyond stdin/stdout/stderr. Tracks open/close/read/write/seek
/// operations. Forking is O(1) via `Arc<HashMap<...>>` — the inner map is only cloned
/// (via `Arc::make_mut`) when a path actually mutates its file descriptors.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(from = "FileSystemData", into = "FileSystemData")]
pub struct FileSystem {
    /// Open file descriptors. Standard fds: 0=stdin, 1=stdout, 2=stderr.
    /// Wrapped in Arc for cheap fork; copy-on-write via Arc::make_mut on mutation.
    fds: Arc<HashMap<u32, FileDescriptor>>,
    /// Next file descriptor number to allocate.
    next_fd: u32,
    /// Current working directory as raw bytes (matches Python
    /// `state.fs.cwd` shape: `bytes`, default `b"/"`). chdir / getcwd
    /// (angr-0hif.2) read & write this directly; no normalization is
    /// applied, mirroring `procedures/linux_kernel/cwd.py::chdir` which
    /// also assigns the raw concrete path.
    cwd: Vec<u8>,
}

/// Serde shadow form for [`FileSystem`].
///
/// Collapses `Arc<HashMap<u32, FileDescriptor>>` to a deterministic
/// `BTreeMap<u32, FileDescriptor>` on the wire and carries `next_fd` /
/// `cwd` explicitly. Mirrors the `MemoryPage` / `RegisterFile` snapshot
/// shadow pattern (angr-x04s.1.2).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileSystemData {
    pub fds: std::collections::BTreeMap<u32, FileDescriptor>,
    pub next_fd: u32,
    pub cwd: Vec<u8>,
}

impl From<FileSystem> for FileSystemData {
    fn from(fs: FileSystem) -> Self {
        let fds: std::collections::BTreeMap<u32, FileDescriptor> =
            fs.fds.iter().map(|(k, v)| (*k, v.clone())).collect();
        FileSystemData {
            fds,
            next_fd: fs.next_fd,
            cwd: fs.cwd,
        }
    }
}

impl From<FileSystemData> for FileSystem {
    fn from(d: FileSystemData) -> Self {
        let fds: HashMap<u32, FileDescriptor> = d.fds.into_iter().collect();
        FileSystem {
            fds: Arc::new(fds),
            next_fd: d.next_fd,
            cwd: d.cwd,
        }
    }
}

impl Default for FileSystem {
    fn default() -> Self {
        let mut fds = HashMap::new();
        // Pre-register standard file descriptors
        fds.insert(
            0,
            FileDescriptor::new("/dev/stdin".to_string(), FdFlags::ReadOnly),
        );
        fds.insert(
            1,
            FileDescriptor::new("/dev/stdout".to_string(), FdFlags::WriteOnly),
        );
        fds.insert(
            2,
            FileDescriptor::new("/dev/stderr".to_string(), FdFlags::WriteOnly),
        );
        FileSystem {
            fds: Arc::new(fds),
            next_fd: 3,
            cwd: b"/".to_vec(),
        }
    }
}

impl FileSystem {
    /// Open a new file descriptor. Returns the allocated fd number.
    pub fn open(&mut self, name: String, flags: FdFlags) -> u32 {
        let fd = self.next_fd;
        self.next_fd += 1;
        Arc::make_mut(&mut self.fds).insert(fd, FileDescriptor::new(name, flags));
        fd
    }

    /// Open a file descriptor with pre-loaded content (for file-backed SimFiles).
    pub fn open_with_content(&mut self, name: String, flags: FdFlags, content: Vec<u8>) -> u32 {
        let fd = self.next_fd;
        self.next_fd += 1;
        Arc::make_mut(&mut self.fds).insert(fd, FileDescriptor::with_content(name, flags, content));
        fd
    }

    /// Close a file descriptor. Returns true if it was open.
    pub fn close(&mut self, fd: u32) -> bool {
        // Read-first to avoid CoW clone if the fd is missing or already closed.
        if !self.fds.get(&fd).is_some_and(|d| d.is_open) {
            return false;
        }
        if let Some(desc) = Arc::make_mut(&mut self.fds).get_mut(&fd) {
            desc.is_open = false;
            true
        } else {
            false
        }
    }

    /// Write data to a file descriptor's content buffer.
    pub fn write(&mut self, fd: u32, data: &[u8]) {
        Arc::make_mut(&mut self.fds)
            .entry(fd)
            .or_insert_with(|| FileDescriptor::new(String::new(), FdFlags::WriteOnly))
            .content
            .extend_from_slice(data);
    }

    /// Read up to `count` bytes from a file descriptor at its current position.
    /// Advances the position. Returns bytes read.
    pub fn read(&mut self, fd: u32, count: usize) -> Vec<u8> {
        // Peek to compute byte count without forcing CoW when nothing is readable.
        let n = match self.fds.get(&fd) {
            Some(desc) => {
                let pos = desc.position as usize;
                let available = desc.content.len().saturating_sub(pos);
                count.min(available)
            }
            None => return Vec::new(),
        };
        if n == 0 {
            return Vec::new();
        }
        let desc = Arc::make_mut(&mut self.fds)
            .get_mut(&fd)
            .expect("fd existed above");
        let pos = desc.position as usize;
        let data = desc.content[pos..pos + n].to_vec();
        desc.position += n as u64;
        data
    }

    /// Seek a file descriptor. Returns the new position.
    ///
    /// whence: 0=SEEK_SET, 1=SEEK_CUR, 2=SEEK_END
    pub fn seek(&mut self, fd: u32, offset: i64, whence: u32) -> Option<u64> {
        // Compute new position without CoW first; only mutate if the fd exists
        // and the whence value is valid.
        let desc = self.fds.get(&fd)?;
        let new_pos = match whence {
            0 => offset.max(0) as u64,                               // SEEK_SET
            1 => (desc.position as i64 + offset).max(0) as u64,      // SEEK_CUR
            2 => (desc.content.len() as i64 + offset).max(0) as u64, // SEEK_END
            _ => return None,
        };
        Arc::make_mut(&mut self.fds).get_mut(&fd)?.position = new_pos;
        Some(new_pos)
    }

    /// Get the content buffer for a file descriptor (read-only).
    pub fn fd_content(&self, fd: u32) -> &[u8] {
        self.fds
            .get(&fd)
            .map(|d| d.content.as_slice())
            .unwrap_or(&[])
    }

    /// Check if a file descriptor is open.
    pub fn is_open(&self, fd: u32) -> bool {
        self.fds.get(&fd).is_some_and(|d| d.is_open)
    }

    /// Get file descriptor info: (name, position, flags, content_len, is_open).
    pub fn fd_info(&self, fd: u32) -> Option<(&str, u64, u32, usize, bool)> {
        self.fds.get(&fd).map(|d| {
            (
                d.name.as_str(),
                d.position,
                d.flags.to_posix(),
                d.content.len(),
                d.is_open,
            )
        })
    }

    /// List all file descriptor numbers (including closed ones).
    pub fn all_fds(&self) -> Vec<u32> {
        let mut fds: Vec<u32> = self.fds.keys().copied().collect();
        fds.sort();
        fds
    }

    /// List only open file descriptor numbers.
    pub fn open_fds(&self) -> Vec<u32> {
        let mut fds: Vec<u32> = self
            .fds
            .iter()
            .filter(|(_, d)| d.is_open)
            .map(|(k, _)| *k)
            .collect();
        fds.sort();
        fds
    }

    /// Get the next fd number (for pre-allocating).
    pub fn next_fd(&self) -> u32 {
        self.next_fd
    }

    /// Current working directory bytes (mirrors Python `state.fs.cwd`).
    pub fn cwd(&self) -> &[u8] {
        &self.cwd
    }

    /// Replace the current working directory bytes. `chdir(2)` semantics —
    /// the raw concrete path is stored verbatim (no `_normalize_path`
    /// applied, matching `procedures/linux_kernel/cwd.py::chdir`).
    pub fn set_cwd(&mut self, cwd: Vec<u8>) {
        self.cwd = cwd;
    }

    /// Duplicate an open file descriptor, allocating the lowest unused fd.
    /// Returns the new fd, or None if `oldfd` is not open.
    ///
    /// Like POSIX `dup(2)`: the new fd refers to the same underlying state.
    /// We model this by cloning the `FileDescriptor` (name/position/flags/content).
    pub fn dup(&mut self, oldfd: u32) -> Option<u32> {
        if !self.fds.get(&oldfd).is_some_and(|d| d.is_open) {
            return None;
        }
        let cloned = self.fds.get(&oldfd).cloned()?;
        let newfd = self.next_fd;
        self.next_fd += 1;
        Arc::make_mut(&mut self.fds).insert(newfd, cloned);
        Some(newfd)
    }

    /// Duplicate `oldfd` to `newfd`. If `newfd` was open, it is closed first.
    /// If `oldfd == newfd` and `oldfd` is open, returns `newfd` unchanged.
    /// Returns the new fd on success, or None if `oldfd` is not open.
    ///
    /// Like POSIX `dup2(2)`. Bumps `next_fd` past `newfd` if necessary so future
    /// allocations don't collide.
    pub fn dup2(&mut self, oldfd: u32, newfd: u32) -> Option<u32> {
        if !self.fds.get(&oldfd).is_some_and(|d| d.is_open) {
            return None;
        }
        if oldfd == newfd {
            return Some(newfd);
        }
        let cloned = self.fds.get(&oldfd).cloned()?;
        Arc::make_mut(&mut self.fds).insert(newfd, cloned);
        if newfd >= self.next_fd {
            self.next_fd = newfd + 1;
        }
        Some(newfd)
    }

    /// Create a pipe: returns `(read_fd, write_fd)`, allocated as two
    /// consecutive fds.
    ///
    /// Like POSIX `pipe(2)`. The read end is opened ReadOnly and the write end
    /// WriteOnly. We do NOT model write→read data flow (each end has its own
    /// content buffer); this matches angr's existing SimPacketsStream-light
    /// modeling — the procedure exists so binaries that allocate fds via pipe()
    /// don't fall through to Python on every fd op.
    pub fn pipe(&mut self) -> (u32, u32) {
        let read_fd = self.next_fd;
        let write_fd = self.next_fd + 1;
        self.next_fd += 2;
        let map = Arc::make_mut(&mut self.fds);
        map.insert(
            read_fd,
            FileDescriptor::new("<pipe:r>".to_string(), FdFlags::ReadOnly),
        );
        map.insert(
            write_fd,
            FileDescriptor::new("<pipe:w>".to_string(), FdFlags::WriteOnly),
        );
        (read_fd, write_fd)
    }
}

/// Types of inspection events that can be tracked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum InspectEvent {
    /// Memory read: (addr, size)
    MemRead = 0,
    /// Memory write: (addr, size)
    MemWrite = 1,
    /// Register read: (offset, size)
    RegRead = 2,
    /// Register write: (offset, size)
    RegWrite = 3,
    /// State fork (branch)
    Fork = 4,
    /// State exit/deadend
    Exit = 5,
}

impl InspectEvent {
    /// Number of event types.
    pub const COUNT: usize = 6;

    /// Convert from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(InspectEvent::MemRead),
            1 => Some(InspectEvent::MemWrite),
            2 => Some(InspectEvent::RegRead),
            3 => Some(InspectEvent::RegWrite),
            4 => Some(InspectEvent::Fork),
            5 => Some(InspectEvent::Exit),
            _ => None,
        }
    }

    /// Convert to string name.
    pub fn name(&self) -> &'static str {
        match self {
            InspectEvent::MemRead => "mem_read",
            InspectEvent::MemWrite => "mem_write",
            InspectEvent::RegRead => "reg_read",
            InspectEvent::RegWrite => "reg_write",
            InspectEvent::Fork => "fork",
            InspectEvent::Exit => "exit",
        }
    }
}

/// A recorded inspection event with address/offset and size.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InspectRecord {
    /// Event type.
    pub event: InspectEvent,
    /// Address (for mem events) or register offset (for reg events).
    pub addr: u64,
    /// Size in bytes.
    pub size: u32,
    /// Block address where the event occurred.
    pub block_addr: u64,
}

/// Inspection/breakpoint manager for state events.
///
/// Tracks which event types are enabled for logging and maintains a
/// ring buffer of recent events. Designed for minimal overhead when
/// no inspections are registered (single bool check).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InspectionManager {
    /// Bitmask of enabled event types (bit N = InspectEvent with value N).
    enabled: u8,
    /// Ring buffer of recent events (capacity = max_events).
    events: Vec<InspectRecord>,
    /// Maximum number of events to retain (ring buffer capacity).
    max_events: usize,
    /// Total event count per type (never reset, for statistics).
    event_counts: [u64; InspectEvent::COUNT],
}

impl Default for InspectionManager {
    fn default() -> Self {
        InspectionManager {
            enabled: 0,
            events: Vec::new(),
            max_events: 1024,
            event_counts: [0; InspectEvent::COUNT],
        }
    }
}

impl InspectionManager {
    /// Check if any inspections are enabled. O(1).
    #[inline(always)]
    pub fn is_active(&self) -> bool {
        self.enabled != 0
    }

    /// Check if a specific event type is enabled.
    #[inline(always)]
    pub fn is_enabled(&self, event: InspectEvent) -> bool {
        self.enabled & (1 << event as u8) != 0
    }

    /// Enable tracking for an event type.
    pub fn enable(&mut self, event: InspectEvent) {
        self.enabled |= 1 << event as u8;
    }

    /// Disable tracking for an event type.
    pub fn disable(&mut self, event: InspectEvent) {
        self.enabled &= !(1 << event as u8);
    }

    /// Enable all event types.
    pub fn enable_all(&mut self) {
        self.enabled = (1 << InspectEvent::COUNT) - 1;
    }

    /// Disable all event types.
    pub fn disable_all(&mut self) {
        self.enabled = 0;
    }

    /// Set the maximum number of events to retain.
    pub fn set_max_events(&mut self, max: usize) {
        self.max_events = max;
        if self.events.len() > max {
            let drain = self.events.len() - max;
            self.events.drain(0..drain);
        }
    }

    /// Record an event. Only called when the event type is enabled.
    pub fn record(&mut self, event: InspectEvent, addr: u64, size: u32, block_addr: u64) {
        self.event_counts[event as usize] += 1;
        if self.events.len() >= self.max_events {
            self.events.remove(0);
        }
        self.events.push(InspectRecord {
            event,
            addr,
            size,
            block_addr,
        });
    }

    /// Get all recorded events.
    pub fn events(&self) -> &[InspectRecord] {
        &self.events
    }

    /// Get event counts per type.
    pub fn event_counts(&self) -> &[u64; InspectEvent::COUNT] {
        &self.event_counts
    }

    /// Get events filtered by type.
    pub fn events_of_type(&self, event: InspectEvent) -> Vec<&InspectRecord> {
        self.events.iter().filter(|e| e.event == event).collect()
    }

    /// Clear all recorded events (keeps enabled state and counts).
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Get the enabled bitmask (for serialization).
    pub fn enabled_mask(&self) -> u8 {
        self.enabled
    }

    /// Set the enabled bitmask (for deserialization).
    pub fn set_enabled_mask(&mut self, mask: u8) {
        self.enabled = mask;
    }
}

/// Entry in the execution history trace.
///
/// Records block-level execution events with jumpkind and jump target.
/// Appended at each block execution, cloned on state fork.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HistoryEntry {
    /// Block address that was executed.
    pub addr: u64,
    /// Jump kind at block exit (0=Boring, 1=Call, 2=Ret, 3=Syscall, 4=Other).
    pub jumpkind: u8,
    /// Target address of the jump (where execution went after this block).
    pub jump_target: u64,
}

impl HistoryEntry {
    /// Jumpkind constants matching VEX conventions.
    pub const JK_BORING: u8 = 0;
    pub const JK_CALL: u8 = 1;
    pub const JK_RET: u8 = 2;
    pub const JK_SYSCALL: u8 = 3;
    pub const JK_OTHER: u8 = 4;

    /// Create from JumpKind enum.
    pub fn jumpkind_from_vex(jk: &crate::vex::JumpKind) -> u8 {
        if jk.is_call() {
            Self::JK_CALL
        } else if jk.is_ret() {
            Self::JK_RET
        } else if jk.is_syscall() {
            Self::JK_SYSCALL
        } else {
            Self::JK_BORING
        }
    }

    /// Convert jumpkind byte to string (for Python API).
    pub fn jumpkind_str(jk: u8) -> &'static str {
        match jk {
            Self::JK_BORING => "Ijk_Boring",
            Self::JK_CALL => "Ijk_Call",
            Self::JK_RET => "Ijk_Ret",
            Self::JK_SYSCALL => "Ijk_Sys_syscall",
            _ => "Ijk_Other",
        }
    }
}

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

/// Execution event from stepping a state.
///
/// `#[non_exhaustive]` per angr-irwe: minor versions may add new
/// variants as new stepping outcomes emerge; match sites must include
/// a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ExecutionEvent {
    /// Reached end of a basic block, continuing to next address.
    BlockEnd { next_addr: u64 },
    /// Encountered a symbolic branch condition.
    SymbolicBranch {
        condition: RustBV,
        true_target: u64,
        false_target: u64,
    },
    /// Hit a hook address (SimProcedure).
    Hook { addr: u64 },
    /// Syscall instruction.
    Syscall { num: u64 },
    /// Unmapped memory access - need Python callback.
    UnmappedMemory { addr: u64, size: u64 },
    /// Error during execution.
    Error { message: String },
}

/// Incremental state changes for efficient sync.
///
/// Instead of syncing entire state, we track only what changed.
#[derive(Debug, Clone, Default)]
pub struct StateChanges {
    /// Register changes: (offset, size, value_bytes).
    pub register_writes: Vec<(u32, u32, Vec<u8>)>,
    /// Memory writes: (addr, value_bytes).
    pub memory_writes: Vec<(u64, Vec<u8>)>,
    /// New constraints added (as Z3 AST indices or serialized form).
    pub new_constraints: Vec<u64>,
    /// Updated PC.
    pub new_pc: Option<u64>,
}

impl StateChanges {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.register_writes.is_empty()
            && self.memory_writes.is_empty()
            && self.new_constraints.is_empty()
            && self.new_pc.is_none()
    }
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
        self.registers.put_reg(name, value)
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
        self.solver
            .borrow()
            .set_use_shared_lineage_solver(enabled);
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

    /// Clone the three Python-AST metadata maps. Each PyObject ref-count is
    /// incremented under the GIL so the parent and fork share strong refs.
    fn clone_py_metadata(
        &self,
    ) -> (
        HashMap<u64, Py<PyAny>>,
        HashMap<u64, (Py<PyAny>, u32)>,
        HashMap<u64, (Py<PyAny>, u32)>,
    ) {
        Python::attach(|py| {
            let pages = self
                .symbolic_pages
                .iter()
                .map(|(k, v)| (*k, v.clone_ref(py)))
                .collect();
            let hook = self
                .hook_symbolic_memory
                .iter()
                .map(|(k, (v, sz))| (*k, (v.clone_ref(py), *sz)))
                .collect();
            let addr_map = self
                .addr_to_ast
                .iter()
                .map(|(k, (v, sz))| (*k, (v.clone_ref(py), *sz)))
                .collect();
            (pages, hook, addr_map)
        })
    }

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
    // Forking
    // =========================================================================

    /// Fork the state (O(1) copy-on-write).
    ///
    /// Creates a new state that shares memory pages via CoW.
    /// The solver context is forked to preserve constraints.
    ///
    /// See module-level `state-id-never-reused` (child gets a fresh
    /// monotonic ID, parent's ID is preserved on the parent),
    /// `arc-make-mut-cow` (registers/memory/hooks/environment/fs share Arc
    /// or persistent backing with the parent), and `state-metadata-dataclass`
    /// (the three `Py<PyAny>` metadata maps are cloned under the GIL).
    ///
    /// # Returns
    /// A new state with the same register/memory/constraint state.
    pub fn fork(&self) -> Self {
        let forked_solver = Rc::new(RefCell::new(self.solver.borrow().fork()));
        let (symbolic_pages, hook_symbolic_memory, addr_to_ast) = self.clone_py_metadata();

        let child_id = next_state_id();
        // `state-id-never-reused`: monotonic counter must produce a value
        // strictly greater than the parent's ID. Tautological today; this
        // assert fires if a future refactor reorders the allocation or
        // (worse) introduces ID recycling.
        debug_assert!(
            child_id > self.state_id,
            "next_state_id() must monotonically increase; got child={} parent={}",
            child_id,
            self.state_id,
        );

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            solver: forked_solver,
            pc: self.pc,
            state_id: child_id,
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            stdin_symbols: self.stdin_symbols.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment: self.environment.clone(),
            symbolic_pages,
            hook_symbolic_memory,
            addr_to_ast,
            last_time: self.last_time.clone(),
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
        }
    }

    /// Fork with a constraint on the true branch.
    pub fn fork_true(&self, condition: &RustBV) -> Self {
        let forked_solver = Rc::new(RefCell::new(self.solver.borrow().fork_true(condition)));
        let (symbolic_pages, hook_symbolic_memory, addr_to_ast) = self.clone_py_metadata();

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            stdin_symbols: self.stdin_symbols.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment: self.environment.clone(),
            symbolic_pages,
            hook_symbolic_memory,
            addr_to_ast,
            last_time: self.last_time.clone(),
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
        }
    }

    /// Fork with a constraint on the false branch.
    pub fn fork_false(&self, condition: &RustBV) -> Self {
        let forked_solver = Rc::new(RefCell::new(self.solver.borrow().fork_false(condition)));
        let (symbolic_pages, hook_symbolic_memory, addr_to_ast) = self.clone_py_metadata();

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            stdin_symbols: self.stdin_symbols.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment: self.environment.clone(),
            symbolic_pages,
            hook_symbolic_memory,
            addr_to_ast,
            last_time: self.last_time.clone(),
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
        }
    }

    /// Replace the solver context with a different one.
    /// Used for deferred fork processing where the alternate path needs a solver
    /// snapshot from before the branch constraint was added.
    pub fn replace_solver(&mut self, ctx: crate::symbolic::SymContext) {
        self.solver = Rc::new(RefCell::new(ctx));
    }

    /// Create a forked state using a full branch snapshot (solver + registers + memory).
    /// The resulting state has the correct state from the branch point, not from
    /// the continuation of the taken path.
    pub fn fork_from_snapshot(&self, snapshot: crate::interpreter::BranchSnapshot) -> Self {
        let forked_solver = Rc::new(RefCell::new(snapshot.solver));
        let (symbolic_pages, hook_symbolic_memory, addr_to_ast) = self.clone_py_metadata();
        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: snapshot.registers,
            memory: snapshot.memory.unwrap_or_else(|| self.memory.fork()),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            stdin_symbols: self.stdin_symbols.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment: self.environment.clone(),
            symbolic_pages,
            hook_symbolic_memory,
            addr_to_ast,
            last_time: self.last_time.clone(),
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
        }
    }

    /// Merge this state with one or more other states using symbolic merge conditions.
    ///
    /// Creates a new merged state where registers and memory that differ between
    /// states are represented as ITE expressions guarded by merge conditions.
    /// The solver receives guarded constraints from all input states.
    ///
    /// `merge_conditions` has one entry per state: self first, then each of `others`.
    /// Each condition is a fresh 1-bit symbolic variable indicating that path is active.
    ///
    /// Returns a new merged RustSimState.
    pub fn merge(&self, others: &[&RustSimState], merge_conditions: &[RustBV]) -> Self {
        assert_eq!(
            others.len() + 1,
            merge_conditions.len(),
            "merge_conditions must have one entry per state (self + others)"
        );

        // Merge solver contexts
        let other_solvers: Vec<_> = others.iter().map(|s| s.solver.borrow()).collect();
        let other_solver_refs: Vec<&SymContext> = other_solvers.iter().map(|s| &**s).collect();
        let merged_solver = self
            .solver
            .borrow()
            .merge(&other_solver_refs, merge_conditions);

        // Start with a clone of self's registers and merge each other into it
        let mut merged_regs = self.registers.fork();
        for (i, other) in others.iter().enumerate() {
            let cond = &merge_conditions[i + 1]; // skip self's condition
            merged_regs.merge(&other.registers, cond, &merged_solver);
        }

        // Start with a clone of self's memory and merge each other into it
        let mut merged_mem = self.memory.fork();
        for (i, other) in others.iter().enumerate() {
            let cond = &merge_conditions[i + 1];
            merged_mem.merge(&other.memory, cond, &merged_solver);
        }

        // Merge stdout buffers: pick the longest (heuristic — full merge would need ITE on bytes)
        let mut best_fs = self.fs.clone();
        let mut best_len = self.stdout_buffer().len();
        for other in others {
            let other_len = other.stdout_buffer().len();
            if other_len > best_len {
                best_fs = other.fs.clone();
                best_len = other_len;
            }
        }

        // Merge stdin symbols (union)
        let mut merged_stdin = self.stdin_symbols.clone();
        for other in others {
            for sym in &other.stdin_symbols {
                if !merged_stdin.iter().any(|(n, _)| n == &sym.0) {
                    merged_stdin.push(sym.clone());
                }
            }
        }

        let (symbolic_pages, hook_symbolic_memory, addr_to_ast) = self.clone_py_metadata();
        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: merged_regs,
            memory: merged_mem,
            solver: Rc::new(RefCell::new(merged_solver)),
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: best_fs,
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            stdin_symbols: merged_stdin,
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment: self.environment.clone(),
            symbolic_pages,
            hook_symbolic_memory,
            addr_to_ast,
            last_time: self.last_time.clone(),
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
        }
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

impl Clone for RustSimState {
    fn clone(&self) -> Self {
        self.fork()
    }
}

// =============================================================================
// Snapshot / Serialization (angr-x04s.1.3)
// =============================================================================

/// Format-version byte at the head of every [`RustSimState::to_serialized`]
/// envelope. Bump on any breaking shape change to [`RustSimStateSnapshot`]
/// so a stale snapshot fails fast with `SnapshotError::VersionMismatch`
/// instead of silently producing a wrong-shaped state.
pub const SNAPSHOT_VERSION: u8 = 1;

/// Errors raised by [`RustSimState::from_serialized`] /
/// [`StashManager::load_snapshot`].
#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("empty snapshot envelope")]
    EmptyEnvelope,
    #[error("snapshot version mismatch: have {found}, expected {expected}")]
    VersionMismatch { found: u8, expected: u8 },
    #[error("decode error: {0}")]
    Decode(String),
}

/// Snapshot of a [`RustSimState`]'s persistable state (angr-x04s.1.3).
///
/// Covers all bucket A/B/C fields per the `rustsimstate-field-buckets` bd
/// memory:
///
/// * **Bucket A (trivials)** — pc, state_id, parent_id, history,
///   detailed_history, max_history, heap_brk, posix_brk, mmap_base,
///   stdin_symbols, call_stack, heap_metadata, no_ip_concretization,
///   no_symbolic_jump_resolution, keep_ip_symbolic, vex_arch,
///   inspection, concretizer, fs, track_history, drop_terminal flag
///   (carried on StashManager side).
/// * **Bucket B (concrete + symbolic overlay)** — registers
///   ([`RegisterFile`] serde), memory ([`SymbolicMemorySnapshot`]).
/// * **Bucket C (Arc-shared collapse)** — hooks (Vec<u64>), environment
///   (BTreeMap<bytes, bytes>).
/// * **SymContext** — captured via [`SymContextSnapshot`] (replays
///   `assumed_constraints` into a fresh Z3 solver on restore).
///
/// **Bucket D (`Py<PyAny>` overlays)** — symbolic_pages,
/// hook_symbolic_memory, addr_to_ast, last_time — deferred per the task
/// acceptance. After restore these fields are empty / None; Python-side
/// integration tests (.1.4) will route them through claripy.dumps/loads.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RustSimStateSnapshot {
    pub arch_name: String,
    pub vex_arch: crate::vex::VexArch,
    pub registers: crate::arch::RegisterFile,
    pub memory: crate::memory::SymbolicMemorySnapshot,
    pub solver: crate::symbolic::SymContextSnapshot,
    pub pc: u64,
    pub state_id: u64,
    pub parent_id: Option<u64>,
    pub history: Vec<u64>,
    pub detailed_history: Vec<HistoryEntry>,
    pub max_history: usize,
    pub hooks: Vec<u64>,
    pub concretizer: AddressConcretizer,
    pub track_history: bool,
    pub fs: FileSystem,
    pub heap_brk: u64,
    pub posix_brk: u64,
    pub mmap_base: u64,
    pub stdin_symbols: Vec<(String, u32)>,
    pub call_stack: Vec<CallStackEntry>,
    pub heap_metadata: HeapMetadata,
    pub inspection: InspectionManager,
    /// `(key, value)` byte pairs sorted by key for deterministic ordering.
    /// Not a `BTreeMap<Vec<u8>, Vec<u8>>` because `serde_json` only allows
    /// string-shaped map keys; the angr environment is byte-keyed.
    pub environment: Vec<(Vec<u8>, Vec<u8>)>,
    pub no_ip_concretization: bool,
    pub no_symbolic_jump_resolution: bool,
    pub keep_ip_symbolic: bool,
}

impl RustSimState {
    /// Build a serializable snapshot of this state (angr-x04s.1.3).
    ///
    /// Bucket-D `Py<PyAny>` overlays (symbolic_pages, hook_symbolic_memory,
    /// addr_to_ast, last_time) are NOT captured here — see
    /// [`RustSimStateSnapshot`].
    pub fn to_snapshot(&self) -> RustSimStateSnapshot {
        let mut hooks: Vec<u64> = self.hooks.iter().copied().collect();
        hooks.sort_unstable();
        let mut environment: Vec<(Vec<u8>, Vec<u8>)> = self
            .environment
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        environment.sort_by(|a, b| a.0.cmp(&b.0));
        RustSimStateSnapshot {
            arch_name: self.arch.name().to_string(),
            vex_arch: self.vex_arch,
            registers: self.registers.clone(),
            memory: self.memory.to_snapshot(),
            solver: self.solver.borrow().to_snapshot(),
            pc: self.pc,
            state_id: self.state_id,
            parent_id: self.parent_id,
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks,
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            stdin_symbols: self.stdin_symbols.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment,
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
        }
    }

    /// Restore a snapshot into a fresh [`RustSimState`]. Replays solver
    /// constraints via [`SymContext::restore_from_snapshot`] so the Z3
    /// solver, sat/model caches, and `assumed_constraints` log all rebuild
    /// consistently. Bucket-D `Py<PyAny>` overlays restore to empty (see
    /// [`RustSimStateSnapshot`]).
    pub fn from_snapshot(snap: RustSimStateSnapshot) -> Result<Self, String> {
        let arch = arch_from_name(&snap.arch_name)
            .ok_or_else(|| format!("unknown architecture: {}", snap.arch_name))?;
        let solver = Rc::new(RefCell::new(SymContext::new()));
        solver.borrow().restore_from_snapshot(&snap.solver);
        let memory = SymbolicMemory::from_snapshot(snap.memory);
        let environment: HashMap<Vec<u8>, Vec<u8>> =
            snap.environment.into_iter().collect();
        let hooks: HashSet<u64> = snap.hooks.into_iter().collect();
        Ok(RustSimState {
            arch,
            vex_arch: snap.vex_arch,
            registers: snap.registers,
            memory,
            solver,
            pc: snap.pc,
            state_id: snap.state_id,
            parent_id: snap.parent_id,
            history: snap.history,
            detailed_history: snap.detailed_history,
            max_history: snap.max_history,
            hooks: Arc::new(hooks),
            concretizer: snap.concretizer,
            track_history: snap.track_history,
            fs: snap.fs,
            heap_brk: snap.heap_brk,
            posix_brk: snap.posix_brk,
            mmap_base: snap.mmap_base,
            stdin_symbols: snap.stdin_symbols,
            call_stack: snap.call_stack,
            heap_metadata: snap.heap_metadata,
            inspection: snap.inspection,
            environment: Arc::new(environment),
            symbolic_pages: HashMap::new(),
            hook_symbolic_memory: HashMap::new(),
            addr_to_ast: HashMap::new(),
            last_time: None,
            no_ip_concretization: snap.no_ip_concretization,
            no_symbolic_jump_resolution: snap.no_symbolic_jump_resolution,
            keep_ip_symbolic: snap.keep_ip_symbolic,
        })
    }

    /// Serialize this state to a versioned envelope:
    /// `[SNAPSHOT_VERSION: u8] ++ serde_json(RustSimStateSnapshot)`.
    /// The format-version byte lets [`Self::from_serialized`] reject a
    /// stale on-disk snapshot fast. `serde_json` was chosen over postcard
    /// for the prototype because the inner [`RustBV`] op-tree carries
    /// `Arc<...>` boxed enums whose postcard schema would lock the format
    /// to today's [`crate::symbolic::value::BVOp`] layout; JSON tolerates
    /// minor variant churn without a breaking change.
    pub fn to_serialized(&self) -> Vec<u8> {
        let snap = self.to_snapshot();
        let body = serde_json::to_vec(&snap).expect("snapshot encode");
        let mut out = Vec::with_capacity(1 + body.len());
        out.push(SNAPSHOT_VERSION);
        out.extend_from_slice(&body);
        out
    }

    /// Inverse of [`Self::to_serialized`]. Rejects an empty envelope or a
    /// version-byte mismatch with [`SnapshotError`].
    pub fn from_serialized(bytes: &[u8]) -> Result<Self, SnapshotError> {
        if bytes.is_empty() {
            return Err(SnapshotError::EmptyEnvelope);
        }
        let version = bytes[0];
        if version != SNAPSHOT_VERSION {
            return Err(SnapshotError::VersionMismatch {
                found: version,
                expected: SNAPSHOT_VERSION,
            });
        }
        // serde_json's default recursion limit (128) is hit by deep
        // RustBV op-trees that real benches accumulate (per-byte memory
        // loads nest store/load chains hundreds of levels deep).
        // `disable_recursion_limit()` lifts the cap; the on-disk envelope
        // is trusted (written by our own `to_serialized`) so the DoS
        // hardening the limit provides is not load-bearing here.
        let mut de = serde_json::Deserializer::from_slice(&bytes[1..]);
        de.disable_recursion_limit();
        let snap: RustSimStateSnapshot = serde::Deserialize::deserialize(&mut de)
            .map_err(|e| SnapshotError::Decode(e.to_string()))?;
        Self::from_snapshot(snap).map_err(SnapshotError::Decode)
    }
}

// =============================================================================
// Python Bindings
// =============================================================================

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
        let inner = RustSimState::new_with_endian(arch, little_endian)
            .map_err(|e| PyValueError::new_err(e))?;
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
        // Safety: caller guarantees pointer is a valid Z3_ast in shared context.
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
    pub fn memory_store(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        let width = (data.len() * 8) as u32;
        let mut value: u128 = 0;
        for (i, &b) in data.iter().enumerate() {
            value |= (b as u128) << (i * 8);
        }
        let bv = RustBV::concrete(value, width);
        self.inner
            .memory_store(addr, bv)
            .map_err(|e| PyValueError::new_err(e.to_string()))
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
    #[pyo3(signature = (use_approximate, read_range_limit=None, write_range_limit=None, symbolic_write_addresses=false))]
    pub fn configure_concretization_strategies(
        &mut self,
        use_approximate: bool,
        read_range_limit: Option<u64>,
        write_range_limit: Option<u64>,
        symbolic_write_addresses: bool,
    ) {
        self.inner.concretizer.configure_strategies(
            use_approximate,
            read_range_limit,
            write_range_limit,
            symbolic_write_addresses,
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

// =============================================================================
// State Snapshot for Exploration Export
// =============================================================================

/// Internal memory page data (addr, data, permissions, symbolic_offsets).
type PageData = (u64, Vec<u8>, u8, Vec<u16>);

/// Complete state snapshot for exploration export.
///
/// This contains all information needed to reconstruct an angr SimState
/// from a Rust execution state.
#[pyclass(name = "ExplorationStateSnapshot")]
pub struct ExplorationStateSnapshot {
    /// Unique state identifier.
    #[pyo3(get)]
    pub state_id: u64,
    /// Parent state ID (for fork tracking).
    #[pyo3(get)]
    pub parent_id: Option<u64>,
    /// Program counter.
    #[pyo3(get)]
    pub pc: u64,
    /// Architecture name.
    #[pyo3(get)]
    pub arch_name: String,
    /// Raw register bytes.
    registers_raw: Vec<u8>,
    /// Memory pages: (addr, data, permissions, symbolic_offsets).
    memory_pages: Vec<PageData>,
    /// Number of constraints in the solver.
    #[pyo3(get)]
    pub constraint_count: usize,
    /// Basic block history.
    history: Vec<u64>,
    /// Named register values: (name, concrete_value, size_bits).
    /// Pre-computed at export time so Python doesn't need offset tables.
    named_registers: Vec<(String, u128, u32)>,
    /// Call stack entries: (call_site_addr, callee_addr, return_addr, stack_ptr).
    call_stack: Vec<(u64, u64, u64, u64)>,
    /// Detailed execution history: (addr, jumpkind, jump_target).
    detailed_history: Vec<(u64, u8, u64)>,
    /// Heap allocations: (addr, size) for active allocations.
    heap_allocated: Vec<(u64, u64)>,
    /// Heap freed addresses.
    heap_freed: Vec<u64>,
    /// Open file descriptors: (fd, name, position, flags, content_len, is_open).
    open_fds: Vec<(u32, String, u64, u32, usize, bool)>,
    /// Inspection event counts per type.
    inspection_counts: Vec<(String, u64)>,
    /// Inspection enabled bitmask.
    #[pyo3(get)]
    pub inspection_enabled: u8,
}

#[pymethods]
impl ExplorationStateSnapshot {
    /// Get raw register bytes.
    pub fn get_registers_raw(&self) -> Vec<u8> {
        self.registers_raw.clone()
    }

    /// Get named register values as a dict: {name: (value, size_bits)}.
    ///
    /// Pre-computed at export time using Rust's register tables,
    /// so Python doesn't need architecture-specific offset mapping.
    pub fn get_registers_named(&self) -> std::collections::HashMap<String, (u128, u32)> {
        self.named_registers
            .iter()
            .map(|(name, value, bits)| (name.clone(), (*value, *bits)))
            .collect()
    }

    /// Get history (basic block addresses visited).
    pub fn get_history(&self) -> Vec<u64> {
        self.history.clone()
    }

    /// Get call stack as list of (call_site_addr, callee_addr, return_addr, stack_ptr) tuples.
    pub fn get_call_stack(&self) -> Vec<(u64, u64, u64, u64)> {
        self.call_stack.clone()
    }

    /// Get call stack depth.
    pub fn get_call_stack_depth(&self) -> usize {
        self.call_stack.len()
    }

    /// Get detailed history as list of (addr, jumpkind, jump_target) tuples.
    ///
    /// jumpkind: 0=Boring, 1=Call, 2=Ret, 3=Syscall, 4=Other
    pub fn get_detailed_history(&self) -> Vec<(u64, u8, u64)> {
        self.detailed_history.clone()
    }

    /// Get detailed history with string jumpkinds.
    ///
    /// Returns list of (addr, jumpkind_str, jump_target) tuples.
    pub fn get_detailed_history_str(&self) -> Vec<(u64, String, u64)> {
        self.detailed_history
            .iter()
            .map(|(addr, jk, target)| (*addr, HistoryEntry::jumpkind_str(*jk).to_string(), *target))
            .collect()
    }

    /// Get the number of memory pages.
    pub fn page_count(&self) -> usize {
        self.memory_pages.len()
    }

    /// Get a memory page by index.
    /// Returns (addr, data, permissions, symbolic_offsets) or None.
    pub fn get_page(&self, index: usize) -> Option<(u64, Vec<u8>, u8, Vec<u16>)> {
        self.memory_pages
            .get(index)
            .map(|p| (p.0, p.1.clone(), p.2, p.3.clone()))
    }

    /// Get all memory page addresses.
    pub fn page_addresses(&self) -> Vec<u64> {
        self.memory_pages.iter().map(|p| p.0).collect()
    }

    /// Load bytes from memory at a given address.
    /// Returns None if the address is not mapped.
    pub fn memory_load(&self, addr: u64, size: usize) -> Option<Vec<u8>> {
        let page_addr = addr & !0xFFF;
        let offset = (addr & 0xFFF) as usize;

        // Find the page
        for page in &self.memory_pages {
            if page.0 == page_addr {
                if offset + size <= page.1.len() {
                    return Some(page.1[offset..offset + size].to_vec());
                }
            }
        }
        None
    }

    /// Get heap allocations as list of (addr, size) tuples.
    pub fn get_heap_allocated(&self) -> Vec<(u64, u64)> {
        self.heap_allocated.clone()
    }

    /// Get heap freed addresses.
    pub fn get_heap_freed(&self) -> Vec<u64> {
        self.heap_freed.clone()
    }

    /// Get number of active heap allocations.
    pub fn get_heap_alloc_count(&self) -> usize {
        self.heap_allocated.len()
    }

    /// Get number of heap free calls.
    pub fn get_heap_free_count(&self) -> usize {
        self.heap_freed.len()
    }

    /// Get open file descriptors as list of (fd, name, position, flags, content_len, is_open).
    pub fn get_open_fds(&self) -> Vec<(u32, String, u64, u32, usize, bool)> {
        self.open_fds.clone()
    }

    /// Get the number of tracked file descriptors.
    pub fn get_fd_count(&self) -> usize {
        self.open_fds.len()
    }

    /// Get inspection event counts as list of (event_name, count) tuples.
    pub fn get_inspection_counts(&self) -> Vec<(String, u64)> {
        self.inspection_counts.clone()
    }

    /// Get symbolic byte offsets for a page.
    /// Returns empty vec if page not found.
    pub fn get_symbolic_offsets(&self, page_addr: u64) -> Vec<u16> {
        for page in &self.memory_pages {
            if page.0 == page_addr {
                return page.3.clone();
            }
        }
        Vec::new()
    }
}

impl RustSimState {
    /// Export the complete state as a snapshot.
    ///
    /// This creates a self-contained snapshot that can be used to
    /// reconstruct an angr SimState.
    pub fn export_full(&self) -> ExplorationStateSnapshot {
        // Export registers
        let registers_raw = self.get_registers_raw();

        // Export named registers: read each GP register by name
        let mut named_registers = Vec::new();
        let ctx = self.solver.borrow();
        for &name in self.arch.register_names() {
            if let Some(size) = self.arch.register_size(name) {
                let bv = self.registers.get_reg(name, &ctx);
                if let Some(bv) = bv {
                    if let Some(val) = bv.as_u128() {
                        named_registers.push((name.to_string(), val, size * 8));
                    }
                    // Skip symbolic registers (they'll need AST recovery)
                }
            }
        }

        // Flush pending writes before exporting memory pages.
        // We need a mutable borrow, but export_full takes &self. Use an
        // unsafe interior mutability pattern is not ideal, so we just report
        // unflushed writes via pending_writes_count on the snapshot.
        // Callers should call flush_pending_writes() before export_full()
        // if they need materialized memory.

        // Export memory pages as tuples: (addr, data, permissions, symbolic_offsets)
        let mut memory_pages: Vec<PageData> = Vec::new();
        for (page_num, page) in self.memory.pages().iter() {
            let page_addr = page_num << 12;
            let data = page.load_concrete(0, crate::memory::PAGE_SIZE as u16);
            let permissions = page.permissions().to_bits();
            let symbolic_offsets = page.symbolic_offsets();

            memory_pages.push((page_addr, data, permissions, symbolic_offsets));
        }

        // Get constraint count
        let constraint_count = self.solver.borrow().num_constraints();

        // Export call stack
        let call_stack: Vec<(u64, u64, u64, u64)> = self
            .call_stack
            .iter()
            .map(|e| (e.call_site_addr, e.callee_addr, e.return_addr, e.stack_ptr))
            .collect();

        // Export detailed history
        let detailed_history: Vec<(u64, u8, u64)> = self
            .detailed_history
            .iter()
            .map(|e| (e.addr, e.jumpkind, e.jump_target))
            .collect();

        ExplorationStateSnapshot {
            state_id: self.state_id,
            parent_id: self.parent_id,
            pc: self.pc,
            arch_name: self.arch.name().to_string(),
            registers_raw,
            memory_pages,
            constraint_count,
            history: self.history.clone(),
            named_registers,
            call_stack,
            detailed_history,
            heap_allocated: self
                .heap_metadata
                .allocated
                .iter()
                .map(|(&addr, &size)| (addr, size))
                .collect(),
            heap_freed: self.heap_metadata.freed.clone(),
            open_fds: self
                .fs
                .all_fds()
                .iter()
                .filter_map(|&fd| {
                    let info = self.fs.fd_info(fd)?;
                    Some((fd, info.0.to_string(), info.1, info.2, info.3, info.4))
                })
                .collect(),
            inspection_counts: self
                .inspection
                .event_counts()
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 0)
                .filter_map(|(i, &count)| {
                    InspectEvent::from_u8(i as u8).map(|e| (e.name().to_string(), count))
                })
                .collect(),
            inspection_enabled: self.inspection.enabled_mask(),
        }
    }

    /// Flush pending writes and then export.
    /// This materializes any deferred symbolic stores before creating the snapshot.
    pub fn flush_and_export_full(&mut self) -> ExplorationStateSnapshot {
        // Flush pending writes using the current solver context
        {
            let ctx = self.solver.borrow();
            let _ = self.memory.flush_pending_writes(&*ctx, &self.concretizer);
        }
        self.export_full()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = RustSimState::new("amd64").unwrap();
        assert_eq!(state.vex_arch(), VexArch::AMD64);
        assert_eq!(state.pc(), 0);
    }

    #[test]
    fn test_state_fork() {
        let mut state1 = RustSimState::new("amd64").unwrap();
        state1.set_pc(0x1000);
        state1.set_register("rax", RustBV::concrete(42, 64));

        let state2 = state1.fork();

        // Both should have same values
        assert_eq!(state2.pc(), 0x1000);
        assert_eq!(state2.get_register("rax").unwrap().as_u64(), Some(42));

        // Different state IDs
        assert_ne!(state1.state_id(), state2.state_id());

        // state2's parent should be state1
        assert_eq!(state2.parent_id(), Some(state1.state_id()));
    }

    #[test]
    fn test_set_detailed_history_honors_cap() {
        // set_detailed_history (called once per step from interpreter results)
        // must drain the oldest entries when the incoming buffer exceeds the
        // configured cap. Otherwise long blocks bypass max_history.
        let mut state = RustSimState::new("amd64").unwrap();
        state.set_max_history(3);
        let entries: Vec<HistoryEntry> = (0..10)
            .map(|i| HistoryEntry {
                addr: 0x1000 + i,
                jumpkind: 0,
                jump_target: 0,
            })
            .collect();
        state.set_detailed_history(entries);
        let kept = state.detailed_history();
        assert_eq!(kept.len(), 3);
        // FIFO eviction: should retain the most-recent 3 entries.
        assert_eq!(kept[0].addr, 0x1007);
        assert_eq!(kept[2].addr, 0x1009);
    }

    #[test]
    fn test_set_max_history_trims_retroactively() {
        // Lowering max_history on a state that already exceeds the new cap
        // must FIFO-evict the oldest entries down to the cap immediately.
        // Without this, add_to_history (which only removes one entry per
        // push when over cap) never converges and the buffer stays bloated.
        let mut state = RustSimState::new("amd64").unwrap();
        state.set_max_history(0); // unlimited
        for i in 0..20u64 {
            state.add_to_history(0x3000 + i);
            state.add_history_entry(0x3000 + i, 0, 0);
        }
        assert_eq!(state.history().len(), 20);
        assert_eq!(state.detailed_history().len(), 20);

        // Retroactively cap to 4 — both buffers shrink to the most-recent 4.
        state.set_max_history(4);
        let kept = state.history();
        assert_eq!(kept.len(), 4);
        assert_eq!(kept, &[0x3010, 0x3011, 0x3012, 0x3013]);
        let kept_detailed = state.detailed_history();
        assert_eq!(kept_detailed.len(), 4);
        assert_eq!(kept_detailed[0].addr, 0x3010);
        assert_eq!(kept_detailed[3].addr, 0x3013);
    }

    #[test]
    fn test_set_max_history_zero_no_trim() {
        // Switching to max=0 (unlimited) must NOT trim — existing entries stay.
        let mut state = RustSimState::new("amd64").unwrap();
        state.set_max_history(5);
        for i in 0..5u64 {
            state.add_to_history(0x4000 + i);
        }
        state.set_max_history(0);
        assert_eq!(state.history().len(), 5);
    }

    #[test]
    fn test_set_detailed_history_unlimited() {
        // max_history = 0 means no cap (legacy behavior).
        let mut state = RustSimState::new("amd64").unwrap();
        state.set_max_history(0);
        let entries: Vec<HistoryEntry> = (0..50)
            .map(|i| HistoryEntry {
                addr: 0x2000 + i,
                jumpkind: 0,
                jump_target: 0,
            })
            .collect();
        state.set_detailed_history(entries);
        assert_eq!(state.detailed_history().len(), 50);
    }

    #[test]
    fn test_state_memory() {
        let mut state = RustSimState::new("amd64").unwrap();

        // Map and write
        state.map_memory(0x1000, 0x1000, Permission::RWX);
        state
            .memory_store(0x1000, RustBV::concrete(0xDEADBEEF, 32))
            .unwrap();

        // Read back
        let val = state.memory_load(0x1000, 4).unwrap();
        assert_eq!(val.as_u64(), Some(0xDEADBEEF));
    }

    #[test]
    fn test_state_fork_memory_cow() {
        let mut state1 = RustSimState::new("amd64").unwrap();
        state1.map_memory(0x1000, 0x1000, Permission::RWX);
        state1
            .memory_store(0x1000, RustBV::concrete(0xAAAA, 16))
            .unwrap();

        let mut state2 = state1.fork();

        // Modify state2
        state2
            .memory_store(0x1000, RustBV::concrete(0xBBBB, 16))
            .unwrap();

        // state1 should still have original value
        let val1 = state1.memory_load(0x1000, 2).unwrap();
        assert_eq!(val1.as_u64(), Some(0xAAAA));

        // state2 should have new value
        let val2 = state2.memory_load(0x1000, 2).unwrap();
        assert_eq!(val2.as_u64(), Some(0xBBBB));
    }

    #[test]
    fn test_filesystem_default() {
        let fs = FileSystem::default();
        assert!(fs.is_open(0)); // stdin
        assert!(fs.is_open(1)); // stdout
        assert!(fs.is_open(2)); // stderr
        assert!(!fs.is_open(3));
        assert_eq!(fs.next_fd(), 3);
    }

    #[test]
    fn test_filesystem_open_close() {
        let mut fs = FileSystem::default();
        let fd = fs.open("test.txt".to_string(), FdFlags::ReadOnly);
        assert_eq!(fd, 3);
        assert!(fs.is_open(3));

        let closed = fs.close(3);
        assert!(closed);
        assert!(!fs.is_open(3));

        // Double close returns false
        assert!(!fs.close(3));
    }

    #[test]
    fn test_filesystem_write_read() {
        let mut fs = FileSystem::default();
        let fd = fs.open_with_content(
            "data.bin".to_string(),
            FdFlags::ReadOnly,
            b"hello world".to_vec(),
        );

        let data = fs.read(fd, 5);
        assert_eq!(data, b"hello");

        let data2 = fs.read(fd, 6);
        assert_eq!(data2, b" world");

        // Read past end
        let data3 = fs.read(fd, 10);
        assert!(data3.is_empty());
    }

    #[test]
    fn test_filesystem_seek() {
        let mut fs = FileSystem::default();
        let fd = fs.open_with_content("data.bin".to_string(), FdFlags::ReadOnly, vec![0u8; 100]);

        // SEEK_SET
        assert_eq!(fs.seek(fd, 50, 0), Some(50));
        // SEEK_CUR
        assert_eq!(fs.seek(fd, 10, 1), Some(60));
        // SEEK_END
        assert_eq!(fs.seek(fd, -5, 2), Some(95));
        // Invalid whence
        assert_eq!(fs.seek(fd, 0, 99), None);
    }

    #[test]
    fn test_filesystem_fork_isolation() {
        let mut state = RustSimState::new("amd64").unwrap();
        state
            .file_system()
            .open("test.txt".to_string(), FdFlags::ReadOnly);
        assert!(state.file_system_ref().is_open(3));

        let mut forked = state.fork();
        // Close in forked state
        forked.file_system().close(3);
        assert!(!forked.file_system_ref().is_open(3));
        // Original should still be open
        assert!(state.file_system_ref().is_open(3));
    }

    #[test]
    fn test_filesystem_backward_compat() {
        // fd_buffer/write_fd should still work through FileSystem
        let mut state = RustSimState::new("amd64").unwrap();
        state.write_stdout(b"hello");
        assert_eq!(state.stdout_buffer(), b"hello");
        assert!(state.has_stdout());

        state.write_fd(2, b"err");
        assert_eq!(state.fd_buffer(2), b"err");
    }

    #[test]
    fn test_filesystem_open_fds() {
        let mut fs = FileSystem::default();
        let open = fs.open_fds();
        assert_eq!(open, vec![0, 1, 2]); // stdin, stdout, stderr

        fs.open("a.txt".to_string(), FdFlags::ReadOnly);
        fs.open("b.txt".to_string(), FdFlags::WriteOnly);
        let open = fs.open_fds();
        assert_eq!(open, vec![0, 1, 2, 3, 4]);

        fs.close(3);
        let open = fs.open_fds();
        assert_eq!(open, vec![0, 1, 2, 4]);
    }

    #[test]
    fn test_filesystem_dup_dup2_pipe() {
        let mut fs = FileSystem::default();
        let fd = fs.open("a.txt".to_string(), FdFlags::ReadOnly);
        assert_eq!(fd, 3);

        // dup
        let dup_fd = fs.dup(fd).unwrap();
        assert_eq!(dup_fd, 4);
        assert_eq!(fs.fd_info(dup_fd).unwrap().0, "a.txt");

        // dup of closed fd returns None
        fs.close(fd);
        assert!(fs.dup(fd).is_none());

        // dup2 with fresh state
        let mut fs2 = FileSystem::default();
        let src = fs2.open("src.txt".to_string(), FdFlags::ReadOnly);
        let dst = fs2.dup2(src, 10).unwrap();
        assert_eq!(dst, 10);
        assert_eq!(fs2.fd_info(10).unwrap().0, "src.txt");
        // next_fd advanced past 10
        assert!(fs2.next_fd() > 10);

        // dup2(self, self) returns self when open
        assert_eq!(fs2.dup2(src, src), Some(src));

        // pipe allocates two consecutive fds
        let mut fs3 = FileSystem::default();
        let (r, w) = fs3.pipe();
        assert_eq!((r, w), (3, 4));
        assert!(fs3.is_open(r));
        assert!(fs3.is_open(w));
        assert_eq!(fs3.fd_info(r).unwrap().2, 0); // ReadOnly
        assert_eq!(fs3.fd_info(w).unwrap().2, 1); // WriteOnly
    }

    #[test]
    fn test_inspection_default_inactive() {
        let mgr = InspectionManager::default();
        assert!(!mgr.is_active());
        assert!(!mgr.is_enabled(InspectEvent::MemRead));
    }

    #[test]
    fn test_inspection_enable_disable() {
        let mut mgr = InspectionManager::default();
        mgr.enable(InspectEvent::MemRead);
        assert!(mgr.is_active());
        assert!(mgr.is_enabled(InspectEvent::MemRead));
        assert!(!mgr.is_enabled(InspectEvent::MemWrite));

        mgr.enable_all();
        assert!(mgr.is_enabled(InspectEvent::Fork));
        assert!(mgr.is_enabled(InspectEvent::Exit));

        mgr.disable(InspectEvent::MemRead);
        assert!(!mgr.is_enabled(InspectEvent::MemRead));
        assert!(mgr.is_enabled(InspectEvent::MemWrite));

        mgr.disable_all();
        assert!(!mgr.is_active());
    }

    #[test]
    fn test_inspection_record_events() {
        let mut mgr = InspectionManager::default();
        mgr.enable(InspectEvent::MemWrite);

        mgr.record(InspectEvent::MemWrite, 0x1000, 4, 0x400000);
        mgr.record(InspectEvent::MemWrite, 0x1004, 8, 0x400010);

        assert_eq!(mgr.events().len(), 2);
        assert_eq!(mgr.event_counts()[InspectEvent::MemWrite as usize], 2);

        let e = &mgr.events()[0];
        assert_eq!(e.event, InspectEvent::MemWrite);
        assert_eq!(e.addr, 0x1000);
        assert_eq!(e.size, 4);
        assert_eq!(e.block_addr, 0x400000);
    }

    #[test]
    fn test_inspection_ring_buffer() {
        let mut mgr = InspectionManager::default();
        mgr.set_max_events(3);
        mgr.enable(InspectEvent::MemRead);

        for i in 0..5 {
            mgr.record(InspectEvent::MemRead, i * 0x100, 4, 0);
        }

        // Only last 3 should remain
        assert_eq!(mgr.events().len(), 3);
        assert_eq!(mgr.events()[0].addr, 0x200);
        assert_eq!(mgr.events()[2].addr, 0x400);
        // But total count should be 5
        assert_eq!(mgr.event_counts()[InspectEvent::MemRead as usize], 5);
    }

    #[test]
    fn test_inspection_on_state() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.inspection_mut().enable(InspectEvent::MemWrite);
        state.inspection_mut().enable(InspectEvent::MemRead);

        state.set_pc(0x400000);
        state.inspect_mem_write(0x1000, 8);
        state.inspect_mem_read(0x2000, 4);

        assert_eq!(
            state.inspection().event_counts()[InspectEvent::MemWrite as usize],
            1
        );
        assert_eq!(
            state.inspection().event_counts()[InspectEvent::MemRead as usize],
            1
        );
    }

    #[test]
    fn test_inspection_fork_isolation() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.inspection_mut().enable(InspectEvent::MemWrite);
        state.inspect_mem_write(0x1000, 4);

        let mut forked = state.fork();
        forked.inspect_mem_write(0x2000, 4);

        // Parent should have 1 event
        assert_eq!(
            state.inspection().event_counts()[InspectEvent::MemWrite as usize],
            1
        );
        // Forked should have 2 (inherited 1 + new 1)
        assert_eq!(
            forked.inspection().event_counts()[InspectEvent::MemWrite as usize],
            2
        );
    }

    #[test]
    fn test_inspection_disabled_no_record() {
        let mut state = RustSimState::new("amd64").unwrap();
        // Don't enable anything
        state.inspect_mem_write(0x1000, 4);
        state.inspect_mem_read(0x2000, 4);

        assert_eq!(state.inspection().events().len(), 0);
        assert_eq!(
            state.inspection().event_counts()[InspectEvent::MemWrite as usize],
            0
        );
    }

    // =========================================================================
    // Snapshot / Serialization tests (angr-x04s.1.3)
    // =========================================================================

    /// Build a representative RustSimState that touches each bucket A/B/C
    /// field (concrete + symbolic registers, mapped memory pages, solver
    /// constraints, history, call stack, file system, hooks, environment,
    /// flags). The Z3 ASTs are minted inside whichever Z3 context the test
    /// is currently running under.
    #[cfg(feature = "vex-engine-z3")]
    fn build_populated_state() -> RustSimState {
        let mut s = RustSimState::new("amd64").unwrap();
        s.set_pc(0x4012a0);

        // Bucket B (registers): one concrete, one symbolic.
        s.set_register("rax", RustBV::concrete(0xdead_beef, 64));
        let rbx_sym = {
            let ctx = s.solver().borrow();
            RustBV::symbolic(&ctx, "rbx_sym", 64)
        };
        s.set_register("rbx", rbx_sym.clone());

        // Bucket B (memory): map a page with concrete bytes.
        s.memory_mut()
            .map_data(0x10_0000u64, &[1u8, 2, 3, 4, 5], Permission::RWX);

        // Bucket A: history + call stack.
        s.add_to_history(0x4011a0);
        s.add_to_history(0x4012a0);
        s.push_call(0x4012a0, 0x401400, 0x4012a5, 0x7fff_ffff_0000);
        // Heap metadata via the public heap_alloc/heap_free helpers.
        let a1 = s.heap_alloc(32);
        let _a2 = s.heap_alloc(64);
        let _ = s.heap_free(a1);

        // Bucket C: hook + environment + stdin symbols.
        s.add_hook(0x401200);
        s.setenv(b"PATH".to_vec(), b"/usr/bin".to_vec());
        s.setenv(b"HOME".to_vec(), b"/root".to_vec());
        s.record_stdin_symbol("stdin_chunk_0".to_string(), 16);

        // Flags.
        s.set_no_ip_concretization(true);
        s.set_keep_ip_symbolic(false);
        s.set_no_symbolic_jump_resolution(true);
        s.set_posix_brk(0x1B0_4000);
        s.set_mmap_base(0xC100_8000);

        // Solver constraints — `rbx > 10` must hold after restore.
        let cmp = {
            let ctx = s.solver().borrow();
            let ten = RustBV::concrete(10, 64);
            rbx_sym.ugt(&ten, &ctx)
        };
        s.solver().borrow().assume_true(&cmp);

        s
    }

    /// Round-trip assertions: bucket A scalars + bucket B field counts +
    /// bucket C collection contents must match. Solver SAT + concretize
    /// proves the assumed_constraints were faithfully replayed (the
    /// SymContextSnapshot path is already covered separately, this just
    /// confirms the wiring through RustSimStateSnapshot).
    #[cfg(feature = "vex-engine-z3")]
    fn assert_state_round_trip(orig: &RustSimState, restored: &RustSimState) {
        // Bucket A scalars.
        assert_eq!(restored.pc(), orig.pc());
        assert_eq!(restored.state_id(), orig.state_id());
        assert_eq!(restored.parent_id(), orig.parent_id());
        assert_eq!(restored.history().to_vec(), orig.history().to_vec());
        assert_eq!(restored.detailed_history().len(), orig.detailed_history().len());
        assert_eq!(restored.heap_brk(), orig.heap_brk());
        assert_eq!(restored.posix_brk(), orig.posix_brk());
        assert_eq!(restored.mmap_base(), orig.mmap_base());
        assert_eq!(restored.no_ip_concretization(), orig.no_ip_concretization());
        assert_eq!(restored.keep_ip_symbolic(), orig.keep_ip_symbolic());
        assert_eq!(
            restored.no_symbolic_jump_resolution(),
            orig.no_symbolic_jump_resolution()
        );
        assert_eq!(restored.call_stack().len(), orig.call_stack().len());
        assert_eq!(restored.vex_arch(), orig.vex_arch());
        assert_eq!(restored.arch().name(), orig.arch().name());

        // Bucket A subset: heap metadata.
        assert_eq!(
            restored.heap_metadata().alloc_count(),
            orig.heap_metadata().alloc_count()
        );
        assert_eq!(
            restored.heap_metadata().free_count(),
            orig.heap_metadata().free_count()
        );

        // Bucket A: stdin symbols.
        assert_eq!(restored.stdin_symbols(), orig.stdin_symbols());

        // Bucket B: registers — concrete rax and symbolic rbx width.
        assert_eq!(
            restored.get_register("rax").and_then(|bv| bv.as_u64()),
            Some(0xdead_beef)
        );
        let restored_rbx = restored.get_register("rbx").expect("rbx present");
        assert_eq!(restored_rbx.width(), 64);

        // Bucket B: memory — first concrete bytes survived.
        let restored_bytes = restored
            .memory()
            .read_concrete_bytes_for_lift(crate::memory::Address::new(0x10_0000), 5)
            .expect("memory readable");
        assert_eq!(restored_bytes, vec![1, 2, 3, 4, 5]);

        // Bucket C: hooks + env.
        assert!(restored.is_hooked(0x401200));
        assert_eq!(
            restored.getenv(b"PATH").map(|v| v.to_vec()),
            Some(b"/usr/bin".to_vec())
        );
        assert_eq!(
            restored.getenv(b"HOME").map(|v| v.to_vec()),
            Some(b"/root".to_vec())
        );

        // Solver replayed — restored context must be SAT with the rbx > 10
        // constraint honored.
        assert!(restored.solver().borrow().is_sat());
        let restored_rbx_val = restored
            .solver()
            .borrow()
            .eval(&restored_rbx)
            .expect("rbx evaluable");
        assert!(
            restored_rbx_val > 10,
            "constraint rbx > 10 not honored after restore (got {restored_rbx_val})"
        );
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_state_snapshot_round_trip_buckets_a_b_c() {
        let orig = build_populated_state();
        let snap = orig.to_snapshot();
        let restored = RustSimState::from_snapshot(snap).expect("from_snapshot");
        assert_state_round_trip(&orig, &restored);
    }

    #[cfg(feature = "vex-engine-z3")]
    #[test]
    fn test_state_to_from_serialized_round_trip() {
        let orig = build_populated_state();
        let bytes = orig.to_serialized();
        assert_eq!(bytes[0], SNAPSHOT_VERSION, "envelope must carry version byte");
        let restored = RustSimState::from_serialized(&bytes).expect("from_serialized");
        assert_state_round_trip(&orig, &restored);
    }

    #[test]
    fn test_state_from_serialized_empty_envelope() {
        match RustSimState::from_serialized(&[]) {
            Err(SnapshotError::EmptyEnvelope) => {}
            Err(other) => panic!("expected EmptyEnvelope, got {other:?}"),
            Ok(_) => panic!("empty envelope must fail"),
        }
    }

    #[test]
    fn test_state_from_serialized_version_mismatch() {
        // Bump-byte trick: build a real envelope, replace version byte, expect
        // a fast VersionMismatch.
        let orig = RustSimState::new("amd64").unwrap();
        let mut bytes = orig.to_serialized();
        bytes[0] = SNAPSHOT_VERSION.wrapping_add(1);
        match RustSimState::from_serialized(&bytes) {
            Err(SnapshotError::VersionMismatch { found, expected }) => {
                assert_eq!(expected, SNAPSHOT_VERSION);
                assert_eq!(found, SNAPSHOT_VERSION.wrapping_add(1));
            }
            Err(other) => panic!("expected VersionMismatch, got {other:?}"),
            Ok(_) => panic!("bad version must fail"),
        }
    }
}
