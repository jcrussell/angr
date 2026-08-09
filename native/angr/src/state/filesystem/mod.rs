//! FileSystem subsystem (POSIX fd model) for `RustSimState`.
//!
//! Split across five submodules (angr-nbim4.4), all operating on the one
//! [`FileSystem`] struct defined here:
//!
//! - [`fd`] — [`FdFlags`] / [`FileDescriptor`] and the symbolic-serve cap.
//! - [`ops`] — the mutating POSIX surface (open family, close, read/write,
//!   seek, dup/pipe, path & cwd setters).
//! - [`query`] — read-only accessors (path normalization, fd metadata, fd
//!   listings, symlink lookup).
//! - [`symbolic`] — bounded symbolic file content: the `file_contents`
//!   registry, the `read_sym` serve paths, and write-demotion.
//! - [`persist`] — the serde shadow form plus the cross-Z3-context
//!   `translate_into`.

use super::*;

mod fd;
mod ops;
mod persist;
mod query;
mod symbolic;

pub use fd::{FdFlags, FileDescriptor, MAX_FS_FILE_SIZE, MAX_SYMFILE_SERVE_SIZE};
pub use persist::FileSystemData;

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
    /// Paths known to exist (Rust-side mirror of Python `state.fs._files`
    /// keys). Populated by `open` / `open_with_content` so that a
    /// subsequent `access(path)` syscall sees the file. Pre-populated
    /// Python entries (via `state.fs.insert` before the Rust state is
    /// built) are NOT mirrored — same trade-off as `open` / `openat`
    /// (angr-k3ol.1). Queried by `NativeAccessSyscall` (angr-k3ol.2).
    known_paths: Arc<HashSet<String>>,
    /// Minimal symlink table: link path → raw target bytes (as
    /// `readlink(2)` would return — NOT NUL-terminated). Empty by
    /// default, so `readlink` / `readlinkat` keep returning `-1` for
    /// every path (the prior behavior). Populated only via the
    /// `add_symlink` API (tests / state export, mirroring the
    /// `register_known_path` precedent — Python `state.fs` symlinks are
    /// NOT auto-mirrored). Queried by `NativeReadlinkSyscall` /
    /// `NativeReadlinkatSyscall` (angr-11djq.6.2).
    symlinks: Arc<HashMap<String, Vec<u8>>>,
    /// Path-keyed symbolic-content registry: cwd-normalized absolute path
    /// → per-byte symbolic content (see [`FileDescriptor::content_sym`]).
    /// Seeded via [`register_file_content`](FileSystem::register_file_content)
    /// (`RustExplorationManager._export_fs_files_to_rust` pushes Python
    /// `state.fs._files` symbolic SimFile content here at state-add time);
    /// consulted by `open` so a native open attaches
    /// `content_sym` without a Python bounce (`open_symbolic` does NOT
    /// attach — the stream model owns those fds). Arc'd for O(1) fork,
    /// mirroring `known_paths`.
    file_contents: Arc<HashMap<String, Arc<Vec<RustBV>>>>,
    /// Cwd-normalized paths whose registered symbolic content was demoted
    /// to Python ownership by a native write ([`demote_symbolic_content`]
    /// / [`demote_all_symbolic_content`]). Unlike `file_contents`, entries
    /// here are *never* removed — the set accumulates over the lineage so a
    /// mid-run Python re-add (merge / legacy-fork push / cross-manager
    /// transfer) can query it and avoid re-registering a path an ancestor
    /// already demoted, keeping the write-demotion in effect (angr-qluof).
    /// Arc'd for O(1) fork, mirroring `known_paths`.
    ///
    /// [`demote_symbolic_content`]: FileSystem::demote_symbolic_content
    /// [`demote_all_symbolic_content`]: FileSystem::demote_all_symbolic_content
    demoted_paths: Arc<HashSet<String>>,
}

impl Default for FileSystem {
    // !Send Arcs go through `crate::arc_shared` (see its doc comment).
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
            fds: crate::arc_shared(fds),
            next_fd: 3,
            cwd: b"/".to_vec(),
            known_paths: Arc::new(HashSet::new()),
            symlinks: Arc::new(HashMap::new()),
            file_contents: crate::arc_shared(HashMap::new()),
            demoted_paths: Arc::new(HashSet::new()),
        }
    }
}

#[cfg(test)]
#[path = "../filesystem_tests.rs"]
mod filesystem_tests;
