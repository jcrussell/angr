//! FileSystem subsystem (POSIX fd model) for `RustSimState`.

use super::*;

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
    /// When true and the concrete `content` buffer is empty/consumed, reads
    /// mint fresh symbolic bytes natively (the stdin model) instead of
    /// falling back to Python. Models an fd backed by a symbolic stream
    /// (`SimPackets`), like stdin — NOT a bounded symbolic *file* with a
    /// finite size (that EOF-aware case is `content_sym` below, angr-0xyq2;
    /// reads on a symbolic-stream fd never hit EOF). Defaults false; `#[serde(default)]`
    /// keeps pre-angr-11djq.6.1 snapshots loadable (reconstitutes to a
    /// concrete-only fd, the prior behavior). Set only via `open_symbolic`.
    #[serde(default)]
    pub symbolic: bool,
    /// Bounded symbolic file content: one 8-bit `RustBV` per byte (concrete
    /// bytes in mixed files are concrete `RustBV` entries — uniform
    /// representation). Models a cle `SimFile` with symbolic content and a
    /// *finite* size, unlike the unbounded-stream `symbolic` flag above.
    /// `Arc` keeps the per-read CoW position advance (`Arc::make_mut` on
    /// the fd map) a refcount bump rather than a deep BV-vector clone.
    /// Attached by [`FileSystem::open`] from the path-keyed
    /// `file_contents` registry (NOT by `open_symbolic` — the
    /// symbolic-stream and bounded-file models have conflicting EOF
    /// semantics, see [`FileSystem::open_symbolic`]). Served natively by
    /// [`FileSystem::read_sym`] / [`FileSystem::read_sym_at`]
    /// (angr-0xyq2 Phase 2). `#[serde(default)]` keeps pre-angr-0xyq2
    /// snapshots loadable (reconstitutes to `None`).
    /// Serialized via serde's `rc` feature: the `Arc<T>` goes on the wire
    /// as `T` and each load rebuilds a fresh `Arc`, so Arc *sharing*
    /// across fds / the registry is not preserved through serde —
    /// acceptable, sharing is only a fork-time perf optimization.
    #[serde(default)]
    pub content_sym: Option<Arc<Vec<RustBV>>>,
    /// The `file_contents` registry key this fd is linked to, recorded at
    /// attach/registration time (cwd-normalized-at-that-moment absolute
    /// path). [`FileSystem::demote_symbolic_content`] keys off this rather
    /// than re-normalizing `name` against the *current* cwd, so a guest
    /// `chdir` between open and write cannot decouple the fd from its
    /// registry entry (and demotion needs no per-write path allocation).
    /// Set by [`FileSystem::open`] when it attaches `content_sym`, and by
    /// [`FileSystem::register_file_content`] on already-open fds of the
    /// registered path. Cleared on demotion. `#[serde(default)]` keeps
    /// earlier snapshots loadable (reconstitutes to `None`).
    #[serde(default)]
    pub registry_key: Option<String>,
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
            symbolic: false,
            content_sym: None,
            registry_key: None,
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
            symbolic: false,
            content_sym: None,
            registry_key: None,
        }
    }

    /// Create a new symbolic-stream file descriptor (empty content, reads
    /// mint fresh symbolic bytes). See [`FileDescriptor::symbolic`].
    pub fn new_symbolic(name: String, flags: FdFlags) -> Self {
        FileDescriptor {
            name,
            position: 0,
            flags,
            content: Vec::new(),
            is_open: true,
            symbolic: true,
            content_sym: None,
            registry_key: None,
        }
    }

    /// Length in bytes of the fd's backing content: the max of the
    /// concrete buffer length and the symbolic byte count (when
    /// `content_sym` is attached), so a concrete write past the symbolic
    /// end is not masked — mirrors Python `SimFile.write` size semantics
    /// (size = max(old size, write end)). The Phase 2 write choke point
    /// ([`FileSystem::write`]) refuses concrete writes on attached fds, so
    /// the mixed state can only arise from hand-built/legacy snapshots —
    /// this max is defense-in-depth. Length consumers (SEEK_END,
    /// stat/fstat st_size, feof) go through this.
    pub fn effective_len(&self) -> usize {
        self.content
            .len()
            .max(self.content_sym.as_ref().map_or(0, |v| v.len()))
    }
}

/// Serve cap (bytes per call) for reads on bounded-symbolic-content fds
/// (`content_sym`), shared by the read/fread/readv/pread64 serve paths.
/// Matches the Python export gate (`_FS_EXPORT_MAX_FILE_SIZE` in
/// rust_manager.py), so any registered file can be consumed in ONE guest
/// read — Python's `SimFile.read` serves the full request in one call, and
/// a smaller cap would silently diverge (short read Python never produces;
/// for fread, `items = served / size` could even round to 0 forever).
/// `read_sym`/`read_sym_at` clamp to the remaining content anyway; this cap
/// only bounds the per-call `Vec` allocation for absurd guest counts.
pub const MAX_SYMFILE_SERVE_SIZE: u64 = 65536;

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
    /// (Phase 3 of angr-0xyq2 will push Python `state.fs._files` symbolic
    /// SimFile content here); consulted by `open` so a native open attaches
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
    /// `#[serde(default)]` keeps pre-angr-k3ol.2 snapshots loadable —
    /// the field reconstitutes to an empty set, matching the previous
    /// behavior (no native access lookups would succeed).
    #[serde(default)]
    pub known_paths: std::collections::BTreeSet<String>,
    /// `#[serde(default)]` keeps pre-angr-11djq.6.2 snapshots loadable —
    /// reconstitutes to an empty map (no symlinks, `readlink` → `-1`).
    #[serde(default)]
    pub symlinks: std::collections::BTreeMap<String, Vec<u8>>,
    /// Symbolic-content registry. The `Arc`s go on the wire as their
    /// inner `Vec<RustBV>` via serde's `rc` feature (symbolic leaves
    /// rebuild by (name, width) via the `RustBVData` shadow); each load
    /// rebuilds fresh `Arc`s, so fd/registry sharing is not preserved
    /// through serde. `#[serde(default)]` keeps pre-angr-0xyq2 snapshots
    /// loadable — reconstitutes to an empty registry. BTreeMap keeps the
    /// wire ordering deterministic.
    #[serde(default)]
    pub file_contents: std::collections::BTreeMap<String, Arc<Vec<RustBV>>>,
    /// Accumulated demoted-path set (angr-qluof). `#[serde(default)]` keeps
    /// pre-angr-qluof snapshots loadable — reconstitutes to an empty set
    /// (no lineage-demotion memory, matching the previous behavior).
    /// BTreeSet keeps the wire ordering deterministic.
    #[serde(default)]
    pub demoted_paths: std::collections::BTreeSet<String>,
}

impl From<FileSystem> for FileSystemData {
    fn from(fs: FileSystem) -> Self {
        let fds: std::collections::BTreeMap<u32, FileDescriptor> =
            fs.fds.iter().map(|(k, v)| (*k, v.clone())).collect();
        let known_paths: std::collections::BTreeSet<String> =
            fs.known_paths.iter().cloned().collect();
        let symlinks: std::collections::BTreeMap<String, Vec<u8>> = fs
            .symlinks
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let file_contents: std::collections::BTreeMap<String, Arc<Vec<RustBV>>> = fs
            .file_contents
            .iter()
            .map(|(k, v)| (k.clone(), Arc::clone(v)))
            .collect();
        let demoted_paths: std::collections::BTreeSet<String> =
            fs.demoted_paths.iter().cloned().collect();
        FileSystemData {
            fds,
            next_fd: fs.next_fd,
            cwd: fs.cwd,
            known_paths,
            symlinks,
            file_contents,
            demoted_paths,
        }
    }
}

impl From<FileSystemData> for FileSystem {
    // RustBV's Z3 AST makes FileDescriptor !Send; the Arcs here are
    // per-state CoW handles that never cross threads (states migrate
    // between workers via the serde snapshot, not by moving Arcs) — hence
    // the !Send Arcs go through `crate::arc_shared` (see its doc comment).
    fn from(d: FileSystemData) -> Self {
        let fds: HashMap<u32, FileDescriptor> = d.fds.into_iter().collect();
        let symlinks: HashMap<String, Vec<u8>> = d.symlinks.into_iter().collect();
        let file_contents: HashMap<String, Arc<Vec<RustBV>>> =
            d.file_contents.into_iter().collect();
        // demoted_paths keys are always normalized at insertion, so no
        // re-normalization pass is needed (unlike known_paths below).
        let demoted_paths: HashSet<String> = d.demoted_paths.into_iter().collect();
        let mut fs = FileSystem {
            fds: crate::arc_shared(fds),
            next_fd: d.next_fd,
            cwd: d.cwd,
            known_paths: Arc::new(HashSet::new()),
            symlinks: Arc::new(symlinks),
            file_contents: crate::arc_shared(file_contents),
            demoted_paths: Arc::new(demoted_paths),
        };
        // Re-normalize known_paths on load: the key space is normalized at
        // insertion (see `open`), but pre-normalization snapshots may carry
        // raw relative entries. Absolute canonical entries are fixed points,
        // so this is a no-op for current-format snapshots.
        let known_paths: HashSet<String> =
            d.known_paths.iter().map(|p| fs.normalize_path(p)).collect();
        fs.known_paths = Arc::new(known_paths);
        fs
    }
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

impl FileSystem {
    /// Open a new file descriptor. Returns the allocated fd number.
    ///
    /// Also registers `name` in `known_paths` so that a subsequent
    /// `NativeAccessSyscall` against the same path returns 0
    /// (file-exists). Mirrors the way Python `procedures/posix/open.py`
    /// drops a fresh `SimFile` into `state.fs` on creation — our model
    /// treats any successfully-opened path as "existing" from that
    /// point forward.
    ///
    /// When the (cwd-normalized) path has registered symbolic content
    /// (see [`register_file_content`](Self::register_file_content)), the
    /// new fd shares that content via `content_sym` (refcount bump, no
    /// deep clone). Paths without a registry entry behave exactly as
    /// before (`content_sym: None`).
    pub fn open(&mut self, name: String, flags: FdFlags) -> u32 {
        let fd = self.next_fd;
        self.next_fd += 1;
        // known_paths keys on normalized paths; normalizing at insertion
        // freezes cwd-at-open, which is POSIX-correct for relative paths.
        let norm = self.normalize_path(&name);
        let content_sym = self.file_contents.get(&norm).cloned();
        let mut desc = FileDescriptor::new(name, flags);
        if content_sym.is_some() {
            // Freeze the registry key on the descriptor so demotion can
            // find the entry (and every sibling fd) without re-normalizing
            // against a possibly-changed cwd. See `registry_key`.
            desc.registry_key = Some(norm.clone());
        }
        desc.content_sym = content_sym;
        Arc::make_mut(&mut self.known_paths).insert(norm);
        Arc::make_mut(&mut self.fds).insert(fd, desc);
        fd
    }

    /// Open a file descriptor with pre-loaded content (for file-backed
    /// SimFiles). Intentionally bypasses the `file_contents` registry — the
    /// caller supplies explicit concrete content (test-seeding API).
    pub fn open_with_content(&mut self, name: String, flags: FdFlags, content: Vec<u8>) -> u32 {
        let fd = self.next_fd;
        self.next_fd += 1;
        // Normalized at insertion (freezes cwd-at-open) — see `open`.
        let norm = self.normalize_path(&name);
        Arc::make_mut(&mut self.known_paths).insert(norm);
        Arc::make_mut(&mut self.fds).insert(fd, FileDescriptor::with_content(name, flags, content));
        fd
    }

    /// Open a symbolic-stream file descriptor: empty content, flagged so
    /// that reads mint fresh symbolic bytes natively (the stdin model)
    /// rather than falling back to Python. Returns the allocated fd number.
    ///
    /// Like `open` / `open_with_content`, this is a seeding API with no
    /// production Python caller yet (tests + future state-export wiring) —
    /// the open/openat syscall path still uses the content-less `open`, so
    /// a real binary's `open()` is unaffected. See
    /// `FileDescriptor::symbolic` for the stream-vs-bounded-file distinction.
    ///
    /// Unlike `open`, registered bounded content (`file_contents`) is NOT
    /// attached: the two models conflict rather than compose — a bounded
    /// file returns 0 at EOF forever, while the stream model mints fresh
    /// bytes forever. The stream model wins for `open_symbolic`; bounded
    /// symbolic files come via `open()` on a registered path.
    pub fn open_symbolic(&mut self, name: String, flags: FdFlags) -> u32 {
        let fd = self.next_fd;
        self.next_fd += 1;
        // Normalized at insertion (freezes cwd-at-open) — see `open`.
        let norm = self.normalize_path(&name);
        Arc::make_mut(&mut self.known_paths).insert(norm);
        Arc::make_mut(&mut self.fds).insert(fd, FileDescriptor::new_symbolic(name, flags));
        fd
    }

    /// True if `fd` is open and flagged as a symbolic stream (reads mint
    /// fresh symbolic bytes once the concrete buffer is consumed). Drives
    /// the symbolic-read branch of `NativeReadSyscall`.
    pub fn is_symbolic(&self, fd: u32) -> bool {
        self.fds.get(&fd).is_some_and(|d| d.is_open && d.symbolic)
    }

    /// Register an existing-file path without allocating an fd. Used by
    /// the Python state-export path to seed `state.fs._files` entries
    /// (`register_known_path` PyO3 setter) and by tests.
    pub fn register_known_path(&mut self, name: String) {
        // Normalized at insertion (freezes cwd-at-registration) — see `open`.
        let norm = self.normalize_path(&name);
        Arc::make_mut(&mut self.known_paths).insert(norm);
    }

    /// Normalize `path` against the current working directory, mirroring
    /// Python `SimFilesystem._normalize_path` + `_join_chunks`
    /// (`angr/state_plugins/filesystem.py`): truncate at the first NUL,
    /// prefix `cwd` when relative, drop empty / `.` components, resolve
    /// `..` against its parent (saturating at root), and re-join from
    /// root (`/a/b`; bare root is `/`). Python `_files` keys are
    /// cwd-normalized absolute paths, so the `file_contents` registry
    /// keys and lookups both go through this.
    ///
    /// The native path model is UTF-8-lossy (paths enter via
    /// `from_utf8_lossy` repo-wide), so Phase 3's export must skip
    /// non-UTF-8 `_files` keys (Python fallback handles those).
    pub fn normalize_path(&self, path: &str) -> String {
        let path = path.split('\0').next().unwrap_or("");
        let cwd = String::from_utf8_lossy(&self.cwd);
        let full = if path.starts_with('/') {
            path.to_string()
        } else {
            format!("{cwd}/{path}")
        };
        let mut keys: Vec<&str> = Vec::new();
        for k in full.split('/') {
            match k {
                "" | "." => {}
                ".." => {
                    keys.pop();
                }
                _ => keys.push(k),
            }
        }
        format!("/{}", keys.join("/"))
    }

    /// Register bounded symbolic content for `path`: one 8-bit `RustBV`
    /// per byte (concrete bytes as concrete entries — see
    /// [`FileDescriptor::content_sym`]). The path is cwd-normalized so a
    /// later relative `open` of the same file still matches. Also
    /// registers the normalized path in `known_paths` so `access(2)` /
    /// `stat(2)` see the file before any fd exists. Overwrites any
    /// previous registration for the same normalized path. Paths are
    /// UTF-8-lossy (see [`normalize_path`](Self::normalize_path)) —
    /// Phase 3 export must skip non-UTF-8 `_files` keys.
    ///
    /// Fds already open on the registered path get their `registry_key`
    /// stamped (their `content_sym` stays `None` — mid-stream reads keep
    /// their pre-registration semantics), so a later native write through
    /// such an fd still demotes the registry entry rather than leaving a
    /// fresh `open` to serve stale content.
    // !Send Arcs go through `crate::arc_shared` (see its doc comment).
    pub fn register_file_content(&mut self, path: &str, bytes: Vec<RustBV>) {
        let norm = self.normalize_path(path);
        Arc::make_mut(&mut self.known_paths).insert(norm.clone());
        let stamp: Vec<u32> = self
            .fds
            .iter()
            .filter(|(_, d)| d.registry_key.is_none() && self.normalize_path(&d.name) == norm)
            .map(|(k, _)| *k)
            .collect();
        if !stamp.is_empty() {
            let fds = Arc::make_mut(&mut self.fds);
            for k in stamp {
                fds.get_mut(&k).expect("fd existed above").registry_key = Some(norm.clone());
            }
        }
        Arc::make_mut(&mut self.file_contents).insert(norm, crate::arc_shared(bytes));
    }

    /// Look up registered symbolic content for a (possibly relative)
    /// path. Returns a shared handle (refcount bump) when the
    /// cwd-normalized path has a registry entry.
    pub fn file_content_for_path(&self, path: &str) -> Option<Arc<Vec<RustBV>>> {
        // Empty-registry fast path: keeps opens zero-alloc (no normalized
        // String) when no symbolic content is registered — all production
        // opens today.
        if self.file_contents.is_empty() {
            return None;
        }
        self.file_contents.get(&self.normalize_path(path)).cloned()
    }

    /// True if `name` was previously registered via `open` /
    /// `open_with_content` / `register_known_path` /
    /// `register_file_content`. Both the stored key space and this query
    /// are cwd-normalized, so relative and absolute spellings of the same
    /// file agree. Drives `NativeAccessSyscall`.
    pub fn is_path_known(&self, name: &str) -> bool {
        self.known_paths.contains(&self.normalize_path(name))
    }

    /// Register a symlink: `link` resolves to `target` (raw bytes, as
    /// `readlink(2)` returns — NOT NUL-terminated). Overwrites any
    /// existing entry. Used by tests and the state-export path,
    /// mirroring the `register_known_path` precedent (Python `state.fs`
    /// symlinks are NOT auto-mirrored). Drives
    /// `NativeReadlinkSyscall` / `NativeReadlinkatSyscall`.
    pub fn add_symlink(&mut self, link: String, target: Vec<u8>) {
        Arc::make_mut(&mut self.symlinks).insert(link, target);
    }

    /// Look up a symlink target by path. Returns the raw target bytes if
    /// `path` is a registered symlink, else `None` (the path is not a
    /// symlink, so `readlink` returns `-1`).
    pub fn readlink_target(&self, path: &str) -> Option<&[u8]> {
        self.symlinks.get(path).map(std::vec::Vec::as_slice)
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

    /// True when `fd` has bounded symbolic content or a live registry link
    /// — the write choke point must refuse to mutate it. O(1), no
    /// allocation: one hash lookup and two `Option` flag checks, so plain
    /// concrete fds (all production fds today) pay nothing.
    #[inline]
    fn write_refused(&self, fd: u32) -> bool {
        self.fds
            .get(&fd)
            .is_some_and(|d| d.content_sym.is_some() || d.registry_key.is_some())
    }

    /// Write data to a file descriptor at its current position, advancing the
    /// position by `data.len()` (POSIX `write(2)` semantics).
    ///
    /// For the common sequential case (a write-only fd that is only ever
    /// written, so `position` starts at 0 and tracks the content length) this
    /// is byte-identical to a plain append. The position-aware path matters
    /// only after a `seek` or an interleaved `read` moved the offset away from
    /// EOF: there, append-only would corrupt the buffer relative to Python's
    /// position-aware `simfd.write`. Zero-fills any gap when `position` is at or
    /// past EOF (a sparse seek-then-write), mirroring [`write_at`].
    ///
    /// **Choke point (angr-0xyq2 Phase 2):** an fd carrying bounded
    /// symbolic content (`content_sym` / a live `registry_key`) is never
    /// mutated here. Instead the content is demoted
    /// ([`demote_symbolic_content`](Self::demote_symbolic_content)) and
    /// `false` is returned — the caller must bounce the write to Python
    /// (a fallback, NOT a state-killing error), which owns the file from
    /// then on. Zero-length writes are a POSIX no-op: they return `true`
    /// without demoting (and without creating a missing fd entry).
    #[must_use = "false means the write was refused (symbolic content demoted); bounce to Python"]
    pub fn write(&mut self, fd: u32, data: &[u8]) -> bool {
        if data.is_empty() {
            return true;
        }
        if self.write_refused(fd) {
            self.demote_symbolic_content(fd);
            return false;
        }
        let desc = Arc::make_mut(&mut self.fds)
            .entry(fd)
            .or_insert_with(|| FileDescriptor::new(String::new(), FdFlags::WriteOnly));
        let start = desc.position as usize;
        let end = start + data.len();
        if end > desc.content.len() {
            desc.content.resize(end, 0);
        }
        desc.content[start..end].copy_from_slice(data);
        desc.position = end as u64;
        true
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

    /// Read up to `count` bytes of bounded symbolic content from `fd` at its
    /// current position, advancing the position (CoW via `Arc::make_mut`,
    /// like the concrete `read`). Returns `None` when the fd is absent, is
    /// not open (`close` only flips the flag; `content_sym` stays attached
    /// but must not serve), or has no `content_sym` attached — the caller
    /// keeps its concrete-serve / Python-fallback logic. Returns
    /// `Some(vec)` otherwise; an empty vec at EOF (caller returns 0),
    /// matching Python `SimFile.read`'s `max(0, min(count, size - pos))`.
    /// Each returned entry is a cheap `RustBV` clone of the shared
    /// per-byte content.
    ///
    /// The slice is clamped to `content_sym` bounds only: positions beyond
    /// the symbolic end (e.g. an unchecked `SEEK_SET`) serve 0 bytes
    /// rather than panicking.
    ///
    /// Counter note: `symfile_reads_native` is bumped by the *call sites*
    /// (once per guest call that served >0 bytes), not here — `readv`
    /// calls this once per iovec segment.
    pub fn read_sym(&mut self, fd: u32, count: usize) -> Option<Vec<RustBV>> {
        // Peek to compute the byte count without forcing CoW at EOF.
        let (start, n) = {
            let desc = self.fds.get(&fd)?;
            if !desc.is_open {
                return None;
            }
            let content = desc.content_sym.as_ref()?;
            let start = (desc.position as usize).min(content.len());
            (start, count.min(content.len() - start))
        };
        if n == 0 {
            return Some(Vec::new());
        }
        let desc = Arc::make_mut(&mut self.fds)
            .get_mut(&fd)
            .expect("fd existed above");
        let content = desc.content_sym.as_ref().expect("content_sym peeked above");
        let bytes = content[start..start + n].to_vec();
        desc.position += n as u64;
        Some(bytes)
    }

    /// Positioned twin of [`read_sym`](Self::read_sym): read up to `count`
    /// bytes of bounded symbolic content starting at absolute `offset`,
    /// WITHOUT touching the fd's position (POSIX `pread` semantics, like
    /// [`read_at`](Self::read_at)). Same `None` / `Some(empty)` contract
    /// (including the not-open guard and the call-site counter note).
    pub fn read_sym_at(&self, fd: u32, offset: u64, count: usize) -> Option<Vec<RustBV>> {
        let desc = self.fds.get(&fd)?;
        if !desc.is_open {
            return None;
        }
        let content = desc.content_sym.as_ref()?;
        let start = (offset as usize).min(content.len());
        let n = count.min(content.len() - start);
        Some(content[start..start + n].to_vec())
    }

    /// Demote a fd's bounded symbolic content ahead of a native write:
    /// drops the `file_contents` registry entry recorded on the fd
    /// (`registry_key`, frozen at attach/registration time — NOT
    /// re-normalized against the current cwd, so a guest `chdir` between
    /// open and write cannot decouple siblings) AND clears
    /// `content_sym`/`registry_key` on EVERY fd sharing that key (dup'd /
    /// re-opened siblings included), so no stale native serving survives
    /// anywhere. Returns `true` when anything was demoted — the caller
    /// must then return its Python-fallback error, leaving the write
    /// itself and ALL subsequent I/O on the file consistently Python-owned
    /// (the Python SimFile model has the authoritative content from
    /// Phase 3's export). No-op (`false`) for fds without symbolic content
    /// — the common case, O(1) with no allocation, so native writes stay
    /// cheap.
    ///
    /// Post-demotion trade-off (deliberate v1 scope): metadata ops on the
    /// file — `feof` / `fstat` / `SEEK_END` / stat-by-path — keep
    /// answering natively from the concrete buffer (size 0), identical to
    /// the pre-feature behavior for natively-opened fds whose writes
    /// bounce. Bouncing them to Python could not work either: natively
    /// minted fds are not mirrored into Python's fd table (the angr-8j16
    /// unsynced-fd trade-off). Target workloads (parsers *reading* a
    /// symbolic input file) don't write the input file, so this only
    /// bites guests that write their own symbolic input.
    pub fn demote_symbolic_content(&mut self, fd: u32) -> bool {
        let Some(desc) = self.fds.get(&fd) else {
            return false;
        };
        let Some(key) = desc.registry_key.clone() else {
            // Defensive: `content_sym` without a `registry_key` cannot be
            // minted by `open` (which sets both), but a hand-built or
            // legacy-snapshot descriptor could carry it — demote the
            // single fd so it never serves stale bytes.
            if desc.content_sym.is_none() {
                return false;
            }
            Arc::make_mut(&mut self.fds)
                .get_mut(&fd)
                .expect("fd existed above")
                .content_sym = None;
            crate::symbolic::record_symfile_write_demotion();
            return true;
        };
        if self.file_contents.contains_key(&key) {
            Arc::make_mut(&mut self.file_contents).remove(&key);
        }
        // `fd` itself carries the key, so `matching` is never empty.
        let matching: Vec<u32> = self
            .fds
            .iter()
            .filter(|(_, d)| d.registry_key.as_deref() == Some(key.as_str()))
            .map(|(k, _)| *k)
            .collect();
        let fds = Arc::make_mut(&mut self.fds);
        for k in matching {
            let d = fds.get_mut(&k).expect("matching fd existed above");
            d.content_sym = None;
            d.registry_key = None;
        }
        // Remember the demoted path so a later Python re-add (merge /
        // legacy-fork push) does not re-register it (angr-qluof).
        Arc::make_mut(&mut self.demoted_paths).insert(key);
        crate::symbolic::record_symfile_write_demotion();
        true
    }

    /// Nuke ALL bounded symbolic content: every `file_contents` registry
    /// entry plus every fd's `content_sym` / `registry_key`. Insurance for
    /// write paths whose fd is symbolic/unresolvable — any registered file
    /// could be the target, so Python must own them all from here on.
    /// Returns `true` when anything was dropped. O(1) empty-registry check
    /// plus a flag scan over the handful of open fds when nothing is
    /// attached (all production states today), no allocation.
    // !Send Arcs go through `crate::arc_shared` (see its doc comment).
    pub fn demote_all_symbolic_content(&mut self) -> bool {
        let any_fd = self
            .fds
            .values()
            .any(|d| d.content_sym.is_some() || d.registry_key.is_some());
        if !any_fd && self.file_contents.is_empty() {
            return false;
        }
        if !self.file_contents.is_empty() {
            // Record every registry path as demoted before nuking so a
            // later Python re-add skips re-registering them (angr-qluof).
            let keys: Vec<String> = self.file_contents.keys().cloned().collect();
            let demoted = Arc::make_mut(&mut self.demoted_paths);
            demoted.extend(keys);
            self.file_contents = crate::arc_shared(HashMap::new());
        }
        if any_fd {
            let fds = Arc::make_mut(&mut self.fds);
            for d in fds.values_mut() {
                d.content_sym = None;
                d.registry_key = None;
            }
        }
        crate::symbolic::record_symfile_write_demotion();
        true
    }

    /// The cwd-normalized paths this lineage has demoted (angr-qluof).
    /// Sorted for a deterministic FFI order. Consumed by the Python
    /// re-add path to skip re-registering an ancestor's demoted content.
    pub fn demoted_paths(&self) -> Vec<String> {
        let mut v: Vec<String> = self.demoted_paths.iter().cloned().collect();
        v.sort_unstable();
        v
    }

    /// Re-apply a demotion for `path` (angr-qluof): drop any registered
    /// content and clear matching fds, exactly as a native write would,
    /// WITHOUT bumping the write-demotion counter (this is a re-add
    /// correction, not a fresh guest write). Records the path in
    /// `demoted_paths` even when nothing was currently registered, so the
    /// demotion sticks across future re-adds. Returns `true` if any
    /// registered content or fd state was cleared.
    pub fn demote_path(&mut self, path: &str) -> bool {
        let norm = self.normalize_path(path);
        let mut changed = false;
        if self.file_contents.contains_key(&norm) {
            Arc::make_mut(&mut self.file_contents).remove(&norm);
            changed = true;
        }
        let matching: Vec<u32> = self
            .fds
            .iter()
            .filter(|(_, d)| d.registry_key.as_deref() == Some(norm.as_str()))
            .map(|(k, _)| *k)
            .collect();
        if !matching.is_empty() {
            let fds = Arc::make_mut(&mut self.fds);
            for k in matching {
                let d = fds.get_mut(&k).expect("matching fd existed above");
                d.content_sym = None;
                d.registry_key = None;
            }
            changed = true;
        }
        Arc::make_mut(&mut self.demoted_paths).insert(norm);
        changed
    }

    /// Seek a file descriptor. Returns the new position.
    ///
    /// whence: 0=SEEK_SET, 1=SEEK_CUR, 2=SEEK_END
    pub fn seek(&mut self, fd: u32, offset: i64, whence: u32) -> Option<u64> {
        // Compute new position without CoW first; only mutate if the fd exists
        // and the whence value is valid.
        let desc = self.fds.get(&fd)?;
        let new_pos = match whence {
            0 => offset.max(0) as u64,                                 // SEEK_SET
            1 => (desc.position as i64 + offset).max(0) as u64,        // SEEK_CUR
            2 => (desc.effective_len() as i64 + offset).max(0) as u64, // SEEK_END
            _ => return None,
        };
        Arc::make_mut(&mut self.fds).get_mut(&fd)?.position = new_pos;
        Some(new_pos)
    }

    /// Positioned read: read up to `count` bytes starting at absolute
    /// `offset`, WITHOUT touching the fd's current position. Mirrors POSIX
    /// `pread` (the file offset is unaffected). Returns the bytes read
    /// (empty if `offset` is past EOF or the fd is absent).
    pub fn read_at(&self, fd: u32, offset: u64, count: usize) -> Vec<u8> {
        match self.fds.get(&fd) {
            Some(desc) => {
                let pos = offset as usize;
                if pos >= desc.content.len() {
                    return Vec::new();
                }
                let n = count.min(desc.content.len() - pos);
                desc.content[pos..pos + n].to_vec()
            }
            None => Vec::new(),
        }
    }

    /// Positioned write: overwrite `data` at absolute `offset`, WITHOUT
    /// touching the fd's current position. Mirrors POSIX `pwrite` (the file
    /// offset is unaffected). Extends the content buffer (zero-filling any
    /// gap) when `offset` is at or past EOF, so it is not append-only like
    /// `write`. Creates the fd entry if missing, matching `write`.
    ///
    /// Same choke-point contract as [`write`](Self::write): symbolic-content
    /// fds are demoted and refused (`false`), zero-length writes are a
    /// no-demotion no-op (`true`).
    #[must_use = "false means the write was refused (symbolic content demoted); bounce to Python"]
    pub fn write_at(&mut self, fd: u32, offset: u64, data: &[u8]) -> bool {
        if data.is_empty() {
            return true;
        }
        if self.write_refused(fd) {
            self.demote_symbolic_content(fd);
            return false;
        }
        let desc = Arc::make_mut(&mut self.fds)
            .entry(fd)
            .or_insert_with(|| FileDescriptor::new(String::new(), FdFlags::WriteOnly));
        let start = offset as usize;
        let end = start + data.len();
        if end > desc.content.len() {
            desc.content.resize(end, 0);
        }
        desc.content[start..end].copy_from_slice(data);
        true
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

    /// Largest [`effective_len`](FileDescriptor::effective_len) across all
    /// fds (open or closed) that share the given (cwd-normalized) name,
    /// maxed with the `file_contents` registry entry for that path — so a
    /// registered-but-never-opened file still reports its content length.
    /// Returns `None` when no fd has been opened with that name and the
    /// registry has no entry (content-less `register_known_path` paths
    /// keep reporting size 0 via the caller's default). Drives
    /// `NativeStatSyscall`, which needs a content length for the path
    /// without minting a fresh fd.
    pub fn content_size_for_path(&self, name: &str) -> Option<usize> {
        let norm = self.normalize_path(name);
        // Stored `d.name` stays raw (fd_info exposes it to Python), so
        // normalize the fd names on the fly for comparison.
        let fd_max = self
            .fds
            .values()
            .filter(|d| self.normalize_path(&d.name) == norm)
            .map(FileDescriptor::effective_len)
            .max();
        let reg_len = self.file_contents.get(&norm).map(|v| v.len());
        fd_max.into_iter().chain(reg_len).max()
    }

    /// Shared handle to an fd's bounded symbolic content, if attached
    /// (refcount bump, no deep clone). Prefer
    /// [`has_content_sym`](Self::has_content_sym) when only a predicate is
    /// needed.
    pub fn fd_content_sym(&self, fd: u32) -> Option<Arc<Vec<RustBV>>> {
        self.fds.get(&fd).and_then(|d| d.content_sym.clone())
    }

    /// True when `fd` is open with bounded symbolic content attached — the
    /// serve-gate predicate for the read paths (no Arc clone, matching
    /// [`read_sym`](Self::read_sym)'s not-open guard).
    pub fn has_content_sym(&self, fd: u32) -> bool {
        self.fds
            .get(&fd)
            .is_some_and(|d| d.is_open && d.content_sym.is_some())
    }

    /// Position and [`effective_len`](FileDescriptor::effective_len) for an
    /// fd, in a single map lookup. Drives `NativeFeof`, which is hot in
    /// `while (!feof(f))` guest loops.
    pub fn fd_pos_and_size(&self, fd: u32) -> Option<(u64, usize)> {
        self.fds.get(&fd).map(|d| (d.position, d.effective_len()))
    }

    /// Effective content length for an fd — symbolic byte count when
    /// `content_sym` is attached, else the concrete buffer length. Drives
    /// `NativeFstatSyscall` st_size; `NativeFeof` uses the combined
    /// [`fd_pos_and_size`](Self::fd_pos_and_size) accessor instead.
    pub fn effective_size(&self, fd: u32) -> Option<usize> {
        self.fds.get(&fd).map(FileDescriptor::effective_len)
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

    /// Cross-context twin of `clone` for `RustSimState::translate_state`
    /// (angr-ahypj): every context-bound `RustBV` — each fd's
    /// `content_sym` vec and each `file_contents` registry value — is
    /// `Z3_translate`d into `target_ctx` via [`RustBV::translate_into`];
    /// every other field is context-independent and cloned. Like the
    /// serde path, Arc *sharing* between an fd and the registry is not
    /// preserved through translation (each gets a fresh translated Arc) —
    /// acceptable, sharing is only a fork-time perf optimization.
    // !Send Arcs go through `crate::arc_shared` (see its doc comment).
    #[cfg(feature = "vex-engine-z3")]
    pub fn translate_into(&self, target_ctx: &z3::Context) -> Self {
        // Fast path: no symbolic file content anywhere → nothing is
        // context-bound, so the O(1) Arc-sharing clone (the pre-angr-0xyq2
        // behavior) is correct. All production states today take this.
        if self.file_contents.is_empty() && self.fds.values().all(|d| d.content_sym.is_none()) {
            return self.clone();
        }
        let translate_vec = |v: &Arc<Vec<RustBV>>| -> Arc<Vec<RustBV>> {
            crate::arc_shared(v.iter().map(|bv| bv.translate_into(target_ctx)).collect())
        };
        let fds: HashMap<u32, FileDescriptor> = self
            .fds
            .iter()
            .map(|(k, d)| {
                let mut d = d.clone();
                d.content_sym = d.content_sym.as_ref().map(&translate_vec);
                (*k, d)
            })
            .collect();
        let file_contents: HashMap<String, Arc<Vec<RustBV>>> = self
            .file_contents
            .iter()
            .map(|(k, v)| (k.clone(), translate_vec(v)))
            .collect();
        FileSystem {
            fds: crate::arc_shared(fds),
            next_fd: self.next_fd,
            cwd: self.cwd.clone(),
            known_paths: Arc::clone(&self.known_paths),
            symlinks: Arc::clone(&self.symlinks),
            file_contents: crate::arc_shared(file_contents),
            demoted_paths: Arc::clone(&self.demoted_paths),
        }
    }
}

#[cfg(test)]
#[path = "filesystem_tests.rs"]
mod filesystem_tests;
