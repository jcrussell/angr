//! File descriptors: [`FdFlags`], [`FileDescriptor`], and the symbolic-serve cap.

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
    /// The cwd-normalized absolute path this fd was opened as, frozen at
    /// open time — the same "freeze cwd-at-open" rule `known_paths` and
    /// `registry_key` already implement. Path-keyed queries
    /// (`FileSystem::content_size_for_path`, `register_file_content`'s
    /// stamp scan, `is_demoted_fd`) compare against this instead of
    /// re-normalizing `name` against the *current* cwd, so a guest
    /// `chdir` between open and query cannot decouple an fd from its
    /// path (angr-9ke6b.120). `name` itself stays raw because `fd_info`
    /// exposes it to Python verbatim.
    ///
    /// `None` for descriptors minted outside the `open` family (the three
    /// std fds, pipe ends, the write-side auto-vivified fds) and for
    /// pre-angr-9ke6b.120 snapshots; [`FileSystem::fd_norm_name`] falls
    /// back to normalizing `name` in that case, which is exact for the
    /// absolute pseudo-paths those descriptors carry.
    #[serde(default)]
    pub norm_name: Option<String>,
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
            norm_name: None,
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
            norm_name: None,
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
            norm_name: None,
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
