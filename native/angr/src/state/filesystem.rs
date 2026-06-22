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
    /// finite symbolic size (that EOF-aware case is still deferred; reads on
    /// a symbolic-stream fd never hit EOF). Defaults false; `#[serde(default)]`
    /// keeps pre-angr-11djq.6.1 snapshots loadable (reconstitutes to a
    /// concrete-only fd, the prior behavior). Set only via `open_symbolic`.
    #[serde(default)]
    pub symbolic: bool,
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
        FileSystemData {
            fds,
            next_fd: fs.next_fd,
            cwd: fs.cwd,
            known_paths,
            symlinks,
        }
    }
}

impl From<FileSystemData> for FileSystem {
    fn from(d: FileSystemData) -> Self {
        let fds: HashMap<u32, FileDescriptor> = d.fds.into_iter().collect();
        let known_paths: HashSet<String> = d.known_paths.into_iter().collect();
        let symlinks: HashMap<String, Vec<u8>> = d.symlinks.into_iter().collect();
        FileSystem {
            fds: Arc::new(fds),
            next_fd: d.next_fd,
            cwd: d.cwd,
            known_paths: Arc::new(known_paths),
            symlinks: Arc::new(symlinks),
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
            known_paths: Arc::new(HashSet::new()),
            symlinks: Arc::new(HashMap::new()),
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
    pub fn open(&mut self, name: String, flags: FdFlags) -> u32 {
        let fd = self.next_fd;
        self.next_fd += 1;
        Arc::make_mut(&mut self.known_paths).insert(name.clone());
        Arc::make_mut(&mut self.fds).insert(fd, FileDescriptor::new(name, flags));
        fd
    }

    /// Open a file descriptor with pre-loaded content (for file-backed SimFiles).
    pub fn open_with_content(&mut self, name: String, flags: FdFlags, content: Vec<u8>) -> u32 {
        let fd = self.next_fd;
        self.next_fd += 1;
        Arc::make_mut(&mut self.known_paths).insert(name.clone());
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
    pub fn open_symbolic(&mut self, name: String, flags: FdFlags) -> u32 {
        let fd = self.next_fd;
        self.next_fd += 1;
        Arc::make_mut(&mut self.known_paths).insert(name.clone());
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
        Arc::make_mut(&mut self.known_paths).insert(name);
    }

    /// True if `name` was previously registered via `open` /
    /// `open_with_content` / `register_known_path`. Drives
    /// `NativeAccessSyscall`.
    pub fn is_path_known(&self, name: &str) -> bool {
        self.known_paths.contains(name)
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
        self.symlinks.get(path).map(|v| v.as_slice())
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
    pub fn write(&mut self, fd: u32, data: &[u8]) {
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
    pub fn write_at(&mut self, fd: u32, offset: u64, data: &[u8]) {
        let desc = Arc::make_mut(&mut self.fds)
            .entry(fd)
            .or_insert_with(|| FileDescriptor::new(String::new(), FdFlags::WriteOnly));
        let start = offset as usize;
        let end = start + data.len();
        if end > desc.content.len() {
            desc.content.resize(end, 0);
        }
        desc.content[start..end].copy_from_slice(data);
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

    /// Largest `content_len` across all fds (open or closed) that share
    /// the given name. Returns `None` when no fd has been opened with
    /// that name. Drives `NativeStatSyscall`, which needs a content
    /// length for the path without minting a fresh fd.
    pub fn content_size_for_path(&self, name: &str) -> Option<usize> {
        self.fds
            .values()
            .filter(|d| d.name == name)
            .map(|d| d.content.len())
            .max()
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
