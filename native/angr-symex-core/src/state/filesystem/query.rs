//! Read-only accessors over the [`FileSystem`]: path normalization, fd
//! metadata (name/position/size/open-ness), fd listings, symlink lookup and
//! the cwd getter. Nothing here mutates state — the mutating POSIX ops live
//! in [`super::ops`].

use super::*;
use std::borrow::Cow;

impl FileSystem {
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
        String::from_utf8_lossy(&self.normalize_path_bytes(path.as_bytes())).into_owned()
    }

    /// Byte-level body of [`normalize_path`](Self::normalize_path): Python's
    /// `SimFilesystem._normalize_path` + `_join_chunks` (truncate at the
    /// first NUL, join a relative path onto the cwd, drop empty and `.`
    /// components, fold `..` lexically without climbing above `/`).
    pub fn normalize_path_bytes(&self, path: &[u8]) -> Vec<u8> {
        let path = path.split(|&b| b == 0).next().unwrap_or(&[]);
        let mut keys: Vec<&[u8]> = Vec::new();
        let relative = !path.starts_with(b"/");
        let cwd_keys = if relative { self.cwd.as_slice() } else { &[] };
        for k in cwd_keys
            .split(|&b| b == b'/')
            .chain(path.split(|&b| b == b'/'))
        {
            match k {
                b"" | b"." => {}
                b".." => {
                    keys.pop();
                }
                _ => keys.push(k),
            }
        }
        let mut out = b"/".to_vec();
        out.extend_from_slice(&keys.join(&b'/'));
        out
    }

    /// The cwd-normalized absolute path a descriptor was opened as.
    ///
    /// Prefers the [`norm_name`](FileDescriptor::norm_name) frozen at open
    /// time; only descriptors minted outside the `open` family (std fds,
    /// pipe ends, auto-vivified write fds) and older snapshots
    /// (angr-9ke6b.120) fall back to normalizing the raw `name` against the
    /// *current* cwd. Every path-keyed fd scan goes through this so a
    /// guest `chdir` between open and query cannot mis-key an fd.
    pub(super) fn fd_norm_name<'a>(&self, d: &'a FileDescriptor) -> Cow<'a, str> {
        match d.norm_name.as_deref() {
            Some(n) => Cow::Borrowed(n),
            None => Cow::Owned(self.normalize_path(&d.name)),
        }
    }

    /// True if `fd` is open and flagged as a symbolic stream (reads mint
    /// fresh symbolic bytes once the concrete buffer is consumed). Drives
    /// the symbolic-read branch of `NativeReadSyscall`.
    pub fn is_symbolic(&self, fd: u32) -> bool {
        let fd = self.desc_key(fd);
        self.fds.get(&fd).is_some_and(|d| d.is_open && d.symbolic)
    }

    /// True if `name` was previously registered via `open` /
    /// `open_with_content` / `register_known_path` /
    /// `register_file_content`. Both the stored key space and this query
    /// are cwd-normalized, so relative and absolute spellings of the same
    /// file agree. Drives `NativeAccessSyscall`.
    pub fn is_path_known(&self, name: &str) -> bool {
        self.known_paths.contains(&self.normalize_path(name))
    }

    /// True when a native `open` of `name` with access mode `flags` gives
    /// the answer Python's `SimSystemPosix.open` would: the path is known
    /// here, or the mode writes (Python then creates the file whatever the
    /// options). A read-only open of an unknown path is Python's call — it
    /// consults `state.fs` files this side never saw, and
    /// `ALL_FILES_EXIST` / `ANY_FILE_MIGHT_EXIST` (angr-mx00p.132.15).
    pub fn native_open_decides(&self, name: &str, flags: FdFlags) -> bool {
        flags != FdFlags::ReadOnly || self.is_path_known(name)
    }

    /// Look up a symlink target by path. Returns the raw target bytes if
    /// `path` is a registered symlink, else `None` (the path is not a
    /// symlink, so `readlink` returns `-1`).
    pub fn readlink_target(&self, path: &str) -> Option<&[u8]> {
        self.symlinks.get(path).map(std::vec::Vec::as_slice)
    }

    /// Get the content buffer for a file descriptor (read-only).
    pub fn fd_content(&self, fd: u32) -> &[u8] {
        let fd = self.desc_key(fd);
        self.fds
            .get(&fd)
            .map(|d| d.content.as_slice())
            .unwrap_or(&[])
    }

    /// The end offsets of the writes that appended to `fd`'s content, in
    /// write order; see [`FileDescriptor::write_ends`].
    pub fn fd_write_ends(&self, fd: u32) -> &[usize] {
        let fd = self.desc_key(fd);
        self.fds
            .get(&fd)
            .map(|d| d.write_ends.as_slice())
            .unwrap_or(&[])
    }

    /// Check if a file descriptor is open.
    pub fn is_open(&self, fd: u32) -> bool {
        let fd = self.desc_key(fd);
        self.fds.get(&fd).is_some_and(|d| d.is_open)
    }

    /// Get file descriptor info: (name, position, flags, content_len, is_open).
    pub fn fd_info(&self, fd: u32) -> Option<(&str, u64, u32, usize, bool)> {
        let fd = self.desc_key(fd);
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
        // Stored `d.name` stays raw (fd_info exposes it to Python); compare
        // against each fd's cwd-at-open normalization rather than
        // re-normalizing against the *current* cwd (angr-9ke6b.120).
        let fd_max = self
            .fds
            .values()
            .filter(|d| self.fd_norm_name(d) == norm)
            .map(FileDescriptor::effective_len)
            .max();
        let reg_len = self.file_contents.get(&norm).map(|v| v.len());
        let closed_len = self.closed_contents.get(&norm).map(Vec::len);
        fd_max.into_iter().chain(reg_len).chain(closed_len).max()
    }

    /// The path-keyed bytes of closed files (`closed_contents`), sorted by
    /// path. Drives the bounced callback state's `state.fs` mirror, so a
    /// Python `unlink`/`stat` of a file native code wrote and closed finds
    /// its `SimFile` (angr-mx00p.132.46).
    pub fn closed_files(&self) -> Vec<(&str, &[u8])> {
        let mut files: Vec<(&str, &[u8])> = self
            .closed_contents
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        files.sort_unstable();
        files
    }

    /// Position and [`effective_len`](FileDescriptor::effective_len) for an
    /// fd, in a single map lookup. Drives `NativeFeof`, which is hot in
    /// `while (!feof(f))` guest loops.
    pub fn fd_pos_and_size(&self, fd: u32) -> Option<(u64, usize)> {
        let fd = self.desc_key(fd);
        self.fds.get(&fd).map(|d| (d.position, d.effective_len()))
    }

    /// Effective content length for an fd — symbolic byte count when
    /// `content_sym` is attached, else the concrete buffer length. Drives
    /// `NativeFstatSyscall` st_size; `NativeFeof` uses the combined
    /// [`fd_pos_and_size`](Self::fd_pos_and_size) accessor instead.
    pub fn effective_size(&self, fd: u32) -> Option<usize> {
        let fd = self.desc_key(fd);
        self.fds.get(&fd).map(FileDescriptor::effective_len)
    }

    /// List all file descriptor numbers (including closed ones).
    pub fn all_fds(&self) -> Vec<u32> {
        let mut fds: Vec<u32> = self
            .fds
            .keys()
            .chain(self.aliases.keys())
            .copied()
            .collect();
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
            .chain(self.aliases.keys().copied())
            .collect();
        fds.sort();
        fds
    }

    /// True when any fd above stderr is open — i.e. native code has opened a
    /// file the standard three descriptors do not cover.
    ///
    /// The fast-path gate for the inbound callback fd sync
    /// (angr-op0dn.14.1.6): a bounced SimProcedure only needs its posix fd
    /// table seeded when this is true, so the common case pays one bool over
    /// the FFI instead of a `Vec<(u32, String, ..)>` of the std fds.
    pub fn has_fds_above_stderr(&self) -> bool {
        self.aliases.keys().any(|&fd| fd > 2) || self.fds.iter().any(|(&fd, d)| fd > 2 && d.is_open)
    }

    /// True when fd 0 no longer points at the pristine harness-seeded stdin
    /// placeholder [`Default for FileSystem`](struct@FileSystem)'s `impl`
    /// registers at process start -- i.e. a guest `dup2(real_fd, 0)` rewired
    /// it onto a real, Rust-tracked file (angr-qmrrp). A `dup2` onto fd 0
    /// makes it an alias (see `FileSystem::aliases`), which counts whatever
    /// the shared description is — `dup2(1, 0)` shares stdout's placeholder.
    /// Otherwise fd 0's own descriptor decides through its
    /// [`is_std_placeholder`](FileDescriptor::is_std_placeholder) flag, which
    /// is `false` for every descriptor `open`/`pipe`/`register_fd_at` mint.
    ///
    /// Keying off provenance rather than the descriptor's `name` is what
    /// catches a guest `open("/dev/stdin")` + `dup2(fd, 0)`: `open` stores
    /// the caller-supplied path verbatim, so the shared description's name
    /// is still literally `"/dev/stdin"` and the old string comparison read
    /// it as pristine (angr-tqw60).
    pub fn fd0_is_dup2d(&self) -> bool {
        self.aliases.contains_key(&0) || self.fds.get(&0).is_some_and(|d| !d.is_std_placeholder)
    }

    /// Get the next fd number (for pre-allocating).
    pub fn next_fd(&self) -> u32 {
        self.next_fd
    }

    /// Current working directory bytes (mirrors Python `state.fs.cwd`).
    pub fn cwd(&self) -> &[u8] {
        &self.cwd
    }
}
