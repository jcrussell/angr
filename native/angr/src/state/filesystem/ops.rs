//! POSIX fd operations that mutate the [`FileSystem`]: the `open` family,
//! `close`, concrete `read`/`write` (+ their positioned twins), `seek`,
//! `dup`/`dup2`/`pipe`, and the path/symlink/cwd registration setters.
//!
//! The bounded-symbolic-content half of the model lives in
//! [`super::symbolic`]; read-only accessors live in [`super::query`].
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** every `fd`
//! and path here is an untrusted guest syscall argument, so this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` and an absent fd is a
//! documented empty/`false`/`-1` return on every entry point. There are no
//! `unwrap`/`expect` sites left: `read`'s post-`Arc::make_mut` re-lookup folds
//! into that same unknown-fd contract with a `let ... else` (matching
//! [`super::symbolic`]'s `read_sym`).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

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
    ///
    /// Returns `None` — changing nothing — when the fd space is exhausted;
    /// see `alloc_fd` for why that is not unreachable.
    pub fn open(&mut self, name: String, flags: FdFlags) -> Option<u32> {
        let fd = self.alloc_fd()?;
        // Normalized before install (freezes cwd-at-open) — see `install_fd`.
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
        self.install_fd(fd, norm, desc);
        Some(fd)
    }

    /// Allocate the next free fd number, refusing rather than wrapping when
    /// the fd space is exhausted.
    ///
    /// `next_fd` is not bounded by [`MAX_FD`] —
    /// [`register_fd_at`](Self::register_fd_at) imports whatever fd numbers
    /// the Python state carries and bumps `next_fd` past them (saturating), so
    /// an imported `u32::MAX` leaves `next_fd == u32::MAX` and a plain `+= 1`
    /// would wrap to 0 in the shipped release profile, handing out stdin as a
    /// fresh file. Refusing (rather than saturating) is required because an fd
    /// is an *identity*, not a size: clamping aliases two files onto one
    /// number. See `invariant-overflow-fix-refuse-not-saturate-identities`.
    ///
    /// On refusal `next_fd` is left untouched and no descriptor is inserted.
    fn alloc_fd(&mut self) -> Option<u32> {
        let fd = self.next_fd;
        self.next_fd = fd.checked_add(1)?;
        Some(fd)
    }

    /// Shared tail of the `open` family ([`open`](Self::open),
    /// [`open_with_content`](Self::open_with_content),
    /// [`register_fd_at`](Self::register_fd_at),
    /// [`open_symbolic`](Self::open_symbolic)): freeze the already-normalized
    /// path on the descriptor, register it as an existing path, and install
    /// the descriptor under `fd`.
    ///
    /// `norm` must be `self.normalize_path(name)` computed by the caller
    /// *before* any cwd change — normalizing at insertion is what freezes
    /// cwd-at-open, which is POSIX-correct for relative paths. Callers
    /// normalize rather than passing the raw name because two of them
    /// ([`open`](Self::open) via `file_contents` / `registry_key`, and
    /// [`open_symbolic`](Self::open_symbolic)'s siblings) need the normalized
    /// key while building the descriptor; doing it here as well would
    /// normalize twice on the hot `open` path.
    ///
    /// Registering the path mirrors Python `procedures/posix/open.py` dropping
    /// a fresh `SimFile` into `state.fs`: any successfully-opened path is
    /// "existing" from that point on, so a later `NativeAccessSyscall` on it
    /// returns 0.
    fn install_fd(&mut self, fd: u32, norm: String, mut desc: FileDescriptor) {
        desc.norm_name = Some(norm.clone());
        self.note_known_path(norm);
        Arc::make_mut(&mut self.fds).insert(fd, desc);
    }

    /// Record `norm` (already normalized) in `known_paths`, peeking the read
    /// path first so re-registering an already-known path is a no-op rather
    /// than a deep clone of the set away from every forked sibling sharing it
    /// — the `arc-make-mut-cow` invariant in the `state` module header.
    /// `known_paths` is its own `Arc`, so the peek pays off even for callers
    /// like [`install_fd`](Self::install_fd) that go on to mutate `fds`
    /// unconditionally: re-opening a path this state already knows leaves the
    /// set shared. Covered by `filesystem/tests.rs`'s `*_skips_cow_clone`
    /// tests, which assert `Arc::ptr_eq` against a fork (contents alone
    /// cannot distinguish a skipped clone from an equal one).
    pub(super) fn note_known_path(&mut self, norm: String) {
        if !self.known_paths.contains(&norm) {
            Arc::make_mut(&mut self.known_paths).insert(norm);
        }
    }

    /// Open a file descriptor with pre-loaded content (for file-backed
    /// SimFiles). Intentionally bypasses the `file_contents` registry — the
    /// caller supplies explicit concrete content (test-seeding API).
    ///
    /// Returns `None` when the fd space is exhausted — see `alloc_fd`.
    pub fn open_with_content(
        &mut self,
        name: String,
        flags: FdFlags,
        content: Vec<u8>,
    ) -> Option<u32> {
        let fd = self.alloc_fd()?;
        // Normalized before install (freezes cwd-at-open) — see `install_fd`.
        let norm = self.normalize_path(&name);
        let desc = FileDescriptor::with_content(name, flags, content);
        self.install_fd(fd, norm, desc);
        Some(fd)
    }

    /// Adopt a file descriptor that was opened *outside* the native engine,
    /// at a caller-chosen number: a bounced Python SimProcedure's
    /// `open`/`fopen`/`dup` allocates the fd in `state.posix`, and without
    /// this the native side never learns of it (a later native read/write on
    /// that fd hits a closed fd — angr-op0dn.14.1.5). Unlike
    /// [`open`](Self::open) the number is dictated by Python rather than
    /// drawn from `next_fd`, so `next_fd` is bumped past it to keep later
    /// native opens from colliding.
    ///
    /// Returns false — changing nothing — when `fd` is already known, which
    /// is what keeps the caller's diff-against-[`all_fds`](Self::all_fds)
    /// idempotent across the repeated callbacks that share one cached state.
    ///
    /// Content is the explicit concrete buffer the caller extracted from the
    /// Python `SimFile`; like [`open_with_content`](Self::open_with_content)
    /// this intentionally bypasses the `file_contents` symbolic registry.
    pub fn register_fd_at(
        &mut self,
        fd: u32,
        name: String,
        flags: FdFlags,
        content: Vec<u8>,
        position: u64,
    ) -> bool {
        if self.fds.contains_key(&fd) {
            return false;
        }
        // Normalized before install (freezes cwd-at-open) — see `install_fd`.
        let norm = self.normalize_path(&name);
        let mut desc = FileDescriptor::with_content(name, flags, content);
        desc.position = position;
        self.install_fd(fd, norm, desc);
        self.next_fd = self.next_fd.max(fd.saturating_add(1));
        true
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
    ///
    /// Returns `None` when the fd space is exhausted — see `alloc_fd`.
    pub fn open_symbolic(&mut self, name: String, flags: FdFlags) -> Option<u32> {
        let fd = self.alloc_fd()?;
        // Normalized before install (freezes cwd-at-open) — see `install_fd`.
        let norm = self.normalize_path(&name);
        let desc = FileDescriptor::new_symbolic(name, flags);
        self.install_fd(fd, norm, desc);
        Some(fd)
    }

    /// Register an existing-file path without allocating an fd. Used by
    /// the Python state-export path to seed `state.fs._files` entries
    /// (`register_known_path` PyO3 setter) and by tests.
    pub fn register_known_path(&mut self, name: String) {
        // Normalized at insertion (freezes cwd-at-registration) — see
        // `install_fd`, which does the same for the fd-allocating openers.
        let norm = self.normalize_path(&name);
        self.note_known_path(norm);
    }

    /// Register a symlink: `link` resolves to `target` (raw bytes, as
    /// `readlink(2)` returns — NOT NUL-terminated). Overwrites any
    /// existing entry. Used by tests and the state-export path,
    /// mirroring the `register_known_path` precedent (Python `state.fs`
    /// symlinks are NOT auto-mirrored). Drives
    /// `NativeReadlinkSyscall` / `NativeReadlinkatSyscall`.
    pub fn add_symlink(&mut self, link: String, target: Vec<u8>) {
        // Peek first (`arc-make-mut-cow`): re-adding an identical link must
        // not deep-clone the map away from forked siblings. Overwriting with a
        // *different* target is a real mutation and falls through.
        if self.symlinks.get(&link).is_some_and(|t| *t == target) {
            return;
        }
        Arc::make_mut(&mut self.symlinks).insert(link, target);
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

    /// True when `fd` is present in the table but already closed — the write
    /// choke point must refuse rather than mutate it (angr-9ke6b.118).
    ///
    /// `read`/`read_at` guard on `is_open`, so bytes appended to a closed
    /// descriptor could never be read back: the fd would silently accumulate
    /// unreachable writes. A *missing* fd is deliberately NOT refused —
    /// `write`/`write_at` auto-vivify a write-only entry for it, which is how
    /// `write_fd(2, ..)` works on a state whose stderr was never opened.
    #[inline]
    fn write_closed(&self, fd: u32) -> bool {
        self.fds.get(&fd).is_some_and(|d| !d.is_open)
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
    /// past EOF (a sparse seek-then-write), mirroring [`write_at`](Self::write_at).
    ///
    /// **Choke point (angr-0xyq2 Phase 2):** an fd carrying bounded
    /// symbolic content (`content_sym` / a live `registry_key`) is never
    /// mutated here. Instead the content is demoted
    /// ([`demote_symbolic_content`](Self::demote_symbolic_content)) and
    /// `false` is returned — the caller must bounce the write to Python
    /// (a fallback, NOT a state-killing error), which owns the file from
    /// then on. Zero-length writes are a POSIX no-op: they return `true`
    /// without demoting (and without creating a missing fd entry).
    ///
    /// **Closed fds (angr-9ke6b.118):** a tracked-but-closed fd is refused
    /// (`false`) *before* the symbolic check, with no demotion — the write is
    /// an `EBADF` that never reaches file content, so native serving of the
    /// file's sibling fds must stay intact. See
    /// `write_closed`.
    ///
    /// **Size cap (angr-c7xno.67):** a write whose end offset would push the
    /// content buffer past [`MAX_FS_FILE_SIZE`] is refused (`false`) rather
    /// than resized into, since `position` comes from an unbounded guest
    /// `lseek`.
    #[must_use = "false means the write was refused (closed fd, symbolic content demoted, or past MAX_FS_FILE_SIZE); bounce to Python"]
    pub fn write(&mut self, fd: u32, data: &[u8]) -> bool {
        // Read the position before the shared body runs: none of the gates it
        // applies mutate `position` on a path that goes on to copy bytes
        // (`demote_symbolic_content` only runs on a refusal).
        let position = self.fds.get(&fd).map_or(0, |d| d.position);
        self.write_bytes_at(fd, position, data, true)
    }

    /// Shared body of [`write`](Self::write) and [`write_at`](Self::write_at):
    /// the refusal gates plus the compute/resize/copy sequence. `start` is the
    /// absolute byte offset to write at (the fd's `position` for `write`, the
    /// caller-supplied offset for `write_at`); `advance_position` distinguishes
    /// `write(2)` (moves the offset to the end of the data) from `pwrite(2)`
    /// (leaves it untouched).
    ///
    /// Kept as one function so the choke-point contract documented on `write`
    /// — and in particular the [`MAX_FS_FILE_SIZE`] cap — has a single
    /// enforcement point rather than two copies that can drift
    /// (angr-c7xno.71).
    #[must_use = "false means the write was refused; bounce to Python"]
    fn write_bytes_at(&mut self, fd: u32, start: u64, data: &[u8], advance_position: bool) -> bool {
        if data.is_empty() {
            return true;
        }
        if self.write_closed(fd) {
            return false;
        }
        if self.write_refused(fd) {
            self.demote_symbolic_content(fd);
            return false;
        }
        // Host-safety cap (angr-c7xno.67). `start` is guest-controlled and
        // unbounded (`lseek(fd, huge, SEEK_SET)` for `write`, the raw
        // `pwrite64` offset for `write_at`), so the `resize` below would
        // otherwise be a multi-exabyte allocation → `panic = "abort"`. Checked
        // AFTER the demotion gate so an oversized write on a symbolic-content
        // fd still hands ownership to Python (refusing first would leave Rust
        // serving stale content while Python performs the write), but BEFORE
        // `Arc::make_mut` so a refusal costs no CoW. See `MAX_FS_FILE_SIZE`.
        if start.saturating_add(data.len() as u64) > MAX_FS_FILE_SIZE {
            let op = if advance_position {
                "write"
            } else {
                "write_at"
            };
            log::warn!(
                "FileSystem::{op} on fd={fd} refused: start offset {start} + {} bytes exceeds \
                 MAX_FS_FILE_SIZE ({MAX_FS_FILE_SIZE}); falling back to Python",
                data.len()
            );
            return false;
        }
        let desc = Arc::make_mut(&mut self.fds)
            .entry(fd)
            .or_insert_with(|| FileDescriptor::new(String::new(), FdFlags::WriteOnly));
        let start = start as usize;
        // overflow-ok: the `start.saturating_add(data.len()) > MAX_FS_FILE_SIZE`
        // refusal above bounds both terms by `MAX_FS_FILE_SIZE` (0x100_0000).
        let end = start + data.len();
        if end > desc.content.len() {
            desc.content.resize(end, 0);
        }
        desc.content[start..end].copy_from_slice(data);
        if advance_position {
            desc.position = end as u64;
        }
        true
    }

    /// Read up to `count` bytes from a file descriptor at its current position.
    /// Advances the position. Returns bytes read.
    pub fn read(&mut self, fd: u32, count: usize) -> Vec<u8> {
        // Peek to compute byte count without forcing CoW when nothing is readable.
        let n = match self.fds.get(&fd) {
            // Guard on is_open for parity with read_sym/read_sym_at; a closed
            // fd serves no bytes even though its content buffer survives.
            Some(desc) if desc.is_open => {
                let pos = desc.position as usize;
                let available = desc.content.len().saturating_sub(pos);
                count.min(available)
            }
            _ => return Vec::new(),
        };
        if n == 0 {
            return Vec::new();
        }
        // Unreachable (the peek above matched `Some(desc)`), but `fd` is a
        // guest syscall argument, so fold the miss into the documented
        // "unknown fd reads nothing" contract rather than panicking.
        let Some(desc) = Arc::make_mut(&mut self.fds).get_mut(&fd) else {
            return Vec::new();
        };
        let pos = desc.position as usize;
        let data = desc.content[pos..pos + n].to_vec();
        desc.position += n as u64;
        data
    }

    /// Seek a file descriptor. Returns the new position.
    ///
    /// whence: 0=SEEK_SET, 1=SEEK_CUR, 2=SEEK_END
    ///
    /// The SEEK_CUR/SEEK_END bases are combined with the guest's signed
    /// `offset` through `offset_position`, not through `base as i64 +
    /// offset` (angr-03vl4.54).
    pub fn seek(&mut self, fd: u32, offset: i64, whence: u32) -> Option<u64> {
        // Compute new position without CoW first; only mutate if the fd exists
        // and the whence value is valid.
        let desc = self.fds.get(&fd)?;
        let new_pos = match whence {
            0 => offset.max(0) as u64,                                 // SEEK_SET
            1 => offset_position(desc.position, offset),               // SEEK_CUR
            2 => offset_position(desc.effective_len() as u64, offset), // SEEK_END
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
            // Guard on is_open for parity with read_sym_at (POSIX pread on a
            // closed fd reads nothing).
            Some(desc) if desc.is_open => {
                let pos = offset as usize;
                if pos >= desc.content.len() {
                    return Vec::new();
                }
                let n = count.min(desc.content.len() - pos);
                desc.content[pos..pos + n].to_vec()
            }
            _ => Vec::new(),
        }
    }

    /// Positioned write: overwrite `data` at absolute `offset`, WITHOUT
    /// touching the fd's current position. Mirrors POSIX `pwrite` (the file
    /// offset is unaffected). Extends the content buffer (zero-filling any
    /// gap) when `offset` is at or past EOF, so it is not append-only like
    /// `write`. Creates the fd entry if missing, matching `write`.
    ///
    /// Same choke-point contract as [`write`](Self::write): closed fds are
    /// refused without demotion, symbolic-content fds are demoted and refused
    /// (`false`), zero-length writes are a no-demotion no-op (`true`), and an
    /// `offset` + length past [`MAX_FS_FILE_SIZE`] is refused rather than
    /// resized into (angr-c7xno.67).
    #[must_use = "false means the write was refused (closed fd, symbolic content demoted, or past MAX_FS_FILE_SIZE); bounce to Python"]
    pub fn write_at(&mut self, fd: u32, offset: u64, data: &[u8]) -> bool {
        self.write_bytes_at(fd, offset, data, false)
    }

    /// Replace the current working directory bytes. `chdir(2)` semantics —
    /// the raw concrete path is stored verbatim (no `_normalize_path`
    /// applied, matching `procedures/linux_kernel/cwd.py::chdir`).
    pub fn set_cwd(&mut self, cwd: Vec<u8>) {
        self.cwd = cwd;
    }

    /// Lowest fd number that is not currently open — the POSIX allocation
    /// rule for [`dup`](Self::dup). A number that was opened and later
    /// `close`d counts as free even though its (closed) descriptor is still
    /// parked in `fds`; reusing it overwrites that tombstone, which is
    /// exactly what a real kernel does when the slot is handed out again.
    ///
    /// O(k) hash lookups for k = the returned fd, and k is bounded by the
    /// number of open fds, so the scan always terminates.
    ///
    /// Returns `None` when every fd from 0 to `u32::MAX` is open, for the same
    /// reason [`alloc_fd`](Self::alloc_fd) refuses rather than saturates: a
    /// wrapped or clamped scan cursor would hand back an fd that is still
    /// open, and an fd is an *identity*, not a size. See
    /// `invariant-overflow-fix-refuse-not-saturate-identities`. (Only
    /// reachable with ~4 billion live descriptors, so this is consistency
    /// with the rest of the file's fd arithmetic, not a practical hazard.)
    fn lowest_free_fd(&self) -> Option<u32> {
        let mut fd = 0u32;
        while self.fds.get(&fd).is_some_and(|d| d.is_open) {
            fd = fd.checked_add(1)?;
        }
        Some(fd)
    }

    /// Duplicate an open file descriptor, allocating the lowest unused fd.
    /// Returns the new fd, or None if `oldfd` is not open or the fd space is
    /// exhausted (see `lowest_free_fd`).
    ///
    /// Like POSIX `dup(2)`: the new fd is the lowest number not currently
    /// open (see `lowest_free_fd`), and it refers to
    /// the same underlying state. We model the sharing by cloning the
    /// `FileDescriptor` (name/position/flags/content). `next_fd` is bumped
    /// past the allocated number so a later [`open`](Self::open) cannot
    /// hand out the same slot.
    ///
    /// Divergence from Python `procedures/posix/dup.py` (angr-9ke6b.119):
    /// its gap scan keeps the *last* mismatching index rather than breaking
    /// at the first, so with two or more gaps it can return an fd that is
    /// still open and clobber it. We return the true lowest free fd; the two
    /// agree for the single-gap case that real binaries hit.
    pub fn dup(&mut self, oldfd: u32) -> Option<u32> {
        if !self.fds.get(&oldfd).is_some_and(|d| d.is_open) {
            return None;
        }
        let cloned = self.fds.get(&oldfd).cloned()?;
        let newfd = self.lowest_free_fd()?;
        self.next_fd = self.next_fd.max(newfd.saturating_add(1));
        Arc::make_mut(&mut self.fds).insert(newfd, cloned);
        Some(newfd)
    }

    /// Duplicate `oldfd` to `newfd`. If `newfd` was open, it is closed first.
    /// If `oldfd == newfd` and `oldfd` is open, returns `newfd` unchanged.
    /// Returns the new fd on success, or None if `oldfd` is not open.
    ///
    /// Like POSIX `dup2(2)`. Bumps `next_fd` past `newfd` if necessary so future
    /// allocations don't collide.
    ///
    /// `newfd` at or above [`MAX_FD`] is refused (`None` → `EBADF` at both the
    /// syscall and SimProcedure layers) rather than allowed to run the `next_fd`
    /// bump past `u32::MAX` (angr-03vl4.52). The refusal sits *after* the
    /// `oldfd == newfd` early return so the check order still matches Python's
    /// `procedures/posix/dup.py` — see `MAX_FD` and the
    /// `invariant-dup-python-parity-no-einval` memory.
    pub fn dup2(&mut self, oldfd: u32, newfd: u32) -> Option<u32> {
        if !self.fds.get(&oldfd).is_some_and(|d| d.is_open) {
            return None;
        }
        if oldfd == newfd {
            return Some(newfd);
        }
        if newfd >= MAX_FD {
            return None;
        }
        let cloned = self.fds.get(&oldfd).cloned()?;
        Arc::make_mut(&mut self.fds).insert(newfd, cloned);
        if newfd >= self.next_fd {
            self.next_fd = newfd.saturating_add(1);
        }
        Some(newfd)
    }

    /// Create a pipe: returns `Some((read_fd, write_fd))`, allocated as two
    /// consecutive fds, or `None` when the fd space cannot supply the pair.
    ///
    /// Like POSIX `pipe(2)`. The read end is opened ReadOnly and the write end
    /// WriteOnly. We do NOT model write→read data flow (each end has its own
    /// content buffer); this matches angr's existing SimPacketsStream-light
    /// modeling — the procedure exists so binaries that allocate fds via pipe()
    /// don't fall through to Python on every fd op.
    ///
    /// `next_fd` is not bounded by [`MAX_FD`] — [`register_fd_at`](Self::register_fd_at)
    /// imports whatever fd numbers the Python state carries and bumps `next_fd`
    /// past them — so the two-fd bump is `checked_add`, and a `next_fd` within
    /// two of `u32::MAX` refuses the pipe (nothing inserted, `next_fd`
    /// unchanged) instead of wrapping. Saturating here would be worse than the
    /// plain `+` it replaced: it would hand out `u32::MAX` for *both* ends and
    /// alias them onto whatever the previous allocation put there
    /// (angr-03vl4.55). Refusing bounces the caller to Python, mirroring
    /// [`dup2`](Self::dup2)'s `MAX_FD` refusal.
    pub fn pipe(&mut self) -> Option<(u32, u32)> {
        let read_fd = self.next_fd;
        let write_fd = read_fd.checked_add(1)?;
        self.next_fd = write_fd.checked_add(1)?;
        let map = Arc::make_mut(&mut self.fds);
        map.insert(
            read_fd,
            FileDescriptor::new("<pipe:r>".to_string(), FdFlags::ReadOnly),
        );
        map.insert(
            write_fd,
            FileDescriptor::new("<pipe:w>".to_string(), FdFlags::WriteOnly),
        );
        Some((read_fd, write_fd))
    }
}

/// Combine an unsigned file position/length `base` with a guest-supplied
/// signed `offset`, clamping at both ends of the `u64` range.
///
/// `lseek`'s offset arrives from the guest unbounded (`NativeLseekSyscall`'s
/// `extract_concrete_arg`), and `base` is itself guest-steerable — a prior
/// `SEEK_SET` accepts any non-negative `i64`. The former spelling,
/// `(base as i64 + offset).max(0) as u64`, was wrong twice over
/// (angr-03vl4.54): the add overflows `i64` (panic under
/// `[profile.release-checked]`, silent wrap in the shipped
/// `[profile.release]`), and `base as i64` alone turns any position above
/// `i64::MAX` negative before the offset is even applied. `i128` has room for
/// every `u64` base plus every `i64` offset, so the clamp is the only
/// approximation left.
fn offset_position(base: u64, offset: i64) -> u64 {
    (i128::from(base) + i128::from(offset)).clamp(0, i128::from(u64::MAX)) as u64
}
