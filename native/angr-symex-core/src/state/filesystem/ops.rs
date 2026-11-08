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
    ///
    /// Opens in non-append mode; callers that have an `O_APPEND` /
    /// `fopen("a")` request go through
    /// [`open_with_append`](Self::open_with_append) instead.
    pub fn open(&mut self, name: String, flags: FdFlags) -> Option<u32> {
        self.open_with_append(name, flags, false)
    }

    /// [`open`](Self::open), plus the orthogonal `O_APPEND` bit — see
    /// [`FileDescriptor::append`] for the semantics it buys and
    /// [`O_APPEND`] for decoding it out of raw
    /// `open(2)` flags.
    ///
    /// Kept as one entry point taking the bool rather than an `open_append`
    /// sibling so the three production callers (`fopen`, the `open` native
    /// proc, the `open`/`openat` syscalls) each name their append decision at
    /// the call site and none can be silently added without one.
    pub fn open_with_append(&mut self, name: String, flags: FdFlags, append: bool) -> Option<u32> {
        let fd = self.alloc_fd()?;
        // Normalized before install (freezes cwd-at-open) — see `install_fd`.
        let norm = self.normalize_path(&name);
        let content_sym = self.file_contents.get(&norm).cloned();
        let mut desc = match content_sym {
            Some(_) => FileDescriptor::new(name, flags),
            None => FileDescriptor::with_content(name, flags, self.concrete_content_of(&norm)),
        };
        desc.append = append;
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

    /// The concrete bytes a fresh open of `norm` starts from: an open
    /// sibling fd's buffer, else what the path held when its last fd closed
    /// (`closed_contents`), else empty. Python shares one `SimFile` across
    /// every fd of a path; this copies a snapshot instead, so writes through
    /// two simultaneously open fds still do not see each other
    /// (angr-vmi2l.196). No truncation: Python's `posix.open` ignores
    /// `O_TRUNC` too. A demoted path is Python-owned and starts empty, as
    /// before.
    fn concrete_content_of(&self, norm: &str) -> Vec<u8> {
        if self.demoted_paths.contains(norm) {
            return Vec::new();
        }
        self.fds
            .values()
            .find(|d| {
                d.is_open && Self::holds_path_content(d) && d.norm_name.as_deref() == Some(norm)
            })
            .map(|d| d.content.clone())
            .or_else(|| self.closed_contents.get(norm).cloned())
            .unwrap_or_default()
    }

    /// Whether `desc`'s concrete buffer is the content of the file it names:
    /// not a symbolic stream, not bounded symbolic content (the
    /// `file_contents` registry owns that), and opened by path (std fds and
    /// pipe ends have no `norm_name`).
    fn holds_path_content(desc: &FileDescriptor) -> bool {
        !desc.symbolic
            && desc.content_sym.is_none()
            && desc.registry_key.is_none()
            && !desc.is_std_placeholder
            && desc.norm_name.is_some()
    }

    /// The `fds` key holding `fd`'s open file description: the fd a `dup`
    /// aliased it to, else `fd` itself. See `FileSystem::aliases`. Every
    /// per-fd entry point resolves its argument through this before touching
    /// `fds`.
    #[inline]
    pub(super) fn desc_key(&self, fd: u32) -> u32 {
        if self.aliases.is_empty() {
            return fd;
        }
        self.aliases.get(&fd).copied().unwrap_or(fd)
    }

    /// Allocate the lowest fd number that is not open — the POSIX `open(2)`
    /// rule, and Python `SimSystemPosix::_pick_fd`'s — so a closed number is
    /// handed out again: `fclose(fopen(p)); open(p)` returns the same fd on
    /// both engines (angr-mx00p.132.13). See
    /// [`lowest_free_fd_from`](Self::lowest_free_fd_from) for why exhaustion refuses.
    ///
    /// `next_fd` is only a high-water mark and is bumped past the result. On
    /// refusal it is left untouched and no descriptor is inserted.
    fn alloc_fd(&mut self) -> Option<u32> {
        let fd = self.lowest_free_fd_from(0)?;
        self.next_fd = self.next_fd.max(fd.saturating_add(1));
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
    /// that fd hits a closed fd; angr-op0dn.14.1.5). Unlike
    /// [`open`](Self::open) the number is dictated by Python rather than
    /// drawn from `next_fd`, so `next_fd` is bumped past it to keep later
    /// native opens from colliding.
    ///
    /// Returns false — changing nothing — when `fd` is already open, which
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
        // A closed tombstone is a free slot, as in `lowest_free_fd_from`:
        // Python reuses the number, so the adopted fd overwrites it.
        if self.is_open(fd) {
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

    /// Overwrite an open path-opened fd's concrete bytes and position with
    /// what a bounced Python SimProcedure left on its mirror of it — a
    /// bounced `getc` advances the position, a bounced write extends the
    /// content — so the native side stays the authority the next bounce's
    /// mirror is rebuilt from (angr-mx00p.132.46).
    ///
    /// Returns false — changing nothing — for an fd that is closed, unknown,
    /// carries symbolic content (see `holds_path_content`), or already
    /// matches; a content larger than [`MAX_FS_FILE_SIZE`] is refused too.
    pub fn update_fd(&mut self, fd: u32, content: Vec<u8>, position: u64) -> bool {
        let fd = self.desc_key(fd);
        let unchanged_or_foreign = self.fds.get(&fd).is_none_or(|d| {
            !d.is_open
                || !Self::holds_path_content(d)
                || (d.position == position && d.content == content)
        });
        if unchanged_or_foreign || content.len() as u64 > MAX_FS_FILE_SIZE {
            return false;
        }
        let Some(desc) = Arc::make_mut(&mut self.fds).get_mut(&fd) else {
            return false;
        };
        desc.content = content;
        desc.position = position;
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

    /// Forget `name`, the inverse of [`register_known_path`]: it leaves
    /// `known_paths`, the `file_contents` registry and `closed_contents`, so a
    /// later read-only open, `access` or `stat` of it fails as on Python. An fd
    /// still open on the path keeps its bytes, as POSIX unlink does (a
    /// write-mode open of the path while one is still open starts from them,
    /// where Python creates an empty file). Drives the
    /// write-back of a bounced Python `unlink` (angr-mx00p.132.71). Returns
    /// whether the path was known.
    ///
    /// [`register_known_path`]: FileSystem::register_known_path
    pub fn forget_path(&mut self, name: &str) -> bool {
        let norm = self.normalize_path(name);
        // Peek first (`arc-make-mut-cow`): forgetting an unknown path must not
        // deep-clone the maps away from forked siblings.
        let known = self.known_paths.contains(&norm);
        if known {
            Arc::make_mut(&mut self.known_paths).remove(&norm);
        }
        if self.file_contents.contains_key(&norm) {
            Arc::make_mut(&mut self.file_contents).remove(&norm);
        }
        if self.closed_contents.contains_key(&norm) {
            Arc::make_mut(&mut self.closed_contents).remove(&norm);
        }
        known
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
    ///
    /// A path-opened fd with bytes leaves them in `closed_contents`, so a
    /// later open of the path reads them even after this tombstone's number
    /// is reused (angr-mx00p.132.46).
    ///
    /// Closing one of several fds sharing a description (see
    /// `FileSystem::aliases`) leaves the description open for the others
    /// and records nothing in `closed_contents`; the closed number keeps a
    /// tombstone so a later write to it is refused, not auto-vivified.
    pub fn close(&mut self, fd: u32) -> bool {
        if self.aliases.contains_key(&fd) {
            Arc::make_mut(&mut self.aliases).remove(&fd);
            self.insert_tombstone(fd);
            return true;
        }
        if self.aliases.values().any(|&k| k == fd) {
            self.promote_alias_of(fd);
            self.insert_tombstone(fd);
            return true;
        }
        // Read-first to avoid CoW clone if the fd is missing or already closed.
        let Some(desc) = self.fds.get(&fd).filter(|d| d.is_open) else {
            return false;
        };
        if let Some(norm) = desc.norm_name.clone()
            && Self::holds_path_content(desc)
            && !desc.content.is_empty()
        {
            let content = desc.content.clone();
            Arc::make_mut(&mut self.closed_contents).insert(norm, content);
        }
        if let Some(desc) = Arc::make_mut(&mut self.fds).get_mut(&fd) {
            desc.is_open = false;
            true
        } else {
            false
        }
    }

    /// Hand `owner`'s description to its lowest alias, which becomes the new
    /// `fds` key, and repoint the remaining aliases at it. The caller then
    /// overwrites `owner`'s slot. The moved description loses its
    /// [`is_std_placeholder`](FileDescriptor::is_std_placeholder) stamp: it no
    /// longer sits at its pre-registered number, so `dup2(1, 0); close(1)`
    /// still reads as a rewired fd 0.
    fn promote_alias_of(&mut self, owner: u32) {
        let Some(heir) = self
            .aliases
            .iter()
            .filter(|&(_, &k)| k == owner)
            .map(|(&a, _)| a)
            .min()
        else {
            return;
        };
        let aliases = Arc::make_mut(&mut self.aliases);
        aliases.remove(&heir);
        for target in aliases.values_mut() {
            if *target == owner {
                *target = heir;
            }
        }
        let fds = Arc::make_mut(&mut self.fds);
        if let Some(mut desc) = fds.remove(&owner) {
            desc.is_std_placeholder = false;
            fds.insert(heir, desc);
        }
    }

    /// Park a closed descriptor at `fd`, which `close` has just detached
    /// from a description another fd still holds open.
    fn insert_tombstone(&mut self, fd: u32) {
        let desc = self.fds.get(&self.desc_key(fd));
        let (name, flags) = desc.map_or((String::new(), FdFlags::ReadOnly), |d| {
            (d.name.clone(), d.flags)
        });
        let mut tomb = FileDescriptor::new(name, flags);
        tomb.is_open = false;
        Arc::make_mut(&mut self.fds).insert(fd, tomb);
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
        //
        // An `O_APPEND` fd ignores its tracked position and writes at the
        // current end of content, recomputed per write so an intervening
        // `lseek` cannot move the write off the end. `write_bytes_at` then
        // advances `position` to the new end, exactly as Python
        // `SimFileDescriptor::write_data` does after its
        // `self._pos = self.file.size`. See [`FileDescriptor::append`].
        let fd = self.desc_key(fd);
        let position = self.fds.get(&fd).map_or(0, |d| {
            if d.append {
                d.effective_len() as u64
            } else {
                d.position
            }
        });
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
        let fd = self.desc_key(fd);
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
        if start == desc.content.len() {
            // Ends a truncation left past the new end are no boundaries.
            while desc.write_ends.last().is_some_and(|&e| e > start) {
                desc.write_ends.pop();
            }
            desc.write_ends.push(end);
        }
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
        let n = self.read_len(fd, count);
        let fd = self.desc_key(fd);
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
        // overflow-ok: `n <= content.len() - pos` (the peek's `count.min(available)`).
        let data = desc.content[pos..pos + n].to_vec();
        desc.position += n as u64;
        data
    }

    /// How many bytes [`read`](Self::read) would return for `(fd, count)`,
    /// without reading or moving the position. A native syscall checks its
    /// destination for that many bytes before the read advances the
    /// position (angr-vmi2l.140).
    pub fn read_len(&self, fd: u32, count: usize) -> usize {
        match self.fds.get(&self.desc_key(fd)) {
            // Guard on is_open for parity with read_sym/read_sym_at; a closed
            // fd serves no bytes even though its content buffer survives.
            Some(desc) if desc.is_open => {
                let available = desc.content.len().saturating_sub(desc.position as usize);
                count.min(available)
            }
            _ => 0,
        }
    }

    /// Seek a file descriptor. Returns the new position.
    ///
    /// whence: 0=SEEK_SET, 1=SEEK_CUR, 2=SEEK_END
    ///
    /// A negative target is `None` with the position unchanged, as in glibc
    /// (`EINVAL`) and Python's `SimFileDescriptor.seek` (angr-vmi2l.113).
    ///
    /// The SEEK_CUR/SEEK_END bases are combined with the guest's signed
    /// `offset` through `offset_position`, not through `base as i64 +
    /// offset` (angr-03vl4.54).
    pub fn seek(&mut self, fd: u32, offset: i64, whence: u32) -> Option<u64> {
        // Compute new position without CoW first; only mutate if the fd exists
        // and the whence value is valid.
        let new_pos = self.seek_target(fd, offset, whence)?;
        let fd = self.desc_key(fd);
        Arc::make_mut(&mut self.fds).get_mut(&fd)?.position = new_pos;
        Some(new_pos)
    }

    /// Whether [`seek`](Self::seek) would move `fd` past the end of its
    /// content. glibc allows that, but Python's `SimFileDescriptor.seek`
    /// fails, and on a file of unknown size its answer is symbolic, so the
    /// guest-facing `lseek` / `fseek` defer to Python (angr-vmi2l.115).
    /// Internal position advances call `seek` directly.
    pub fn seeks_past_eof(&self, fd: u32, offset: i64, whence: u32) -> bool {
        let Some(new_pos) = self.seek_target(fd, offset, whence) else {
            return false;
        };
        self.fds
            .get(&self.desc_key(fd))
            .is_some_and(|desc| new_pos > desc.effective_len() as u64)
    }

    /// The position [`seek`](Self::seek) would move `fd` to, or `None` for
    /// an absent fd, a bad `whence` or a negative target.
    fn seek_target(&self, fd: u32, offset: i64, whence: u32) -> Option<u64> {
        let desc = self.fds.get(&self.desc_key(fd))?;
        match whence {
            0 => u64::try_from(offset).ok(),             // SEEK_SET
            1 => offset_position(desc.position, offset), // SEEK_CUR
            2 => offset_position(desc.effective_len() as u64, offset), // SEEK_END
            _ => None,
        }
    }

    /// Positioned read: read up to `count` bytes starting at absolute
    /// `offset`, WITHOUT touching the fd's current position. Mirrors POSIX
    /// `pread` (the file offset is unaffected). Returns the bytes read
    /// (empty if `offset` is past EOF or the fd is absent).
    pub fn read_at(&self, fd: u32, offset: u64, count: usize) -> Vec<u8> {
        let fd = self.desc_key(fd);
        match self.fds.get(&fd) {
            // Guard on is_open for parity with read_sym_at (POSIX pread on a
            // closed fd reads nothing).
            Some(desc) if desc.is_open => {
                let pos = offset as usize;
                if pos >= desc.content.len() {
                    return Vec::new();
                }
                let n = count.min(desc.content.len() - pos);
                // overflow-ok: `n <= content.len() - pos` (the `min` above).
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

    /// Replace the current working directory bytes verbatim. The inbound
    /// sync of Python's already-normalized `state.fs.cwd`; a guest
    /// `chdir(2)` goes through [`chdir`](Self::chdir) instead.
    pub fn set_cwd(&mut self, cwd: Vec<u8>) {
        self.cwd = cwd;
    }

    /// `chdir(2)`: store `path` resolved against the cwd with `.`/`..`
    /// folded, as Python's `SimFilesystem.chdir` does (angr-vmi2l.136).
    pub fn chdir(&mut self, path: &[u8]) {
        self.cwd = self.normalize_path_bytes(path);
    }

    /// Lowest fd number at or above `start` that is not currently open — the
    /// POSIX allocation rule for [`alloc_fd`](Self::alloc_fd),
    /// [`dup`](Self::dup) and [`pipe`](Self::pipe). A number that was opened and later
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
    fn lowest_free_fd_from(&self, start: u32) -> Option<u32> {
        let mut fd = start;
        while self.is_open(fd) {
            fd = fd.checked_add(1)?;
        }
        Some(fd)
    }

    /// Duplicate an open file descriptor, allocating the lowest unused fd.
    /// Returns the new fd, or None if `oldfd` is not open or the fd space is
    /// exhausted (see `lowest_free_fd_from`).
    ///
    /// Like POSIX `dup(2)`: the new fd is the lowest number not currently
    /// open (see `lowest_free_fd_from`), and it refers to the same open file
    /// description — an entry in `FileSystem::aliases`, so offset, content
    /// and flags stay shared (angr-vmi2l.196). `next_fd` is bumped past the
    /// allocated number so a later [`open`](Self::open) cannot hand out the
    /// same slot.
    ///
    /// Divergence from Python `procedures/posix/dup.py` (angr-9ke6b.119):
    /// its gap scan keeps the *last* mismatching index rather than breaking
    /// at the first, so with two or more gaps it can return an fd that is
    /// still open and clobber it. We return the true lowest free fd; the two
    /// agree for the single-gap case that real binaries hit.
    pub fn dup(&mut self, oldfd: u32) -> Option<u32> {
        if !self.is_open(oldfd) {
            return None;
        }
        let newfd = self.lowest_free_fd_from(0)?;
        self.next_fd = self.next_fd.max(newfd.saturating_add(1));
        self.alias_fd(newfd, oldfd);
        Some(newfd)
    }

    /// Make the free number `newfd` share `oldfd`'s open description,
    /// dropping any closed tombstone parked at `newfd`.
    fn alias_fd(&mut self, newfd: u32, oldfd: u32) {
        let key = self.desc_key(oldfd);
        if self.fds.contains_key(&newfd) {
            Arc::make_mut(&mut self.fds).remove(&newfd);
        }
        Arc::make_mut(&mut self.aliases).insert(newfd, key);
    }

    /// Duplicate `oldfd` to `newfd`, sharing its open description as
    /// [`dup`](Self::dup) does. If `newfd` was open, it is closed first.
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
        if !self.is_open(oldfd) {
            return None;
        }
        if oldfd == newfd {
            return Some(newfd);
        }
        if newfd >= MAX_FD {
            return None;
        }
        // Already sharing one description: POSIX dup2 leaves it as is.
        if self.desc_key(oldfd) == self.desc_key(newfd) {
            return Some(newfd);
        }
        // Closing first may promote `newfd`'s aliases, possibly `oldfd`
        // itself, so the alias resolves `oldfd` only afterwards.
        self.close(newfd);
        self.alias_fd(newfd, oldfd);
        if newfd >= self.next_fd {
            self.next_fd = newfd.saturating_add(1);
        }
        Some(newfd)
    }

    /// Create a pipe: returns `Some((read_fd, write_fd))`, the two lowest fds
    /// that are not open (so a closed number is reused, as POSIX `pipe(2)`
    /// does), or `None` when the fd space cannot supply the pair.
    ///
    /// Like POSIX `pipe(2)`. The read end is opened ReadOnly and the write end
    /// WriteOnly. We do NOT model write→read data flow (each end has its own
    /// content buffer); this matches angr's existing SimPacketsStream-light
    /// modeling — the procedure exists so binaries that allocate fds via pipe()
    /// don't fall through to Python on every fd op.
    ///
    /// Exhaustion refuses — nothing inserted, `next_fd` unchanged — rather
    /// than wrapping or saturating, which would alias both ends onto one
    /// number (angr-03vl4.55); see `lowest_free_fd_from`. Refusing bounces the
    /// caller to Python, mirroring [`dup2`](Self::dup2)'s `MAX_FD` refusal.
    pub fn pipe(&mut self) -> Option<(u32, u32)> {
        let read_fd = self.lowest_free_fd_from(0)?;
        let write_fd = self.lowest_free_fd_from(read_fd.checked_add(1)?)?;
        self.next_fd = self.next_fd.max(write_fd.saturating_add(1));
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
/// signed `offset`: `None` below zero, saturating above `u64::MAX`.
///
/// `lseek`'s offset arrives from the guest unbounded (`NativeLseekSyscall`'s
/// `extract_concrete_arg`), and `base` is itself guest-steerable — a prior
/// `SEEK_SET` accepts any non-negative `i64`. The former spelling,
/// `(base as i64 + offset).max(0) as u64`, was wrong twice over
/// (angr-03vl4.54): the add overflows `i64` (panic under
/// `[profile.release-checked]`, silent wrap in the shipped
/// `[profile.release]`), and `base as i64` alone turns any position above
/// `i64::MAX` negative before the offset is even applied. `i128` has room for
/// every `u64` base plus every `i64` offset. A negative sum is `None` (the
/// seek fails); a sum past `u64::MAX` saturates.
fn offset_position(base: u64, offset: i64) -> Option<u64> {
    let target = i128::from(base) + i128::from(offset);
    if target < 0 {
        return None;
    }
    Some(target.min(i128::from(u64::MAX)) as u64)
}
