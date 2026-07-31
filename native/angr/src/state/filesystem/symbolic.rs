//! Bounded symbolic file content (angr-0xyq2): the path-keyed
//! `file_contents` registry, the native symbolic-read serve paths
//! (`read_sym` / `read_sym_at`), and the write-demotion machinery that hands
//! a file over to Python ownership (`demote_*`, angr-qluof).
//!
//! The unbounded symbolic *stream* model (`FileDescriptor::symbolic`, the
//! stdin model) is deliberately NOT here — it is an fd flag consumed by the
//! read syscalls, see [`FileDescriptor::symbolic`] for the distinction.
//!
//! **Panic policy (angr-9ke6b.212):** every `fd: u32` reaching this module is
//! an *untrusted* value — it comes from a guest syscall argument or a
//! Python-side seeding call — so nothing here may panic on a bad fd. Absent
//! fds are a documented `None` / `false` return on every entry point. The
//! two surviving `expect`s are both the second half of a
//! collect-keys-then-`Arc::make_mut`-and-re-look-up pair
//! ([`FileSystem::register_file_content`] and
//! [`FileSystem::clear_key_state`]): the key set is collected from
//! `self.fds` immediately above and `Arc::make_mut` only deep-clones the map,
//! it never re-keys it, so the second lookup cannot miss. They read as
//! fallible-on-input only because the forced re-borrow hides the preceding
//! check.
//!
//! **Enforcement (angr-qwyti.11):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` so a future
//! panic-on-untrusted-fd landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

impl FileSystem {
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
    #[allow(
        clippy::expect_used,
        reason = "`stamp` is collected from `self.fds` two statements above and `Arc::make_mut` only deep-clones the map (never re-keys it), so the re-lookup under the mutable borrow cannot miss — see the module Panic policy header"
    )]
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

    /// Attach bounded symbolic content directly to an already-open fd,
    /// without going through the path registry. This is the seed-time
    /// channel for a Python-side `posix.stdin` stream whose content the
    /// harness filled in (`state.posix.stdin.content.append((BVS, n))`):
    /// fd 0 is open from `FileSystem::default`, so `register_file_content`
    /// (which only attaches on a later `open`) can never reach it. Reads
    /// then serve the harness's own BVS bytes through `read_sym` instead of
    /// minting fresh stdin symbols, which is what makes the seeding BVS
    /// evaluable on an exported found state (angr-mb09c).
    ///
    /// Position is left as-is (0 on a seed state). No-op on an unknown fd.
    // !Send Arcs go through `crate::arc_shared` (see its doc comment).
    pub fn set_fd_content_sym(&mut self, fd: u32, bytes: Vec<RustBV>) {
        // The `contains_key` peek keeps the unknown-fd no-op from forcing a
        // CoW deep clone of the fd table; the `if let` is the same single
        // lookup the write needs anyway, so the pair costs what one
        // `get_mut().expect(..)` did without the panic edge.
        if !self.fds.contains_key(&fd) {
            return;
        }
        if let Some(desc) = Arc::make_mut(&mut self.fds).get_mut(&fd) {
            desc.content_sym = Some(crate::arc_shared(bytes));
        }
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
        // Both `?`s are unreachable — the peek block above already proved the
        // fd is present with `content_sym` attached, and `Arc::make_mut` only
        // deep-clones the map — but folding them into the existing
        // fd-absent/no-content `None` contract keeps an untrusted fd from
        // reaching a panic even if that reasoning ever stops holding.
        let desc = Arc::make_mut(&mut self.fds).get_mut(&fd)?;
        let content = desc.content_sym.as_ref()?;
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
            // Unreachable (the `self.fds.get(&fd)` at the top of this fn
            // succeeded), but a missing fd here means nothing was demoted,
            // so report that rather than panicking on an untrusted fd.
            let Some(desc) = Arc::make_mut(&mut self.fds).get_mut(&fd) else {
                return false;
            };
            desc.content_sym = None;
            crate::symbolic::record_symfile_write_demotion();
            return true;
        };
        // `fd` itself carries the key, so at least one fd always matches.
        self.clear_key_state(&key);
        // Remember the demoted path so a later Python re-add (merge /
        // legacy-fork push) does not re-register it (angr-qluof).
        Arc::make_mut(&mut self.demoted_paths).insert(key);
        crate::symbolic::record_symfile_write_demotion();
        true
    }

    /// Drop any registered content for `key` and clear `content_sym` /
    /// `registry_key` on every fd whose `registry_key` matches it (dup'd /
    /// re-opened siblings included). Returns `true` when anything was
    /// cleared. Shared clear-by-key core of
    /// [`demote_symbolic_content`](Self::demote_symbolic_content) and
    /// [`demote_path`](Self::demote_path); it deliberately does NOT touch
    /// `demoted_paths` or the write-demotion counter — those differ between
    /// the callers (a re-add correction bumps neither the counter nor relies
    /// on this method's return), so each layers them on itself.
    #[allow(
        clippy::expect_used,
        reason = "`matching` is collected from `self.fds` immediately above and `Arc::make_mut` only deep-clones the map (never re-keys it), so the re-lookup under the mutable borrow cannot miss — same shape as `register_file_content`, see the module Panic policy header"
    )]
    fn clear_key_state(&mut self, key: &str) -> bool {
        let mut changed = false;
        if self.file_contents.contains_key(key) {
            Arc::make_mut(&mut self.file_contents).remove(key);
            changed = true;
        }
        let matching: Vec<u32> = self
            .fds
            .iter()
            .filter(|(_, d)| d.registry_key.as_deref() == Some(key))
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
        changed
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

    /// Whether `fd`'s backing path was demoted by a native write
    /// (angr-8kk32). Demotion hands the file to Python for good — the write
    /// itself bounced, so Python's `SimFile` holds bytes the native
    /// `FileSystem` never saw. Native reads must therefore keep bouncing on
    /// such an fd rather than minting fresh symbolic bytes for it
    /// (`read_file_symbolic`, angr-gorvf.15), which would silently discard
    /// the Python-side write.
    pub fn is_demoted_fd(&self, fd: u32) -> bool {
        if self.demoted_paths.is_empty() {
            return false;
        }
        self.fds
            .get(&fd)
            .is_some_and(|d| self.demoted_paths.contains(&self.normalize_path(&d.name)))
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
        let changed = self.clear_key_state(&norm);
        Arc::make_mut(&mut self.demoted_paths).insert(norm);
        changed
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
}
