"""Persistent disk cache for Python init results (save + load paths).

The :class:`RustDiskCacheManager` mixin owns both sides of the on-disk init
cache. The *write* side hashes a binary into a stable cache key, snapshots
the post-init :class:`~angr.SimState` (registers, stack page, loader pages,
callstack frames, section patches, posix entry pointers) and pickles it under
``~/.cache/angr_rust_init/<key>.pkl``. The *read* side
(``_load_init_from_disk_cache`` and its ``_load_init_pickle`` /
``_deserialize_init_state`` / ``_apply_init_side_effects`` phases, plus the
``_state_has_user_symbolic`` guard) reverses that: unpickling the file,
rebuilding a SimState off a cached blank state, and repopulating the
manager's continuation/precomputed-register metadata.

``RustExplorationManager`` mixes this in, so every method is invoked as
``self._save_init_to_disk_cache(...)`` / ``self._load_init_from_disk_cache(...)``
with no change at the call sites. ``_deserialize_init_state`` reaches back
to ``self._get_cached_blank_state`` (which stays on the host class because
it owns the ``_blank_state_cache`` pool) and ``_apply_init_side_effects``
mutates ``self._pending_procedure_data`` / ``self._precomputed_regs``; both
resolve through the MRO.

The module-level ``_extract_*`` functions are pure snapshot helpers over a
SimState / loader — kept at module scope (not on the mixin) because they take
no manager state and are independently testable. ``_restore_posix_entry_fields``
is their read-side mirror, module-scope for the same reason.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import pickle
from typing import TYPE_CHECKING

import claripy

from angr.exploration._constants import (
    MAX_OVERLAY_SECTION_SIZE,
    PAGE_MASK,
    PAGE_SIZE,
    STACK_SIZE,
)

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)

# Disk cache versioning is split across two axes so each side can invalidate
# without forcing a full cache rebuild on the other:
#   _RUST_CACHE_VERSION:      bump when Rust engine changes affect serialized
#                             init state (memory page format, register values,
#                             callstack frame layout produced by Rust).
#   _PYTHON_METADATA_VERSION: bump when Python-side SimState attributes that
#                             we read or restore change shape (e.g., new
#                             callstack frame field, new register imports).
# Both are mixed into the cache key together with the arch name, so a key
# from a different (rust, python, arch) tuple lands at a different file and
# is treated as a miss — never deserialized into a current-format slot.
_RUST_CACHE_VERSION = 2
_PYTHON_METADATA_VERSION = 3

# Bounded-cache policy for ~/.cache/angr_rust_init. The cache key is an MD5 of
# (binary content, version axes, arch), so a version bump orphans the whole
# previous generation under names we cannot recognize by inspection — the only
# workable GC is a usage-ordered budget. Files are ranked newest-mtime-first
# (the load path touches a file on every hit, so mtime is last-use, not
# last-write) and everything past either budget is unlinked. Both budgets are
# overridable per-process; 0 on either disables that dimension, and 0 on both
# disables pruning entirely.
_DISK_CACHE_MAX_BYTES = 512 * 1024 * 1024
_DISK_CACHE_MAX_FILES = 128


def _disk_cache_budgets() -> tuple[int, int]:
    """Return the (max_bytes, max_files) budget, honoring env overrides.

    ``ANGR_RUST_INIT_CACHE_MAX_BYTES`` / ``ANGR_RUST_INIT_CACHE_MAX_FILES``
    override the module defaults. A negative or unparseable value falls back
    to the default; 0 disables that dimension.
    """
    budgets = []
    for env_name, default in (
        ("ANGR_RUST_INIT_CACHE_MAX_BYTES", _DISK_CACHE_MAX_BYTES),
        ("ANGR_RUST_INIT_CACHE_MAX_FILES", _DISK_CACHE_MAX_FILES),
    ):
        raw = os.environ.get(env_name)
        try:
            value = default if raw is None else int(raw)
        except ValueError:
            value = default
        budgets.append(default if value < 0 else value)
    return budgets[0], budgets[1]


def _extract_register_snapshot(state, arch) -> dict[str, int]:
    """Extract concrete register values from a SimState.

    Skips symbolic registers and any access errors. Returns a dict
    mapping register name → concrete int value.
    """
    registers: dict[str, int] = {}
    for reg_name in arch.register_names.values():
        try:
            val = getattr(state.regs, reg_name)
            if not val.symbolic:
                registers[reg_name] = state.solver.eval(val)
        except (AttributeError, KeyError, TypeError, ValueError):
            # cat-(a) EXPECTED CONTROL FLOW: arch lists a register the SimState
            # doesn't expose, or the read errors on a symbolic value — skip.
            pass
    return registers


# The ``state.posix`` entry-stack pointers ``SimLinux.state_entry`` sets and a
# blank_state does not: the argv/envp/auxv arrays it dumps just above SP. Each
# is a plain address, so they share one extract/restore path; ``argc`` is a
# claripy BV and gets its own (see ``_extract_posix_argc``).
_POSIX_POINTER_FIELDS = ("argv", "environ", "auxv")


def _extract_posix_pointer(state, field: str) -> int | None:
    """One of the ``_POSIX_POINTER_FIELDS`` pointers as a plain int, or None.

    ``simos/linux.py::state_entry`` stores these alongside the argv/``KEY=VALUE``
    string table it dumps onto the entry stack; they are the only handle either
    engine has on ``entry_state(args=..., env=...)`` (angr-6cp06.12 did
    ``environ`` for the Rust getenv seed bridge, angr-7tmoz the other two for
    the Python SimProcedures — e.g. ``__libc_start_main`` — that read
    ``state.posix.argv``). Returns None when the plugin never set the field, or
    when it is symbolic — the pickle must stay plain-data, and a symbolic
    pointer is unusable to those consumers anyway.
    """
    ptr = getattr(getattr(state, "posix", None), field, None)
    if ptr is None:
        return None
    try:
        if isinstance(ptr, claripy.ast.Base):
            if ptr.symbolic:
                return None
            ptr = state.solver.eval(ptr)
        return int(ptr)
    except Exception:
        # cat-(b) FALLBACK WITH LOSS: an unreadable pointer is cached as absent,
        # so a warm hit restores that field as None — the pre-angr-6cp06.12
        # behavior — rather than aborting the cache write.
        return None


def _extract_posix_argc(state) -> tuple[int, int] | None:
    """``posix.argc`` as a ``(value, width_in_bits)`` pair, or None.

    Unlike the three pointer fields, ``argc`` is a claripy BV that consumers
    call BV methods on (``simos/linux.py`` does ``state.posix.argc.sign_extend``),
    so it must come back as a BV rather than an int — hence the width travels
    with the value, keeping a caller-supplied ``entry_state(argc=BVV(n, w))`` of
    non-default width round-tripping exactly. ``state_entry`` otherwise builds
    it as ``claripy.BVV(len(args), 32)``.

    A *symbolic* argc returns None, but never actually reaches here: the BVS is
    also stored to the entry stack page, where ``_state_has_user_symbolic``
    finds it and disables the cache for that state entirely. The None arm is
    defense in depth against a symbolic argc that leaves no stack footprint.
    """
    argc = getattr(getattr(state, "posix", None), "argc", None)
    if argc is None:
        return None
    try:
        if isinstance(argc, claripy.ast.Base):
            if argc.symbolic:
                return None
            return (int(state.solver.eval(argc)), argc.size())
        return (int(argc), state.arch.bits)
    except Exception:
        # cat-(b) FALLBACK WITH LOSS: same policy as _extract_posix_pointer —
        # cache the field as absent rather than abort the write.
        return None


def _restore_posix_entry_fields(state, data: dict) -> None:
    """Put the ``posix`` entry pointers + argc back onto a rebuilt blank_state.

    ``_deserialize_init_state`` rebuilds a *blank* state, whose ``posix`` plugin
    has no argv/argc/envp/auxv — only memory and registers are restored — so a
    warm disk hit used to hand back a state on which ``getenv`` found no
    environment (angr-6cp06.12) and ``__libc_start_main`` read ``argv is None``
    (angr-7tmoz), diverging from the cold run. The arrays and string table
    themselves live in the cached ``stack_page``; only the pointers are missing.
    They are restored as claripy BVs of the same shape ``state_entry`` produced,
    not as ints, so a warm state is indistinguishable from a cold one.

    Fields absent from the pickle (older generation, or unreadable/symbolic at
    save time) are left at the blank-state default.
    """
    for field in _POSIX_POINTER_FIELDS:
        ptr = data.get(f"posix_{field}")
        if ptr is not None:
            setattr(state.posix, field, claripy.BVV(ptr, state.arch.bits))
    argc = data.get("posix_argc")
    if argc is not None:
        value, width = argc
        state.posix.argc = claripy.BVV(value, width)


def _extract_stack_page(state, page_size: int):
    """Extract the stack page at SP and the lazy stack region descriptor.

    Returns (stack_page, lazy_region) where stack_page is
    ``(page_addr, bytes)`` and lazy_region is ``(start_addr, length)``.
    Both are None if extraction fails.
    """
    try:
        sp = state.solver.eval(state.regs.sp)
        sp_page = sp & ~(page_size - 1)
        page_val = state.memory.load(sp_page, page_size, endness="Iend_BE", inspect=False, disable_actions=True)
        concrete = state.solver.eval(page_val).to_bytes(page_size, "big")
        stack_base = (sp & ~(page_size - 1)) + page_size
        stack_start = stack_base - STACK_SIZE
        return (sp_page, concrete), (stack_start, STACK_SIZE)
    except (AttributeError, TypeError, ValueError):
        # cat-(a) EXPECTED CONTROL FLOW: SP is symbolic or stack page can't
        # be loaded; caller treats (None, None) as 'no eager extraction' and
        # falls back to the lazy stack region.
        return None, None


def _extract_loader_pages(loader, page_size: int):
    """Extract concrete loader memory pages and per-object lazy regions.

    Returns (batch_pages, lazy_regions, mapped_page_addrs):
    - batch_pages: list of (page_addr, bytes, perms) tuples
    - lazy_regions: list of (start_addr, length) tuples (one per loader object)
    - mapped_page_addrs: set of page_addr ints already captured
    """
    batch_pages = []
    lazy_regions = []
    mapped_page_addrs = set()
    for obj in loader.all_objects:
        try:
            if hasattr(obj, "segments") and obj.segments:
                ranges = [
                    (s.min_addr & ~(page_size - 1), (s.max_addr + page_size) & ~(page_size - 1))
                    for s in obj.segments
                    if s.memsize > 0
                ]
            else:
                ranges = [(obj.min_addr & ~(page_size - 1), (obj.max_addr + page_size) & ~(page_size - 1))]
            for start_page, end_page in ranges:
                for page_addr in range(start_page, end_page, page_size):
                    if page_addr in mapped_page_addrs:
                        continue
                    try:
                        page_data = loader.memory.load(page_addr, page_size)
                        if page_data and len(page_data) == page_size:
                            batch_pages.append((page_addr, bytes(page_data), 7))
                            mapped_page_addrs.add(page_addr)
                    except (KeyError, TypeError, ValueError):
                        # cat-(a) EXPECTED CONTROL FLOW: per-page loader read failed (e.g.
                        # unmapped gap inside the segment range); skip and continue.
                        pass
            region_start = obj.min_addr & ~(page_size - 1)
            region_end = (obj.max_addr + page_size) & ~(page_size - 1)
            if region_end - region_start > 0:
                lazy_regions.append((region_start, region_end - region_start))
        except (AttributeError, KeyError, TypeError):
            # cat-(b) FALLBACK WITH LOSS: per-object iteration failed (loader
            # object lacks expected attributes); that object's pages won't be
            # eagerly mapped — Rust will fetch them on demand via fetch_page.
            pass
    return batch_pages, lazy_regions, mapped_page_addrs


def _extract_section_patches(state, loader) -> list:
    """Extract concrete post-init section bytes (e.g., GOT fixups).

    Returns a list of (min_addr, bytes) tuples for sections smaller
    than MAX_OVERLAY_SECTION_SIZE whose memory loads as concrete.
    """
    section_patches = []
    for obj in loader.all_objects:
        if obj.binary is None or not hasattr(obj, "sections"):
            continue
        for section in obj.sections:
            if 0 < section.memsize < MAX_OVERLAY_SECTION_SIZE:
                try:
                    val = state.memory.load(
                        section.min_addr, section.memsize, endness="Iend_BE", inspect=False, disable_actions=True
                    )
                    if not val.symbolic:
                        section_patches.append(
                            (section.min_addr, state.solver.eval(val).to_bytes(section.memsize, "big"))
                        )
                except (AttributeError, TypeError, ValueError):
                    # cat-(b) FALLBACK WITH LOSS: section overlay extraction failed;
                    # Rust sees raw loader bytes for this section without the post-init
                    # concrete patches (e.g. GOT relocations).
                    pass
    return section_patches


def _extract_extra_pages(state, page_size: int, mapped_page_addrs: set, stack_page_addr: int | None) -> list:
    """Extract non-loader pages created during init (e.g., ctype tables).

    Skips pages already captured as loader pages or as the stack page.
    Returns a list of (page_addr, bytes) tuples.
    """
    extra_pages = []
    if hasattr(state.memory, "_pages"):
        mem_page_size = getattr(state.memory, "page_size", page_size)
        for page_num in state.memory._pages:
            page_addr = page_num * mem_page_size
            if page_addr in mapped_page_addrs:
                continue
            if stack_page_addr is not None and page_addr == stack_page_addr:
                continue
            page = state.memory._pages[page_num]
            if page is None:
                continue
            try:
                page_data = page.concrete_load(0, mem_page_size)
                if any(page_data):
                    extra_pages.append((page_addr, bytes(page_data)))
            except (AttributeError, TypeError, ValueError):
                # cat-(b) FALLBACK WITH LOSS: extra (non-loader) page extraction
                # failed (e.g., concrete_load on a symbolic page). Rust will fetch
                # the page lazily via the fetch_page callback if it is accessed.
                pass
    return extra_pages


def _extract_callstack_snapshot(state):
    """Walk the SimState callstack and collect frames + continuation addrs.

    Returns (callstack_frames, continuation_addrs).
    """
    callstack_frames = []
    continuation_addrs = []
    frame = state.callstack.top if hasattr(state, "callstack") else None
    while frame is not None:
        pdata = getattr(frame, "procedure_data", None)
        frame_data = {
            "call_site_addr": frame.call_site_addr,
            "func_addr": frame.func_addr,
            "ret_addr": frame.ret_addr,
            "stack_ptr": frame.stack_ptr,
        }
        if pdata is not None and len(pdata) >= 5:
            try:
                continuation_addrs.append(int(pdata[4]))
            except (TypeError, ValueError):
                # cat-(a) EXPECTED CONTROL FLOW: continuation addr in procedure_data
                # is not int-castable (symbolic). Skip — saves no continuation, the
                # normal procedure-data restore path handles it later.
                pass
        callstack_frames.append(frame_data)
        frame = getattr(frame, "next", None)
    return callstack_frames, continuation_addrs


class RustDiskCacheManager:
    """Mixin owning the disk-cache *save* and *load* paths for Python init.

    Expects the host class to provide ``self._project``, the metadata dicts
    ``self._pending_procedure_data`` / ``self._precomputed_regs`` (mutated by
    the load path), and ``self._get_cached_blank_state`` (the blank-state
    pool stays on the host). The class-level ``_disk_key_cache`` memo dict is
    declared here so the mixin is usable standalone. See the module docstring
    for the save/load split.
    """

    # Class-level cache for disk cache keys (MD5 of binary content).
    # Keyed by (binary_path, arch_name, loader_digest) so cross-arch lookups and
    # differing shared-library sets don't collide.
    # Avoids re-hashing the same file on every RustExplorationManager construction.
    _disk_key_cache: dict[tuple[str, str, str], str] = {}

    @staticmethod
    def _disk_cache_dir() -> str:
        """Return the disk cache directory for init state data."""
        return os.path.join(os.path.expanduser("~"), ".cache", "angr_rust_init")

    @staticmethod
    def _loader_identity_digest(loader) -> str:
        """Digest the identity of every loaded object, not just the main binary.

        The cached payload embeds pages and post-relocation section patches for
        *all* ``loader.all_objects`` (see ``_extract_loader_pages`` /
        ``_extract_section_patches``), so keying only on the main binary lets a
        shared-library upgrade, a changed ``auto_load_libs`` / ``force_load_libs``
        setting, or a different CLE base layout collide with a stale entry and
        replay old library bytes and GOT fixups into Rust memory (angr-uv7z5).

        Each object contributes its binary path, on-disk size + mtime, and
        mapped base. Stat metadata rather than content: shared libs are large
        and rarely edited in place, and a libc upgrade changes the path's
        size/mtime (or the path itself). Synthetic CLE objects (``cle##...``,
        externs, kernel) have no readable file; they contribute their class name
        and base, which is what distinguishes their presence from absence.
        Objects are sorted so loader ordering jitter doesn't change the digest.
        """
        parts = []
        for obj in getattr(loader, "all_objects", []) or []:
            path = getattr(obj, "binary", None) or ""
            base = getattr(obj, "mapped_base", 0) or 0
            try:
                st = os.stat(path)
                ident = f"{st.st_size}:{st.st_mtime_ns}"
            except OSError:
                # cat-(a) EXPECTED CONTROL FLOW: synthetic/backing-less object
                # (cle##externs, kernel, or a path that no longer exists) —
                # fall back to the class name, which still separates a loader
                # that has this object from one that does not.
                ident = type(obj).__name__
            parts.append(f"{path}|{ident}|{base:x}")
        parts.sort()
        return hashlib.md5("\n".join(parts).encode()).hexdigest()[:16]

    @classmethod
    def _disk_cache_key(cls, binary_path: str, arch_name: str = "", loader_digest: str = "") -> str:
        """Compute a cache key from binary content hash, version axes, and arch.

        Combines (binary_hash, _RUST_CACHE_VERSION, _PYTHON_METADATA_VERSION,
        arch_name, loader_digest) so a change on any axis lands at a different
        filename and treats stale entries as misses. ``loader_digest`` covers
        the shared libraries baked into the payload — see
        ``_loader_identity_digest``. Results are memoized per
        (binary_path, arch_name, loader_digest) to avoid re-hashing the same
        file on every RustExplorationManager construction (~0.5ms for 100KB
        binary).
        """
        memo_key = (binary_path, arch_name, loader_digest)
        cached = cls._disk_key_cache.get(memo_key)
        if cached is not None:
            return cached
        try:
            h = hashlib.md5()
            # Mix all version dimensions into the hash so any one bumping
            # produces a fresh key without colliding with old cache files.
            h.update(f"r{_RUST_CACHE_VERSION}:p{_PYTHON_METADATA_VERSION}:a{arch_name}:l{loader_digest}:".encode())
            with open(binary_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            result = h.hexdigest()
            cls._disk_key_cache[memo_key] = result
            return result
        except OSError:
            # cat-(b) FALLBACK WITH LOSS: cannot read binary for hash; disk
            # cache disabled for this binary (empty key returns ''-keyed nothing).
            return ""

    @staticmethod
    def _prune_disk_cache(cache_dir: str, keep_key: str = "") -> int:
        """Evict cache files past the size/count budget, least-recently-used first.

        Ranks ``<cache_dir>/*.pkl`` by mtime (newest first) and unlinks
        everything past ``_disk_cache_budgets()``. ``keep_key`` is never
        evicted — the just-written entry stays even if it alone exceeds the
        byte budget, so a single oversized binary degrades to a one-entry
        cache instead of an empty one. Returns the number of files removed.
        Best-effort: any OSError is swallowed like the write path.
        """
        max_bytes, max_files = _disk_cache_budgets()
        if max_bytes <= 0 and max_files <= 0:
            return 0
        try:
            entries = []
            with os.scandir(cache_dir) as it:
                for entry in it:
                    if not entry.name.endswith(".pkl") or not entry.is_file():
                        continue
                    try:
                        st = entry.stat()
                    except OSError:
                        continue
                    entries.append((st.st_mtime, st.st_size, entry.name, entry.path))
        except OSError:
            return 0

        keep_name = f"{keep_key}.pkl" if keep_key else None
        # Newest first; ties broken by name so the order is deterministic.
        entries.sort(key=lambda e: (-e[0], e[2]))

        removed = 0
        total_bytes = 0
        kept = 0
        # Charge the retained entry against the budget up front, so the
        # remaining files are ranked against what is actually left rather
        # than overshooting by its size wherever it lands in mtime order.
        for _mtime, size, name, _path in entries:
            if name == keep_name:
                total_bytes += size
                kept += 1
                break

        for _mtime, size, name, path in entries:
            if name == keep_name:
                continue
            over_bytes = max_bytes > 0 and total_bytes + size > max_bytes
            over_files = max_files > 0 and kept + 1 > max_files
            if over_bytes or over_files:
                try:
                    os.unlink(path)
                except OSError:
                    # cat-(b) FALLBACK WITH LOSS: eviction failed (races with a
                    # concurrent run, read-only dir). The cache stays over
                    # budget until the next save retries.
                    continue
                removed += 1
                continue
            total_bytes += size
            kept += 1
        if removed:
            l.debug(f"Pruned {removed} init cache file(s) from {cache_dir} ({kept} kept, {total_bytes} bytes)")
        return removed

    def _save_init_to_disk_cache(self, cache_key: str, state: angr.SimState):
        """Save essential post-init state data to disk cache.

        Stores: addr, registers, stack page, continuation addrs, and loader
        memory pages + lazy regions for fast memory sync on warm runs.
        """
        try:
            cache_dir = self._disk_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)

            page_size = PAGE_SIZE
            loader = self._project.loader

            stack_page, stack_lazy_region = _extract_stack_page(state, page_size)
            batch_pages, loader_lazy_regions, mapped_page_addrs = _extract_loader_pages(loader, page_size)

            lazy_regions = []
            if stack_lazy_region is not None:
                lazy_regions.append(stack_lazy_region)
            lazy_regions.extend(loader_lazy_regions)

            callstack_frames, continuation_addrs = _extract_callstack_snapshot(state)

            data = {
                "addr": state.addr,
                "registers": _extract_register_snapshot(state, self._project.arch),
                "stack_page": stack_page,
                "continuation_addrs": continuation_addrs,
                "batch_pages": batch_pages,
                "lazy_regions": lazy_regions,
                "section_patches": _extract_section_patches(state, loader),
                "extra_pages": _extract_extra_pages(
                    state, page_size, mapped_page_addrs, stack_page[0] if stack_page is not None else None
                ),
                "callstack_frames": callstack_frames,
                # The entry-stack pointers a rebuilt blank_state lacks — see
                # `_restore_posix_entry_fields` for what breaks without them.
                # The arrays and string table they point at already ride along
                # inside `stack_page`.
                **{f"posix_{field}": _extract_posix_pointer(state, field) for field in _POSIX_POINTER_FIELDS},
                "posix_argc": _extract_posix_argc(state),
            }

            cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
            with open(cache_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            l.debug(f"Saved init cache to {cache_path} ({os.path.getsize(cache_path)} bytes, {len(batch_pages)} pages)")
            self._prune_disk_cache(cache_dir, keep_key=cache_key)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: disk cache write failed (e.g., OOM,
            # permission, ENOSPC). Run continues without persistent caching.
            l.debug(f"Failed to save disk cache: {e}")

    # ------------------------------------------------------------------
    # Load path: file -> SimState + manager metadata.
    #
    # _load_init_from_disk_cache orchestrates three independently-testable
    # phases (_load_init_pickle / _deserialize_init_state /
    # _apply_init_side_effects). _state_has_user_symbolic is the guard the
    # caller consults BEFORE the load to avoid clobbering a user's symbolic
    # state with a cached blank_state (see angr-g9hy).
    # ------------------------------------------------------------------

    @staticmethod
    def _state_has_user_symbolic(state) -> bool:
        """Check if state has user-created symbolic data in memory or registers.

        Detects symbolic argv, symbolic input buffers, ``state.regs.a0 =
        BVS(...)``-style register mutations, etc. by scanning:
        1. The stack page near SP for BVS variables that aren't unconstrained fill.
        2. All memory pages with symbolic_data for user-created variables
           (e.g., state.memory.store(addr, BVS(...))).
        3. Architectural registers for user-set symbolic values
           (e.g., state.regs.a0 = BVS(...)). Without this, the disk init
           cache silently replaces the user's state with a cached blank_state,
           losing the user's symbolic register mutations — see angr-g9hy.
        4. The filesystem for user-inserted SimFiles (e.g.,
           ``state.fs.insert(name, SimFile(...))``). The init-cache state is
           built from a plain blank_state with an empty filesystem, and
           ``_apply_state_metadata`` copies constraints/options/globals but
           NOT the ``fs`` plugin. Without this guard a cache hit hands the
           callback path a state whose ``state.fs`` is empty, so ``fopen``
           falls into the ALL_FILES_EXIST branch and mints a fresh
           symbolic-size SimFile. ``ftell`` then returns an unbounded
           symbolic file size and ``fread`` over-reads, exploding into tens
           of thousands of fill bytes (asisctffinals2015_license TIMEOUT;
           see angr-ql3ja).
        """
        # 4. User-inserted SimFiles can't round-trip through the cached
        # blank_state — disable caching if the filesystem is non-empty.
        try:
            files = getattr(state.fs, "_files", None)
            if files:
                return True
        except AttributeError:
            # cat-(a) EXPECTED CONTROL FLOW: state has no fs plugin or it
            # lacks ``_files``; treat as no user filesystem data.
            pass
        _user_prefixes = ("mem_", "reg_", "unconstrained")
        try:
            sp = state.solver.eval(state.regs.sp)
            sp_page = sp & ~PAGE_MASK
            page_data = state.memory.load(sp_page, PAGE_SIZE, endness="Iend_BE", inspect=False, disable_actions=True)
            if page_data.symbolic:
                for name in page_data.variables:
                    if not name.startswith(_user_prefixes):
                        return True
        except (AttributeError, KeyError, TypeError):
            # cat-(a) EXPECTED CONTROL FLOW: stack-page probe for user symbolic
            # data failed; fall through to the per-page bitmap scan below.
            pass
        # Also check non-stack pages with symbolic_data (e.g., user stores
        # a BVS into .data/.bss segment via state.memory.store()).
        try:
            mem = state.memory
            for page_num, page in mem._pages.items():
                sd = getattr(page, "symbolic_data", None)
                if not sd:
                    continue
                # Page has symbolic data — check if any variable is user-created
                for offset, bv in sd.items():
                    if hasattr(bv, "variables"):
                        for name in bv.variables:
                            if not name.startswith(_user_prefixes):
                                return True
        except (AttributeError, KeyError, TypeError):
            # cat-(a) EXPECTED CONTROL FLOW: per-page symbolic-data scan hit
            # missing attribute; conclude no user symbolic data, return False.
            pass
        # Check the register file for user-set symbolic values. blank_state's
        # default symbol-fill is lazy (BVS allocated only on first read), so
        # uninitialized registers do NOT appear in `symbolic_data`. The
        # values that DO appear are either initialization writes or user
        # mutations like `state.regs.a0 = BVS(...)`. A non-default variable
        # prefix on any of these is treated as user-supplied — the cache
        # can't round-trip it.
        try:
            regs_mem = state.registers
            for _page_num, page in regs_mem._pages.items():
                sd = getattr(page, "symbolic_data", None)
                if not sd:
                    continue
                for _offset, bv in sd.items():
                    if hasattr(bv, "variables"):
                        for name in bv.variables:
                            if not name.startswith(_user_prefixes):
                                return True
        except (AttributeError, KeyError, TypeError):
            # cat-(a) EXPECTED CONTROL FLOW: registers storage lacks _pages
            # (non-DefaultMemory plugin?); skip — caller treats False as "no
            # user symbolic registers detected".
            pass
        return False

    @staticmethod
    def _concrete_input_digest(state) -> str:
        """Short hex digest of the concrete argv/env surface on the entry stack.

        The init caches are keyed only by binary hash + arch, and
        ``_state_has_user_symbolic`` gates only on *symbolic* data — concrete
        argv/env strings pass clean. Two runs of the same binary that differ
        only in CONCRETE args or environment would therefore collide, and a
        warm hit silently replays the first run's argv/env (angr-gxaht).

        ``SimLinux.state_entry`` writes argc, the argv/envp pointer arrays, and
        the arg + env string bytes onto the stack just above SP
        (``table.dump(state, sp - 16)``), so those inputs live in the stack
        page containing SP. Digesting that page's concrete bytes makes the
        cache key input-sensitive: same args -> same digest (stable across
        processes, since ``eval(..., cast_to=bytes)`` resolves default-fill
        symbolic bytes to their deterministic minimum), different args ->
        different digest -> cache miss.

        Callers only reach this after ``_state_has_user_symbolic`` returns
        False, so the page holds no user symbolic data. Returns '' on any
        failure — the digest then simply doesn't differentiate, matching the
        pre-fix behavior rather than crashing init.
        """
        try:
            sp = state.solver.eval(state.regs.sp)
            sp_page = sp & ~PAGE_MASK
            page = state.memory.load(sp_page, PAGE_SIZE, endness="Iend_BE", inspect=False, disable_actions=True)
            raw = state.solver.eval(page, cast_to=bytes)
            return hashlib.md5(raw).hexdigest()[:16]
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: best-effort digest only. SP may be
            # unresolvable, the page load may fail, or the state's constraint
            # set may be unsat (SimUnsatError) — e.g. an error-recovery state
            # deliberately seeded unsat. Any of these fall back to a
            # non-differentiating (empty) digest, matching the pre-fix behavior
            # rather than aborting manager construction.
            return ""

    def _load_init_pickle(self, cache_key: str):
        """Read and unpickle the disk cache file. Returns the raw data dict
        on hit, None on miss or read failure. Pure I/O — no state mutation."""
        try:
            cache_path = os.path.join(self._disk_cache_dir(), f"{cache_key}.pkl")
            if not os.path.exists(cache_path):
                return None
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            # Touch on hit so _prune_disk_cache's mtime ranking is last-USE,
            # not last-write: a hot binary rebuilt rarely must not be evicted
            # ahead of a one-shot binary cached yesterday.
            with contextlib.suppress(OSError):
                os.utime(cache_path, None)
            return data
        except (OSError, pickle.UnpicklingError, EOFError) as e:
            # cat-(b) FALLBACK WITH LOSS: disk cache read failed / corrupt;
            # treated as a cache miss. Caller pays full Python init.
            l.debug(f"Disk cache read failed: {e}")
            return None

    def _deserialize_init_state(self, data: dict):
        """Build a SimState + memory_cache from a cache data dict. Pure
        function over `self._project` and the cached blank-state pool — no
        mutation of manager-owned metadata dicts (those happen in
        `_apply_init_side_effects`)."""
        state = self._get_cached_blank_state(data["addr"])
        for reg_name, val in data["registers"].items():
            try:
                setattr(state.regs, reg_name, val)
            except (AttributeError, TypeError, ValueError):
                # cat-(b) FALLBACK WITH LOSS: per-register restore failed; that
                # register stays at the blank-state default — may diverge from the
                # pre-cached value.
                pass
        if data.get("stack_page"):
            sp_page, page_bytes = data["stack_page"]
            state.memory.store(sp_page, claripy.BVV(page_bytes), endness="Iend_BE", inspect=False, disable_actions=True)

        # Restore extra memory pages created during init
        # (e.g., ctype tables at 0xc0000000 written by SimProcedures)
        for page_addr, page_bytes in data.get("extra_pages", []):
            try:
                state.memory.store(
                    page_addr, claripy.BVV(page_bytes), endness="Iend_BE", inspect=False, disable_actions=True
                )
            except (TypeError, ValueError):
                # cat-(b) FALLBACK WITH LOSS: extra page restore failed; that page
                # stays blank and Rust sees concrete zeros there.
                pass

        # Restore posix.argv/argc/environ/auxv (angr-6cp06.12, angr-7tmoz).
        # Absent from pre-p3 pickles, which the version bump already orphans.
        _restore_posix_entry_fields(state, data)

        # Restore callstack frames
        callstack_frames = data.get("callstack_frames", [])
        if callstack_frames and len(callstack_frames) > 1:
            # The first frame is the top of the callstack (main's frame).
            # We need to push frames from bottom to top.
            try:
                from angr.state_plugins.callstack import CallStack  # noqa: F401

                cs = state.callstack
                for frame_data in reversed(callstack_frames[:-1]):
                    # Skip the bottom sentinel frame (all zeros)
                    if frame_data["call_site_addr"] == 0 and frame_data["func_addr"] == 0:
                        continue
                    cs.call(
                        frame_data["call_site_addr"],
                        frame_data["func_addr"],
                        return_address=frame_data["ret_addr"],
                        stack_pointer=frame_data["stack_ptr"],
                    )
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: callstack restoration failed; the
                # state has a top-of-stack frame but ret-chain may be incomplete.
                # Debug-logs.
                l.debug(f"Failed to restore callstack: {e}")

        mem_cache = None
        if data.get("batch_pages") is not None:
            mem_cache = {
                "batch_pages": data["batch_pages"],
                "lazy_regions": data.get("lazy_regions", []),
                "section_patches": data.get("section_patches", []),
                "stack_page": data.get("stack_page"),
            }
        return state, mem_cache

    def _apply_init_side_effects(self, data: dict) -> None:
        """Populate manager-owned metadata from a cache data dict:
        `_pending_procedure_data` (continuation slots) and `_precomputed_regs`
        (fast Rust register sync). Separated from deserialization so the
        SimState construction can be tested in isolation."""
        for cont_addr in data.get("continuation_addrs", []):
            if cont_addr > 0:
                self._pending_procedure_data.setdefault(cont_addr, None)
        self._precomputed_regs = data.get("registers", {})

    def _load_init_from_disk_cache(self, cache_key: str):
        """Load post-init state from disk cache.

        Returns (SimState, memory_cache_data) on hit, (None, None) on miss.
        memory_cache_data contains pre-computed loader pages and lazy regions
        for fast memory sync.

        Phases (each independently testable):
        1. `_load_init_pickle` — pure I/O.
        2. `_deserialize_init_state` — pure SimState construction.
        3. `_apply_init_side_effects` — manager-owned metadata mutation.
        """
        data = self._load_init_pickle(cache_key)
        if data is None:
            return None, None
        try:
            state, mem_cache = self._deserialize_init_state(data)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: pickle deserialization succeeded but
            # state construction failed; treat as cache miss. Debug-logs.
            l.debug(f"Disk cache deserialization failed: {e}")
            return None, None
        self._apply_init_side_effects(data)
        l.info(f"Disk cache hit: restored state at 0x{data['addr']:x}")
        return state, mem_cache
