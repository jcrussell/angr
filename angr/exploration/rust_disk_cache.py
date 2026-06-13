"""Persistent disk cache for Python init results (save path).

The :class:`RustDiskCacheManager` mixin owns the *write* side of the
on-disk init cache: hashing a binary into a stable cache key, snapshotting
the post-init :class:`~angr.SimState` (registers, stack page, loader pages,
callstack frames, section patches) and pickling it under
``~/.cache/angr_rust_init/<key>.pkl``.

``RustExplorationManager`` mixes this in, so the methods are invoked as
``self._save_init_to_disk_cache(...)`` / ``self._disk_cache_key(...)`` with
no change at the call sites. The *read* side (``_load_init_from_disk_cache``
and its deserialization helpers) still lives in ``rust_manager.py`` and is
slated to move into this same mixin (bd angr-wqao.2); the shared
``_disk_cache_dir`` / ``_disk_cache_key`` helpers live here so both halves
can reach them.

The module-level ``_extract_*`` functions are pure snapshot helpers over a
SimState / loader — kept at module scope (not on the mixin) because they take
no manager state and are independently testable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from angr.exploration._constants import (
    PAGE_SIZE,
    STACK_SIZE,
    MAX_OVERLAY_SECTION_SIZE,
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
_PYTHON_METADATA_VERSION = 1


def _extract_register_snapshot(state, arch) -> Dict[str, int]:
    """Extract concrete register values from a SimState.

    Skips symbolic registers and any access errors. Returns a dict
    mapping register name → concrete int value.
    """
    registers: Dict[str, int] = {}
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


def _extract_stack_page(state, page_size: int):
    """Extract the stack page at SP and the lazy stack region descriptor.

    Returns (stack_page, lazy_region) where stack_page is
    ``(page_addr, bytes)`` and lazy_region is ``(start_addr, length)``.
    Both are None if extraction fails.
    """
    try:
        sp = state.solver.eval(state.regs.sp)
        sp_page = sp & ~(page_size - 1)
        page_val = state.memory.load(
            sp_page, page_size, endness='Iend_BE',
            inspect=False, disable_actions=True)
        concrete = state.solver.eval(page_val).to_bytes(page_size, 'big')
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
            if hasattr(obj, 'segments') and obj.segments:
                ranges = [(s.min_addr & ~(page_size - 1),
                           (s.max_addr + page_size) & ~(page_size - 1))
                          for s in obj.segments if s.memsize > 0]
            else:
                ranges = [(obj.min_addr & ~(page_size - 1),
                           (obj.max_addr + page_size) & ~(page_size - 1))]
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
        if obj.binary is None or not hasattr(obj, 'sections'):
            continue
        for section in obj.sections:
            if 0 < section.memsize < MAX_OVERLAY_SECTION_SIZE:
                try:
                    val = state.memory.load(
                        section.min_addr, section.memsize,
                        endness='Iend_BE', inspect=False, disable_actions=True)
                    if not val.symbolic:
                        section_patches.append(
                            (section.min_addr,
                             state.solver.eval(val).to_bytes(section.memsize, 'big')))
                except (AttributeError, TypeError, ValueError):
                    # cat-(b) FALLBACK WITH LOSS: section overlay extraction failed;
                    # Rust sees raw loader bytes for this section without the post-init
                    # concrete patches (e.g. GOT relocations).
                    pass
    return section_patches


def _extract_extra_pages(state, page_size: int, mapped_page_addrs: set,
                         stack_page_addr: Optional[int]) -> list:
    """Extract non-loader pages created during init (e.g., ctype tables).

    Skips pages already captured as loader pages or as the stack page.
    Returns a list of (page_addr, bytes) tuples.
    """
    extra_pages = []
    if hasattr(state.memory, '_pages'):
        mem_page_size = getattr(state.memory, 'page_size', page_size)
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
    frame = state.callstack.top if hasattr(state, 'callstack') else None
    while frame is not None:
        pdata = getattr(frame, 'procedure_data', None)
        frame_data = {
            'call_site_addr': frame.call_site_addr,
            'func_addr': frame.func_addr,
            'ret_addr': frame.ret_addr,
            'stack_ptr': frame.stack_ptr,
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
        frame = getattr(frame, 'next', None)
    return callstack_frames, continuation_addrs


class RustDiskCacheManager:
    """Mixin owning the disk-cache *save* path for Python init results.

    Expects the host class to provide ``self._project`` and the class-level
    ``_disk_key_cache`` memo dict (declared here so the mixin is usable
    standalone). See the module docstring for the save/load split.
    """

    # Class-level cache for disk cache keys (MD5 of binary content).
    # Keyed by (binary_path, arch_name) so cross-arch lookups don't collide.
    # Avoids re-hashing the same file on every RustExplorationManager construction.
    _disk_key_cache: Dict[Tuple[str, str], str] = {}

    @staticmethod
    def _disk_cache_dir() -> str:
        """Return the disk cache directory for init state data."""
        return os.path.join(os.path.expanduser("~"), ".cache", "angr_rust_init")

    @classmethod
    def _disk_cache_key(cls, binary_path: str, arch_name: str = "") -> str:
        """Compute a cache key from binary content hash, version axes, and arch.

        Combines (binary_hash, _RUST_CACHE_VERSION, _PYTHON_METADATA_VERSION,
        arch_name) so a change on any axis lands at a different filename and
        treats stale entries as misses. Results are memoized per
        (binary_path, arch_name) to avoid re-hashing the same file on every
        RustExplorationManager construction (~0.5ms for 100KB binary).
        """
        memo_key = (binary_path, arch_name)
        cached = cls._disk_key_cache.get(memo_key)
        if cached is not None:
            return cached
        try:
            h = hashlib.md5()
            # Mix all version dimensions into the hash so any one bumping
            # produces a fresh key without colliding with old cache files.
            h.update(
                f"r{_RUST_CACHE_VERSION}:p{_PYTHON_METADATA_VERSION}:"
                f"a{arch_name}:".encode()
            )
            with open(binary_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    h.update(chunk)
            result = h.hexdigest()
            cls._disk_key_cache[memo_key] = result
            return result
        except OSError:
            # cat-(b) FALLBACK WITH LOSS: cannot read binary for hash; disk
            # cache disabled for this binary (empty key returns ''-keyed nothing).
            return ""

    def _save_init_to_disk_cache(self, cache_key: str, state: "angr.SimState"):
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
            batch_pages, loader_lazy_regions, mapped_page_addrs = _extract_loader_pages(
                loader, page_size)

            lazy_regions = []
            if stack_lazy_region is not None:
                lazy_regions.append(stack_lazy_region)
            lazy_regions.extend(loader_lazy_regions)

            callstack_frames, continuation_addrs = _extract_callstack_snapshot(state)

            data = {
                'addr': state.addr,
                'registers': _extract_register_snapshot(state, self._project.arch),
                'stack_page': stack_page,
                'continuation_addrs': continuation_addrs,
                'batch_pages': batch_pages,
                'lazy_regions': lazy_regions,
                'section_patches': _extract_section_patches(state, loader),
                'extra_pages': _extract_extra_pages(
                    state, page_size, mapped_page_addrs,
                    stack_page[0] if stack_page is not None else None),
                'callstack_frames': callstack_frames,
            }

            cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            l.debug(f"Saved init cache to {cache_path} "
                    f"({os.path.getsize(cache_path)} bytes, "
                    f"{len(batch_pages)} pages)")
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: disk cache write failed (e.g., OOM,
            # permission, ENOSPC). Run continues without persistent caching.
            l.debug(f"Failed to save disk cache: {e}")
