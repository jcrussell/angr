"""Mixin for State synchronization between Python and Rust."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING

from sortedcontainers import SortedDict

from angr import claripy
from angr.rustylib.vex_engine import arch_register_info
from angr.storage.memory_mixins.paged_memory.pages import UltraPage
from angr.storage.memory_object import SimMemoryObject

from ._constants import MAX_OVERLAY_SECTION_SIZE, PAGE_MASK, PAGE_SIZE, STACK_SIZE

# Precomputed all-zero buffers for _sync_extra_python_pages fast-path
# zero-page detection (memcmp-based instead of any() byte iteration, which
# dominates per-page cost in mma_howtouse-style workloads; angr-b58a).
_ZERO_PAGE_BA = bytearray(PAGE_SIZE)
_ZERO_PAGE_BYTES = bytes(PAGE_SIZE)


def _ultrapage_bytes(page, *, writing: bool = False):
    """An UltraPage's own page-sized concrete backing store, or ``None`` when it has none the fast paths can use.

    UltraPage allocates ``concrete_data`` on first use, and until then every byte reads as zero. A read-only caller
    gets the shared immutable all-zero buffer for such a page; ``writing=True`` allocates the real store so the caller
    can write into it.
    """
    if not isinstance(page, UltraPage) or page.symbolic_bitmap.size != PAGE_SIZE:
        return None
    data = page.concrete_data
    if data is None:
        return page._concrete() if writing else _ZERO_PAGE_BYTES  # pylint:disable=protected-access
    # A page still backed by the loader holds a (Not)memoryview into the shared backer: not ours to read as a plain
    # buffer, nor to write through. Those take the generic path.
    if not isinstance(data, bytearray) or len(data) != PAGE_SIZE:
        return None
    return data


if TYPE_CHECKING:
    import angr
    from angr.rustylib.vex_engine import RustSimState as _RustSimState

l = logging.getLogger(name=__name__)
_DBG = l.isEnabledFor(logging.DEBUG)

_RUST_REGISTER_TABLES: dict[str, dict | None] = {}


def rust_register_table(arch) -> dict | None:
    """The engine's register table for ``arch``, cached per arch name.

    ``arch_register_info`` (``native/angr/src/engine.rs``) plus ``offsets``,
    ``{name: (offset, size)}`` over every name the engine resolves,
    ``sync_map``, the same for the ``sync`` registers, and ``return_regs``
    (see ``RustStateSyncMixin._get_reg_map_and_return_regs``). ``None``
    for an arch the engine does not implement. Rust's ``arch/*.rs`` tables
    are the source of truth for what crosses the Python/Rust boundary
    (angr-mx00p.34); ``tests/engines/rust/test_arch_offset_parity.py`` checks
    them against archinfo's VEX guest layout.
    """
    try:
        return _RUST_REGISTER_TABLES[arch.name]
    except KeyError:
        pass
    info = arch_register_info(arch.name)
    if info is not None:
        info = dict(info)
        offsets = {name: (offset, size) for name, offset, size, _ in info["registers"]}
        info["offsets"] = offsets
        info["sync_map"] = {name: offsets[name] for name in info["sync"]}
        # MIPS also returns in v1 (the high half of a 64-bit result), which a
        # proc may write as part of its return value (angr-hrcds).
        mips_hi = {"v1"} if arch.name in ("MIPS32", "MIPS64") else set()
        info["return_regs"] = frozenset({info["cc_ret"]} | mips_hi)
    _RUST_REGISTER_TABLES[arch.name] = info
    return info


class RustStateSyncMixin:
    """State synchronization between Python and Rust

    This mixin expects the host class to have the standard
    RustExplorationManager attributes (self._rust_mgr, self._project, etc.).
    """

    def _sync_hooks_before_step(self):
        """Sync dynamically created hooks (like continuations) before stepping.

        SimProcedures can create continuation hooks via self.call() which are
        added to the project dynamically. This method ensures those hooks are
        registered with Rust before exploration continues.

        This is critical for __libc_start_main and other procedures that use
        continuations to chain function calls (init -> main -> fini).
        """
        self._stats_hook_sync_calls += 1

        if not hasattr(self._project, "_sim_procedures"):
            self._stats_hook_sync_skips += 1
            return

        # Fast path: identical key sets -> no hooks added or removed.
        # `set == dict_keys` compares as sets without materializing a copy of
        # the keys, so this stays cheap for small hook tables. Replaces the old
        # len-equality fast path, which missed equal-count remove+add swaps and
        # never detected removals (proj.unhook on a live manager) at all
        # (angr-969g).
        sim_procedures = self._project._sim_procedures
        registered = self._registered_hooks
        keys = sim_procedures.keys()
        if registered == keys:
            self._stats_hook_sync_skips += 1
            return

        # Diff registered vs current so both additions (new continuations) and
        # removals (proj.unhook) propagate to Rust.
        removed = [addr for addr in registered if addr not in sim_procedures]
        if removed:
            self._rust_mgr.unregister_simprocedures(removed)
            registered.difference_update(removed)
            if _DBG:
                l.debug(f"Unregistered {len(removed)} stale hooks from Rust")

        procs = []
        for addr in keys:
            if addr in registered:
                continue
            proc = sim_procedures[addr]
            name = proc.__class__.__name__ if hasattr(proc, "__class__") else str(proc)
            num_args = getattr(proc, "num_args", 0) or 0
            no_return = getattr(proc, "NO_RET", False)
            procs.append((addr, name, num_args, no_return))
            registered.add(addr)
            if _DBG:
                l.debug(f"Syncing dynamically created hook at 0x{addr:x}: {name}")

        if procs:
            self._rust_mgr.register_simprocedures(procs)
            if _DBG:
                l.debug(f"Synced {len(procs)} dynamically created hooks")

    @staticmethod
    def _supported_register_names(arch) -> list:
        """Registers that the Rust engine models for `arch`.

        The engine's exported register list (``rust_register_table``), the
        filter applied to every register dict that crosses the FFI boundary.
        archinfo defines many registers (cr0..8, ymm0..15, fs_seg, cmstart, ...)
        the Rust engine does not model; the Rust list excludes them, so
        `set_registers_bulk` will not raise ValueError on the names returned
        here.
        """
        table = rust_register_table(arch)
        return list(table["exported"]) if table is not None else []

    def _sync_registers_to_rust(self, angr_state: angr.SimState, rust_state: _RustSimState):
        """Sync registers from angr state to Rust state."""
        arch = angr_state.arch
        reg_names = self._supported_register_names(arch)
        regs = angr_state.regs

        # Build dict of all register values, then send in single FFI call.
        bulk_regs = {}
        for reg_name in reg_names:
            try:
                reg_val = getattr(regs, reg_name)
                if reg_val.op == "BVV":
                    bulk_regs[reg_name] = reg_val.args[0]
                elif not reg_val.symbolic:
                    bulk_regs[reg_name] = angr_state.solver.eval(reg_val)
                else:
                    # Import the symbolic register as a claripy AST. If
                    # conversion fails, the register is missing on the Rust
                    # side and any read will see an uninitialized BVS — wrong
                    # value, not a crash. Log loudly.
                    try:
                        # Layer 2 (angr-21vi5): route the full claripy AST
                        # through claripy_to_rustbv so leaf symbols (e.g. the
                        # ecx in `state.regs.ecx = BVS('ecx')`) intern into the
                        # shared symbol cache and round-trip back to the user's
                        # symbol on export.
                        rust_state.set_register_symbolic_ast(reg_name, reg_val)
                    except Exception as e:
                        # cat-(c) WRONG-ANSWER RISK: symbolic register not
                        # synced; Rust will see stale/uninit BVS for this
                        # register. Already logged at warn (angr-i1wg).
                        l.warning(
                            "Symbolic register %s not synced to Rust "
                            "(Z3 conversion failed: %s) — Rust will "
                            "see stale/uninit value",
                            reg_name,
                            e,
                        )
            except AttributeError:
                # cat-(a) EXPECTED CONTROL FLOW: arch defines register name
                # but state doesn't expose it (rare, but harmless to skip).
                if _DBG:
                    l.debug("Register %s not on state, skipping", reg_name)
            except Exception as e:
                # cat-(c) WRONG-ANSWER RISK: unexpected failure (e.g.
                # solver.eval on concrete value raises) — register won't
                # be synced; Rust may diverge from Python.
                l.warning("Failed to sync register %s to Rust: %s", reg_name, e)
        if bulk_regs:
            rust_state.set_registers_bulk(bulk_regs)

    def _sync_memory_to_rust(self, angr_state: angr.SimState, rust_state: _RustSimState):
        """Sync memory from angr state to Rust state.

        Strategy: map pages for each loaded segment (not the entire address
        space). Uses the Python state's memory for relocations/initialized data.
        """
        page_size = PAGE_SIZE
        arch = self._project.arch

        symbolic_pages = self._find_user_symbolic_pages(angr_state, page_size)
        if symbolic_pages:
            l.debug(f"Skipping {len(symbolic_pages)} pages with symbolic data during memory sync")

        mapped_page_addrs, pages_mapped = self._map_loader_pages(rust_state, symbolic_pages, page_size)
        self._overlay_relocated_sections(angr_state, rust_state)
        l.debug(f"Pre-populated {pages_mapped} pages from loaded objects")

        self._overlay_python_state_pages(angr_state, rust_state, mapped_page_addrs, symbolic_pages, page_size)
        self._add_loader_lazy_regions(rust_state, page_size)

        sp_page, stack_start, stack_base = self._setup_stack_region(angr_state, rust_state, arch, page_size)

        symbolic_regions: list = []  # (addr, claripy_ast) pairs to import
        self._sync_stack_page(angr_state, rust_state, sp_page, page_size, arch, symbolic_regions)
        self._sync_extra_python_pages(
            angr_state, rust_state, mapped_page_addrs, symbolic_pages, sp_page, stack_start, stack_base, page_size
        )
        self._scan_user_symbolic_pages(angr_state, stack_start, stack_base, page_size, symbolic_regions)

        if symbolic_regions:
            self._pending_symbolic_imports = symbolic_regions
            l.debug(f"Found {len(symbolic_regions)} symbolic regions for import")

    def _find_user_symbolic_pages(self, angr_state: angr.SimState, page_size: int) -> set:
        """Find pages that contain user-written symbolic data.

        These pages are NOT pre-populated with concrete loader data, so Rust
        falls back to the Python memory_load callback (which returns the
        symbolic AST and enables symbolic forking on comparisons).
        """
        symbolic_pages: set = set()
        if hasattr(angr_state.memory, "get_symbolic_addrs"):
            # cat-(b) FALLBACK WITH LOSS: get_symbolic_addrs() failed;
            # fall through to the _pages bitmap scan below. Both
            # populate symbolic_pages, so this is a graceful fallback.
            with contextlib.suppress(Exception):
                for addr in angr_state.memory.get_symbolic_addrs():
                    symbolic_pages.add(addr & ~(page_size - 1))
        if not symbolic_pages and hasattr(angr_state.memory, "_pages"):
            # angr-7vcx: scan UltraPage.symbolic_data (the dict of explicit
            # symbolic stores), NOT symbolic_bitmap. A freshly map_region'd
            # page initialises symbolic_bitmap to all-ones (every byte will
            # default-fill via SYMBOL_FILL_UNCONSTRAINED_MEMORY on read), so
            # `any(sb)` overflags pages that have NO user-stored symbolic
            # data — those pages then get skipped by both _overlay and
            # _sync_extra paths, leaving Rust without the mapping and
            # silently dropping concrete stores.
            mem_page_size = getattr(angr_state.memory, "page_size", page_size)
            for page_num in list(angr_state.memory._pages.keys()):
                page = angr_state.memory._pages.get(page_num)
                if page is None:
                    continue
                sd = getattr(page, "symbolic_data", None)
                if sd is not None and len(sd) > 0:
                    symbolic_pages.add(page_num * mem_page_size)
        return symbolic_pages

    def _build_loader_pages_cache_entry(self, page_size: int) -> dict:
        """Compute the cacheable (batch_pages, lazy_regions) for the loader.

        Iterates `loader.all_objects` once, reads each in-segment page via
        `loader.memory.load`, and collects every object's full address range
        as a lazy region. Pure function of `self._project.loader` — output
        is reusable across RustExplorationManager constructions for the
        same project (see `RustExplorationManager._loader_pages_cache`).
        """
        batch_pages = []
        lazy_regions = []
        seen_page_addrs: set = set()
        for obj in self._project.loader.all_objects:
            # cat-(b) FALLBACK WITH LOSS: per-object iteration failed
            # (object lacks expected attributes). That object's pages
            # won't be eagerly mapped; lazy_region in
            # _add_loader_lazy_regions catches accesses on demand.
            with contextlib.suppress(Exception):
                if hasattr(obj, "segments") and obj.segments:
                    ranges = []
                    for seg in obj.segments:
                        if seg.memsize > 0:
                            ranges.append(
                                (seg.min_addr & ~(page_size - 1), (seg.max_addr + page_size) & ~(page_size - 1))
                            )
                else:
                    ranges = [(obj.min_addr & ~(page_size - 1), (obj.max_addr + page_size) & ~(page_size - 1))]
                for start_page, end_page in ranges:
                    for page_addr in range(start_page, end_page, page_size):
                        if page_addr in seen_page_addrs:
                            continue
                        # cat-(a) EXPECTED CONTROL FLOW: per-page load
                        # may hit unmapped gaps in object's page range
                        # (esp. ELF .bss gaps). Skip silently — the
                        # lazy region added below catches accesses.
                        with contextlib.suppress(Exception):
                            data = self._project.loader.memory.load(page_addr, page_size)
                            if data and len(data) == page_size:
                                batch_pages.append((page_addr, bytes(data), 7))
                                seen_page_addrs.add(page_addr)
                region_start = obj.min_addr & ~(page_size - 1)
                region_end = (obj.max_addr + page_size) & ~(page_size - 1)
                if region_end - region_start > 0:
                    lazy_regions.append((region_start, region_end - region_start))
        return {"batch_pages": batch_pages, "lazy_regions": lazy_regions}

    def _get_loader_pages_cache(self, page_size: int) -> dict:
        """Return the cached loader pages entry, building it on miss.

        Keyed (weakly) by `self._project.loader` so Callable-style
        workflows that spawn many RustExplorationManagers on the same
        Project hit the cache after the first construction (angr-bzsc).
        Entries auto-evict when the Loader is garbage-collected.
        """
        cls = type(self)
        cache = cls._loader_pages_cache
        loader = self._project.loader
        entry = cache.get(loader)
        if entry is not None:
            return entry
        entry = self._build_loader_pages_cache_entry(page_size)
        cache[loader] = entry
        return entry

    def _map_loader_pages(self, rust_state: _RustSimState, symbolic_pages: set, page_size: int):
        """Map pages for each loaded object's segments via a single FFI batch.

        Returns (mapped_page_addrs, pages_mapped). Skips pages already in
        symbolic_pages so callbacks can serve them. Reads the
        `_get_loader_pages_cache` entry (built on first construction per
        project) to avoid re-loading loader pages on every Callable spawn.
        """
        entry = self._get_loader_pages_cache(page_size)
        cached_pages = entry["batch_pages"]
        if symbolic_pages:
            filtered = [(addr, data, perms) for (addr, data, perms) in cached_pages if addr not in symbolic_pages]
        else:
            filtered = cached_pages
        mapped_page_addrs = {addr for (addr, _data, _perms) in filtered}
        if filtered:
            try:
                rust_state.map_memory_batch(filtered)
            except AttributeError:
                # cat-(a) EXPECTED CONTROL FLOW: probing for batch API on
                # older Rust builds; fall back to per-page mapping.
                for page_addr, data, perms in filtered:
                    rust_state.map_memory_data(page_addr, data, perms)
        return mapped_page_addrs, len(filtered)

    def _overlay_relocated_sections(self, angr_state: angr.SimState, rust_state: _RustSimState) -> None:
        """Overlay GOT entries / relocated section data from the Python state.

        Only writes small concrete sections (<64KB) to avoid expensive Z3 eval.
        """
        for obj in self._project.loader.all_objects:
            if obj.binary is None or not hasattr(obj, "sections"):
                continue
            for section in obj.sections:
                if section.memsize > 0 and section.memsize < MAX_OVERLAY_SECTION_SIZE:
                    # cat-(b) FALLBACK WITH LOSS: GOT/relocation
                    # overlay failed (load OOB or eval timeout). Rust
                    # sees the raw loader-mapped data without
                    # relocations applied — may misresolve external
                    # references. Lazy fetch_page from Python catches
                    # this on access in most cases.
                    with contextlib.suppress(Exception):
                        val = angr_state.memory.load(
                            section.min_addr, section.memsize, endness="Iend_BE", inspect=False, disable_actions=True
                        )
                        if not val.symbolic:
                            data = angr_state.solver.eval(val).to_bytes(section.memsize, "big")
                            rust_state.map_memory_data(section.min_addr, data, 7)

    def _overlay_python_state_pages(
        self,
        angr_state: angr.SimState,
        rust_state: _RustSimState,
        mapped_page_addrs: set,
        symbolic_pages: set,
        page_size: int,
    ) -> None:
        """Overlay Python state's concrete memory on top of loader pages.

        Critical for multi-stage explore: when a found state from stage N is
        re-imported into a new RustExplorationManager for stage N+1, the
        loader's static data (e.g. zeros in .bss) would otherwise overwrite
        runtime modifications (e.g. result buffer at 0x612040 in sakura).
        """
        state_overlay_count = 0
        mem_pages = getattr(angr_state.memory, "_pages", None)
        if mem_pages is None:
            return
        for page_no in list(mem_pages.keys()):
            page_addr = page_no * page_size
            if page_addr not in mapped_page_addrs:
                continue  # Non-loader pages handled separately
            if page_addr in symbolic_pages:
                continue  # Symbolic pages use callback path
            page_obj = mem_pages.get(page_no)
            if page_obj is None:
                continue
            # cat-(b) FALLBACK WITH LOSS: page lacks concrete_load or
            # contains symbolic bytes that can't be eagerly loaded.
            # Rust falls back to memory_load callback for this page.
            with contextlib.suppress(Exception):
                concrete = bytes(page_obj.concrete_load(0, page_size))
                if len(concrete) == page_size:
                    rust_state.map_memory_data(page_addr, concrete, 7)
                    state_overlay_count += 1
        if state_overlay_count:
            l.debug(f"Overlaid {state_overlay_count} loader pages with Python state data")

    def _add_loader_lazy_regions(self, rust_state: _RustSimState, page_size: int) -> None:
        """Register every loaded object as a lazy region for fetch_page callbacks.

        Uses the cached lazy_regions list (computed once per project, shared
        across managers) to avoid re-iterating `loader.all_objects`.
        """
        entry = self._get_loader_pages_cache(page_size)
        for region_start, region_size in entry["lazy_regions"]:
            # cat-(b) FALLBACK WITH LOSS: lazy region registration
            # failed for this object; accesses to its pages can't
            # be auto-fetched and will fail to load on demand.
            with contextlib.suppress(Exception):
                rust_state.add_lazy_region(region_start, region_size)

    def _setup_stack_region(self, angr_state: angr.SimState, rust_state: _RustSimState, arch, page_size: int):
        """Compute the stack region and register it as lazy.

        Returns (sp_page, stack_start, stack_base).
        """
        try:
            sp = angr_state.solver.eval(angr_state.regs.sp)
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: state has symbolic SP and no
            # default; pick the conventional stack-base for this arch.
            # If the state's actual SP differs, the lazy stack region
            # may not cover real accesses and they'll FFI back to Python.
            sp = 0x7FFF_FFF0_0000 if arch.bits == 64 else 0x7FFF_0000
        stack_base = (sp & ~(page_size - 1)) + page_size
        stack_start = stack_base - STACK_SIZE
        rust_state.add_lazy_region(stack_start, STACK_SIZE)
        sp_page = sp & ~(page_size - 1)
        return sp_page, stack_start, stack_base

    def _sync_stack_page(
        self,
        angr_state: angr.SimState,
        rust_state: _RustSimState,
        sp_page: int,
        page_size: int,
        arch,
        symbolic_regions: list,
    ) -> None:
        """Pre-populate just the stack page at SP from the Python state.

        Other stack pages are served lazily via fetch_page when accessed,
        avoiding expensive solver.eval() on unconstrained fill pages.
        """
        page_no = sp_page // page_size
        mem_pages = getattr(angr_state.memory, "_pages", None)
        page_obj = mem_pages.get(page_no) if mem_pages is not None else None
        used_fast_path = False
        # cat-(b) FALLBACK WITH LOSS: stack page sync failed entirely
        # (slow path raised). Rust will fetch the stack page lazily
        # via fetch_page callback on first access — correct but pays
        # FFI roundtrip per page.
        with contextlib.suppress(Exception):
            if page_obj is not None and hasattr(page_obj, "concrete_load"):
                # cat-(a) EXPECTED CONTROL FLOW: concrete_load fast
                # path failed (page has symbolic content). Fall
                # through to slow path below.
                with contextlib.suppress(Exception):
                    concrete = bytes(page_obj.concrete_load(0, page_size))
                    if len(concrete) == page_size:
                        rust_state.map_memory_data(sp_page, concrete, 6)
                        used_fast_path = True
                        # Targeted scan of symbolic_data ranges only, far
                        # cheaper than a full _extract_symbolic_regions scan.
                        sd = getattr(page_obj, "symbolic_data", None)
                        if sd:
                            self._extract_stack_symbolic_from_sd(angr_state, sp_page, page_size, sd, symbolic_regions)

            if not used_fast_path:
                # Slow path: load through memory mixin stack + solver.eval()
                page_data = angr_state.memory.load(
                    sp_page, page_size, endness="Iend_BE", inspect=False, disable_actions=True
                )
                concrete = angr_state.solver.eval(page_data).to_bytes(page_size, "big")
                rust_state.map_memory_data(sp_page, concrete, 6)
                if page_data.symbolic and self._has_user_symbolic_var(page_data):
                    self._extract_symbolic_regions(angr_state, sp_page, page_size, arch.bytes, symbolic_regions)
            l.debug("Pre-populated 1 stack page in Rust memory")

    def _extract_stack_symbolic_from_sd(
        self, angr_state: angr.SimState, page_addr: int, page_size: int, sd: dict, symbolic_regions: list
    ) -> None:
        """Targeted scan of a stack page's symbolic_data ranges.

        Avoids the full 4096-byte byte-by-byte scan in _extract_symbolic_regions
        — this only walks the explicitly-stored symbolic byte ranges.
        """
        scan_ranges = self._scan_ranges_from_symbolic_data(sd)
        for start_offset, size in scan_ranges:
            for byte_off in range(size):
                if start_offset + byte_off >= page_size:
                    break
                addr = page_addr + start_offset + byte_off
                # cat-(b) FALLBACK WITH LOSS: per-byte symbolic load
                # failed; that byte position is not added to
                # symbolic_regions and will be served as the concrete
                # fast-path byte. Loss is partial: only that single
                # byte's symbolic identity is missed.
                with contextlib.suppress(Exception):
                    val = angr_state.memory.load(addr, 1, endness="Iend_BE", inspect=False, disable_actions=True)
                    if val.symbolic:
                        symbolic_regions.append((addr, val))

    @staticmethod
    def _has_user_symbolic_var(ast) -> bool:
        """True if the AST references at least one user-created symbolic variable
        (not a synthetic mem_/reg_/unconstrained placeholder)."""
        try:
            return any(
                not n.startswith("mem_") and not n.startswith("reg_") and not n.startswith("unconstrained")
                for n in ast.variables
            )
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: ast lacks .variables
            # (concrete claripy wrapper). Treat as no-user-symbolic.
            return False

    @classmethod
    def _scan_ranges_from_symbolic_data(cls, sd) -> list:
        """Build a list of (offset, ast_size_bytes) ranges from a page's
        symbolic_data dict, keeping only entries that reference a user-created
        symbolic variable.

        Shared by _extract_stack_symbolic_from_sd (stack fast-path) and
        _extract_wide_symbolic_regions (wide-region grouping) so the
        _has_user_symbolic_var filter and the size = ast.size() // 8
        computation stay in one place. Note: _extract_from_ultrapage uses a
        structurally different bitmap walk (SymbolicBitmap scan + MAX_EXTENT + next_key) and
        deliberately does not go through this helper.
        """
        scan_ranges = []
        for sd_offset, sd_ast in sd.items():
            if not hasattr(sd_ast, "variables"):
                continue
            if cls._has_user_symbolic_var(sd_ast):
                ast_size = sd_ast.size() // 8 if hasattr(sd_ast, "size") else 1
                scan_ranges.append((sd_offset, ast_size))
        return scan_ranges

    def _sync_extra_python_pages(
        self,
        angr_state: angr.SimState,
        rust_state: _RustSimState,
        mapped_page_addrs: set,
        symbolic_pages: set,
        sp_page: int,
        stack_start: int,
        stack_base: int,
        page_size: int,
    ) -> None:
        """Sync non-loader, non-stack pages from the Python state.

        Critical for multi-stage explore: pages written during stage 1 (ctype
        tables, heap data, extra stack frames from SimProcedures) must be
        synced or the Rust engine will read zeros and diverge.
        """
        extra_pages_synced = 0
        # angr-8t45: eager-map all-zero pages too when there aren't many.
        # Zero pages stay lazy when numerous, to keep mma_howtouse from
        # blowing up in memory (thousands of zero pages per state across many
        # short-lived states; angr-9maq). But hackcon2016 has only a few dozen
        # zero pages per state, and keeping them lazy multiplies final-solve
        # Z3 time. Cap below: if a state has more all-zero pages than this,
        # keep them all lazy (mma path); otherwise eager-map (hackcon path).
        zero_eager_cap = 200
        # cat-(b) FALLBACK WITH LOSS: outer iteration failed (rare,
        # _pages dict shape mismatch). All non-loader pages skip
        # eager sync; lazy fetch_page handles them.
        with contextlib.suppress(Exception):
            mem_pages = getattr(angr_state.memory, "_pages", None)
            if mem_pages is None:
                return
            # angr-b58a: classify pages without materializing 4KB bytes objects.
            # For UltraPage (the default backend) we read concrete_data
            # directly — a bytearray-vs-bytearray compare is a memcmp, orders
            # of magnitude cheaper than `any(bytes_iter)`. The bytes()
            # copy is deferred until we know the page actually needs to be
            # FFI-mapped (saves 8MB of bytes allocations per Callable on
            # mma_howtouse's ~2000 all-zero pages).
            raw_pages = []  # list[(page_addr, page_obj, is_nonzero, perms, cached_bytes_or_None)]
            for page_no in list(mem_pages.keys()):
                page_addr = page_no * page_size
                if page_addr in mapped_page_addrs:
                    continue  # Loader page — handled by overlay
                if page_addr == sp_page:
                    continue  # Stack page — already synced
                # angr-ric3: do NOT skip symbolic pages here. The page may
                # contain concrete bytes interspersed with the symbolic
                # store (e.g. a SimProc test that writes a 1-byte BVS at
                # offset N and concrete bytes at offset N+M on the same
                # page). Pushing concrete_load gives Rust the surrounding
                # bytes at their actual values; the symbolic import in
                # _add_rust_state then overlays the symbolic positions on
                # top, so symbolic identity is preserved while concrete
                # bytes become visible to the proxy/Rust-side read path.
                page_obj = mem_pages.get(page_no)
                if page_obj is None:
                    continue
                perms = 6 if stack_start <= page_addr < stack_base else 7
                cd = _ultrapage_bytes(page_obj) if page_size == PAGE_SIZE else None
                if cd is not None:
                    # UltraPage fast path: classify via memcmp on the backing
                    # bytearray. No bytes() copy yet — deferred to the FFI
                    # phase (and skipped entirely for lazy-only zero pages).
                    # A never-written page has no store yet and reads as the
                    # shared zero buffer, which is already the bytes to send.
                    is_nonzero = cd != _ZERO_PAGE_BA
                    cached = cd if cd is _ZERO_PAGE_BYTES else None
                    raw_pages.append((page_addr, page_obj, is_nonzero, perms, cached))
                else:
                    # Non-UltraPage backend: materialize bytes once and keep
                    # the buffer for the FFI phase to avoid a second
                    # concrete_load.
                    try:
                        concrete = bytes(page_obj.concrete_load(0, page_size))
                    except Exception:
                        # cat-(b) FALLBACK WITH LOSS: per-page sync failed;
                        # lazy fetch_page handles it on first access.
                        continue
                    if len(concrete) != page_size:
                        continue
                    is_nonzero = concrete != _ZERO_PAGE_BYTES
                    raw_pages.append((page_addr, page_obj, is_nonzero, perms, concrete))

            zero_count = sum(1 for _, _, nz, _, _ in raw_pages if not nz)
            eager_zero = zero_count <= zero_eager_cap

            lazy_batch = []
            for page_addr, page_obj, is_nonzero, perms, cached in raw_pages:
                # angr-7vcx: must record the mapping even for all-zero pages,
                # otherwise user `map_region` calls silently disappear and
                # Rust faults on the first store. Stack region gets RW.
                if is_nonzero or eager_zero:
                    concrete = cached
                    if concrete is None:
                        try:
                            concrete = bytes(page_obj.concrete_load(0, page_size))
                        except Exception:
                            # cat-(b) FALLBACK WITH LOSS: classification path
                            # worked but second load failed; keep the lazy
                            # registration so on-demand fetch can recover.
                            concrete = None
                    if concrete is not None and len(concrete) == page_size:
                        rust_state.map_memory_data(page_addr, concrete, perms)
                # Else: store_concrete_automap_internal will auto-allocate
                # this page on first write via add_lazy_region tracking.
                lazy_batch.append((page_addr, page_size))
                extra_pages_synced += 1

            if lazy_batch:
                try:
                    rust_state.add_lazy_regions_batch(lazy_batch)
                except AttributeError:
                    # cat-(a) EXPECTED CONTROL FLOW: probing for batch API on
                    # older Rust builds; fall back to per-page registration.
                    for start, size in lazy_batch:
                        rust_state.add_lazy_region(start, size)
        if extra_pages_synced:
            l.debug(f"Synced {extra_pages_synced} extra pages from Python state (non-loader)")

    def _scan_user_symbolic_pages(
        self, angr_state: angr.SimState, stack_start: int, stack_base: int, page_size: int, symbolic_regions: list
    ) -> None:
        """Scan non-stack pages for user symbolic data and import as wide regions.

        Importing as a single wide object preserves symbolic identity across
        the Rust/Python boundary (avoids creating rust_sym_XXX aliases).

        Uses get_symbolic_addrs() (or page.symbolic_data fallback) to skip the
        thousands of mem_*/unconstrained fill pages that show up after Python
        init — scanning each of those pays a full per-byte page walk.
        """
        # cat-(b) FALLBACK WITH LOSS: outer scan failed (rare).
        # symbolic_regions is empty, so user symbolic memory is not
        # imported into Rust — same risk class as the inner cat-(c)
        # but at this point the exploration likely won't run at all.
        with contextlib.suppress(Exception):
            pages = getattr(angr_state.memory, "_pages", {})
            user_sym_pages = set()
            if hasattr(angr_state.memory, "get_symbolic_addrs"):
                # cat-(b) FALLBACK WITH LOSS: get_symbolic_addrs() is
                # an opt-in API; if it raises, we fall through to the
                # symbolic_data fallback walk below.
                with contextlib.suppress(Exception):
                    for addr in angr_state.memory.get_symbolic_addrs():
                        user_sym_pages.add(addr // page_size)
            if not user_sym_pages:
                # Cheap filter: symbolic_data is non-empty only for pages with
                # explicit stores (not unconstrained fill from the filler mixin).
                for page_no in pages:
                    page = pages[page_no]
                    if hasattr(page, "symbolic_data"):
                        sd = page.symbolic_data
                        if sd:
                            user_sym_pages.add(page_no)

            for page_no in sorted(user_sym_pages):
                page_addr = page_no * page_size
                if stack_start <= page_addr < stack_base:
                    continue  # Stack pages already handled
                # cat-(c) WRONG-ANSWER RISK: per-page scan failed; user
                # symbolic data on this page is not added to
                # symbolic_regions, so Rust later sees concrete bytes
                # instead of the symbolic AST. Solver will pick a
                # concrete value rather than fork on the symbol.
                # (Wider impact than (b) per-byte miss because the
                # whole page is dropped.)
                with contextlib.suppress(Exception):
                    page_data = angr_state.memory.load(
                        page_addr, page_size, endness="Iend_BE", inspect=False, disable_actions=True
                    )
                    if page_data.symbolic and self._has_user_symbolic_var(page_data):
                        self._extract_wide_symbolic_regions(angr_state, page_addr, page_size, symbolic_regions)

    def _extract_wide_symbolic_regions(self, angr_state, page_addr, page_size, out):
        """Extract WIDE symbolic regions from a page.

        Groups contiguous symbolic bytes that share the same variable and
        imports them as a single wide object. This preserves symbolic identity
        across the Rust/Python boundary (avoids creating rust_sym_XXX aliases).

        Uses page.symbolic_data dict to find symbolic ranges, avoiding a full
        per-byte page scan.
        """
        # Fast path: use symbolic_data dict to find which ranges to scan
        page_no = page_addr // page_size
        mem_pages = getattr(angr_state.memory, "_pages", None)
        page_obj = mem_pages.get(page_no) if mem_pages is not None else None

        if page_obj is not None and hasattr(page_obj, "symbolic_data"):
            sd = page_obj.symbolic_data
            if sd:
                # Determine byte ranges that need scanning from symbolic_data entries
                scan_ranges = self._scan_ranges_from_symbolic_data(sd)

                if scan_ranges:
                    # Scan only the identified ranges
                    for start_offset, size in sorted(scan_ranges):
                        region_start = page_addr + start_offset
                        end_offset = min(start_offset + size, page_size)
                        actual_size = end_offset - start_offset
                        if actual_size <= 0:
                            continue
                        # Load the full region as a single wide object
                        try:
                            wide_val = angr_state.memory.load(
                                region_start, actual_size, endness="Iend_BE", inspect=False, disable_actions=True
                            )
                            if wide_val.symbolic:
                                out.append((region_start, wide_val))
                        except Exception:
                            # cat-(a) EXPECTED CONTROL FLOW: wide load
                            # failed; fall through to per-byte loop
                            # which is the documented fallback.
                            for byte_off in range(actual_size):
                                addr = region_start + byte_off
                                # cat-(b) FALLBACK WITH LOSS: per-byte
                                # load also failed; that byte not
                                # imported, served as concrete instead.
                                with contextlib.suppress(Exception):
                                    val = angr_state.memory.load(
                                        addr, 1, endness="Iend_BE", inspect=False, disable_actions=True
                                    )
                                    if val.symbolic:
                                        out.append((addr, val))
                    return  # Done with fast path

        # Slow fallback: scan entire page byte by byte
        offset = 0
        while offset < page_size:
            addr = page_addr + offset
            try:
                val = angr_state.memory.load(addr, 1, endness="Iend_BE", inspect=False, disable_actions=True)
                if not val.symbolic:
                    offset += 1
                    continue
                leaf_names = list(val.variables)
                if not self._has_user_symbolic_var(val):
                    offset += 1
                    continue

                # Found a symbolic byte — scan forward to find the full region
                region_start = addr
                region_vars = frozenset(leaf_names)
                region_len = 1
                while offset + region_len < page_size:
                    next_addr = page_addr + offset + region_len
                    try:
                        next_val = angr_state.memory.load(
                            next_addr, 1, endness="Iend_BE", inspect=False, disable_actions=True
                        )
                        if next_val.symbolic and frozenset(next_val.variables) == region_vars:
                            region_len += 1
                        else:
                            break
                    except Exception:
                        # cat-(a) EXPECTED CONTROL FLOW: scan stopped at
                        # an inaccessible byte; treat it as a region
                        # boundary and emit what we've collected.
                        break

                try:
                    wide_val = angr_state.memory.load(
                        region_start, region_len, endness="Iend_BE", inspect=False, disable_actions=True
                    )
                    out.append((region_start, wide_val))
                except Exception:
                    # cat-(a) EXPECTED CONTROL FLOW: wide consolidation
                    # load failed; emit just the first byte we already
                    # loaded above. Identity preserved for that byte.
                    out.append((addr, val))

                offset += region_len
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: per-byte scan failed at
                # the start of a candidate region; advance by one and
                # continue. No data lost — this byte just isn't imported.
                offset += 1

    def _extract_symbolic_regions(self, angr_state, page_addr, page_size, ptr_size, out):
        """Extract symbolic memory regions from a page for import to Rust.

        Only imports BYTE-LEVEL symbolic values that contain user-defined
        symbols (BVS with names not starting with 'mem_' or 'reg_').
        Skips unconstrained fill variables.
        """
        for offset in range(0, page_size, 1):
            addr = page_addr + offset
            # cat-(b) FALLBACK WITH LOSS: per-byte symbolic load
            # failed; that byte's symbolic identity is not imported.
            # The page's concrete bytes still load via fast paths,
            # so the user-visible value is wrong only if the missed
            # byte was meaningful (e.g. a constraint leaf).
            with contextlib.suppress(Exception):
                val = angr_state.memory.load(addr, 1, endness="Iend_BE", inspect=False, disable_actions=True)
                # Check if this contains a user-defined variable
                # (not just unconstrained fill from entry_state)
                if val.symbolic and self._has_user_symbolic_var(val):
                    out.append((addr, val))

    def _concretize_stack_registers(self, state: angr.SimState):
        """Concretize stack registers for Rust memory mapping compatibility.

        This prevents symbolic address issues during Rust exploration by
        ensuring stack-relative registers have concrete values.

        Fast path: if the register's symbolic variables don't appear in any
        solver constraint, skip solver.eval and pick a sensible default —
        same value Z3 would have picked, without paying for a model query.
        Common on blank_state cold path where rbp is filled with a
        fresh unconstrained BVS by default_filler_mixin.
        """
        arch = state.arch

        if arch.name in ("AMD64", "X86_64"):
            bp_reg = "rbp"
            sp_reg = "rsp"
        elif arch.name == "X86":
            bp_reg = "ebp"
            sp_reg = "esp"
        elif arch.name.startswith("ARM"):
            bp_reg = None
            sp_reg = "sp"
        else:
            bp_reg = None
            sp_reg = None

        if sp_reg:
            try:
                reg_val = getattr(state.regs, sp_reg)
                if reg_val.symbolic:
                    sp_default = getattr(arch, "initial_sp", None) or 0
                    sp_val = self._eval_or_default(state, reg_val, sp_default)
                    state.solver.add(reg_val == sp_val)
                    setattr(state.regs, sp_reg, sp_val)
                    l.debug(f"Concretized {sp_reg} to 0x{sp_val:x} (constraint added)")
            except Exception as e:
                # cat-(c) WRONG-ANSWER RISK: SP stays symbolic; Rust will
                # not be able to map a concrete stack region. The lazy
                # stack region uses a default base, so symbolic SP
                # accesses will misroute. Log at debug because the
                # downstream stack-page sync also catches this.
                l.debug(f"Could not concretize {sp_reg}: {e}")

        if bp_reg:
            try:
                reg_val = getattr(state.regs, bp_reg)
                if reg_val.symbolic:
                    concrete_val = self._eval_or_default(state, reg_val, 0)
                    setattr(state.regs, bp_reg, concrete_val)
                    l.debug(f"Concretized {bp_reg} to 0x{concrete_val:x}")
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: BP stays symbolic; Rust
                # frame-pointer-relative addressing will see a symbolic
                # base. Most modern compilers omit BP, so this is rarely
                # hit; when it is, the compiler used BP and Rust will
                # need to fork on it.
                l.debug(f"Could not concretize {bp_reg}: {e}")

    @staticmethod
    def _eval_or_default(state, reg_val, default):
        """Fast-path solver.eval for unconstrained symbolic registers.

        Z3 returns 0 (or arbitrary) for a BVS that no constraint references,
        but the call still pays ctx init + check + model. When we
        can prove no constraint mentions any of the value's variables, return
        `default` directly — Z3 would have picked something arbitrary anyway,
        and downstream code only requires *some* concrete value.
        """
        val_vars = reg_val.variables
        for c in state.solver.constraints:
            if val_vars & c.variables:
                return state.solver.eval(reg_val)
        return default

    def _get_arch_register_names(self, arch) -> list:
        """The registers synced around a callback: ``Arch::sync_registers`` in Rust."""
        table = rust_register_table(arch)
        return list(table["sync"]) if table is not None else []

    @staticmethod
    def _get_reg_map_and_return_regs(arch):
        """The callback sync registers' ``{name: (offset, size)}`` and the return registers.

        Both come from the engine's table (``rust_register_table``): the
        ``Arch::sync_registers`` names at the engine's offsets, and the
        default calling convention's return register (plus ``v1`` on MIPS).
        The map is shared and cached per arch; callers must not mutate it.

        Returns:
            Tuple of (reg_map, return_regs) or (None, None) if unsupported.
        """
        table = rust_register_table(arch)
        if table is None:
            l.warning(f"Unknown architecture {arch.name} for register extraction")
            return None, None
        return table["sync_map"], table["return_regs"]

    def _extract_symbolic_pages(self, state: angr.SimState) -> dict:
        """Extract symbolic memory regions from an angr state.

        This identifies memory regions containing symbolic values and caches
        them for later restoration. This is critical for preserving symbolic
        memory when Rust falls back to Python callbacks.

        Args:
            state: The angr state to extract symbolic pages from.

        Returns:
            Dict mapping address -> claripy AST for symbolic memory locations.
        """
        symbolic_regions: dict = {}
        try:
            if self._extract_via_get_symbolic_addrs(state, symbolic_regions):
                return symbolic_regions

            if hasattr(state.memory, "_pages"):
                page_size = getattr(state.memory, "page_size", 4096)
                for page_num in list(state.memory._pages.keys()):
                    page = state.memory._pages.get(page_num)
                    if page is None:
                        continue
                    page_addr = page_num * page_size
                    # First backend that hasattr-matches handles the page.
                    (
                        self._extract_from_ultrapage(state, page, page_addr, symbolic_regions)
                        or self._extract_from_listpage(state, page, page_addr, symbolic_regions)
                        or self._extract_from_alt_bitmap(state, page, page_addr, page_size, symbolic_regions)
                        or self._extract_from_byte_map(page, page_addr, symbolic_regions)
                    )

            if symbolic_regions:
                l.debug(f"Extracted {len(symbolic_regions)} symbolic memory regions")
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: outer extract loop failed.
            # Returns empty regions dict; downstream Rust restoration
            # will not see any symbolic memory from this state.
            l.debug(f"Error extracting symbolic pages: {e}")
        return symbolic_regions

    def _load_symbolic_byte(self, state: angr.SimState, addr: int):
        """Load 1 byte at addr; return the AST if symbolic, else None.

        cat-(b) FALLBACK WITH LOSS on load failure: returns None, so the
        caller drops that byte and Rust will see concrete data (or zeros).
        """
        with contextlib.suppress(Exception):
            val = state.memory.load(addr, 1, endness=state.arch.memory_endness)
            if hasattr(val, "symbolic") and val.symbolic:
                return val
        return None

    def _record_symbolic(self, out: dict, addr: int, val) -> None:
        out[addr] = val
        self._register_handle(id(val), val)

    def _extract_via_get_symbolic_addrs(self, state: angr.SimState, out: dict) -> bool:
        """Strategy 1: angr's internal symbolic tracking. Most accurate.

        Returns True iff at least one symbolic byte was recorded — matching
        the original early-return guard on a non-empty regions dict.
        """
        if not hasattr(state.memory, "get_symbolic_addrs"):
            return False
        try:
            symbolic_addrs = state.memory.get_symbolic_addrs()
        except Exception as e:
            # cat-(a) EXPECTED CONTROL FLOW: opt-in API; fall through.
            l.debug(f"get_symbolic_addrs failed: {e}")
            return False
        if not symbolic_addrs:
            return False
        before = len(out)
        for addr in symbolic_addrs:
            val = self._load_symbolic_byte(state, addr)
            if val is not None:
                self._record_symbolic(out, addr, val)
        added = len(out) - before
        if added:
            l.debug(f"Extracted {added} symbolic bytes via get_symbolic_addrs")
            return True
        return False

    def _extract_from_ultrapage(self, state, page, page_addr: int, out: dict) -> bool:
        """UltraPage: changed-byte segments filtered by symbolic_bitmap, plus
        a symbolic_data fallback for filler-materialised values.

        ``all_bytes_changed_in_history`` only sees bytes touched by ``store``.
        Values materialised via the ``SYMBOL_FILL_UNCONSTRAINED_MEMORY``
        filler at load time (e.g. ``init.memory.load(addr, 8)`` used to
        capture a symbolic input variable, as in sokohashv2's solve.py)
        live in ``symbolic_data`` but never make it into the changed-history
        list. When the history view is empty but ``symbolic_data`` is
        non-empty and small, fall back to iterating the SortedDict
        directly so those user-seeded symbolic values still survive the
        Python↔Rust round trip (angr-ctct). The size cap keeps the cost
        bounded: pages with hundreds of filler entries from binary
        execution would otherwise burn time re-importing each one.
        """
        if not (hasattr(page, "all_bytes_changed_in_history") and hasattr(page, "symbolic_bitmap")):
            return False
        sb = page.symbolic_bitmap
        # A whole-page bitmap scan beats walking every changed byte when
        # nothing is symbolic — common for entry_state pages with concrete
        # loader writes.
        if sb is None or not sb.any_set(0, sb.size):
            return True
        # cat-(b) FALLBACK WITH LOSS: page-level walk failed.
        with contextlib.suppress(Exception):
            changed = page.all_bytes_changed_in_history()
            had_changed = False
            for segment in changed:
                had_changed = True
                start = getattr(segment, "start", None)
                end = getattr(segment, "end", None)
                if start is None or end is None:
                    continue
                for offset in range(start, end):
                    if offset < sb.size and sb.get(offset):
                        addr = page_addr + offset
                        val = self._load_symbolic_byte(state, addr)
                        if val is not None:
                            self._record_symbolic(out, addr, val)
            # angr-ctct: changed-history misses filler-materialised symbols.
            # Fall back to symbolic_data only when no store-driven changes
            # were seen (typical of the user-setup page) and the dict is
            # small enough that walking it can't dominate per-state cost.
            # angr-fv81: walk each entry's full byte extent — symbolic_data
            # is keyed only by region-start offset, so an 8-byte filler load
            # produces a single dict entry but marks 8 bitmap bits. The
            # earlier per-key extraction caught only byte 0 of each region;
            # bytes 1..N silently became concrete on the Rust side.
            #
            # Per-entry extent cap: skip entries spanning > MAX_EXTENT bytes.
            # SYMBOL_FILL_UNCONSTRAINED_MEMORY can produce single entries
            # covering whole 4 KB pages (one fill per page-touch); extracting
            # those byte-by-byte would dominate sync cost on benchmarks that
            # symbolically index untouched pages (ais3_crackme,
            # google2016_unbreakable_0). User-seeded symbolic input vars are
            # almost always <= 64 bytes, so this cap preserves correctness
            # for the sokohashv2 / similar pattern without paying the
            # whole-page tax.
            MAX_EXTENT = 64
            if not had_changed:
                sd = getattr(page, "symbolic_data", None)
                if sd is not None and 0 < len(sd) <= 64:
                    keys = list(sd.keys())
                    sb_len = sb.size
                    for i, offset in enumerate(keys):
                        if offset >= sb_len or not sb.get(offset):
                            continue
                        next_key = keys[i + 1] if i + 1 < len(keys) else sb_len
                        end = offset
                        limit = min(offset + MAX_EXTENT, next_key, sb_len)
                        # First concrete byte of the run, capped at `limit`.
                        end = sb.next_clear(offset, limit)
                        if end - offset >= MAX_EXTENT:
                            continue  # likely a page-fill — skip
                        for byte_off in range(offset, end):
                            addr = page_addr + byte_off
                            val = self._load_symbolic_byte(state, addr)
                            if val is not None:
                                self._record_symbolic(out, addr, val)
        return True

    def _extract_from_listpage(self, state, page, page_addr: int, out: dict) -> bool:
        """ListPage: stored_offset tracks all written bytes."""
        if not (hasattr(page, "stored_offset") and page.stored_offset):
            return False
        for offset in page.stored_offset:
            addr = page_addr + offset
            val = self._load_symbolic_byte(state, addr)
            if val is not None:
                self._record_symbolic(out, addr, val)
        return True

    def _extract_from_alt_bitmap(self, state, page, page_addr: int, page_size: int, out: dict) -> bool:
        """Fallback page with a `_symbolic_bitmap` dict attribute."""
        if not (hasattr(page, "_symbolic_bitmap") and page._symbolic_bitmap):
            return False
        for offset in range(page_size):
            if page._symbolic_bitmap.get(offset, False):
                addr = page_addr + offset
                val = self._load_symbolic_byte(state, addr)
                if val is not None:
                    self._record_symbolic(out, addr, val)
        return True

    def _extract_from_byte_map(self, page, page_addr: int, out: dict) -> bool:
        """Page exposing a `symbolic_byte_map` of offset → AST directly."""
        if not (hasattr(page, "symbolic_byte_map") and page.symbolic_byte_map):
            return False
        for offset, sym_val in page.symbolic_byte_map.items():
            self._record_symbolic(out, page_addr + offset, sym_val)
        return True

    def _install_rust_memory_proxy(self, state: angr.SimState):
        """Sync stack data from Rust to Python callback state.

        Loads the SP page in a single bulk FFI call (pending_memory_load_page)
        then writes non-zero pointer-sized values to the Python state.
        This is much faster than one pending_memory_load call per slot.

        angr-ryf6: skip when the callback memory proxy is already installed
        on ``state.memory`` — the writes would
        round-trip Rust → Python → Rust and overwrite symbolic pointer-slot
        values with concrete witnesses (same loss as
        ``_replay_rust_dirty_pages``).
        """
        from angr.exploration.rust_manager import _is_rust_memory_proxy

        if _is_rust_memory_proxy(state.memory):
            return
        # cat-(b) FALLBACK WITH LOSS: outer setup failed (SP eval).
        # No stack data is synced from Rust to Python; Python
        # callback runs with potentially stale stack contents.
        with contextlib.suppress(Exception):
            sp = state.solver.eval(state.regs._sp) if not state.regs._sp.symbolic else None
            if not sp:
                return
            ptr_size = state.arch.bytes

            # Load the full page containing SP in ONE FFI call
            sp_page = sp & ~PAGE_MASK
            sp_offset = sp - sp_page
            # cat-(a) EXPECTED CONTROL FLOW: bulk-page FFI not
            # available or failed; fall through to per-pointer
            # fallback below.
            with contextlib.suppress(Exception):
                page_data = self._rust_mgr.pending_memory_load_page(self._current_callback_state_id, sp_page)
                if page_data and len(page_data) == PAGE_SIZE:
                    # Write non-zero pointer-sized values from SP upward
                    # Covers 64 slots (~512 bytes on x64) for args + locals
                    end_offset = min(sp_offset + 64 * ptr_size, PAGE_SIZE)
                    for off in range(sp_offset, end_offset, ptr_size):
                        chunk = page_data[off : off + ptr_size]
                        if len(chunk) == ptr_size:
                            int_val = int.from_bytes(chunk, "little")
                            if int_val != 0:
                                addr = sp_page + off
                                # Pass int directly — UltraPage fast path avoids claripy BVV
                                state.memory.store(
                                    addr, int_val, size=ptr_size, endness="Iend_LE", inspect=False, disable_actions=True
                                )
                    return

            # Fallback: individual loads
            for i in range(16):
                addr = sp + i * ptr_size
                # cat-(b) FALLBACK WITH LOSS: individual pointer
                # load failed; that slot is not synced. Python
                # callback may see zeros where Rust has data.
                with contextlib.suppress(Exception):
                    data = self._rust_mgr.pending_memory_load(self._current_callback_state_id, addr, ptr_size)
                    if data and len(data) == ptr_size:
                        int_val = int.from_bytes(data, "little")
                        if int_val != 0:
                            val = claripy.BVV(int_val, ptr_size * 8)
                            state.memory.store(addr, val, endness="Iend_LE", inspect=False, disable_actions=True)

    @staticmethod
    def _acquire_ultrapage(state: angr.SimState, page_addr: int, memory=None):
        """Get the writable UltraPage backing `page_addr`, or None.

        Used by the dirty-page replay and register-apply fast paths to talk to
        the page directly instead of through the memory-mixin stack.
        ``writing=True`` routes through ``acquire_unique()``, so a page shared
        with another state is copy-on-written rather than mutated in place.

        ``memory`` selects which SimMemory to acquire from (default
        ``state.memory``); pass ``state.registers`` for the register file
        (angr-gorvf.19).

        Returns None — meaning "caller must use the generic ``memory.store``" —
        when this is not a 4096-byte UltraPage, or when any of the mixins the
        fast path skips would actually have work to do. Of the mixins above the
        paged store, only ConvenientMappingsMixin has store-side state, and it
        is inert unless one of these three options is set; Inspect and both
        Actions mixins are already disabled by the ``inspect=False`` /
        ``disable_actions=True`` the generic path passes, and the addresses and
        sizes here are concrete and exact, so address/size normalization are
        no-ops.
        """
        mem = memory if memory is not None else state.memory
        get_page = getattr(mem, "_get_page", None)
        if get_page is None or page_addr % PAGE_SIZE:
            return None
        if (
            "MEMORY_SYMBOLIC_BYTES_MAP" in state.options
            or "REVERSE_MEMORY_NAME_MAP" in state.options
            or "REVERSE_MEMORY_HASH_MAP" in state.options
        ):
            return None
        page = get_page(page_addr // PAGE_SIZE, True)
        # Callers write straight into concrete_data, so allocate it.
        if _ultrapage_bytes(page, writing=True) is None:
            return None
        return page

    @staticmethod
    def _blit_concrete_page(page, page_bytes) -> None:
        """Copy a full concrete page straight into its UltraPage backing.

        angr-gorvf.10: routing a whole page through ``state.memory.store`` is
        quadratic. ``UltraPage.store`` walks the value byte-by-byte with
        ``self.concrete_data[subaddr] = ival & 0xFF; ival >>= 8`` — for a
        4096-byte page that right-shifts a 32768-bit int 4096 times, which was
        nearly all of the ``sc_memreplay`` bill on csaw_wyvern (the FFI fetch
        of the same pages was a small fraction). A
        page-aligned, page-sized concrete write is a memcpy; do it as one.

        Semantics match the concrete branch of ``UltraPage.store``: the
        symbolic bitmap is cleared over the range and ``symbolic_data`` is
        left alone (loads consult the bitmap).
        """
        page.concrete_data[0:PAGE_SIZE] = page_bytes
        page.symbolic_bitmap.clear_range(0, PAGE_SIZE)

    @staticmethod
    def _blit_symbolic_page(page, page_addr: int, sym_entries, endness) -> bool:
        """Rebuild an UltraPage's symbolic half from Rust's object list in one pass.

        angr-gorvf.11: the page carries many symbolic objects on csaw_wyvern
        (most of them a single byte wide), and even a direct ``page.store``
        pays per-entry overhead on each: the
        cooperation generator wraps every entry in a ``SimMemoryObject``, then
        the store re-derives the range it must clear with two ``SortedDict``
        ``irange`` walks.

        That clear is dead work here. The caller has just blitted the concrete
        page, which zeroes ``symbolic_bitmap`` across all 4096 bytes, so no
        pre-existing ``symbolic_data`` entry is reachable by a load; and Rust's
        object list is the page's complete symbolic state. So the whole
        ``SortedDict`` can be rebuilt from the list directly.

        ONLY valid when ``_blit_concrete_page`` ran for this page — that is what
        makes the stale ``symbolic_data`` unreachable.

        Returns False (caller falls back to the per-entry ``page.store``) when
        an entry straddles the page boundary or has a bad width. The scan runs
        before anything is mutated, so a False leaves the page untouched.
        """
        decomposed = []
        for addr, ast in sym_entries:
            offset = addr - page_addr
            size = len(ast) // 8
            if size <= 0 or offset < 0 or offset + size > PAGE_SIZE:
                return False
            decomposed.append((offset, size, SimMemoryObject(ast, addr, endness)))

        bitmap = page.symbolic_bitmap
        symbolic_data = {}
        for offset, size, mo in decomposed:
            bitmap.set_range(offset, offset + size)
            symbolic_data[offset] = mo
        page.symbolic_data = SortedDict(symbolic_data)
        return True

    def _replay_rust_dirty_pages(self, state: angr.SimState):
        """Replay Rust-side memory mutations into the cached Python SimState.

        Walks `_get_pending_dirty_pages()` from the Rust pending state and
        writes per-page concrete bytes (`pending_memory_load_page`) then
        symbolic objects (`pending_memory_load_symbolic_page`) into
        `state.memory`. Symbolic stores are applied LAST per page so a
        symbolic byte does not get clobbered by a concrete-zero default
        sitting at the same address.

        This cache sync is what NativeRead/NativeWrite rely on (angr-3tek.2):
        native procs mutate Rust memory but never
        push into the symbolic-page snapshot consumed by
        `_restore_symbolic_pages`, so without this replay a later Python
        SimProc (e.g. strcmp) reads stale concrete-zero bytes.

        angr-ryf6: when the cached state already has the callback memory
        proxy installed from a prior
        callback, ``state.memory`` *is* the Rust state — replaying dirty
        pages would round-trip Rust → Python → Rust through
        ``set_state_memory_concrete``, which evaluates symbolic bytes to
        concrete witnesses on the read side and overwrites the symbolic
        objects on the write side (defcon2016quals_baby-re's flag_chars
        symbols got clobbered to space (0x20) between scanf calls). The
        proxy is already the single source of truth, so the replay is
        both redundant and lossy — skip it.
        """
        from angr.exploration.rust_manager import _is_rust_memory_proxy

        if _is_rust_memory_proxy(state.memory):
            return
        try:
            dirty_pages = self._rust_mgr.get_pending_dirty_pages(self._current_callback_state_id)
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: dirty-page API unavailable on
            # this build; skip replay. NativeRead/NativeWrite-style sync
            # gaps fall back to the older snapshot path.
            return
        if not dirty_pages:
            return

        for page_addr in dirty_pages:
            _ffi_ns = 0
            _store_ns = 0
            _t0 = time.perf_counter_ns()
            try:
                page_bytes = self._rust_mgr.pending_memory_load_page(self._current_callback_state_id, page_addr)
            except Exception:
                page_bytes = None
            _ffi_ns += time.perf_counter_ns() - _t0

            _t0 = time.perf_counter_ns()
            try:
                sym_entries = self._rust_mgr.pending_memory_load_symbolic_page(
                    self._current_callback_state_id, page_addr
                )
            except (AttributeError, RuntimeError):
                # cat-(a) EXPECTED CONTROL FLOW: symbolic-page FFI absent on
                # older builds; symbolic bytes fall through to the older
                # snapshot path on the next sync.
                sym_entries = None
            _ffi_ns += time.perf_counter_ns() - _t0

            # Acquire the target page once and drive it directly for both
            # halves of the replay; None means fall back to memory.store.
            page = self._acquire_ultrapage(state, page_addr)
            blitted = False

            if page_bytes and len(page_bytes) == PAGE_SIZE:
                _t0 = time.perf_counter_ns()
                try:
                    if page is not None:
                        self._blit_concrete_page(page, page_bytes)
                        blitted = True
                    else:
                        state.memory.store(
                            page_addr,
                            claripy.BVV(bytes(page_bytes), PAGE_SIZE * 8),
                            endness="Iend_BE",
                            inspect=False,
                            disable_actions=True,
                        )
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: concrete page replay failed;
                    # cached Python state's memory page stays out of sync with
                    # Rust until the next snapshot refresh.
                    if _DBG:
                        l.debug(f"dirty-page concrete replay failed at 0x{page_addr:x}: {e}")
                _store_ns += time.perf_counter_ns() - _t0

            _t0 = time.perf_counter_ns()
            n_sym = len(sym_entries) if sym_entries else 0
            if sym_entries:
                endness = state.arch.memory_endness
                # angr-gorvf.11: rebuild the whole symbolic half in one pass when
                # the concrete blit ran (it is what makes the page's stale
                # symbolic_data unreachable). Falls through to the per-entry loop
                # below for a straddling entry or a page the blit skipped.
                fast = False
                if page is not None and blitted:
                    try:
                        fast = self._blit_symbolic_page(page, page_addr, sym_entries, endness)
                    except Exception as e:
                        # cat-(a) EXPECTED CONTROL FLOW: unexpected page shape;
                        # the per-entry loop below is the general path.
                        if _DBG:
                            l.debug(f"dirty-page symbolic blit failed at 0x{page_addr:x}: {e}")
                        fast = False
                    if fast:
                        for _addr, ast in sym_entries:
                            self._register_handle(id(ast), ast)
                        sym_entries = ()
                for addr, ast in sym_entries:
                    try:
                        # angr-gorvf.10: one page carries many symbolic objects
                        # (csaw_wyvern) and each generic store pays the full
                        # mixin stack — most of the replay bill. Store
                        # into the already-acquired page instead; UltraPage.store
                        # with cooperate=False runs the same decomposition the
                        # mixin would have handed it.
                        offset = addr - page_addr
                        size = len(ast) // 8
                        if page is not None and size > 0 and offset >= 0 and offset + size <= PAGE_SIZE:
                            page.store(
                                offset,
                                ast,
                                size=size,
                                endness=endness,
                                memory=state.memory,
                                page_addr=page_addr,
                                cooperate=False,
                            )
                        else:
                            # Entry straddles the page boundary (or non-UltraPage
                            # backend): the mixin owns the split.
                            state.memory.store(
                                addr,
                                ast,
                                endness=endness,
                                inspect=False,
                                disable_actions=True,
                            )
                        self._register_handle(id(ast), ast)
                    except Exception as e:
                        # cat-(b) FALLBACK WITH LOSS: symbolic store at addr
                        # raised; that byte stays at the cached Python value
                        # instead of the Rust-side symbolic AST.
                        if _DBG:
                            l.debug(f"dirty-page symbolic replay failed at 0x{addr:x}: {e}")
            _sym_ns = time.perf_counter_ns() - _t0
            _store_ns += _sym_ns
            self._perf_stats.add_replay_page(_ffi_ns, _store_ns, _sym_ns, n_sym)

        try:
            self._rust_mgr.clear_pending_dirty_tracking(self._current_callback_state_id)
        except (AttributeError, RuntimeError):
            # cat-(a) EXPECTED CONTROL FLOW: clear-API missing on older builds;
            # dirty bits stay set but next replay is idempotent (writes the
            # same page bytes again).
            if _DBG:
                l.debug("clear_pending_dirty_tracking failed after replay")

    def _lookup_via_ancestry(self, getter, state_id, what=""):
        """Fetch a falsy-when-empty payload via direct, root, then ancestry lookup.

        ``getter`` is a callable taking a state_id and returning a payload that
        is falsy when empty (e.g. ``get_state_symbolic_pages``). Returns the
        first non-empty payload found walking direct -> pending root -> full
        ancestry chain, or ``None`` if nothing matched.
        """
        payload = getter(state_id) or None
        if payload:
            return payload
        root_id = self._get_pending_root_state_id()
        if root_id is not None:
            root_payload = getter(root_id)
            if root_payload:
                if _DBG:
                    l.debug(f"Using root {root_id} {what} for state {state_id}")
                return root_payload
        for ancestor_id in self._get_pending_ancestry():
            ancestor_payload = getter(ancestor_id)
            if ancestor_payload:
                if _DBG:
                    l.debug(f"Using ancestor {ancestor_id} {what} for state {state_id}")
                return ancestor_payload
        return None

    def _restore_symbolic_pages(self, state: angr.SimState, state_id: int):
        """Restore symbolic memory regions to an angr state.

        This restores symbolic values that were previously extracted and
        cached, ensuring that Python callbacks see the correct symbolic
        memory context after fallback from Rust.

        For forked states, tries parent chain if direct lookup fails.

        Args:
            state: The angr state to restore symbolic pages to.
            state_id: The state ID to look up cached symbolic pages.
        """
        # Try direct lookup, then ancestry chain for forked states
        symbolic_pages = self._lookup_via_ancestry(self._rust_mgr.get_state_symbolic_pages, state_id, "symbolic pages")
        if not symbolic_pages:
            if _DBG:
                l.debug(f"No symbolic pages found for state {state_id} or ancestors")
            return
        restored_count = 0
        failed_count = 0

        # Group contiguous regions for more efficient restoration
        # This reduces the number of store operations
        sorted_addrs = sorted(symbolic_pages.keys())
        i = 0
        while i < len(sorted_addrs):
            start_addr = sorted_addrs[i]
            ast = symbolic_pages[start_addr]

            # Check for single-byte symbolic values (most common case after byte-granular extraction)
            if ast.length == 8:  # 8 bits = 1 byte
                try:
                    state.memory.store(start_addr, ast, endness=state.arch.memory_endness)
                    restored_count += 1
                    self._register_handle(id(ast), ast)
                except Exception as e:
                    # cat-(c) WRONG-ANSWER RISK: per-byte symbolic restore
                    # failed; that byte is left at the state's pre-restore
                    # value (often concrete zero). Counted in failed_count
                    # which is already warn-logged below.
                    if _DBG:
                        l.debug(f"Error restoring symbolic byte at 0x{start_addr:x}: {e}")
                    failed_count += 1
                i += 1
            else:
                # Multi-byte symbolic value - store directly
                try:
                    state.memory.store(start_addr, ast, endness=state.arch.memory_endness)
                    restored_count += 1
                    self._register_handle(id(ast), ast)
                except Exception as e:
                    # cat-(c) WRONG-ANSWER RISK: multi-byte symbolic
                    # restore failed; entire region is wrong. failed_count
                    # already drives the warn-log below.
                    if _DBG:
                        l.debug(f"Error restoring symbolic memory at 0x{start_addr:x}: {e}")
                    failed_count += 1
                i += 1

        if restored_count > 0 and _DBG:
            l.debug(f"Restored {restored_count} symbolic memory regions for state {state_id}")
        if failed_count > 0:
            l.warning(f"Failed to restore {failed_count} symbolic memory regions")

    def _restore_hook_symbolic_memory(self, state: angr.SimState, state_id: int):
        """Restore symbolic memory that was tracked during hook execution.

        When hooks copy or manipulate symbolic memory, the symbolic ASTs are
        tracked in `_hook_symbolic_memory`. This method restores those ASTs
        to the callback state so subsequent operations preserve symbolic
        relationships.

        This is critical for examples like flareon2015_5 where hooks copy
        symbolic password bytes to new memory locations.

        For forked states, tries parent chain if direct lookup fails.

        Args:
            state: The angr state to restore symbolic memory to.
            state_id: The state ID to look up tracked symbolic memory.
        """
        # Try direct lookup, then ancestry chain for forked states
        hook_memory = self._lookup_via_ancestry(
            self._rust_mgr.get_state_hook_symbolic_memory, state_id, "hook symbolic memory"
        )
        if not hook_memory:
            return
        restored_count = 0

        for addr, (ast, size) in hook_memory.items():
            try:
                state.memory.store(addr, ast, endness=state.arch.memory_endness)
                self._register_handle(id(ast), ast)
                restored_count += 1
                if _DBG:
                    l.debug(f"Restored hook symbolic memory at 0x{addr:x} (size={size})")
            except Exception as e:
                # cat-(c) WRONG-ANSWER RISK: hook-tracked symbolic memory
                # not restored; subsequent Python operations on that
                # address will see concrete bytes instead of the hook's
                # symbolic AST (e.g. flareon2015_5 password bytes lost
                # symbolic identity). Debug-only because we don't have
                # a counter to emit a summary warn.
                if _DBG:
                    l.debug(f"Could not restore hook symbolic at 0x{addr:x}: {e}")

        if restored_count > 0 and _DBG:
            l.debug(f"Restored {restored_count} hook symbolic memory regions for state {state_id}")
