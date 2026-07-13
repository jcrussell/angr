"""Mixin for State synchronization between Python and Rust."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import claripy

from angr.rustylib.vex_engine import register_names_for_arch

from ._constants import MAX_OVERLAY_SECTION_SIZE, PAGE_MASK, PAGE_SIZE, STACK_SIZE

# Precomputed all-zero buffers for _sync_extra_python_pages fast-path
# zero-page detection (memcmp-based instead of any() byte iteration; saves
# ~14us / page in mma_howtouse-style workloads — see angr-b58a).
_ZERO_PAGE_BA = bytearray(PAGE_SIZE)
_ZERO_PAGE_BYTES = bytes(PAGE_SIZE)

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)
_DBG = l.isEnabledFor(logging.DEBUG)


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

    def _cached_z3_ast_ptr(self, expr, z3_backend):
        """Return the Z3 AST pointer for a symbolic claripy expression.

        Caches `(hash(expr), expr.length) -> (z3_obj, ast_ptr)` so repeated
        register sync of the same symbolic AST skips both `z3_backend.convert`
        and the `.as_ast().value` attribute walk. Holds a strong ref to the
        Z3 wrapper to keep the AST pointer valid (Z3 ASTs are refcounted in
        the shared context; without our ref, claripy may be the only holder
        and could drop it under memory pressure).
        """
        cache = self._z3_ptr_cache
        key = (hash(expr), expr.length)
        cached = cache.get(key)
        if cached is not None:
            self._z3_ptr_cache_hits += 1
            return cached[1]
        z3_obj = z3_backend.convert(expr)
        ast_ptr = z3_obj.as_ast().value
        if len(cache) >= self._z3_ptr_cache_max:
            cache.clear()
        cache[key] = (z3_obj, ast_ptr)
        self._z3_ptr_cache_misses += 1
        return ast_ptr

    @staticmethod
    def _supported_register_names(arch) -> list:
        """Registers that the Rust engine models for `arch`.

        Single source of truth: delegates to Rust's per-arch canonical
        register table via `register_names_for_arch`. Used by both the slow
        sync path and the disk-cache fast path so they agree on which
        registers cross the FFI boundary. archinfo defines many registers
        (cr0..8, ymm0..15, fs_seg, cmstart, ...) the Rust engine does not
        model; the Rust list excludes them, so `set_registers_bulk` will
        not raise ValueError on the names returned here.
        """
        return register_names_for_arch(arch.name)

    def _sync_registers_to_rust(
        self, angr_state: angr.SimState, rust_state: _RustSimState, precomputed_regs: dict = None
    ):
        """Sync registers from angr state to Rust state.

        Args:
            precomputed_regs: Optional dict of {reg_name: concrete_value}.
                When provided (e.g. from disk cache), skips reading registers
                from the SimState, saving ~0.5ms of angr register plugin overhead.
                The disk cache extractor skips symbolic registers, so the
                caller (RustExplorationManager._compute_disk_init_key /
                _state_has_user_symbolic) must invalidate the cache key when
                the state holds any user-set symbolic register — otherwise
                Rust would see the cached concrete value instead of the
                user's symbolic. See angr-g9hy.
        """
        arch = angr_state.arch
        reg_names = self._supported_register_names(arch)

        if precomputed_regs is not None:
            # Fast path: use pre-computed concrete register values directly.
            # Filter to registers Rust actually models — the disk cache stores
            # everything in arch.register_names but Rust only knows the subset
            # in _supported_register_names.
            supported = set(reg_names)
            filtered = {name: val for name, val in precomputed_regs.items() if name in supported}
            if filtered:
                rust_state.set_registers_bulk(filtered)
            return

        regs = angr_state.regs

        # Build dict of all register values, then send in single FFI call.
        # Symbolic registers are imported via Z3 AST pointers (shared context).
        import claripy as _claripy

        z3_backend = _claripy.backends.z3
        bulk_regs = {}
        for reg_name in reg_names:
            try:
                reg_val = getattr(regs, reg_name)
                if reg_val.op == "BVV":
                    bulk_regs[reg_name] = reg_val.args[0]
                elif not reg_val.symbolic:
                    bulk_regs[reg_name] = angr_state.solver.eval(reg_val)
                else:
                    # Import symbolic register via shared Z3 context.
                    # If conversion fails, the register is missing on the Rust
                    # side and any read will see an uninitialized BVS — wrong
                    # value, not a crash. Log loudly.
                    try:
                        # Layer 2 (angr-21vi5): route the full claripy AST
                        # through claripy_to_rustbv so leaf symbols (e.g. the
                        # ecx in `state.regs.ecx = BVS('ecx')`) intern into the
                        # shared symbol cache and round-trip back to the user's
                        # symbol on export. The raw-Z3-ptr path
                        # (set_register_symbolic) wraps the whole register as an
                        # opaque id:0 symbol and mints a fresh rcx_N on the way
                        # out — see angr-4ju9e. Fall back to the ptr path if the
                        # AST-based method is unavailable (older .so).
                        if hasattr(rust_state, "set_register_symbolic_ast"):
                            rust_state.set_register_symbolic_ast(reg_name, reg_val)
                        else:
                            ast_ptr = self._cached_z3_ast_ptr(reg_val, z3_backend)
                            if ast_ptr:
                                rust_state.set_register_symbolic(reg_name, ast_ptr, reg_val.length)
                    except Exception as e:
                        # cat-(c) WRONG-ANSWER RISK: symbolic register not
                        # synced; Rust will see stale/uninit BVS for this
                        # register. Already logged at warn (commit
                        # 4f16e3792 / angr-i1wg).
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
        if self._try_fast_memory_sync(rust_state):
            return

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

    def _try_fast_memory_sync(self, rust_state: _RustSimState) -> bool:
        """Apply cached memory layout from disk init cache. Returns True if used."""
        if self._mem_cache is None:
            return False
        mem = self._mem_cache
        self._mem_cache = None  # Consume once
        try:
            if mem.get("batch_pages"):
                rust_state.map_memory_batch(mem["batch_pages"])
            for addr, patch_bytes in mem.get("section_patches", []):
                rust_state.map_memory_data(addr, patch_bytes, 7)
            if mem.get("stack_page"):
                sp_page, page_bytes = mem["stack_page"]
                rust_state.map_memory_data(sp_page, page_bytes, 6)
            for start, size in mem.get("lazy_regions", []):
                rust_state.add_lazy_region(start, size)
            l.debug(f"Fast memory sync from cache: {len(mem.get('batch_pages', []))} pages")
            return True
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: fast memory sync from disk cache
            # failed (e.g. cache version skew, FFI mismatch). The slow path
            # in _sync_memory_to_rust then runs from scratch; result is
            # correct but cold-init latency is paid.
            l.debug(f"Fast memory sync failed, falling back: {e}")
            return False

    def _find_user_symbolic_pages(self, angr_state: angr.SimState, page_size: int) -> set:
        """Find pages that contain user-written symbolic data.

        These pages are NOT pre-populated with concrete loader data, so Rust
        falls back to the Python memory_load callback (which returns the
        symbolic AST and enables symbolic forking on comparisons).
        """
        symbolic_pages: set = set()
        if hasattr(angr_state.memory, "get_symbolic_addrs"):
            try:
                for addr in angr_state.memory.get_symbolic_addrs():
                    symbolic_pages.add(addr & ~(page_size - 1))
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: get_symbolic_addrs() failed;
                # fall through to the _pages bitmap scan below. Both
                # populate symbolic_pages, so this is a graceful fallback.
                pass
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
            try:
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
                        try:
                            data = self._project.loader.memory.load(page_addr, page_size)
                            if data and len(data) == page_size:
                                batch_pages.append((page_addr, bytes(data), 7))
                                seen_page_addrs.add(page_addr)
                        except Exception:
                            # cat-(a) EXPECTED CONTROL FLOW: per-page load
                            # may hit unmapped gaps in object's page range
                            # (esp. ELF .bss gaps). Skip silently — the
                            # lazy region added below catches accesses.
                            pass
                region_start = obj.min_addr & ~(page_size - 1)
                region_end = (obj.max_addr + page_size) & ~(page_size - 1)
                if region_end - region_start > 0:
                    lazy_regions.append((region_start, region_end - region_start))
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: per-object iteration failed
                # (object lacks expected attributes). That object's pages
                # won't be eagerly mapped; lazy_region in
                # _add_loader_lazy_regions catches accesses on demand.
                pass
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
        symbolic_pages so callbacks can serve them. Reads the cached
        `_extract_loader_pages` output (built on first construction per
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
                    try:
                        val = angr_state.memory.load(
                            section.min_addr, section.memsize, endness="Iend_BE", inspect=False, disable_actions=True
                        )
                        if not val.symbolic:
                            data = angr_state.solver.eval(val).to_bytes(section.memsize, "big")
                            rust_state.map_memory_data(section.min_addr, data, 7)
                    except Exception:
                        # cat-(b) FALLBACK WITH LOSS: GOT/relocation
                        # overlay failed (load OOB or eval timeout). Rust
                        # sees the raw loader-mapped data without
                        # relocations applied — may misresolve external
                        # references. Lazy fetch_page from Python catches
                        # this on access in most cases.
                        pass

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
            try:
                concrete = bytes(page_obj.concrete_load(0, page_size))
                if len(concrete) == page_size:
                    rust_state.map_memory_data(page_addr, concrete, 7)
                    state_overlay_count += 1
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: page lacks concrete_load or
                # contains symbolic bytes that can't be eagerly loaded.
                # Rust falls back to memory_load callback for this page.
                pass
        if state_overlay_count:
            l.debug(f"Overlaid {state_overlay_count} loader pages with Python state data")

    def _add_loader_lazy_regions(self, rust_state: _RustSimState, page_size: int) -> None:
        """Register every loaded object as a lazy region for fetch_page callbacks.

        Uses the cached lazy_regions list (computed once per project, shared
        across managers) to avoid re-iterating `loader.all_objects`.
        """
        entry = self._get_loader_pages_cache(page_size)
        for region_start, region_size in entry["lazy_regions"]:
            try:
                rust_state.add_lazy_region(region_start, region_size)
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: lazy region registration
                # failed for this object; accesses to its pages can't
                # be auto-fetched and will fail to load on demand.
                pass

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
        try:
            if page_obj is not None and hasattr(page_obj, "concrete_load"):
                try:
                    concrete = bytes(page_obj.concrete_load(0, page_size))
                    if len(concrete) == page_size:
                        rust_state.map_memory_data(sp_page, concrete, 6)
                        used_fast_path = True
                        # Targeted scan of symbolic_data ranges only (~1-5ms)
                        # vs full _extract_symbolic_regions scan (~137ms).
                        sd = getattr(page_obj, "symbolic_data", None)
                        if sd:
                            self._extract_stack_symbolic_from_sd(angr_state, sp_page, page_size, sd, symbolic_regions)
                except Exception:
                    # cat-(a) EXPECTED CONTROL FLOW: concrete_load fast
                    # path failed (page has symbolic content). Fall
                    # through to slow path below.
                    pass

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
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: stack page sync failed entirely
            # (slow path raised). Rust will fetch the stack page lazily
            # via fetch_page callback on first access — correct but pays
            # FFI roundtrip per page.
            pass

    def _extract_stack_symbolic_from_sd(
        self, angr_state: angr.SimState, page_addr: int, page_size: int, sd: dict, symbolic_regions: list
    ) -> None:
        """Targeted scan of a stack page's symbolic_data ranges.

        Avoids the full 4096-byte byte-by-byte scan in _extract_symbolic_regions
        — this only walks the explicitly-stored symbolic byte ranges.
        """
        scan_ranges = []
        for sd_offset, sd_ast in sd.items():
            if not hasattr(sd_ast, "variables"):
                continue
            if self._has_user_symbolic_var(sd_ast):
                ast_size = sd_ast.size() // 8 if hasattr(sd_ast, "size") else 1
                scan_ranges.append((sd_offset, ast_size))
        for start_offset, size in scan_ranges:
            for byte_off in range(size):
                if start_offset + byte_off >= page_size:
                    break
                addr = page_addr + start_offset + byte_off
                try:
                    val = angr_state.memory.load(addr, 1, endness="Iend_BE", inspect=False, disable_actions=True)
                    if val.symbolic:
                        symbolic_regions.append((addr, val))
                except Exception:
                    # cat-(b) FALLBACK WITH LOSS: per-byte symbolic load
                    # failed; that byte position is not added to
                    # symbolic_regions and will be served as the concrete
                    # fast-path byte. Loss is partial: only that single
                    # byte's symbolic identity is missed.
                    pass

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
        # angr-9maq made zero pages lazy-only to avoid mma_howtouse blowing
        # up to 1.9GB (2000+ zero pages per state * 45 short-lived states).
        # But hackcon2016 has only ~35 zero pages per state, and keeping
        # them lazy regressed final-solve Z3 time from ~10s to ~28s (root
        # cause is structural — see angr-8t45 memory). Cap below: if a
        # state has more all-zero pages than this, keep them all lazy
        # (mma path); otherwise eager-map (hackcon path).
        zero_eager_cap = 200
        try:
            mem_pages = getattr(angr_state.memory, "_pages", None)
            if mem_pages is None:
                return
            # angr-b58a: classify pages without materializing 4KB bytes objects.
            # For UltraPage (the default backend) we read concrete_data
            # directly — a bytearray-vs-bytearray compare is ~0.1us / page
            # (memcmp) versus ~15us / page for `any(bytes_iter)`. The bytes()
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
                cd = getattr(page_obj, "concrete_data", None)
                if isinstance(cd, bytearray) and len(cd) == page_size:
                    # UltraPage fast path: classify via memcmp on the backing
                    # bytearray. No bytes() copy yet — deferred to the FFI
                    # phase (and skipped entirely for lazy-only zero pages).
                    is_nonzero = cd != _ZERO_PAGE_BA
                    raw_pages.append((page_addr, page_obj, is_nonzero, perms, None))
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
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: outer iteration failed (rare,
            # _pages dict shape mismatch). All non-loader pages skip
            # eager sync; lazy fetch_page handles them.
            pass
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
        init — scanning those is ~0.3ms/page.
        """
        try:
            pages = getattr(angr_state.memory, "_pages", {})
            user_sym_pages = set()
            if hasattr(angr_state.memory, "get_symbolic_addrs"):
                try:
                    for addr in angr_state.memory.get_symbolic_addrs():
                        user_sym_pages.add(addr // page_size)
                except Exception:
                    # cat-(b) FALLBACK WITH LOSS: get_symbolic_addrs() is
                    # an opt-in API; if it raises, we fall through to the
                    # symbolic_data fallback walk below.
                    pass
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
                try:
                    page_data = angr_state.memory.load(
                        page_addr, page_size, endness="Iend_BE", inspect=False, disable_actions=True
                    )
                    if page_data.symbolic and self._has_user_symbolic_var(page_data):
                        self._extract_wide_symbolic_regions(angr_state, page_addr, page_size, symbolic_regions)
                except Exception:
                    # cat-(c) WRONG-ANSWER RISK: per-page scan failed; user
                    # symbolic data on this page is not added to
                    # symbolic_regions, so Rust later sees concrete bytes
                    # instead of the symbolic AST. Solver will pick a
                    # concrete value rather than fork on the symbol.
                    # (Wider impact than (b) per-byte miss because the
                    # whole page is dropped.)
                    pass
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: outer scan failed (rare).
            # symbolic_regions is empty, so user symbolic memory is not
            # imported into Rust — same risk class as the inner cat-(c)
            # but at this point the exploration likely won't run at all.
            pass

    def _extract_wide_symbolic_regions(self, angr_state, page_addr, page_size, out):
        """Extract WIDE symbolic regions from a page.

        Groups contiguous symbolic bytes that share the same variable and
        imports them as a single wide object. This preserves symbolic identity
        across the Rust/Python boundary (avoids creating rust_sym_XXX aliases).

        Uses page.symbolic_data dict to find symbolic ranges, avoiding full
        4096-byte scan (~137ms → ~1ms).
        """
        # Fast path: use symbolic_data dict to find which ranges to scan
        page_no = page_addr // page_size
        mem_pages = getattr(angr_state.memory, "_pages", None)
        page_obj = mem_pages.get(page_no) if mem_pages is not None else None

        if page_obj is not None and hasattr(page_obj, "symbolic_data"):
            sd = page_obj.symbolic_data
            if sd:
                # Determine byte ranges that need scanning from symbolic_data entries
                scan_ranges = []
                for sd_offset, sd_ast in sd.items():
                    if not hasattr(sd_ast, "variables"):
                        continue
                    if self._has_user_symbolic_var(sd_ast):
                        ast_size = sd_ast.size() // 8 if hasattr(sd_ast, "size") else 1
                        scan_ranges.append((sd_offset, ast_size))

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
                                try:
                                    val = angr_state.memory.load(
                                        addr, 1, endness="Iend_BE", inspect=False, disable_actions=True
                                    )
                                    if val.symbolic:
                                        out.append((addr, val))
                                except Exception:
                                    # cat-(b) FALLBACK WITH LOSS: per-byte
                                    # load also failed; that byte not
                                    # imported, served as concrete instead.
                                    pass
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
            try:
                val = angr_state.memory.load(addr, 1, endness="Iend_BE", inspect=False, disable_actions=True)
                if val.symbolic:
                    # Check if this contains a user-defined variable
                    # (not just unconstrained fill from entry_state)
                    if self._has_user_symbolic_var(val):
                        out.append((addr, val))
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: per-byte symbolic load
                # failed; that byte's symbolic identity is not imported.
                # The page's concrete bytes still load via fast paths,
                # so the user-visible value is wrong only if the missed
                # byte was meaningful (e.g. a constraint leaf).
                pass

    def _concretize_stack_registers(self, state: angr.SimState):
        """Concretize stack registers for Rust memory mapping compatibility.

        This prevents symbolic address issues during Rust exploration by
        ensuring stack-relative registers have concrete values.

        Fast path: if the register's symbolic variables don't appear in any
        solver constraint, skip solver.eval and pick a sensible default —
        same value Z3 would have picked, without paying the ~10 ms model-query
        cost. Common on blank_state cold path where rbp is filled with a
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
        but the call still costs ~10 ms (ctx init + check + model). When we
        can prove no constraint mentions any of the value's variables, return
        `default` directly — Z3 would have picked something arbitrary anyway,
        and downstream code only requires *some* concrete value.
        """
        val_vars = reg_val.variables
        for c in state.solver.constraints:
            if val_vars & c.variables:
                return state.solver.eval(reg_val)
        return default

    def _sync_registers_from_rust_pending(self, state: angr.SimState):
        """Sync register values from Rust pending state to angr state.

        Handles both concrete and symbolic registers. Concrete values are
        set directly. Symbolic values are converted from Rust Z3 BVs to
        claripy ASTs via rustbv_to_claripy, preserving symbolic identity.
        """
        arch = self._project.arch
        reg_names = self._get_arch_register_names(arch)

        reg_map = self._get_register_offset_map(arch)
        for reg_name in reg_names:
            try:
                # Try concrete first (fast path — direct store bypasses claripy)
                val = self._rust_mgr.get_pending_register(self._current_callback_state_id, reg_name)
                if val is not None:
                    offset_size = reg_map.get(reg_name)
                    if offset_size is not None:
                        state.registers.store(offset_size[0], val, size=offset_size[1])
                    else:
                        setattr(state.regs, reg_name, claripy.BVV(val, arch.bits))
                else:
                    # Register is symbolic — convert to claripy AST
                    try:
                        ast = self._rust_mgr.get_pending_register_ast(self._current_callback_state_id, reg_name)
                        if ast is not None:
                            setattr(state.regs, reg_name, ast)
                    except Exception:
                        # cat-(c) WRONG-ANSWER RISK: symbolic register
                        # conversion failed; Python state keeps its old
                        # value while Rust has a different one. Log at
                        # debug — caller hits the warn-level register
                        # sync error site at line 192 (cat-c) only when
                        # this is reachable through the export path.
                        pass
            except Exception:
                # cat-(c) WRONG-ANSWER RISK: register fetch failed; Python
                # state keeps stale value. Same risk class as above —
                # debug logged because higher-level export sites catch
                # divergence.
                pass

    def _is_binary_code_addr(self, addr: int) -> bool:
        """Check if address is in real binary code (not extern/loader space).

        Cached on first call. Only considers ELF objects with actual binary
        files, excluding CLE's ExternObject, KernelObject, TLSObject etc.
        """
        if not hasattr(self, "_binary_addr_ranges"):
            self._binary_addr_ranges = []
            for obj in self._project.loader.all_objects:
                # Only include real binary files (ELF, PE, etc.)
                # Skip CLE's synthetic objects (ExternObject, KernelObject, TLS)
                binary_path = getattr(obj, "binary", None)
                if not binary_path or not isinstance(binary_path, str) or binary_path.startswith("cle##"):
                    continue
                if hasattr(obj, "segments") and obj.segments:
                    for seg in obj.segments:
                        if seg.memsize > 0:
                            self._binary_addr_ranges.append((seg.min_addr, seg.max_addr))
                else:
                    self._binary_addr_ranges.append((obj.min_addr, obj.max_addr))
        return any(lo <= addr <= hi for lo, hi in self._binary_addr_ranges)

    def _get_register_offset_map(self, arch) -> dict:
        """Get cached {name: (offset, size)} mapping for register fast-path writes."""
        if not hasattr(self, "_reg_offset_cache"):
            self._reg_offset_cache = {}
        arch_name = arch.name
        if arch_name not in self._reg_offset_cache:
            mapping = {}
            for name in self._get_arch_register_names(arch):
                try:
                    info = arch.registers.get(name)
                    if info is not None:
                        mapping[name] = (info[0], info[1])  # (offset, size_bytes)
                except Exception:
                    # cat-(a) EXPECTED CONTROL FLOW: probing arch metadata
                    # for an optional register name; absence is normal.
                    pass
            self._reg_offset_cache[arch_name] = mapping
        return self._reg_offset_cache[arch_name]

    def _get_arch_register_names(self, arch) -> list:
        """Get register names for an architecture."""
        if arch.name in ("AMD64", "X86_64"):
            return [
                "rax",
                "rbx",
                "rcx",
                "rdx",
                "rsi",
                "rdi",
                "rbp",
                "rsp",
                "r8",
                "r9",
                "r10",
                "r11",
                "r12",
                "r13",
                "r14",
                "r15",
                "rip",
            ]
        if arch.name == "X86":
            return ["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp", "eip"]
        if arch.name == "AARCH64":
            return ["x%d" % i for i in range(31)] + ["sp", "pc"]
        if arch.name.startswith("ARM"):
            return ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "sp", "lr", "pc"]
        return []

    @staticmethod
    def _get_reg_map_and_return_regs(arch):
        """Get architecture-specific register map and return registers.

        Returns:
            Tuple of (reg_map, return_regs) or (None, None) if unsupported.
        """
        if arch.name in ("AMD64", "X86_64"):
            reg_map = {
                "rax": (16, 8),
                "rcx": (24, 8),
                "rdx": (32, 8),
                "rbx": (40, 8),
                "rsp": (48, 8),
                "rbp": (56, 8),
                "rsi": (64, 8),
                "rdi": (72, 8),
                "r8": (80, 8),
                "r9": (88, 8),
                "r10": (96, 8),
                "r11": (104, 8),
                "r12": (112, 8),
                "r13": (120, 8),
                "r14": (128, 8),
                "r15": (136, 8),
                "rip": (184, 8),
            }
            return_regs = {"rax"}
        elif arch.name == "X86":
            reg_map = {
                "eax": (8, 4),
                "ecx": (12, 4),
                "edx": (16, 4),
                "ebx": (20, 4),
                "esp": (24, 4),
                "ebp": (28, 4),
                "esi": (32, 4),
                "edi": (36, 4),
                "eip": (68, 4),
            }
            return_regs = {"eax"}
        elif arch.name in ("ARMEL", "ARMHF", "ARM"):
            reg_map = {
                "r0": (8, 4),
                "r1": (12, 4),
                "r2": (16, 4),
                "r3": (20, 4),
                "r4": (24, 4),
                "r5": (28, 4),
                "r6": (32, 4),
                "r7": (36, 4),
                "r8": (40, 4),
                "r9": (44, 4),
                "r10": (48, 4),
                "r11": (52, 4),
                "r12": (56, 4),
                "sp": (60, 4),
                "lr": (64, 4),
                "pc": (68, 4),
            }
            return_regs = {"r0"}
        elif arch.name == "AARCH64":
            reg_map = {("x%d" % i): (16 + i * 8, 8) for i in range(31)}
            reg_map["sp"] = (264, 8)
            reg_map["pc"] = (272, 8)
            return_regs = {"x0"}
        else:
            l.warning(f"Unknown architecture {arch.name} for register extraction")
            return None, None
        return reg_map, return_regs

    def _snapshot_registers(self, state) -> dict:
        """Snapshot register values as a lightweight dict for later comparison.

        Much cheaper than state.copy() — only reads register values (~0.1ms
        vs ~1ms for full state copy). Used for non-memory-writing extern
        SimProcedures where we only need to detect register changes.

        Returns:
            Dict mapping reg_name -> (is_symbolic, concrete_value_or_None, offset, size).
        """
        reg_map, _ = self._get_reg_map_and_return_regs(state.arch)
        if reg_map is None:
            return {}

        snapshot = {}
        for reg_name, (offset, size) in reg_map.items():
            try:
                val = getattr(state.regs, reg_name)
                if val.symbolic:
                    snapshot[reg_name] = (True, None, offset, size)
                else:
                    # Fast path: BVV values have concrete int in args[0]
                    concrete = val.args[0] if val.op == "BVV" else state.solver.eval(val)
                    snapshot[reg_name] = (False, concrete, offset, size)
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: register read failed for
                # this entry; snapshot misses it, so a later
                # _extract_register_changes can't compare and the
                # changed register won't be propagated.
                pass
        return snapshot

    def _snapshot_sp(self, state_or_snapshot) -> int | None:
        """Return the concrete stack pointer from a SimState or register snapshot.

        Accepts either a SimState or the dict produced by
        ``_snapshot_registers`` / ``_snapshot_registers_from_bundle``
        (reg_name -> (is_symbolic, concrete, offset, size)). Returns None if
        the sp is symbolic or unavailable. Used by the self.call() continuation
        capture in ``_resume_with_state`` (angr-aca6y).
        """
        if isinstance(state_or_snapshot, dict):
            for _name, (is_sym, concrete, _off, _sz) in state_or_snapshot.items():
                # offset match is the robust key, but sp/rsp/esp names cover it
                if _name in ("rsp", "esp", "sp") and not is_sym:
                    return concrete
            return None
        try:
            sp_ast = state_or_snapshot.regs.sp
            if sp_ast.symbolic:
                return None
            return state_or_snapshot.solver.eval(sp_ast)
        except Exception:
            return None

    def _snapshot_registers_from_bundle(self, bundle_regs: dict, arch) -> dict:
        """Build a register snapshot from a Rust callback bundle's registers.

        Same output format as _snapshot_registers, but skips reading values
        back from state — _create_state_for_callback just stored these same
        values, so we can construct the snapshot directly from the bundle.

        Args:
            bundle_regs: Dict mapping reg_name -> int|None (None = symbolic).
            arch: The state's architecture, used to look up offsets/sizes.

        Returns:
            Dict mapping reg_name -> (is_symbolic, concrete_value_or_None, offset, size).
        """
        reg_map = self._get_register_offset_map(arch)
        snapshot = {}
        for reg_name, val in bundle_regs.items():
            offset_size = reg_map.get(reg_name)
            if offset_size is None:
                continue
            offset, size = offset_size
            if val is not None:
                snapshot[reg_name] = (False, val, offset, size)
            else:
                snapshot[reg_name] = (True, None, offset, size)
        return snapshot

    def _extract_register_changes(self, old_state, new_state: angr.SimState) -> list:
        """Extract register changes between states.

        Handles both concrete and symbolic register values. For symbolic
        values (especially return registers like RAX), the value is converted
        to a claripy AST and stored in Rust's pending symbolic state.

        Args:
            old_state: Either a SimState or a register snapshot dict from
                       _snapshot_registers(). Using a snapshot avoids the
                       cost of state.copy() for non-memory-writing callbacks.
            new_state: The successor SimState after callback execution.

        Returns:
            List of (offset, size, data_bytes) tuples for concrete changes.
        """
        changes = []
        is_snapshot = isinstance(old_state, dict)
        arch = new_state.arch

        reg_map, return_regs = self._get_reg_map_and_return_regs(arch)
        if reg_map is None:
            return []

        for reg_name, (offset, size) in reg_map.items():
            try:
                new_val = getattr(new_state.regs, reg_name)

                # Get old value from snapshot or state
                if is_snapshot:
                    entry = old_state.get(reg_name)
                    if entry is None:
                        continue
                    old_is_symbolic, old_concrete, _, _ = entry
                    old_ast = None
                else:
                    old_ast = old_val = getattr(old_state.regs, reg_name)
                    old_is_symbolic = old_val.symbolic
                    old_concrete = (
                        None
                        if old_is_symbolic
                        else (old_val.args[0] if old_val.op == "BVV" else old_state.solver.eval(old_val))
                    )

                if new_val.symbolic:
                    # Symbolic register value — sync the AST to Rust. This is not
                    # limited to return registers: a hook may assign a symbolic
                    # address expression to any register (flareon2015_5 does
                    # ``state.regs.ecx = ebp - 0x70004``, angr-5rjbq), and dropping
                    # it leaves Rust executing with the stale pre-callback value.
                    if reg_name not in return_regs and not self._symbolic_reg_changed(
                        old_ast, old_is_symbolic, new_val
                    ):
                        continue
                    try:
                        self._sync_symbolic_register_to_rust(reg_name, new_val)
                        if _DBG:
                            l.debug(f"Synced symbolic register {reg_name} to Rust")
                    except Exception as e:
                        # cat-(b) FALLBACK WITH LOSS: symbolic-AST
                        # sync failed; we try to concretize as a
                        # last resort, losing symbolic identity for
                        # the register.
                        if _DBG:
                            l.debug(f"Could not sync symbolic {reg_name}: {e}")
                        try:
                            new_concrete = new_state.solver.eval(new_val)
                            data = new_concrete.to_bytes(size, "little")
                            changes.append((offset, size, bytes(data)))
                        except Exception:
                            # cat-(c) WRONG-ANSWER RISK: both AST
                            # sync and concretization failed; the
                            # register change is dropped
                            # entirely. Rust will continue with the
                            # pre-callback value, possibly diverging
                            # from Python's intent. Debug-only log
                            # because higher-level callback dispatch
                            # surfaces hook errors.
                            pass
                else:
                    # Fast path: extract concrete value without solver.eval()
                    # BVV values have the concrete int in args[0]
                    new_concrete = new_val.args[0] if new_val.op == "BVV" else new_state.solver.eval(new_val)
                    if old_is_symbolic or old_concrete != new_concrete:
                        data = new_concrete.to_bytes(size, "little")
                        changes.append((offset, size, bytes(data)))
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: register diff failed;
                # this register's change isn't propagated to Rust.
                # Rust then runs with stale value for this register —
                # similar to the (c) above but for non-return regs
                # where divergence often goes unnoticed.
                pass

        return changes

    @staticmethod
    def _symbolic_reg_changed(old_ast, old_is_symbolic: bool, new_val) -> bool:
        """Whether a now-symbolic register actually changed during the callback.

        Used to gate the symbolic-AST sync for non-return registers, so an
        already-symbolic register that the callback never touched isn't
        re-exported to Rust on every callback.
        """
        if not old_is_symbolic:
            return True
        if old_ast is None:
            # Register-snapshot path (``_snapshot_registers*``): the old AST was
            # never captured, so an unchanged symbol is indistinguishable from a
            # re-assignment. Skip rather than re-sync every callback; return
            # registers bypass this check and are always synced.
            return False
        return old_ast.hash() != new_val.hash()

    def _sync_symbolic_register_to_rust(self, reg_name: str, value):
        """Sync a symbolic register value to Rust pending state.

        Tries direct AST sync first (best approach), falls back to handle-based
        sync if the direct method is not available.
        """
        try:
            # Best approach: directly sync claripy AST to Rust
            if hasattr(self._rust_mgr, "set_pending_register_symbolic_ast"):
                self._rust_mgr.set_pending_register_symbolic_ast(self._current_callback_state_id, reg_name, value)
                if _DBG:
                    l.debug(f"Synced symbolic register {reg_name} to Rust via AST")
                return

            # Fallback: use handle-based sync
            if hasattr(self._rust_mgr, "claripy_ast_to_handle"):
                handle = self._rust_mgr.claripy_ast_to_handle(value)
                self._rust_mgr.set_pending_register_symbolic(self._current_callback_state_id, reg_name, handle.id())
                if _DBG:
                    l.debug(f"Synced symbolic register {reg_name} to Rust via handle")
                return

            # Final fallback: just register the handle for later retrieval
            handle_id = id(value)
            self._register_handle(handle_id, value)
            if _DBG:
                l.debug(f"Registered symbolic register {reg_name} handle for later retrieval")

        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: symbolic register sync to Rust
            # failed at the entry point. The caller's fallback in
            # _extract_register_changes catches this and tries to
            # concretize, but if that also fails the register change is
            # lost. Debug-only log because the caller's branch logs at
            # warn when both paths fail.
            if _DBG:
                l.debug(f"Could not sync symbolic {reg_name}: {e}")

    def _extract_memory_changes(self, old_state: angr.SimState, new_state: angr.SimState) -> tuple:
        """Extract memory changes between states for Rust sync.

        Uses angr's changed_bytes() to detect memory modifications,
        groups them into contiguous regions, and returns concrete values.
        Also tracks symbolic values for constraint propagation.

        Returns:
            Tuple of:
            - List of (addr, data_bytes) for concrete memory changes
            - List of (addr, ast) for symbolic memory that needs import
        """
        concrete_changes = []
        symbolic_imports = []  # Collect symbolic ASTs for import to Rust
        try:
            # Use angr's changed_bytes to find modifications
            changed = new_state.memory.changed_bytes(old_state.memory)
            if not changed:
                return [], []

            # Limit to prevent timeouts on large diffs (e.g., unconstrained fill)
            if len(changed) > 10000:
                if _DBG:
                    l.debug(f"Too many changed bytes ({len(changed)}), truncating to 10000")
                changed = set(sorted(changed)[:10000])

            # Group consecutive changed bytes into regions
            for item in self._group_changed_bytes(new_state, changed):
                if item[0] == "concrete":
                    _, start, size, data = item
                    concrete_changes.append((start, bytes(data)))
                elif item[0] == "symbolic":
                    _, start, size, data, handle_id, ast = item
                    # Provide concrete witness for Rust memory sync
                    concrete_changes.append((start, bytes(data)))
                    # Cache symbolic value for later constraint sync
                    # The handle is already registered in _emit_memory_region
                    if _DBG:
                        l.debug(f"Tracked symbolic memory change at 0x{start:x} (handle={handle_id})")

                    # Collect symbolic AST for import to Rust
                    symbolic_imports.append((start, ast))

                    # Track symbolic AST for state restoration during callbacks
                    # This is critical: hooks that copy symbolic memory would lose
                    # the symbolic relationship without this tracking
                    state_id = self._current_callback_state_id
                    if state_id is not None:
                        try:
                            self._rust_mgr.set_state_hook_symbolic_memory(state_id, start, ast, size)
                        except Exception:
                            # cat-(b) FALLBACK WITH LOSS: preserving the
                            # symbolic AST per-state failed; subsequent
                            # restoration during callbacks will see the
                            # concrete witness (in symbolic_imports)
                            # without the symbolic relationship.
                            pass
                        if _DBG:
                            l.debug(f"Preserved symbolic memory at 0x{start:x} for state {state_id}")

        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: outer extract failed (e.g.
            # changed_bytes raised). No memory diff is emitted; Rust
            # will diverge from Python on this callback's writes.
            if _DBG:
                l.debug(f"Error extracting memory changes: {e}")

        return concrete_changes, symbolic_imports

    def _group_changed_bytes(self, state: angr.SimState, changed_addrs):
        """Group consecutive changed bytes into contiguous regions.

        Yields tuples from _emit_memory_region:
            ('concrete', start_addr, size, data_bytes) for concrete regions
            ('symbolic', start_addr, size, data_bytes, handle_id, ast) for symbolic regions
        """
        if not changed_addrs:
            return

        sorted_addrs = sorted(changed_addrs)
        start = sorted_addrs[0]
        end = start + 1

        for addr in sorted_addrs[1:]:
            if addr == end:
                # Contiguous with current region
                end += 1
            else:
                # Gap found - emit current region and start new one
                yield from self._emit_memory_region(state, start, end - start)
                start = addr
                end = addr + 1

        # Emit final region
        yield from self._emit_memory_region(state, start, end - start)

    def _emit_memory_region(self, state: angr.SimState, start: int, size: int):
        """Emit a memory region with concrete bytes and optional symbolic info.

        Yields tuples with symbolic value info for constraint reconstruction:
        - ('concrete', start_addr, size, data_bytes) for concrete values
        - ('symbolic', start_addr, size, data_bytes, handle_id, ast) for symbolic values

        For symbolic regions, emits byte-by-byte to produce simple ASTs
        (individual BVS or Extract) that convert cleanly to RustBV. This
        avoids complex Concat trees from multi-byte loads that may fail
        in claripy_to_rustbv conversion.
        """
        # Limit region size to avoid memory issues
        MAX_REGION_SIZE = 4096
        if size > MAX_REGION_SIZE:
            for offset in range(0, size, MAX_REGION_SIZE):
                chunk_size = min(MAX_REGION_SIZE, size - offset)
                yield from self._emit_memory_region(state, start + offset, chunk_size)
            return

        try:
            val = state.memory.load(start, size, endness=state.arch.memory_endness)
            if not val.symbolic:
                concrete = state.solver.eval(val)
                data = concrete.to_bytes(size, "little")
                yield ("concrete", start, size, data)
            else:
                # For small symbolic regions (typical SimProcedure writes),
                # emit byte-by-byte for simple ASTs. For large regions,
                # emit as one chunk to avoid 1000s of solver.eval() calls.
                if size > 128:
                    # Large region: emit whole (may produce complex AST)
                    handle_id = id(val)
                    self._register_handle(handle_id, val)
                    try:
                        concrete = state.solver.eval(val)
                        data = concrete.to_bytes(size, "little")
                    except Exception:
                        # cat-(b) FALLBACK WITH LOSS: large-region eval
                        # failed (often timeout on complex AST). Emit
                        # zeros for the concrete witness; Rust will see
                        # zero bytes if it reads concretely. The
                        # symbolic AST is still attached.
                        data = bytes(size)
                    yield ("symbolic", start, size, data, handle_id, val)
                    return

                # Small region: byte-by-byte for simple ASTs
                for byte_offset in range(size):
                    byte_addr = start + byte_offset
                    try:
                        byte_val = state.memory.load(
                            byte_addr, 1, endness="Iend_BE", inspect=False, disable_actions=True
                        )
                        if byte_val.symbolic:
                            handle_id = id(byte_val)
                            self._register_handle(handle_id, byte_val)
                            try:
                                concrete_byte = state.solver.eval(byte_val)
                                data = bytes([concrete_byte & 0xFF])
                            except Exception:
                                # cat-(b) FALLBACK WITH LOSS: symbolic
                                # byte couldn't be evaluated for the
                                # concrete witness; emit zero. The
                                # downstream Rust memory will be wrong
                                # if it reads this byte concretely.
                                data = bytes(1)
                            yield ("symbolic", byte_addr, 1, data, handle_id, byte_val)
                        else:
                            try:
                                concrete_byte = state.solver.eval(byte_val)
                                data = bytes([concrete_byte & 0xFF])
                            except Exception:
                                # cat-(b) FALLBACK WITH LOSS: concrete
                                # byte eval failed (rare). Emit zero —
                                # Rust will see zero here.
                                data = bytes(1)
                            yield ("concrete", byte_addr, 1, data)
                    except Exception:
                        # cat-(b) FALLBACK WITH LOSS: per-byte load
                        # failed (e.g. memory not mapped). Emit zero
                        # for that byte; Rust may see incorrect data.
                        yield ("concrete", byte_addr, 1, bytes(1))
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: region emit failed entirely
            # (load on the whole region raised). Yields nothing — the
            # caller's Rust sync misses this region's writes.
            l.debug(f"Error emitting memory region at 0x{start:x}: {e}")

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
        try:
            val = state.memory.load(addr, 1, endness=state.arch.memory_endness)
            if hasattr(val, "symbolic") and val.symbolic:
                return val
        except Exception:
            pass
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
        # symbolic_bitmap entries are 0 (concrete) or 1 (symbolic). A C-level
        # `1 in sb` scan beats walking every changed byte when nothing is
        # symbolic — common for entry_state pages with concrete loader writes.
        if sb is None or 1 not in sb:
            return True
        try:
            changed = page.all_bytes_changed_in_history()
            had_changed = False
            for segment in changed:
                had_changed = True
                start = getattr(segment, "start", None)
                end = getattr(segment, "end", None)
                if start is None or end is None:
                    continue
                for offset in range(start, end):
                    if offset < len(sb) and sb[offset]:
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
                    sb_len = len(sb)
                    for i, offset in enumerate(keys):
                        if offset >= sb_len or not sb[offset]:
                            continue
                        next_key = keys[i + 1] if i + 1 < len(keys) else sb_len
                        end = offset
                        limit = min(offset + MAX_EXTENT, next_key, sb_len)
                        while end < limit and sb[end]:
                            end += 1
                        if end - offset >= MAX_EXTENT:
                            continue  # likely a page-fill — skip
                        for byte_off in range(offset, end):
                            addr = page_addr + byte_off
                            val = self._load_symbolic_byte(state, addr)
                            if val is not None:
                                self._record_symbolic(out, addr, val)
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: page-level walk failed.
            pass
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
        This is ~30x faster than 64 individual pending_memory_load calls.

        angr-ryf6: skip when the callback-memory-proxy gate is on and the
        proxy is already installed on ``state.memory`` — the writes would
        round-trip Rust → Python → Rust and overwrite symbolic pointer-slot
        values with concrete witnesses (same loss as
        ``_replay_rust_dirty_pages``).
        """
        from angr.exploration.rust_manager import _is_rust_memory_proxy

        if _is_rust_memory_proxy(state.memory):
            return
        try:
            sp = state.solver.eval(state.regs._sp) if not state.regs._sp.symbolic else None
            if not sp:
                return
            ptr_size = state.arch.bytes

            # Load the full page containing SP in ONE FFI call
            sp_page = sp & ~PAGE_MASK
            sp_offset = sp - sp_page
            try:
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
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: bulk-page FFI not
                # available or failed; fall through to per-pointer
                # fallback below.
                pass

            # Fallback: individual loads
            for i in range(16):
                addr = sp + i * ptr_size
                try:
                    data = self._rust_mgr.pending_memory_load(self._current_callback_state_id, addr, ptr_size)
                    if data and len(data) == ptr_size:
                        int_val = int.from_bytes(data, "little")
                        if int_val != 0:
                            val = claripy.BVV(int_val, ptr_size * 8)
                            state.memory.store(addr, val, endness="Iend_LE", inspect=False, disable_actions=True)
                except Exception:
                    # cat-(b) FALLBACK WITH LOSS: individual pointer
                    # load failed; that slot is not synced. Python
                    # callback may see zeros where Rust has data.
                    pass
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: outer setup failed (SP eval).
            # No stack data is synced from Rust to Python; Python
            # callback runs with potentially stale stack contents.
            pass

    def _replay_rust_dirty_pages(self, state: angr.SimState):
        """Replay Rust-side memory mutations into the cached Python SimState.

        Walks `_get_pending_dirty_pages()` from the Rust pending state and
        writes per-page concrete bytes (`pending_memory_load_page`) then
        symbolic objects (`pending_memory_load_symbolic_page`) into
        `state.memory`. Symbolic stores are applied LAST per page so a
        symbolic byte does not get clobbered by a concrete-zero default
        sitting at the same address.

        This is the cache-sync fix from angr-3tek.2 that re-enables
        NativeRead/NativeWrite: native procs mutate Rust memory but never
        push into the symbolic-page snapshot consumed by
        `_restore_symbolic_pages`, so without this replay a later Python
        SimProc (e.g. strcmp) reads stale concrete-zero bytes.

        angr-ryf6: when the callback-memory-proxy gate is on and the
        cached state already has the proxy installed from a prior
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
            try:
                page_bytes = self._rust_mgr.pending_memory_load_page(self._current_callback_state_id, page_addr)
            except Exception:
                page_bytes = None

            if page_bytes and len(page_bytes) == PAGE_SIZE:
                try:
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

            try:
                sym_entries = self._rust_mgr.pending_memory_load_symbolic_page(
                    self._current_callback_state_id, page_addr
                )
            except (AttributeError, RuntimeError):
                # cat-(a) EXPECTED CONTROL FLOW: symbolic-page FFI absent on
                # older builds; symbolic bytes fall through to the older
                # snapshot path on the next sync.
                sym_entries = None

            if sym_entries:
                for addr, ast in sym_entries:
                    try:
                        state.memory.store(
                            addr,
                            ast,
                            endness=state.arch.memory_endness,
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

        if restored_count > 0:
            if _DBG:
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

        if restored_count > 0:
            if _DBG:
                l.debug(f"Restored {restored_count} hook symbolic memory regions for state {state_id}")
