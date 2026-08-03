"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations

import os
import re

import pytest

import angr

# Rust availability guard, binary-path resolution, and the module-scoped
# fauxware_project fixture all live in tests/engines/conftest.py (angr-7gdp).
from tests.engines.conftest import (  # noqa: F401
    RUST_EXPLORATION_AVAILABLE,
    TEST_BINARIES_DIR,
    ExplorationEvent,
    PythonCallbacks,
    RustExplorationManager,
    _RustExplorationManager,
)

# All tests in this module require the Rust extension; skip the whole module
# when it is unavailable (matches tests/engines/test_rust_public_api.py).
pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


class TestUnconstrainedRet:
    """angr-3uye: `ret` from a blank state with no prior call must route the
    state to the `unconstrained` stash (matching Python), not be silently
    concretized to address 0 and deadended.

    Python's _eval_target_brutal (engines/successors.py) sees the lazy-filled
    symbolic stack as a `<BV64 mem_*>` with >256 solutions and overflows into
    `unconstrained_successors`. Rust's state sync materialises the lazy stack
    to concrete zeros, so the popped IP is concrete 0; the fix in
    interpreter/exits.rs detects `Ijk_Ret` with an empty call stack to
    an out-of-binary target and routes to the unconstrained stash instead.
    """

    def test_ret_with_empty_call_stack_routes_to_unconstrained(self):
        """Single-instruction `ret` on a blank state lands in the
        unconstrained stash, not deadended."""
        import angr

        proj = angr.load_shellcode(b"\xc3", arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rsp = 0x7FFF_0000
        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=2)

        # The state must be in the unconstrained stash (Python's behaviour).
        assert len(mgr.unconstrained) == 1, (
            f"expected 1 unconstrained state, got stashes={ {k: len(v) for k, v in mgr.stashes.items() if v} }"
        )
        assert len(mgr.deadended) == 0, (
            "ret-from-blank-state must not deadend (would have meant the popped IP was silently concretized to 0)"
        )


class TestRustConcreteMemoryStoreRoundTrip:
    """angr-7vcx: A concrete store at an absolute non-stack address inside
    the Rust engine must propagate back to Python state.memory.load() after
    exploration completes. Previously, writes via 'mov [abs], imm' to BSS-
    style pages were lost — sokohashv2 hash bytes read back as zeros.
    """

    def test_concrete_store_to_absolute_addr_propagates(self):
        import angr
        import angr.sim_options as o

        # AMD64:
        #   c7 04 25 c0 16 42 00  78 56 34 12   mov dword [0x4216c0], 0x12345678
        #   c3                                   ret
        shellcode = bytes.fromhex("c70425c016420078563412") + b"\xc3"
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x401000)
        state = proj.factory.blank_state(
            addr=0x401000,
            add_options={
                o.ZERO_FILL_UNCONSTRAINED_MEMORY,
                o.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            },
        )
        # Set up a clean return target so the ret deadends predictably.
        state.regs.rsp = 0x7FFF_0000
        state.memory.store(0x7FFF_0000, b"\x00" * 8)
        # Map the destination page so the Rust engine's store hits a
        # writable page (otherwise it would unmap-fault).
        state.memory.map_region(0x421000, 0x1000, 7)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=10)

        all_states = list(mgr.found) + list(mgr.active) + list(mgr.deadended) + list(mgr.unconstrained)
        assert all_states, "expected at least one state after run"
        s = all_states[0]
        loaded = s.memory.load(0x4216C0, 4, endness=s.arch.memory_endness, inspect=False, disable_actions=True)
        val = s.solver.eval(loaded)
        assert val == 0x12345678, (
            f"expected 0x12345678 at 0x4216c0, got 0x{val:x}. "
            f"Rust engine memory store to absolute address did not "
            f"propagate to Python state.memory.load."
        )


class TestRustManagerCleanup:
    """angr-518z: opt-in AST cache cleanup so Callable-heavy workloads
    (mma_howtouse pattern: many short-lived managers on one thread)
    don't accumulate O(n) thread-local AST cache entries.
    """

    def test_clear_ast_cache_ffi_exposed(self):
        """The Rust-side cache flush helper must be callable from Python."""
        from angr.rustylib.vex_engine import clear_ast_cache

        # No-op on an empty cache; must not raise.
        clear_ast_cache()
        clear_ast_cache()

    def test_cleanup_is_idempotent_and_safe(self):
        """cleanup() can be called multiple times and on a manager that
        never ran exploration; must not raise."""
        import angr.rustylib.vex_engine as vex_engine

        import angr

        proj = angr.load_shellcode(b"\x90\xc3", arch="AMD64", load_address=0x401000)
        state = proj.factory.blank_state(addr=0x401000)
        mgr = RustExplorationManager(proj, [state])

        # Spy on clear_ast_cache: cleanup()'s whole point is flushing the
        # Rust thread-local AST caches (angr-518z). A no-op regression must
        # be caught, not silently swallowed by cleanup()'s try/except.
        calls = {"n": 0}
        orig = vex_engine.clear_ast_cache

        def _spy(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        vex_engine.clear_ast_cache = _spy
        try:
            mgr.cleanup()
            mgr.cleanup()
        finally:
            vex_engine.clear_ast_cache = orig

        assert calls["n"] == 2, f"cleanup() must flush the AST cache each call; saw {calls['n']}"
        # Manager stays usable after repeated cleanup (not left broken).
        assert "active" in mgr.stash_counts()

    def test_cleanup_runs_after_short_exploration(self):
        """Run a tiny exploration, then call cleanup() — must not raise
        and the manager must remain usable for inspecting stash counts."""
        import angr

        # nop; ret — predictable deadend after one step.
        proj = angr.load_shellcode(b"\x90\xc3", arch="AMD64", load_address=0x401000)
        state = proj.factory.blank_state(addr=0x401000)
        state.regs.rsp = 0x7FFF_0000
        state.memory.store(0x7FFF_0000, b"\x00" * 8)
        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=4)
        mgr.cleanup()
        # Contract: stash_counts() stays usable post-cleanup (no raise) and
        # still reports the "active" stash.
        assert "active" in mgr.stash_counts()

    def test_clear_caches_on_cleanup_flag_default_off(self):
        """Default constructor leaves the flag off — single-long-exploration
        users see no behavior change."""
        import angr

        proj = angr.load_shellcode(b"\x90\xc3", arch="AMD64", load_address=0x401000)
        state = proj.factory.blank_state(addr=0x401000)
        mgr = RustExplorationManager(proj, [state])
        assert mgr._clear_caches_on_cleanup is False

    def test_clear_caches_on_cleanup_flag_honored(self):
        """When the constructor flag is set, __del__ wires through to
        cleanup(); explicit drop triggers the cache flush without raising."""
        import gc

        import angr

        proj = angr.load_shellcode(b"\x90\xc3", arch="AMD64", load_address=0x401000)
        state = proj.factory.blank_state(addr=0x401000)
        mgr = RustExplorationManager(
            proj,
            [state],
            clear_caches_on_cleanup=True,
        )
        assert mgr._clear_caches_on_cleanup is True
        # Drop and force collection — cleanup() must not raise during __del__.
        del mgr
        gc.collect()


class TestLoaderPagesCache:
    """angr-bzsc: class-level cache of loader-page output so Callable-style
    workflows (e.g. mma_howtouse, which spawns 45 RustExplorationManagers
    on one Project) skip the per-init `loader.memory.load` + `map_memory_batch`
    cost on the second-and-later constructions.
    """

    def test_loader_pages_cache_hit_on_second_manager(self, fauxware_project):
        """Constructing a second manager with a non-entry start state must
        reuse the cache entry built by the first manager (identity check)."""
        from angr.exploration.rust_manager import RustExplorationManager

        RustExplorationManager._loader_pages_cache.clear()
        loader = fauxware_project.loader
        main_sym = loader.find_symbol("main")
        assert main_sym is not None

        # First manager: cache miss, populates the entry.
        s1 = fauxware_project.factory.blank_state(addr=main_sym.rebased_addr)
        RustExplorationManager(fauxware_project, [s1])
        assert loader in RustExplorationManager._loader_pages_cache, (
            "first manager must populate the loader-pages cache"
        )
        entry1 = RustExplorationManager._loader_pages_cache[loader]
        assert entry1["batch_pages"], "cache must record at least one batch page"
        assert entry1["lazy_regions"], "cache must record at least one lazy region"

        # Second manager: cache hit. Same dict identity proves no rebuild.
        s2 = fauxware_project.factory.blank_state(addr=main_sym.rebased_addr)
        RustExplorationManager(fauxware_project, [s2])
        entry2 = RustExplorationManager._loader_pages_cache[loader]
        assert entry2 is entry1, "second manager must reuse cache entry"

    def test_loader_pages_cache_separate_projects_separate_entries(self):
        """Two distinct projects must hold separate cache entries (the
        WeakKeyDictionary keys by Loader identity, not by file path)."""
        import angr
        from angr.exploration.rust_manager import RustExplorationManager

        RustExplorationManager._loader_pages_cache.clear()
        binary = os.path.join(TEST_BINARIES_DIR, "fauxware")
        if not os.path.exists(binary):
            pytest.skip("fauxware binary not found")

        proj_a = angr.Project(binary, auto_load_libs=False)
        proj_b = angr.Project(binary, auto_load_libs=False)

        main_a = proj_a.loader.find_symbol("main").rebased_addr
        main_b = proj_b.loader.find_symbol("main").rebased_addr

        RustExplorationManager(proj_a, [proj_a.factory.blank_state(addr=main_a)])
        RustExplorationManager(proj_b, [proj_b.factory.blank_state(addr=main_b)])

        assert proj_a.loader in RustExplorationManager._loader_pages_cache
        assert proj_b.loader in RustExplorationManager._loader_pages_cache
        assert (
            RustExplorationManager._loader_pages_cache[proj_a.loader]
            is not RustExplorationManager._loader_pages_cache[proj_b.loader]
        )

    def test_loader_pages_cache_weakref_auto_evicts(self):
        """When the Project (and its Loader) is garbage-collected, the
        WeakKeyDictionary entry must vanish — no stale-id collisions."""
        import gc

        import angr
        from angr.exploration.rust_manager import RustExplorationManager

        RustExplorationManager._loader_pages_cache.clear()
        binary = os.path.join(TEST_BINARIES_DIR, "fauxware")
        if not os.path.exists(binary):
            pytest.skip("fauxware binary not found")

        proj = angr.Project(binary, auto_load_libs=False)
        main = proj.loader.find_symbol("main").rebased_addr
        mgr = RustExplorationManager(proj, [proj.factory.blank_state(addr=main)])
        assert proj.loader in RustExplorationManager._loader_pages_cache

        # Drop all strong references and force collection.
        del mgr
        del proj
        gc.collect()
        # WeakKeyDictionary now has no remaining strong refs to any Loader.
        assert len(RustExplorationManager._loader_pages_cache) == 0


class TestSyncExtraPagesFastPath:
    """angr-b58a: _sync_extra_python_pages skips full bytes() materialization
    for UltraPage backings (using a memcmp on `concrete_data`) and batches
    the per-page lazy-region FFI into a single call.  These tests pin the
    fast-path behaviour so a refactor that drops one branch keeps working.
    """

    @staticmethod
    def _fresh_rust_state(project):
        """Build a bare _RustSimState matching the project's arch."""
        from angr.rustylib.vex_engine import RustSimState

        is_le = project.arch.memory_endness == "Iend_LE"
        return RustSimState(project.arch.name, little_endian=is_le)

    def test_lazy_regions_use_batch_ffi(self, fauxware_project):
        """All synced extra pages should go through add_lazy_regions_batch
        exactly once; the per-page add_lazy_region helper must not be hit
        from inside _sync_extra_python_pages."""
        from angr.exploration.rust_manager import RustExplorationManager
        from angr.exploration.rust_state_sync import RustStateSyncMixin

        state = fauxware_project.factory.entry_state()
        # Add 8 extra non-loader pages so the function has work to do.
        for i in range(8):
            page_addr = 0x4000_0000 + i * 0x1000
            state.memory.store(page_addr, b"\x00" * 0x1000, endness="Iend_BE")
        mgr = RustExplorationManager(fauxware_project, [state])
        rust_state = self._fresh_rust_state(fauxware_project)

        batch_calls: list = []
        single_calls: list = []
        real_batch = type(rust_state).add_lazy_regions_batch
        real_single = type(rust_state).add_lazy_region
        try:
            type(rust_state).add_lazy_regions_batch = lambda self, regions, _r=real_batch, _b=batch_calls: (
                _b.append(list(regions)),
                _r(self, regions),
            )[1]
            type(rust_state).add_lazy_region = lambda self, start, size, _r=real_single, _s=single_calls: (
                _s.append((start, size)),
                _r(self, start, size),
            )[1]
            sp = state.solver.eval(state.regs.sp)
            stack_base = (sp & ~0xFFF) + 0x1000
            stack_start = stack_base - 0x11_0000
            mapped, _ = mgr._map_loader_pages(rust_state, set(), 0x1000)
            RustStateSyncMixin._sync_extra_python_pages(
                mgr, state, rust_state, mapped, set(), sp & ~0xFFF, stack_start, stack_base, 0x1000
            )
        finally:
            type(rust_state).add_lazy_regions_batch = real_batch
            type(rust_state).add_lazy_region = real_single

        assert len(single_calls) == 0, (
            f"_sync_extra_python_pages should batch lazy regions, saw {len(single_calls)} single calls"
        )
        assert len(batch_calls) == 1, f"expected exactly one batch call, got {len(batch_calls)}"
        addrs_in_batch = {addr for addr, _size in batch_calls[0]}
        for i in range(8):
            assert 0x4000_0000 + i * 0x1000 in addrs_in_batch

    def test_zero_page_classified_without_concrete_load(self, fauxware_project):
        """For all-zero pages on the mma lazy path (zero_count > cap),
        the UltraPage fast path classifies via concrete_data memcmp and
        never invokes concrete_load() — even during the FFI phase."""
        from angr.exploration.rust_manager import RustExplorationManager
        from angr.exploration.rust_state_sync import RustStateSyncMixin
        from angr.storage.memory_mixins.paged_memory.pages.ultra_page import (
            UltraPage,
        )

        state = fauxware_project.factory.entry_state()
        # Allocate > zero_eager_cap (200) zero pages so eager_zero=False
        # (mma path).  Then _sync_extra_python_pages will only call
        # add_lazy_regions_batch — no per-page bytes() copy, no FFI map.
        for i in range(220):
            page_addr = 0x4100_0000 + i * 0x1000
            state.memory.store(page_addr, b"\x00" * 0x1000, endness="Iend_BE")
        mgr = RustExplorationManager(fauxware_project, [state])
        rust_state = self._fresh_rust_state(fauxware_project)

        # Verify the UltraPage backing is what the fast path expects.
        any_page = state.memory._pages.get(0x4100_0000 // 0x1000)
        assert isinstance(any_page, UltraPage)
        assert isinstance(any_page.concrete_data, bytearray)

        calls = {"concrete_load": 0}
        orig_cl = UltraPage.concrete_load

        def spy_cl(self, addr, size, **kw):
            calls["concrete_load"] += 1
            return orig_cl(self, addr, size, **kw)

        UltraPage.concrete_load = spy_cl
        try:
            sp = state.solver.eval(state.regs.sp)
            stack_base = (sp & ~0xFFF) + 0x1000
            stack_start = stack_base - 0x11_0000
            mapped, _ = mgr._map_loader_pages(rust_state, set(), 0x1000)
            RustStateSyncMixin._sync_extra_python_pages(
                mgr, state, rust_state, mapped, set(), sp & ~0xFFF, stack_start, stack_base, 0x1000
            )
        finally:
            UltraPage.concrete_load = orig_cl

        # mma path: > 200 zero pages → eager_zero=False → no FFI
        # map_memory_data, no bytes() copy.  concrete_load MUST be zero.
        assert calls["concrete_load"] == 0, (
            f"UltraPage lazy path should never call concrete_load; saw {calls['concrete_load']}"
        )

    def test_add_lazy_regions_batch_ffi_available(self, fauxware_project):
        """Smoke-test the new Rust FFI surface exists and is callable."""
        rust_state = self._fresh_rust_state(fauxware_project)
        assert hasattr(rust_state, "add_lazy_regions_batch"), "Rust FFI must expose add_lazy_regions_batch (angr-b58a)"
        before = rust_state.lazy_region_count()
        rust_state.add_lazy_regions_batch([(0x4200_0000, 0x1000), (0x4200_1000, 0x1000)])
        # The batch must actually register both regions, not no-op.
        assert rust_state.lazy_region_count() == before + 2
        assert rust_state.is_in_lazy_region(0x4200_0000)
        assert rust_state.is_in_lazy_region(0x4200_1000)
        assert rust_state.is_in_lazy_region(0x4200_0800)  # interior byte of region 1
        # An address outside both regions is not lazy.
        assert not rust_state.is_in_lazy_region(0x4200_8000)


class TestStashProxyAccessors:
    """angr-kwpi.3: opt-in ``mgr.<stash>_proxies()`` direct accessors that
    return ``list[RustStateProxy]`` without materializing full SimStates.
    """

    def test_all_five_methods_exposed(self, fauxware_project):
        mgr = RustExplorationManager(fauxware_project, [fauxware_project.factory.entry_state()])
        for name in ("found_proxies", "active_proxies", "avoid_proxies", "deadended_proxies", "unconstrained_proxies"):
            assert callable(getattr(mgr, name)), f"RustExplorationManager.{name}() should be callable"

    def test_active_proxies_returns_rust_state_proxy(self, fauxware_project):
        """active_proxies() returns RustStateProxy objects whose attrs
        round-trip from the Rust state (not full SimStates)."""
        from angr.exploration.rust_state_proxy import RustStateProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        proxies = mgr.active_proxies()
        assert len(proxies) >= 1
        assert all(isinstance(p, RustStateProxy) for p in proxies)
        # PC is reachable through the proxy without SimState materialization
        # — and matches what the full SimState path reports.
        assert proxies[0].addr == mgr.active[0].addr

    def test_both_api_forms_agree_on_state_count(self, fauxware_project):
        """The proxy-returning form and the SimState-returning form must
        report the same number of states in each stash after exploration.

        This is the demo-both-forms test required by angr-kwpi.3
        acceptance criteria: it exercises ``mgr.found`` (full SimState)
        and ``mgr.found_proxies()`` (RustStateProxy) side by side and
        confirms they agree on cardinality and on per-state addr.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        # Step until fauxware forks at the strcmp.
        mgr.run(max_steps=200)

        # Pair each stash property with its proxy counterpart.
        pairs = [
            (mgr.active, mgr.active_proxies()),
            (mgr.found, mgr.found_proxies()),
            (mgr.avoid, mgr.avoid_proxies()),
            (mgr.deadended, mgr.deadended_proxies()),
            (mgr.unconstrained, mgr.unconstrained_proxies()),
        ]
        for full_states, proxy_states in pairs:
            # ``mgr.found`` may include a Python-side predicate-found
            # SimState that has no corresponding Rust stash entry, so the
            # proxy count is a lower bound rather than an exact match.
            assert len(proxy_states) <= len(full_states)
            # Every proxy.addr must appear in the full-states addr set.
            full_addrs = sorted(s.addr for s in full_states)
            proxy_addrs = sorted(p.addr for p in proxy_states)
            for a in proxy_addrs:
                assert a in full_addrs, f"proxy addr {hex(a)} not in full-states addrs {[hex(x) for x in full_addrs]}"

    def test_proxies_skip_simstate_materialization(self, fauxware_project):
        """The proxy accessor must NOT call into the SimState export path.

        Patches ``RustStateExportMixin._get_stash_states`` to raise so
        that any accidental fall-through to the full export path fails
        loudly; the proxy accessor must remain green.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        original = mgr._get_stash_states

        def _fail(*args, **kwargs):
            raise AssertionError("found_proxies() must not call _get_stash_states (SimState materialization)")

        mgr._get_stash_states = _fail
        try:
            proxies = mgr.active_proxies()
            assert len(proxies) >= 1
        finally:
            mgr._get_stash_states = original

    def test_full_addr_matches_rust_pc_under_register_proxy(self, fauxware_project):
        """angr-4rq7 regression: with the register-proxy write-through gate
        on, the full-export ``state.addr`` must equal the live Rust PC
        (``get_state_pc_by_id``) for every stashed state.

        A materialized state that inherits a callback ``RustRegisterProxy``
        used to read its stale per-name cache for ``regs.ip``, so the
        full-export ``addr`` diverged from the ``X_proxies()`` accessor
        (which reads the live pc). ``_sync_rust_registers_to_state`` now
        clears that cache and rebinds the proxy on materialization, so both
        read paths agree.
        """
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], use_callback_register_proxy=True)
        mgr.run(max_steps=200)

        checked = 0
        for stash in ("active", "found", "deadended", "avoid", "unconstrained"):
            for sid in mgr._rust_mgr.get_state_ids(stash):
                rust_pc = mgr._rust_mgr.get_state_pc_by_id(sid)
                full = mgr._materialize_single_state(sid)
                assert full.addr == rust_pc, (
                    f"stash {stash} state {sid}: full-export addr {hex(full.addr)} != Rust pc {hex(rust_pc)}"
                )
                checked += 1
        # fauxware reaches the strcmp fork, so at least one state exists.
        assert checked >= 1


class TestRustExecutionErrorHierarchy:
    """Typed exception classes surfaced by the Rust engine (angr-tkbr.3).

    The Rust engine previously string-coerced internal errors into
    ``ExecutionEvent.error`` so Python could only ``re.search`` the
    message to discriminate. tkbr.3 adds a typed ``RustExecutionError``
    base + five sibling subclasses so user code can ``pytest.raises``
    against a specific failure mode. Acceptance criteria checked by
    this class:

    * Exception classes are importable from ``angr.exploration``.
    * ``RustExecutionError`` is the base; the five siblings each
      inherit from it (so ``except RustExecutionError`` catches all).
    * Each class is also ``Exception``-derived (PyO3 base class
      ``PyException`` shows up in the MRO).
    * Real-path coverage: ``pytest.raises(RustUnsupportedVexOpError,
      match="<op_name>.*<arch>")`` for a NEON op via ``execute_irsb_json``.
    * Demo coverage for the syscall path (full panic-site conversion
      lives in angr-tkbr.1): ``pytest.raises(RustUnsupportedSyscallError,
      match="<syscall_name>")`` via the ``_raise_typed_test_error`` helper.
    """

    def test_classes_importable_from_angr_exploration(self):
        """The 6 typed exception classes re-export from ``angr.exploration``."""
        from angr.exploration import (
            RustExecutionError,
            RustMalformedIRSBError,
            RustOomError,
            RustUnsupportedSyscallError,
            RustUnsupportedVexOpError,
            RustZ3Error,
        )

        # Sanity: each is a class object.
        for cls in (
            RustExecutionError,
            RustMalformedIRSBError,
            RustOomError,
            RustUnsupportedSyscallError,
            RustUnsupportedVexOpError,
            RustZ3Error,
        ):
            assert isinstance(cls, type), f"{cls!r} is not a class"

    def test_subclass_hierarchy(self):
        """All five typed errors inherit from ``RustExecutionError``,
        which inherits from ``Exception``."""
        from angr.exploration import (
            RustExecutionError,
            RustMalformedIRSBError,
            RustOomError,
            RustUnsupportedSyscallError,
            RustUnsupportedVexOpError,
            RustZ3Error,
        )

        subclasses = (
            RustMalformedIRSBError,
            RustOomError,
            RustUnsupportedSyscallError,
            RustUnsupportedVexOpError,
            RustZ3Error,
        )
        for cls in subclasses:
            assert issubclass(cls, RustExecutionError), f"{cls.__name__} must subclass RustExecutionError"
        assert issubclass(RustExecutionError, Exception)

    def test_pytest_raises_catches_subclass_via_base(self):
        """``pytest.raises(RustExecutionError)`` catches every sibling.

        This is the key downstream contract: callers can write a single
        ``except RustExecutionError`` to handle all engine failures.
        """
        from angr.rustylib.vex_engine import _raise_typed_test_error

        from angr.exploration import RustExecutionError

        for kind in (
            "malformed_irsb",
            "unsupported_syscall",
            "unsupported_vex_op",
            "z3",
            "oom",
            "other",
        ):
            with pytest.raises(RustExecutionError):
                _raise_typed_test_error(kind, name="probe", num=1, arch="AMD64")

    def test_raise_unsupported_syscall_match(self):
        """The syscall acceptance: ``pytest.raises(RustUnsupportedSyscallError,
        match="<syscall_name>")`` works.

        The 48 syscall handler panic sites currently still ``panic!`` on
        unexpected dispatcher outcomes; their conversion to typed
        ``SyscallError`` lives in sibling bead angr-tkbr.1. This test
        exercises the typed-exception API via ``_raise_typed_test_error``
        — the API tkbr.1 will route into. Replace with a real-path test
        once tkbr.1 lands.
        """
        from angr.rustylib.vex_engine import _raise_typed_test_error

        from angr.exploration import RustUnsupportedSyscallError

        with pytest.raises(RustUnsupportedSyscallError, match=r"brk"):
            _raise_typed_test_error(
                "unsupported_syscall",
                name="brk",
                num=12,
                arch="AMD64",
                message="no native handler",
            )

    def test_raise_unsupported_neon_op_via_execute_irsb(self):
        """The NEON acceptance: ``pytest.raises(RustUnsupportedVexOpError,
        match="<op_name>.*<arch>")`` for a currently-unimplemented NEON op.

        Constructs a minimal AArch64 IRSB that applies ``Iop_PwAdd32Fx2``
        (a Binop on Ity_I64 routed through ``IROp::NeonUnimplemented``) to
        two zero constants. Before tkbr.3 the dispatcher in ``VEXOps::binop``
        panicked, which PyO3 surfaced as ``pyo3_runtime.PanicException``;
        after tkbr.3 the panic is replaced with ``OpError::UnsupportedNeon``
        and surfaces as ``RustUnsupportedVexOpError`` carrying op name + arch.

        Originally targeted ``Iop_RecipEst32Ux2`` — retargeted in angr-tukg.5
        when that op was implemented. ``Iop_PwAdd32Fx2`` (FP pairwise add) is
        the only remaining NeonUnimplemented placeholder.
        """
        import json

        from angr.rustylib.vex_engine import execute_irsb_for_test

        from angr.exploration import RustUnsupportedVexOpError

        irsb = {
            "addr": 4096,
            "arch": "ARM64",
            "statements": [
                {"tag": "Ist_IMark", "addr": 4096, "len": 4, "delta": 0},
                {
                    "tag": "Ist_WrTmp",
                    "tmp": 0,
                    "data": {
                        "tag": "Iex_Const",
                        "con": {"tag": "Ico_U64", "value": 0},
                    },
                },
                {
                    "tag": "Ist_WrTmp",
                    "tmp": 1,
                    "data": {
                        "tag": "Iex_Binop",
                        "op": "Iop_QShlNsatSU8x8",
                        "args": [
                            {"tag": "Iex_RdTmp", "tmp": 0},
                            {"tag": "Iex_RdTmp", "tmp": 0},
                        ],
                    },
                },
            ],
            "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 4100}},
            "jumpkind": "Ijk_Boring",
            "offsIP": 272,
            "tyenv": {"types": ["Ity_I64", "Ity_I64"]},
        }
        # Iop_PwAdd32Fx2 graduated to IROp::VFPwAdd (angr-cudgw.6); use the
        # still-unimplemented NEON shift-by-immediate as the unsupported probe.
        with pytest.raises(RustUnsupportedVexOpError, match=r"Iop_QShlNsatSU8x8.*arm64"):
            execute_irsb_for_test(json.dumps(irsb), "arm64")

    def test_raise_unmapped_op_via_execute_irsb(self):
        """The angr-tkbr.2 acceptance: an opcode string with NO entry in
        ``parse_opcode`` (not in any ``parse_*`` table, not in
        ``parse_neon_unimplemented``) surfaces as
        ``RustUnsupportedVexOpError`` carrying the original op name +
        arch.

        Before tkbr.2 the unmapped fallback silently rewrote to
        ``IROp::Raw(0)`` (just a ``log::warn!``) and the dispatcher
        produced a fresh-symbolic value of the wrong width. After
        tkbr.2 ``parse_opcode`` returns ``IROp::Unmapped(name)`` and
        the dispatcher emits ``OpError::UnsupportedVexOp`` which the
        engine maps to ``RustUnsupportedVexOpError(op_name, arch)``.

        Uses a deliberately fake name (``Iop_NotARealOp1234``) so the
        test stays valid even as real opcodes get mapped over time.
        """
        import json

        from angr.rustylib.vex_engine import execute_irsb_for_test

        from angr.exploration import RustUnsupportedVexOpError

        fake_op = "Iop_NotARealOp1234"
        irsb = {
            "addr": 4096,
            "arch": "AMD64",
            "statements": [
                {"tag": "Ist_IMark", "addr": 4096, "len": 4, "delta": 0},
                {
                    "tag": "Ist_WrTmp",
                    "tmp": 0,
                    "data": {
                        "tag": "Iex_Const",
                        "con": {"tag": "Ico_U64", "value": 0},
                    },
                },
                {
                    "tag": "Ist_WrTmp",
                    "tmp": 1,
                    "data": {
                        "tag": "Iex_Unop",
                        "op": fake_op,
                        "arg": {"tag": "Iex_RdTmp", "tmp": 0},
                    },
                },
            ],
            "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 4100}},
            "jumpkind": "Ijk_Boring",
            "offsIP": 184,
            "tyenv": {"types": ["Ity_I64", "Ity_I64"]},
        }
        with pytest.raises(RustUnsupportedVexOpError, match=r"Iop_NotARealOp1234.*amd64"):
            execute_irsb_for_test(json.dumps(irsb), "amd64")

    def test_unhandled_ccall_surfaces_error_not_concrete_zero(self):
        """The angr-ppgx acceptance: unhandled ``Iex_CCall`` in the legacy
        ``VEXInterpreter`` must surface as a ``RustExecutionError`` instead
        of silently returning concrete 0.

        Before this fix, ``interpreter.rs::IRExpr::CCall`` fell back to
        ``RustBV::concrete(0, retty.bits())`` when ``ccall::handle_ccall``
        had no entry — on amd64/x86 that corrupts ``rflags``/``eflags``
        and miscompiles downstream conditional branches. The fix returns
        ``ExecutionError::Unsupported("CCall {name}")`` which the engine
        maps through ``execution_error_to_typed`` to the base
        ``RustExecutionError`` class.
        """
        import json

        from angr.rustylib.vex_engine import execute_irsb_for_test

        from angr.exploration import RustExecutionError

        irsb = {
            "addr": 4096,
            "arch": "AMD64",
            "statements": [
                {"tag": "Ist_IMark", "addr": 4096, "len": 4, "delta": 0},
                {
                    "tag": "Ist_WrTmp",
                    "tmp": 0,
                    "data": {
                        "tag": "Iex_CCall",
                        "cee": {
                            "name": "amd64g_NotARealCCall",
                            "addr": 0,
                            "mcx_mask": 0,
                        },
                        "retty": "Ity_I64",
                        "args": [],
                    },
                },
            ],
            "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 4100}},
            "jumpkind": "Ijk_Boring",
            "offsIP": 184,
            "tyenv": {"types": ["Ity_I64"]},
        }
        with pytest.raises(RustExecutionError, match=r"CCall.*amd64g_NotARealCCall"):
            execute_irsb_for_test(json.dumps(irsb), "amd64")

    def test_malformed_irsb_surfaces_typed_error_with_addr(self):
        """The angr-95up.1 acceptance: a structurally-invalid IRSB surfaces
        as ``RustMalformedIRSBError`` carrying the block address — not as a
        panic, ``PyValueError``, or silent fallthrough.

        ``RustMalformedIRSBError`` is the typed-error variant covering VEX
        lift / IR validation failure: ``engine.rs::cb_execution_error_to_typed``
        maps ``CbExecutionError::InvalidIR(reason)`` to
        ``RustExecError::MalformedIRSB { addr, reason }``.

        This pins the contract ahead of the ``angr-tkbr.5`` hot-path-unwrap
        audit, so the audit can rely on lift/IR-validation failures landing
        in a typed class rather than panicking. The trigger here is the
        cheapest InvalidIR site: an ``Ist_LLSC`` whose ``result`` temp index
        sits past the end of the block's ``tyenv``. Dispatch hits
        ``statements.rs::execute_llsc_statement`` line ~708 and returns
        ``InvalidIR("LLSC result temp <n> not in tyenv")``.

        Note: ``RustExplorationManager.run()`` does NOT raise typed errors
        end-to-end on malformed bytes — the Python lift callback in
        ``rust_manager.py::_cb_lift_block`` catches PyVEXError /
        SimEngineError / ClaripyError and returns ``'{}'``, and the
        ``stepping.rs`` heuristic deadends states whose error message
        contains ``"lift"`` (see memory ``rust-engine-lift-failure-silent-deadend``).
        ``execute_irsb_for_test`` is therefore the testbench that exercises
        the typed-error surface for VEX lift validation; the silent-deadend
        contract for the run-loop path is locked down separately via
        ``test_cb_lift_block_*`` in ``TestRustManagerCallbacksUnit``.
        """
        import json

        from angr.rustylib.vex_engine import execute_irsb_for_test

        from angr.exploration import RustMalformedIRSBError

        addr = 0x1000
        irsb = {
            "addr": addr,
            "arch": "AMD64",
            "statements": [
                {"tag": "Ist_IMark", "addr": addr, "len": 4, "delta": 0},
                # LLSC `result` references temp 99, but tyenv only declares
                # one type (t0:Ity_I64). The dispatcher resolves the temp
                # type via `irsb.tyenv.get(99)` -> None -> InvalidIR.
                {
                    "tag": "Ist_LLSC",
                    "result": 99,
                    "addr": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 0x2000}},
                    "storedata": None,
                    "end": "Iend_LE",
                },
            ],
            "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": addr + 4}},
            "jumpkind": "Ijk_Boring",
            "offsIP": 184,
            "tyenv": {"types": ["Ity_I64"]},
        }

        with pytest.raises(RustMalformedIRSBError) as exc_info:
            execute_irsb_for_test(json.dumps(irsb), "amd64")

        # The typed-error contract: address is in the message so callers
        # can correlate the failure to a specific block, and the reason
        # names the IR construct that tripped validation.
        msg = str(exc_info.value)
        assert "0x1000" in msg, f"missing addr context in: {msg}"
        assert "LLSC" in msg, f"missing IR-construct context in: {msg}"


# ---------------------------------------------------------------------------
# angr-uprs: parametrized negative-tests for the RustUnsupported* surface.
# ---------------------------------------------------------------------------

# x87 FPU / AArch64-FRECPX transcendentals. These used to live in the
# unmapped list below: IROp::Raw had no producer, so the string router
# landed them in IROp::Unmapped and the libm fast paths in
# vex/transcendentals.rs were unreachable (angr-9ke6b.233). Since .233
# `opcode_map::parse_transcendental` maps them to IROp::Raw(tag), so they
# execute natively and must NOT raise RustUnsupportedVexOpError. Each entry
# pairs the opcode with its VEX arity: (rm, x) binop vs (rm, x, y) triop.
_MAPPED_X87_TRANSCENDENTALS = [
    ("Iop_SinF64", 2),
    ("Iop_CosF64", 2),
    ("Iop_TanF64", 2),
    ("Iop_2xm1F64", 2),
    ("Iop_RecpExpF64", 2),
    ("Iop_RecpExpF32", 2),
    ("Iop_AtanF64", 3),
    ("Iop_Yl2xF64", 3),
    ("Iop_Yl2xp1F64", 3),
    ("Iop_ScaleF64", 3),
]

# Real NEON saturating shift-left by immediate (UQSHL/SQSHL imm). The
# vector-by-vector forms `Iop_QShl{N}x{M}` / `Iop_QSal{N}x{M}` ARE mapped
# (angr-tukg.8 → IROp::VQShlSat); only the `*sat*` immediate variants remain.
_UNMAPPED_NEON_QSHL_IMM = [
    "Iop_QShlNsatSU8x8",
    "Iop_QShlNsatSU16x4",
    "Iop_QShlNsatSU32x2",
    "Iop_QShlNsatSU64x1",
    "Iop_QShlNsatSU8x16",
    "Iop_QShlNsatSU16x8",
    "Iop_QShlNsatSU32x4",
    "Iop_QShlNsatSU64x2",
    "Iop_QShlNsatSS16x4",
    "Iop_QShlNsatSS32x2",
    "Iop_QShlNsatSS16x8",
    "Iop_QShlNsatSS32x4",
]

# Real but currently-unmapped float/decimal/conversion opcodes from libvex_ir.h.
# Mostly Power/S390 BFP-DFP / ARMv8 FP16 / x86 cvt-with-fixed-rm. Probed
# 2026-06-01 against `IROp::Unmapped` via execute_irsb_for_test.
_UNMAPPED_FP_DECIMAL = [
    "Iop_F128toD32",
    "Iop_F128toI128S",
    "Iop_F16toF32x4",
    "Iop_F32ToFixed32Sx2_RZ",
    "Iop_F64toD128",
    "Iop_Fixed32SToF32x2_RN",
    "Iop_FtoI32Sx2_RZ",
    "Iop_RoundF32x4_RM",
    "Iop_RoundF32x4_RN",
    "Iop_SignificanceRoundD64",
]

# Real but currently-unmapped polynomial-MAC + crypto extensions. Power/ARM
# vector crypto (AES, SHA) that we do not model.
_UNMAPPED_CRYPTO_AND_POLY = [
    "Iop_PolynomialMulAdd8x16",
    "Iop_PolynomialMulAdd16x8",
    "Iop_PolynomialMulAdd32x4",
    "Iop_PolynomialMulAdd64x2",
    "Iop_CipherV128",
    "Iop_NCipherV128",
    "Iop_SHA256",
    "Iop_SHA512",
]

# Union of all real-name unmapped VEX ops to parametrize over. Total >=30
# (was >=40 before angr-9ke6b.233 moved the ten x87 transcendentals to
# _MAPPED_X87_TRANSCENDENTALS) so the acceptance target of ">=50 unsupported
# ops/syscalls" still lands once the syscalls (>=20) are added below.
_UNMAPPED_VEX_OPS_REAL = _UNMAPPED_NEON_QSHL_IMM + _UNMAPPED_FP_DECIMAL + _UNMAPPED_CRYPTO_AND_POLY

# Currently-NeonUnimplemented (routes through OpError::UnsupportedNeon, not
# UnsupportedVexOp, but both PyErr-map to RustUnsupportedVexOpError). Updated
# from native/angr/src/vex/opcode_map.rs::parse_neon_unimplemented.
#
# Empty as of angr-cudgw.6: the last scaffold (Iop_PwAdd32Fx2) graduated to
# IROp::VFPwAdd, so no opcode currently routes to NeonUnimplemented. The
# variant + parse_neon_unimplemented fn are kept as the scaffold point for the
# next NEON op; add it here (and remove from _UNMAPPED_* if applicable) when one
# lands. Still-unsupported NEON shift-by-immediate ops live in
# _UNMAPPED_NEON_QSHL_IMM (they route to IROp::Unmapped).
_NEON_UNIMPLEMENTED: list[str] = []

# Linux syscall names that lack a native handler in native/angr/src/syscalls/
# as of 2026-06-01 (campaigns angr-0hif.{1,5,6,7}). Production code returns
# `SyscallError` for symbolic/unsupported and routes to the Python callback —
# the typed-error surface exists ONLY via `_raise_typed_test_error` until the
# tkbr.1-style conversion lands per-syscall, so this group exercises the
# Python-facing exception API. Each entry pairs (name, num) on amd64.
_UNSUPPORTED_SYSCALLS = [
    # File path operations (0hif.1)
    ("open", 2),
    ("openat", 257),
    ("close", 3),
    ("stat", 4),
    ("fstat", 5),
    ("lstat", 6),
    ("newfstatat", 262),
    ("readlink", 89),
    ("access", 21),
    # FD control (0hif.5)
    ("ioctl", 16),
    ("fcntl", 72),
    ("dup", 32),
    ("dup2", 33),
    ("pipe", 22),
    ("pipe2", 293),
    # Signals + process control (0hif.6)
    ("kill", 62),
    ("tgkill", 234),
    ("pause", 34),
    ("alarm", 37),
    # Resource limits + concurrency (0hif.7)
    ("getrlimit", 97),
    ("setrlimit", 160),
    ("futex", 202),
    ("eventfd", 290),
]


def _build_unop_irsb_json(op_name, arch):
    """Single-Unop IRSB referencing `op_name`. Args/result type are I64 since
    the dispatch hits `IROp::Unmapped` before any width validation runs."""
    import json

    offs_ip = 272 if arch.lower() in ("arm64", "aarch64") else 184
    return json.dumps(
        {
            "addr": 4096,
            "arch": arch,
            "statements": [
                {"tag": "Ist_IMark", "addr": 4096, "len": 4, "delta": 0},
                {
                    "tag": "Ist_WrTmp",
                    "tmp": 0,
                    "data": {
                        "tag": "Iex_Const",
                        "con": {"tag": "Ico_U64", "value": 0},
                    },
                },
                {
                    "tag": "Ist_WrTmp",
                    "tmp": 1,
                    "data": {
                        "tag": "Iex_Unop",
                        "op": op_name,
                        "arg": {"tag": "Iex_RdTmp", "tmp": 0},
                    },
                },
            ],
            "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 4100}},
            "jumpkind": "Ijk_Boring",
            "offsIP": offs_ip,
            "tyenv": {"types": ["Ity_I64", "Ity_I64"]},
        }
    )


def _build_binop_irsb_json(op_name, arch):
    """Single-Binop IRSB referencing `op_name`. Used for QShlN-style ops
    (binary in VEX) and the lone NeonUnimplemented entry Iop_PwAdd32Fx2."""
    import json

    offs_ip = 272 if arch.lower() in ("arm64", "aarch64") else 184
    return json.dumps(
        {
            "addr": 4096,
            "arch": arch,
            "statements": [
                {"tag": "Ist_IMark", "addr": 4096, "len": 4, "delta": 0},
                {
                    "tag": "Ist_WrTmp",
                    "tmp": 0,
                    "data": {
                        "tag": "Iex_Const",
                        "con": {"tag": "Ico_U64", "value": 0},
                    },
                },
                {
                    "tag": "Ist_WrTmp",
                    "tmp": 1,
                    "data": {
                        "tag": "Iex_Binop",
                        "op": op_name,
                        "args": [
                            {"tag": "Iex_RdTmp", "tmp": 0},
                            {"tag": "Iex_RdTmp", "tmp": 0},
                        ],
                    },
                },
            ],
            "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 4100}},
            "jumpkind": "Ijk_Boring",
            "offsIP": offs_ip,
            "tyenv": {"types": ["Ity_I64", "Ity_I64"]},
        }
    )


def _build_triop_irsb_json(op_name, arch):
    """Single-Triop IRSB referencing `op_name`, all three args the same
    concrete I64 zero. Used for the x87 (rm, x, y) transcendentals."""
    import json

    offs_ip = 272 if arch.lower() in ("arm64", "aarch64") else 184
    return json.dumps(
        {
            "addr": 4096,
            "arch": arch,
            "statements": [
                {"tag": "Ist_IMark", "addr": 4096, "len": 4, "delta": 0},
                {
                    "tag": "Ist_WrTmp",
                    "tmp": 0,
                    "data": {
                        "tag": "Iex_Const",
                        "con": {"tag": "Ico_U64", "value": 0},
                    },
                },
                {
                    "tag": "Ist_WrTmp",
                    "tmp": 1,
                    "data": {
                        "tag": "Iex_Triop",
                        "op": op_name,
                        "args": [
                            {"tag": "Iex_RdTmp", "tmp": 0},
                            {"tag": "Iex_RdTmp", "tmp": 0},
                            {"tag": "Iex_RdTmp", "tmp": 0},
                        ],
                    },
                },
            ],
            "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 4100}},
            "jumpkind": "Ijk_Boring",
            "offsIP": offs_ip,
            "tyenv": {"types": ["Ity_I64", "Ity_I64"]},
        }
    )


class TestRustUnsupportedErrorParametrized:
    """angr-uprs: parametrized ``pytest.raises`` coverage of the typed
    ``RustUnsupported*`` surface.

    Goal: guard against silent regressions where adding a new dispatch arm
    accidentally consumes an opcode (returning fresh-symbolic) instead of
    surfacing a typed exception. If any entry in the list below gets
    implemented, the corresponding test fails and the entry should be
    moved out of the list (or replaced with another unmapped op).

    Total cases >= 50 (acceptance target). All run via the lightweight
    `execute_irsb_for_test` mock-context path (no Project / no Solver) so
    the full class is well under the 30s acceptance budget.

    Each case also asserts the EXACT subclass (not just `RustExecutionError`)
    so a regression that flattens the typed hierarchy is caught.
    """

    @pytest.mark.parametrize("op_name", _UNMAPPED_VEX_OPS_REAL)
    def test_unmapped_vex_op_raises_unsupported(self, op_name):
        """Each currently-unmapped real opcode surfaces as
        ``RustUnsupportedVexOpError`` carrying `<op_name>` + arch."""
        from angr.rustylib.vex_engine import execute_irsb_for_test

        from angr.exploration import RustUnsupportedVexOpError

        with pytest.raises(RustUnsupportedVexOpError) as exc_info:
            execute_irsb_for_test(_build_unop_irsb_json(op_name, "AMD64"), "amd64")
        # Exact subclass — flattening the hierarchy would still satisfy the
        # base-class match but is a regression.
        assert type(exc_info.value).__name__ == "RustUnsupportedVexOpError"
        msg = str(exc_info.value)
        assert op_name in msg, f"op name missing from message: {msg}"
        assert "amd64" in msg.lower(), f"arch missing from message: {msg}"

    @pytest.mark.parametrize(("op_name", "arity"), _MAPPED_X87_TRANSCENDENTALS)
    def test_x87_transcendental_is_mapped_not_unsupported(self, op_name, arity):
        """angr-9ke6b.233: the x87 / FRECPX transcendentals reach the libm
        fast paths in ``vex/transcendentals.rs`` from the *string* opcode
        router, so they must execute instead of raising.

        Regression guard for the dead-code window opened by angr-h0ur: with
        no producer for ``IROp::Raw`` these all fell through to
        ``IROp::Unmapped`` and every caller silently degraded to the Python
        fallback.
        """
        from angr.rustylib.vex_engine import execute_irsb_for_test

        builder = _build_binop_irsb_json if arity == 2 else _build_triop_irsb_json
        # Concrete args -> the libm path returns a value; any exception here
        # (typically RustUnsupportedVexOpError) means the op lost its arm.
        execute_irsb_for_test(builder(op_name, "AMD64"), "amd64")

    @pytest.mark.parametrize("op_name", _NEON_UNIMPLEMENTED)
    def test_neon_unimplemented_raises_unsupported(self, op_name):
        """NeonUnimplemented routes through ``OpError::UnsupportedNeon`` but
        Python-side it still surfaces as ``RustUnsupportedVexOpError``."""
        from angr.rustylib.vex_engine import execute_irsb_for_test

        from angr.exploration import RustUnsupportedVexOpError

        with pytest.raises(RustUnsupportedVexOpError) as exc_info:
            execute_irsb_for_test(_build_binop_irsb_json(op_name, "ARM64"), "arm64")
        assert type(exc_info.value).__name__ == "RustUnsupportedVexOpError"
        msg = str(exc_info.value)
        assert op_name in msg, f"op name missing from message: {msg}"
        assert "arm64" in msg.lower(), f"arch missing from message: {msg}"

    @pytest.mark.parametrize(("name", "num"), _UNSUPPORTED_SYSCALLS)
    def test_unsupported_syscall_typed_error(self, name, num):
        """Syscalls without a native handler surface as
        ``RustUnsupportedSyscallError`` via the ``_raise_typed_test_error``
        helper. Production sites currently return ``SyscallError`` and fall
        back to the Python callback path; this parametrize pins the Python
        exception API so a follow-up tkbr.1-style conversion only needs to
        wire each handler into the existing typed-error path."""
        from angr.rustylib.vex_engine import _raise_typed_test_error

        from angr.exploration import RustUnsupportedSyscallError

        with pytest.raises(RustUnsupportedSyscallError) as exc_info:
            _raise_typed_test_error(
                "unsupported_syscall",
                name=name,
                num=num,
                arch="AMD64",
                message="no native handler",
            )
        assert type(exc_info.value).__name__ == "RustUnsupportedSyscallError"
        msg = str(exc_info.value)
        assert name in msg, f"syscall name missing from message: {msg}"
        # Number is rendered as decimal in the Display impl.
        assert str(num) in msg, f"syscall num missing from message: {msg}"

    def test_total_case_count_meets_acceptance(self):
        """Acceptance: >=50 parametrized cases across the three groups.

        Tracked here so a thoughtless trim of any list (e.g. an
        implementation lands that moves an op from this list to the real
        dispatch) still keeps the total above the bar."""
        total = len(_UNMAPPED_VEX_OPS_REAL) + len(_NEON_UNIMPLEMENTED) + len(_UNSUPPORTED_SYSCALLS)
        assert total >= 50, f"only {total} parametrized cases; need >=50"


class TestImportZ3ConstraintPtrsValidation:
    """angr-33t9: validate import_z3_constraint_ptrs rejects malformed input.

    Catches the realistic misuse cases (null ptr, wrong sort) that the PyO3
    trust-model audit (angr-9l9j) flagged. Does NOT exercise truly garbage
    integers (e.g. 0xdeadbeef) — those still segfault inside Z3's deref,
    and ruling them out would require a side-table of blessed ptrs.
    """

    def test_null_pointer_rejected(self, fauxware_project):
        """A null (0) pointer in the ptrs list yields PyValueError."""

        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        sid = next(iter(mgr._rust_mgr.get_state_ids("active")))

        with pytest.raises(ValueError, match="null pointer at index"):
            mgr._rust_mgr.import_z3_constraint_ptrs(sid, [0])

    def test_null_after_valid_ptr_rejected_atomically(self, fauxware_project):
        """A null partway through the list rejects the whole batch — validation
        runs before any constraint is added, so the state's solver is
        unmutated on failure."""
        import claripy

        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        sid = next(iter(mgr._rust_mgr.get_state_ids("active")))

        # Obtain a real Bool Z3 ptr to pair with the null.
        z3_backend = claripy.backends.z3
        bv = claripy.BVS("x_33t9", 32)
        bool_ast = z3_backend.convert(bv == 0)
        bool_ptr = bool_ast.as_ast().value
        assert bool_ptr != 0

        before = len(mgr._rust_mgr.export_z3_constraint_ptrs(sid))
        with pytest.raises(ValueError, match="null pointer at index 1"):
            mgr._rust_mgr.import_z3_constraint_ptrs(sid, [bool_ptr, 0])
        after = len(mgr._rust_mgr.export_z3_constraint_ptrs(sid))
        assert before == after, f"validation failure leaked partial constraints: before={before} after={after}"

    def test_bv_sort_pointer_rejected(self, fauxware_project):
        """A Z3 AST with non-Bool sort (e.g. a BV) is rejected as the wrong
        sort kind, not silently asserted (which would corrupt the solver)."""
        import claripy

        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        sid = next(iter(mgr._rust_mgr.get_state_ids("active")))

        # Convert a BV (not a Bool predicate) to get a non-Bool Z3 ptr.
        z3_backend = claripy.backends.z3
        bv = claripy.BVS("y_33t9", 64)
        bv_ast = z3_backend.convert(bv)
        bv_ptr = bv_ast.as_ast().value
        assert bv_ptr != 0

        with pytest.raises(ValueError, match="expected Bool"):
            mgr._rust_mgr.import_z3_constraint_ptrs(sid, [bv_ptr])

    def test_valid_bool_ptr_round_trip(self, fauxware_project):
        """Sanity: a real Bool ptr is accepted and the constraint becomes
        visible to subsequent export. Guards against the validation
        accidentally rejecting valid input."""
        import claripy

        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        sid = next(iter(mgr._rust_mgr.get_state_ids("active")))

        z3_backend = claripy.backends.z3
        bv = claripy.BVS("z_33t9", 32)
        # Hold a strong ref to the z3 wrapper so the AST stays alive
        # through the import call (z3-side refcount discipline — same as
        # _cached_z3_ast_ptr in rust_state_sync.py).
        bool_ast = z3_backend.convert(bv == 0x4242)
        bool_ptr = bool_ast.as_ast().value
        assert bool_ptr != 0

        before = len(mgr._rust_mgr.export_z3_constraint_ptrs(sid))
        sat = mgr._rust_mgr.import_z3_constraint_ptrs(sid, [bool_ptr])
        assert sat is True, "constraint x==0x4242 should be satisfiable"
        after = len(mgr._rust_mgr.export_z3_constraint_ptrs(sid))
        assert after == before + 1, f"expected 1 constraint to be added; got before={before} after={after}"


class TestStashNameValidation:
    """angr-630x: warn on unknown stash names in create_state/move_state/etc.

    Follow-up from the angr-9l9j PyO3 trust-model audit. A typo like
    ``actve`` for ``active`` previously created an invisible stash and the
    state vanished from ``mgr.active`` / ``mgr.found``. Now the first
    creation of a non-standard stash emits a ``log::warn!`` line so the typo
    is surfaced at the boundary.
    """

    def _set_warn_level(self):
        from angr.rustylib.vex_engine import set_rust_log_level

        set_rust_log_level("warn")

    def _new_low_level_mgr(self):
        from angr.rustylib.vex_engine import RustExplorationManager as _LL

        return _LL("amd64")

    def test_unknown_dest_stash_in_create_state_warns(self, capfd):
        """create_state with a typo'd stash name logs a one-time warning."""
        self._set_warn_level()
        mgr = self._new_low_level_mgr()
        _ = capfd.readouterr()

        sid = mgr.create_state("actve")
        captured = capfd.readouterr()
        assert "creating new stash 'actve'" in captured.err, (
            f"expected warn for typo stash; got stderr:\n{captured.err!r}"
        )
        # State is still actually placed (warn, don't reject) — typo-tolerant.
        assert sid in mgr.get_state_ids("actve")
        assert mgr.stash_count("actve") == 1

    def test_standard_stash_names_do_not_warn(self, capfd):
        """create_state with one of the seven standard stash names is silent."""
        self._set_warn_level()
        mgr = self._new_low_level_mgr()
        _ = capfd.readouterr()

        for stash in (
            "active",
            "found",
            "avoid",
            "deadended",
            "errored",
            "pruned",
            "unconstrained",
        ):
            mgr.create_state(stash)

        captured = capfd.readouterr()
        assert "creating new stash" not in captured.err, (
            f"standard stash names should not warn; got stderr:\n{captured.err!r}"
        )

    def test_unknown_stash_warns_only_once(self, capfd):
        """Repeated use of the same non-standard stash warns only on first
        creation — the stash exists in the map after that and subsequent
        operations are silent."""
        self._set_warn_level()
        mgr = self._new_low_level_mgr()
        _ = capfd.readouterr()

        mgr.create_state("custom_stash_630x")
        mgr.create_state("custom_stash_630x")
        mgr.create_state("custom_stash_630x")

        captured = capfd.readouterr()
        warn_count = captured.err.count("creating new stash 'custom_stash_630x'")
        assert warn_count == 1, (
            f"expected exactly one warn for repeated use; got {warn_count} in stderr:\n{captured.err!r}"
        )
        assert mgr.stash_count("custom_stash_630x") == 3

    def test_move_state_to_unknown_stash_warns(self, capfd):
        """move_state into a typo'd destination also warns on first creation."""
        self._set_warn_level()
        mgr = self._new_low_level_mgr()
        sid = mgr.create_state("active")
        _ = capfd.readouterr()

        moved = mgr.move_state(sid, "active", "fnd")
        assert moved is True
        captured = capfd.readouterr()
        assert "creating new stash 'fnd'" in captured.err, (
            f"expected warn for move_state typo; got stderr:\n{captured.err!r}"
        )
        assert mgr.stash_count("fnd") == 1

    def test_per_module_filter_silences_other_modules(self, capfd):
        """angr-c242: RUST_LOG-style spec keeps only the targeted module live.

        Sets the filter to ``rustylib::stash=warn,off`` — the stash module
        still warns on typo creation, but other modules drop to ``off``. We
        can't easily emit a non-stash log from Python, so the positive half
        (stash still warns) covers the parser path and the negative half is
        the absence of any non-stash output.
        """
        from angr.rustylib.vex_engine import set_rust_log_level

        set_rust_log_level("rustylib::stash=warn,off")
        try:
            mgr = self._new_low_level_mgr()
            _ = capfd.readouterr()
            mgr.create_state("c242_typo")
            captured = capfd.readouterr()
            assert "creating new stash 'c242_typo'" in captured.err, (
                f"per-module filter should keep stash warns live; got stderr:\n{captured.err!r}"
            )
            # Only the stash warn should be present; nothing from any other
            # rustylib::* module slipped through.
            for line in captured.err.splitlines():
                if not line.startswith("[rust:"):
                    continue
                assert "rustylib::stash" in line, (
                    f"non-stash log leaked through under per-module 'off' default: {line!r}"
                )
        finally:
            # Reset so we don't leak this filter into later tests.
            set_rust_log_level("off")

    def test_set_rust_log_level_accepts_full_filter_spec(self):
        """angr-c242: full RUST_LOG-style specs (with ``=`` or ``,``) are
        accepted; invalid single-word levels still raise."""
        from angr.rustylib.vex_engine import set_rust_log_level

        try:
            # Specs with `=` or `,` bypass the strict single-word check and
            # are handed to env_logger's parser. Acceptance is that the call
            # returns None (no exception).
            set_rust_log_level("rustylib::stash=warn,info")
            set_rust_log_level("rustylib::exploration=debug")
            set_rust_log_level("warn,rustylib::stash=trace")
            # Single-word typo still fails (matches pre-existing behavior).
            with pytest.raises(ValueError):
                set_rust_log_level("bogus")
        finally:
            set_rust_log_level("off")


class TestSnapshotRoundTrip:
    """Snapshot dump/load via the Python wrapper (angr-x04s.1.4).

    Closes the angr-x04s.1 prototype: end-to-end round-trip on the fauxware
    binary, plus the format-version asymmetric-mismatch error path.
    """

    def test_dump_load_fauxware_round_trip_preserves_stash_shape(self, fauxware_project, tmp_path):
        """Mid-exploration snapshot + restore preserves the structural shape
        of every stash (state_id set, per-state pc, per-state constraint
        count). This is the structural-equivalence half of the round-trip
        contract; the model-equivalence half (``posix.dumps(0)`` parity)
        is exercised separately and depends on Z3 heuristic latitude.

        Steps ``run_mgr`` 10 times so it has executed past the entry
        block and accumulated some lineage, dumps to a tempfile, loads
        into a fresh manager constructed from the same entry state, and
        compares the two on:

        - active/found/deadended/avoid stash sizes,
        - the set of state_ids per stash (a state's id is preserved
          across the round-trip — the snapshot serializes ``state_id``
          and ``from_snapshot`` does not rebump the global counter),
        - the (state_id, pc, constraint_count) tuple per state in active.

        Then continues exploration on the resumed manager and asserts it
        still terminates at ``find=0x4006ed`` (i.e. exploration is
        functional after restore, not stuck).
        """

        find_addr = 0x4006ED

        run_state = fauxware_project.factory.entry_state()
        run_mgr = RustExplorationManager(fauxware_project, [run_state])
        run_mgr.step(n=10)

        snapshot_path = tmp_path / "fauxware.snap"
        run_mgr.dump_snapshot(str(snapshot_path))
        assert snapshot_path.stat().st_size > 0, "snapshot file must be non-empty"

        pre_counts = dict(run_mgr.stash_counts())
        pre_active_ids = sorted(run_mgr._rust_mgr.get_state_ids("active"))
        pre_active_pcs = [run_mgr._rust_mgr.get_state_pc_by_id(sid) for sid in pre_active_ids]
        pre_active_constraint_counts = [run_mgr._rust_mgr.state_constraint_count(sid) for sid in pre_active_ids]

        resumed_state = fauxware_project.factory.entry_state()
        resumed_mgr = RustExplorationManager(fauxware_project, [resumed_state])
        resumed_mgr.load_snapshot(str(snapshot_path))

        post_counts = dict(resumed_mgr.stash_counts())
        post_active_ids = sorted(resumed_mgr._rust_mgr.get_state_ids("active"))
        post_active_pcs = [resumed_mgr._rust_mgr.get_state_pc_by_id(sid) for sid in post_active_ids]
        post_active_constraint_counts = [resumed_mgr._rust_mgr.state_constraint_count(sid) for sid in post_active_ids]

        assert post_counts == pre_counts, f"stash counts differ post-restore: {pre_counts} -> {post_counts}"
        assert post_active_ids == pre_active_ids, (
            f"active state_ids differ post-restore: {pre_active_ids} -> {post_active_ids}"
        )
        assert post_active_pcs == pre_active_pcs, (
            f"active state PCs differ post-restore: {pre_active_pcs} -> {post_active_pcs}"
        )

        # angr-82g6: `state_constraint_count` (the
        # `SymContext::num_constraints` counter) now round-trips because
        # the snapshot captures the full Z3 solver as SMT-LIB2 and
        # restore replays every assertion through `add_constraint_raw`.
        # The `assumed_constraints` BV log is restored separately via
        # `assumed_constraints_push` (no second solver assert), so the
        # two captures don't double-count. Pre-82g6, the raw-path
        # constraints from the Python claripy sync layer were dropped —
        # see `snapshot-add-constraint-raw-not-tracked` for the old gap.
        assert post_active_constraint_counts == pre_active_constraint_counts, (
            f"per-state num_constraints differ post-restore: "
            f"{pre_active_constraint_counts} -> "
            f"{post_active_constraint_counts}"
        )

        resumed_mgr.explore(find=find_addr, num_find=1)
        assert len(resumed_mgr.found) > 0, (
            "resumed manager must continue exploration after load and reach the find address"
        )
        resumed_stdin = bytes(resumed_mgr.found[0].posix.dumps(0))
        assert len(resumed_stdin) > 0, "resumed exploration's first found state must have a non-empty stdin model"

    def test_load_snapshot_rejects_stale_version_byte(self, fauxware_project, tmp_path):
        """Format-version asymmetric mismatch: load must refuse a snapshot
        whose first byte is not the current ``STASH_SNAPSHOT_VERSION``.

        Builds a valid snapshot from a fresh manager, flips the version
        byte, asserts that :meth:`load_snapshot` raises ``ValueError``.
        Symmetric to the Rust-side ``test_stash_manager_load_snapshot_errors``
        in ``native/angr/src/stash.rs``.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        snapshot_path = tmp_path / "good.snap"
        mgr.dump_snapshot(str(snapshot_path))
        good = snapshot_path.read_bytes()
        assert len(good) > 0
        bad = bytes([good[0] ^ 0xFF]) + good[1:]
        bad_path = tmp_path / "bad.snap"
        bad_path.write_bytes(bad)

        resumed_state = fauxware_project.factory.entry_state()
        resumed_mgr = RustExplorationManager(fauxware_project, [resumed_state])
        with pytest.raises(ValueError, match="version mismatch"):
            resumed_mgr.load_snapshot(str(bad_path))

    def test_load_snapshot_rejects_empty_envelope(self, fauxware_project, tmp_path):
        """Empty snapshot file routes through the typed ``SnapshotError::EmptyEnvelope``
        variant and surfaces as a ``ValueError`` on the Python side."""

        empty_path = tmp_path / "empty.snap"
        empty_path.write_bytes(b"")

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        with pytest.raises(ValueError, match="empty snapshot envelope"):
            mgr.load_snapshot(str(empty_path))

    def test_load_snapshot_disables_phase2_eager_retry(self, fauxware_project, tmp_path):
        """angr-op0dn.13.12: a resumed manager must not re-seed the placeholder
        state its constructor was handed.

        The angr-027h phase-2 retry re-seeds ``_initial_seed_states`` when an
        address-based find exhausts. After ``load_snapshot`` those are the
        pre-load placeholder, not the restored frontier, so replaying them would
        explore a path the caller never asked to resume (and flip
        ``use_deferred_forks`` off globally, which drops stored branch conditions
        on the parallel path). ``load_snapshot`` therefore retires phase 2.
        """
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.step(n=1)
        assert mgr._initial_seed_states, "pre-condition: the constructor kept its seed copies"
        assert getattr(mgr, "_phase2_retried", False) is False

        snapshot_path = tmp_path / "phase2.snap"
        mgr.dump_snapshot(str(snapshot_path))

        placeholder = fauxware_project.factory.entry_state()
        resumed = RustExplorationManager(fauxware_project, [placeholder])
        resumed.load_snapshot(str(snapshot_path))

        assert resumed._initial_seed_states is None
        assert resumed._phase2_retried is True
        # The guard is what the retry itself checks, so the retry is now a no-op
        # even with an address-based find pending and nothing found.
        resumed._explore_find_addrs = {fauxware_project.entry}
        assert resumed._maybe_phase2_eager_retry(num_find=1) is False

    def test_load_from_disk_constructs_fresh_manager_without_placeholder(self, fauxware_project, tmp_path):
        """angr-9o4n: the v1.0 classmethod constructs a fresh manager
        without requiring the caller to pre-build an entry state purely
        for the constructor's signature. Functionally equivalent to
        ``mgr = RustExplorationManager(project); mgr.load_snapshot(path)``
        but with the two-step collapsed and active_states=None made
        explicit.

        Round-trip contract: save N states from a stepped manager, load
        into a fresh one via the classmethod, exploration continues to
        ``find`` and produces a non-empty stdin model.
        """

        find_addr = 0x4006ED

        run_state = fauxware_project.factory.entry_state()
        run_mgr = RustExplorationManager(fauxware_project, [run_state])
        run_mgr.step(n=10)

        pre_active = sorted(run_mgr._rust_mgr.get_state_ids("active"))
        assert pre_active, "stepped manager must have active states"

        snap = tmp_path / "fauxware.snap"
        run_mgr.dump_snapshot(str(snap))

        resumed = RustExplorationManager.load_from_disk(snap, fauxware_project)
        assert sorted(resumed._rust_mgr.get_state_ids("active")) == pre_active

        resumed.explore(find=find_addr, num_find=1)
        assert len(resumed.found) > 0, "resumed exploration must reach find"
        assert len(bytes(resumed.found[0].posix.dumps(0))) > 0, (
            "resumed exploration must produce a non-empty stdin model"
        )

    def test_load_from_disk_passes_kwargs_to_constructor(self, fauxware_project, tmp_path):
        """Manager-level configuration (not captured by the snapshot) must
        be overridable on resume by passing kwargs through to ``__init__``.
        Smoke-checks that ``exploration_strategy='dfs'`` and a custom
        ``solver_timeout_ms`` flow through without raising and the
        resumed manager remains functional."""

        run_state = fauxware_project.factory.entry_state()
        run_mgr = RustExplorationManager(fauxware_project, [run_state])
        run_mgr.step(n=5)
        snap = tmp_path / "fauxware.snap"
        run_mgr.dump_snapshot(str(snap))

        resumed = RustExplorationManager.load_from_disk(
            snap,
            fauxware_project,
            exploration_strategy="dfs",
            solver_timeout_ms=12345,
        )
        # Active stash carries the same state_ids as the source.
        assert sorted(resumed._rust_mgr.get_state_ids("active")) == sorted(run_mgr._rust_mgr.get_state_ids("active"))
        # Resume is functional after the kwargs path.
        resumed.step(n=1)


class TestAvoidMultivaluedOptions:
    """angr-tfic: AVOID_MULTIVALUED_READS / AVOID_MULTIVALUED_WRITES SimOptions.

    Python's `address_concretization_mixin` (storage/memory_mixins/
    address_concretization_mixin.py:272/327) short-circuits symbolic-addr
    loads to an unconstrained value and silently drops symbolic-addr writes
    when these options are set. The Rust engine mirrors that via
    `AddressConcretizer::should_avoid_multivalued_read/write` gates at the
    memory-layer (load_symbolic_unified / store_symbolic_unified) and
    interpreter (try_rust_memory_load / try_rust_memory_store /
    load_symbolic_addr / handle_symbolic_store) entry points.
    """

    def test_configure_accepts_avoid_multivalued_kwargs(self):
        """The Rust-side `_RustExplorationManager.configure_concretization_strategies`
        accepts the two new kwargs and stores them on the concretizer config.
        Positional ordering matches the PyO3 signature."""
        mgr = _RustExplorationManager("amd64")
        # Positional call with all six params: (use_approximate, read_limit,
        # write_limit, symbolic_write_addresses, avoid_reads, avoid_writes).
        mgr.configure_concretization_strategies(False, 1024, 128, False, True, True)
        cfg = mgr.get_concretization_config()
        assert cfg["avoid_multivalued_reads"] == 1
        assert cfg["avoid_multivalued_writes"] == 1
        # Keyword call — reads on, writes off. A transposed/dropped kwarg at
        # the PyO3 boundary would flip these.
        mgr.configure_concretization_strategies(
            False,
            avoid_multivalued_reads=True,
            avoid_multivalued_writes=False,
        )
        cfg = mgr.get_concretization_config()
        assert cfg["avoid_multivalued_reads"] == 1
        assert cfg["avoid_multivalued_writes"] == 0
        # Keyword call — reads off, writes on.
        mgr.configure_concretization_strategies(
            False,
            avoid_multivalued_reads=False,
            avoid_multivalued_writes=True,
        )
        cfg = mgr.get_concretization_config()
        assert cfg["avoid_multivalued_reads"] == 0
        assert cfg["avoid_multivalued_writes"] == 1

    def test_avoid_multivalued_reads_smoke(self, fauxware_project):
        """Exploration with AVOID_MULTIVALUED_READS completes without
        crashing. fauxware contains symbolic-addr loads via the password
        comparison loop, so the gate is exercised at least once per state."""

        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.AVOID_MULTIVALUED_READS},
        )
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED, avoid=0x4006FD, max_steps=50000)
        # We don't assert found > 0 because the option can prune the
        # success path; the contract is that the engine doesn't crash.
        total = len(mgr.active) + len(mgr.deadended) + len(mgr.avoid) + len(mgr.errored) + len(mgr.found)
        assert total >= 1, f"AVOID_MULTIVALUED_READS exploration lost all states; counts={mgr.stash_counts()}"

    def test_avoid_multivalued_writes_smoke(self, fauxware_project):
        """Exploration with AVOID_MULTIVALUED_WRITES completes without
        crashing. Symbolic-addr writes silently no-op, which can affect
        reachability but must not error the engine."""

        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.AVOID_MULTIVALUED_WRITES},
        )
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED, avoid=0x4006FD, max_steps=50000)
        total = len(mgr.active) + len(mgr.deadended) + len(mgr.avoid) + len(mgr.errored) + len(mgr.found)
        assert total >= 1, f"AVOID_MULTIVALUED_WRITES exploration lost all states; counts={mgr.stash_counts()}"

    def test_avoid_multivalued_read_returns_unconstrained_via_memory_api(self, fauxware_project):
        """End-to-end: with AVOID_MULTIVALUED_READS set, a symbolic-address
        load via the Rust `memory_load_symbolic` path returns an
        unconstrained value — solver eval is NOT pinned to any concrete
        backer byte, so two evals under different constraints differ.
        Without the option, the same load enumerates and pins to backer
        data (or fails with SymbolicAddress).
        """
        import claripy

        # Pre-place a known concrete pattern at a target address so a
        # successful concretization would clearly resolve to it.
        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.AVOID_MULTIVALUED_READS},
        )
        marker_addr = 0x500500
        state.memory.store(
            marker_addr,
            claripy.BVV(0xCAFEBABEDEADBEEF, 64),
            endness=state.arch.memory_endness,
        )

        mgr = RustExplorationManager(fauxware_project, [state])
        # Take 1 step to ensure the state is loaded into the Rust manager
        # so the option propagation through `_add_rust_state` has fired.
        mgr.run(max_steps=1)

        # Contract: the AVOID_MULTIVALUED_READS SimOption set on the source
        # state propagates through `_add_rust_state` and lands as
        # avoid_multivalued_reads=1 on the Rust concretizer config. Reading
        # the config back here (rather than re-calling configure and
        # discarding the result) is what makes a no-op/dropped-assignment
        # regression detectable — the original test never inspected anything.
        cfg = mgr._rust_mgr.get_concretization_config()
        assert cfg["avoid_multivalued_reads"] == 1, (
            f"AVOID_MULTIVALUED_READS did not propagate to Rust config; cfg={cfg}"
        )
        # AVOID_MULTIVALUED_WRITES was NOT set, so it must stay 0 — proves the
        # two flags are stored independently, not coupled/transposed.
        assert cfg["avoid_multivalued_writes"] == 0, f"avoid_multivalued_writes leaked on; cfg={cfg}"

        # The manager remains healthy after the gated symbolic-addr loads
        # fauxware performs under the option.
        assert (len(mgr.active) + len(mgr.deadended) + len(mgr.errored)) >= 1

    def test_avoid_multivalued_read_behavioral_unconstrained_vs_pinned(self):
        """angr-vkkny: the *behavioral* AVOID_MULTIVALUED_READS contract, via
        the native `probe_symbolic_load_value_range` surface.

        The config read-back in
        `test_avoid_multivalued_read_returns_unconstrained_via_memory_api`
        only proves the flag propagated. It cannot catch a deletion of the
        `should_avoid_multivalued_read` branch in
        `memory/load.rs::load_symbolic_unified`: the flag stays set but the
        load pins to the backer byte. This test exercises the actual load:

        * option ON  -> a symbolic-address load returns a FRESH UNCONSTRAINED
          value spanning the full width (min == 0, max == 2**(8*size) - 1),
          NOT pinned to the concrete backer.
        * option OFF -> the same load concretizes the (pinned) address and
          reads the backer, so min == max == the stored marker.

        Deleting the avoid branch collapses the ON case to min == max, which
        the `hi_on > lo_on` assertion below catches.
        """
        from angr.rustylib.vex_engine import RustSimState

        marker_addr = 0x500500
        marker_val = 0xCAFEBABEDEADBEEF
        marker_bytes = marker_val.to_bytes(8, "little")

        # Option ON: unconstrained load, not pinned to the backer.
        state_on = RustSimState("amd64")
        state_on.map_memory_data(marker_addr, marker_bytes)
        # (use_approximate, read_limit, write_limit, sym_write_addrs,
        #  avoid_reads, avoid_writes)
        state_on.configure_concretization_strategies(False, None, None, False, True, False)
        lo_on, hi_on = state_on.probe_symbolic_load_value_range(marker_addr, 8)
        assert hi_on > lo_on, (
            f"AVOID_MULTIVALUED_READS load was pinned (lo={lo_on:#x} hi={hi_on:#x}); "
            "expected an unconstrained range — the should_avoid_multivalued_read "
            "branch in memory/load.rs may have regressed"
        )
        # A fresh unconstrained 8-byte symbol spans the full 64-bit range.
        assert lo_on == 0
        assert hi_on == (1 << 64) - 1

        # Option OFF: the load concretizes the pinned address and reads the
        # backer, so it is single-valued at the marker.
        state_off = RustSimState("amd64")
        state_off.map_memory_data(marker_addr, marker_bytes)
        state_off.configure_concretization_strategies(False, None, None, False, False, False)
        lo_off, hi_off = state_off.probe_symbolic_load_value_range(marker_addr, 8)
        assert lo_off == hi_off == marker_val, (
            f"without AVOID_MULTIVALUED_READS the load should pin to the backer "
            f"{marker_val:#x}; got lo={lo_off:#x} hi={hi_off:#x}"
        )


class TestConcretizationOptionPropagation:
    """angr-1kdi: APPROXIMATE_MEMORY_INDICES / SYMBOLIC_WRITE_ADDRESSES SimOptions.

    ``rust_manager._add_rust_state`` reads both options off ``state.options``
    plus the read/write strategy ``_limit`` values off
    ``state.memory.read_strategies`` / ``write_strategies`` and forwards all
    six into ``_RustExplorationManager.configure_concretization_strategies``
    via a positional call. The ``get_concretization_config`` getter (added
    alongside these tests) reads the Rust-side concretizer config back so a
    positional-arg swap or a strategy-sniffing regression is no longer silent.

    Both options are documented Honored in
    ``docs/advanced-topics/rust_engine.rst`` yet had zero test coverage before
    this class.
    """

    def test_get_concretization_config_defaults(self):
        """A fresh manager carries Python's default concretizer config:
        all flags off, read limit 1024, write limit 128."""
        mgr = _RustExplorationManager("amd64")
        cfg = mgr.get_concretization_config()
        assert cfg["use_approximate"] == 0
        assert cfg["symbolic_write_addresses"] == 0
        assert cfg["avoid_multivalued_reads"] == 0
        assert cfg["avoid_multivalued_writes"] == 0
        assert cfg["read_range_limit"] == 1024
        assert cfg["write_range_limit"] == 128

    def test_approximate_memory_indices_propagates_to_rust(self, fauxware_project):
        """Setting APPROXIMATE_MEMORY_INDICES on a state flips the Rust
        ``use_approximate`` flag once the state is loaded into the manager,
        and the read range limit is bumped to the approximate minimum
        (4096) because the sniffed default (1024) sits below it."""
        plain = fauxware_project.factory.entry_state()
        plain_mgr = RustExplorationManager(fauxware_project, [plain])
        assert plain_mgr._rust_mgr.get_concretization_config()["use_approximate"] == 0

        approx = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.APPROXIMATE_MEMORY_INDICES},
        )
        approx_mgr = RustExplorationManager(fauxware_project, [approx])
        cfg = approx_mgr._rust_mgr.get_concretization_config()
        assert cfg["use_approximate"] == 1
        # configure_strategies bumps read_range_limit to APPROXIMATE_MIN_RANGE
        # (4096) when approximate is on and the sniffed limit is below it.
        assert cfg["read_range_limit"] >= 4096

    def test_symbolic_write_addresses_propagates_to_rust(self, fauxware_project):
        """Setting SYMBOLIC_WRITE_ADDRESSES on a state flips the Rust
        ``symbolic_write_addresses`` flag."""
        plain = fauxware_project.factory.entry_state()
        plain_mgr = RustExplorationManager(fauxware_project, [plain])
        assert plain_mgr._rust_mgr.get_concretization_config()["symbolic_write_addresses"] == 0

        sym = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.SYMBOLIC_WRITE_ADDRESSES},
        )
        sym_mgr = RustExplorationManager(fauxware_project, [sym])
        assert sym_mgr._rust_mgr.get_concretization_config()["symbolic_write_addresses"] == 1

    def test_custom_strategy_limits_propagate_to_rust(self, fauxware_project):
        """Non-default read/write strategy ``_limit`` values are sniffed off
        ``state.memory.{read,write}_strategies`` and forwarded into the Rust
        concretizer config. Guards the ``_limit`` sniffing loop against a
        regression that would silently fall back to the 1024/128 defaults."""
        state = fauxware_project.factory.entry_state()
        # The default Range strategies expose ``_limit``; the sniff loop picks
        # the first strategy carrying the attribute, so mutate index 0.
        state.memory.read_strategies[0]._limit = 777
        state.memory.write_strategies[0]._limit = 55

        mgr = RustExplorationManager(fauxware_project, [state])
        cfg = mgr._rust_mgr.get_concretization_config()
        assert cfg["read_range_limit"] == 777
        assert cfg["write_range_limit"] == 55

    def test_approximate_memory_indices_smoke(self, fauxware_project):
        """Exploration with APPROXIMATE_MEMORY_INDICES completes without
        crashing. The approximate read strategy is engaged on symbolic-addr
        loads in fauxware's password comparison loop."""
        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.APPROXIMATE_MEMORY_INDICES},
        )
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED, avoid=0x4006FD, max_steps=50000)
        total = len(mgr.active) + len(mgr.deadended) + len(mgr.avoid) + len(mgr.errored) + len(mgr.found)
        assert total >= 1, f"APPROXIMATE_MEMORY_INDICES exploration lost all states; counts={mgr.stash_counts()}"

    def test_symbolic_write_addresses_smoke(self, fauxware_project):
        """Exploration with SYMBOLIC_WRITE_ADDRESSES completes without
        crashing. Symbolic-addr writes are permitted (multi-valued) rather
        than concretized to a single address, which can broaden reachability
        but must not error the engine."""
        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.SYMBOLIC_WRITE_ADDRESSES},
        )
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED, avoid=0x4006FD, max_steps=50000)
        total = len(mgr.active) + len(mgr.deadended) + len(mgr.avoid) + len(mgr.errored) + len(mgr.found)
        assert total >= 1, f"SYMBOLIC_WRITE_ADDRESSES exploration lost all states; counts={mgr.stash_counts()}"


class TestCounterParity:
    """angr-ah3s: contract between ``mgr.stats()`` and the bench-diff tooling.

    ``tests/benchmarks/run_single.py`` categorizes counters via
    ``_DUMP_EXPLICIT_GROUPS`` (literal names) and ``_DUMP_PREFIX_GROUPS``
    (counter families); ``tests/benchmarks/bench_diff.py`` flattens the
    same dict to drive the per-bench regression triage. A silent rename
    on either side of the FFI boundary would drop a counter from the
    surfaced dict until a bench engineer noticed by hand.

    This test runs a single exploration step on a real binary so every
    code path that lazily attaches a counter has fired, then asserts
    that every name and every family expected by the bench tooling
    is present in ``mgr.stats()``.
    """

    @staticmethod
    def _load_dump_groups():
        """Import ``_DUMP_EXPLICIT_GROUPS`` / ``_DUMP_PREFIX_GROUPS`` from
        ``tests/benchmarks/run_single.py`` without making ``tests/benchmarks``
        a package (it intentionally is not — it's a script directory).
        """
        import importlib.util

        run_single_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "benchmarks",
            "run_single.py",
        )
        spec = importlib.util.spec_from_file_location("_bench_run_single", run_single_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._DUMP_EXPLICIT_GROUPS, module._DUMP_PREFIX_GROUPS

    def test_stats_dict_carries_all_expected_counter_names(self, fauxware_project):
        """Every literal name in ``_DUMP_EXPLICIT_GROUPS`` (the curated
        sections of ``run_single.py --dump-counters``) must appear in
        ``mgr.stats`` after one step. Failure prints the missing keys.
        """

        explicit_groups, _ = self._load_dump_groups()

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        # Route through the predicate path so ``_time_in_*_ns`` attrs are
        # populated — passing a callable ``find`` flips the explore() router
        # in rust_manager.py:3880 to ``_explore_with_predicates``. The
        # address path doesn't attach those attrs and would drop the
        # ``time_in_rust_run`` family from stats.
        mgr.run(find=lambda s: False, max_steps=1)
        stats = mgr.stats

        expected = set().union(*explicit_groups.values())
        missing = sorted(expected - set(stats))
        assert not missing, (
            f"mgr.stats() is missing {len(missing)} counter(s) expected by "
            f"run_single.py --dump-counters curated sections:\n  "
            + "\n  ".join(missing)
            + f"\n(total expected: {len(expected)}, present: "
            f"{len(expected) - len(missing)})"
        )

    def test_stats_dict_has_at_least_one_key_per_prefix_family(self, fauxware_project):
        """Each prefix family in ``_DUMP_PREFIX_GROUPS`` (``rust_*``,
        ``z3_*``, ``vex_*``, ``mem_*``, ``concretize_*``, ``bvop_*``,
        ``zext_*``) must contribute at least one key to ``mgr.stats()``.
        A missing family means an entire counter group fell off the
        Rust → Python bridge.
        """

        _, prefix_groups = self._load_dump_groups()

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=1)
        stats = mgr.stats

        missing_families = []
        for label, prefix in prefix_groups:
            if not any(k.startswith(prefix) for k in stats):
                missing_families.append(f"{label!r} (prefix={prefix!r})")
        assert not missing_families, (
            "mgr.stats() is missing entire counter families expected by "
            "run_single.py --dump-counters prefix groups:\n  " + "\n  ".join(missing_families)
        )


class TestProxyWriteCounters:
    """angr-7jv5: instrumentation for SimProc callback proxy traffic.

    The ``RustStateProxy`` sub-proxies (memory / registers / solver)
    write through to the underlying Rust state via PyO3 on every store.
    To decide whether a within-callback write buffer would pay off, we
    need per-FFI counts surfaced in ``mgr.stats``. This test pins the
    contract: each proxy write-through site increments the corresponding
    counter on the Python wrapper, the four counter keys are surfaced
    in ``stats``, and a no-op run leaves them at zero (so they only
    move when real proxy traffic happens).
    """

    PROXY_COUNTER_KEYS = (
        "proxy_mem_concrete_writes",
        "proxy_mem_ast_writes",
        "proxy_reg_writes",
        "proxy_solver_adds",
    )

    def test_counters_present_and_zero_on_clean_run(self, fauxware_project):
        """After construction (no exploration yet), all four proxy counters
        must be present in ``mgr.stats`` and equal to zero. This is the
        baseline a bench engineer reads when no proxy gate is on.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        stats = mgr.stats
        for key in self.PROXY_COUNTER_KEYS:
            assert key in stats, f"missing proxy counter: {key}"
            assert stats[key] == 0, f"expected {key} == 0 on a clean manager, got {stats[key]}"

    def test_register_proxy_write_bumps_counter(self, fauxware_project):
        """``RustRegisterProxy.__setattr__`` increments ``_stats_proxy_reg_writes``
        once per assigned register. Direct unit test against the proxy
        avoids depending on which simprocedures fire during exploration.
        """
        from angr.exploration.rust_state_proxy import RustRegisterProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        # Step once to materialize the Rust state. ``state_ids`` exposes the
        # currently-active state ids so the proxy can bind to a live one.
        mgr.run(max_steps=1)
        state_ids = list(mgr._rust_mgr.get_state_ids("active"))
        assert state_ids, "expected at least one active state after one step"
        proxy = RustRegisterProxy(mgr._rust_mgr, state_ids[0], fauxware_project.arch, python_mgr=mgr)
        before = mgr._stats_proxy_reg_writes
        proxy.rax = 0xDEADBEEF
        proxy.rbx = 0xCAFEBABE
        after = mgr._stats_proxy_reg_writes
        assert after - before == 2, f"expected 2 register writes, got delta {after - before}"
        assert mgr.stats["proxy_reg_writes"] == after

    def test_memory_proxy_concrete_write_bumps_counter(self, fauxware_project):
        """``RustMemoryProxy.store`` increments ``_stats_proxy_mem_concrete_writes``
        on a concrete-payload path. Three concrete-payload branches
        (concrete BVV, raw bytes, raw int) all bump the same counter.
        """
        from angr.exploration.rust_state_proxy import RustMemoryProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=1)
        state_ids = list(mgr._rust_mgr.get_state_ids("active"))
        assert state_ids, "expected at least one active state after one step"
        proxy = RustMemoryProxy(mgr._rust_mgr, state_ids[0], fauxware_project.arch, python_mgr=mgr)
        # Pick an address inside the binary's loaded text region — angr maps
        # those pages by default. The exact location does not matter; we
        # just need a concrete-addr concrete-payload write to flow through
        # ``set_state_memory_concrete``.
        addr = fauxware_project.entry
        before = mgr._stats_proxy_mem_concrete_writes
        proxy.store(addr, b"abcd")  # bytes branch
        proxy.store(addr + 8, 0x1234, size=2)  # int branch
        after = mgr._stats_proxy_mem_concrete_writes
        assert after - before == 2, f"expected 2 concrete-memory writes, got delta {after - before}"
        assert mgr.stats["proxy_mem_concrete_writes"] == after

    def _memory_proxy_on(self, project, state):
        """Build a ``RustMemoryProxy`` bound to the first active state."""
        from angr.exploration.rust_state_proxy import RustMemoryProxy

        mgr = RustExplorationManager(project, [state])
        mgr.run(max_steps=1)
        state_ids = list(mgr._rust_mgr.get_state_ids("active"))
        assert state_ids, "expected at least one active state after one step"
        return RustMemoryProxy(mgr._rust_mgr, state_ids[0], project.arch, python_mgr=mgr)

    def test_memory_proxy_unmapped_access_segfaults_under_strict(self, fauxware_project):
        """Under STRICT_PAGE_ACCESS, a proxy load *or* store to a page nothing
        has mapped raises the same ``SimSegfaultException`` angr's paged memory
        raises — page-base address, reason ``unmapped`` (angr-s0x0v).

        The read half is the regression guard: it used to return zeros, so a
        libc SimProcedure running under the callback-memory-proxy gate would
        dereference a garbage pointer, read zeros and sail on where the Python
        engine segfaults. ``unmapped_analysis`` never terminated because of it.
        """
        from angr.errors import SimSegfaultException

        state = fauxware_project.factory.entry_state(add_options={angr.options.STRICT_PAGE_ACCESS})
        proxy = self._memory_proxy_on(fauxware_project, state)

        for op in (lambda: proxy.load(0x44444444, 4), lambda: proxy.store(0x44444444, b"abcd")):
            with pytest.raises(SimSegfaultException) as exc:
                op()
            assert exc.value.addr == 0x44444000
            assert exc.value.reason == "unmapped"

    def test_memory_proxy_unmapped_load_zero_fills_without_strict(self, fauxware_project):
        """Without STRICT_PAGE_ACCESS the page maps on demand, so an unmapped
        proxy load reads back the untouched fill value rather than segfaulting.
        """
        state = fauxware_project.factory.entry_state()
        proxy = self._memory_proxy_on(fauxware_project, state)

        value = proxy.load(0x44444444, 4)
        assert value.concrete
        assert value.concrete_value == 0

    def test_solver_proxy_add_bumps_counter(self, fauxware_project):
        """``RustSolverProxyPlugin.add`` increments ``_stats_proxy_solver_adds``
        by the number of (non-tautology) constraints accepted. Tautologies
        (Python ``True``) are filtered out by ``add`` itself so they
        must not count.
        """
        import claripy

        from angr.exploration.rust_state_proxy import RustSolverProxyPlugin

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=1)
        state_ids = list(mgr._rust_mgr.get_state_ids("active"))
        assert state_ids, "expected at least one active state after one step"
        proxy = RustSolverProxyPlugin(mgr._rust_mgr, state_ids[0], python_mgr=mgr)
        sym = claripy.BVS("x", 32)
        before = mgr._stats_proxy_solver_adds
        proxy.add(sym == 1, sym != 2)  # 2 real constraints
        proxy.add(True)  # tautology — must not bump
        after = mgr._stats_proxy_solver_adds
        assert after - before == 2, f"expected 2 solver adds (tautology filtered), got delta {after - before}"
        assert mgr.stats["proxy_solver_adds"] == after


class TestCgcReceiveStdinSync:
    """Regression for angr-vx8p.3: CGC ``receive(fd=0, ...)`` symbolic bytes
    must be tracked under ``state.stdin_symbols`` so the Python-side
    ``_inject_rust_stdin`` can feed ``posix.dumps(0)``.

    Before the fix, ``cgc.rs::receive`` wrote fresh symbolic bytes into the
    buffer but never called ``record_stdin_symbol``, so ``posix.dumps(0)``
    over the exported state returned ``b''`` even though the binary read
    user-controlled stdin (see bd memory benchmark-cadet-cgc-partial-unblock).
    Exercises only the buffer-overflow phase of CADET_00001 (reaches
    ``unconstrained`` in ~0.1s); the heavy easter-egg explore is deliberately
    avoided to keep the test fast and OOM-safe.
    """

    def test_cadet_buffer_overflow_dumps_stdin(self):
        examples_dir = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
        cadet = os.path.join(examples_dir, "CADET_00001", "CADET_00001")
        if not os.path.exists(cadet):
            pytest.skip(f"CADET_00001 binary not found at {cadet}")

        proj = angr.Project(cadet, auto_load_libs=False)
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)

        # Buffer-overflow phase: step until the return address is overwritten
        # with user-controlled stdin, producing an unconstrained state.
        for _ in range(50):
            if mgr.unconstrained:
                break
            mgr.run(max_steps=1)

        assert mgr.unconstrained, f"CADET buffer-overflow never reached unconstrained; counts={mgr.stash_counts()}"

        crashing_input = bytes(mgr.unconstrained[0].posix.dumps(0))
        # The fix: stdin symbols recorded by cgc_receive are evaluated via the
        # Rust solver and injected, so the crashing input is non-empty.
        assert len(crashing_input) > 0, (
            "posix.dumps(0) returned empty bytes — CGC receive() stdin symbols were not synced into state.posix.fd[0]"
        )


class TestCgcEndToEndNativeDispatch:
    """Integration test for angr-isru: pin the
    ``docs/advanced-topics/rust_engine.rst`` claim that the CADET_00001
    buffer-overflow phase "runs end-to-end under Rust".

    Complements ``TestCgcReceiveStdinSync`` (which only asserts the crashing
    input is non-empty) by proving the seven native CGC syscall handlers
    (``native/angr/src/syscalls/cgc.rs``) actually fired through the
    ``os_name=="cgc"`` dispatch table: ``stats["syscall_native_count"] > 0``
    with zero Python syscall fallbacks. Until this test, the only coverage of
    CGC dispatch was Rust ``#[test]``s on the handlers in isolation — no
    Python-suite test loaded a DECREE binary and ran the engine on it.

    Exercises only the buffer-overflow phase (reaches ``unconstrained`` in
    ~0.1s under Rust); the heavy easter-egg explore (angr-027h) is avoided to
    keep the test fast and OOM-safe.
    """

    def test_cadet_buffer_overflow_native_cgc_dispatch(self):
        examples_dir = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
        cadet = os.path.join(examples_dir, "CADET_00001", "CADET_00001")
        if not os.path.exists(cadet):
            pytest.skip(f"CADET_00001 binary not found at {cadet}")

        proj = angr.Project(cadet, auto_load_libs=False)
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)

        for _ in range(50):
            if mgr.unconstrained:
                break
            mgr.run(max_steps=1)

        assert mgr.unconstrained, f"CADET buffer-overflow never reached unconstrained; counts={mgr.stash_counts()}"

        # Found-state semantics: the unconstrained state is the crash, and its
        # stdin (recorded by native cgc.rs::receive) is the crashing input.
        crashing_input = bytes(mgr.unconstrained[0].posix.dumps(0))
        assert len(crashing_input) > 0, "crashing input empty — native CGC receive() did not record stdin symbols"

        # The point of this test: native CGC syscall dispatch fired with no
        # Python round-trip. receive()/transmit() are the minimum the
        # buffer-overflow phase exercises.
        stats = mgr.stats
        assert stats["syscall_native_count"] > 0, (
            f"expected native CGC syscall dispatch (>0), got {stats['syscall_native_count']}; "
            f"native_by_num={stats.get('syscall_native_by_num')}"
        )
        assert stats["syscall_python_fallback_count"] == 0, (
            f"CGC syscalls fell back to Python: {stats['syscall_python_fallback_by_num']}"
        )


class TestCadetEasterEggStepLoop:
    """End-to-end coverage for angr-ckdy: the CADET solve.py phase-3 idiom
    (``while True: sm.step(); break if any active.addr == 0x804833E``) converges
    under the Rust engine.

    Two opt-in flags make this work together (neither alone suffices — see bd
    memories ``block-granular-step-mode`` and
    ``avoid-cadet-phase3-sticky-eager-retry``):

    - ``set_block_granular(True)`` (angr-bmyx) — the VEX interpreter otherwise
      chains basic blocks within one ``step(n=1)`` and runs THROUGH the egg
      block 0x804833E without ever parking on it (observability blocker #1).
    - ``set_materialize_unconstrained_forks(True)`` (angr-ckdy) — deferred-fork
      mode otherwise DROPS the loop-exit forks at the unconstrained jump
      (buffer overflow → ret goes unconstrained), collapsing the active stash
      to empty so the step-loop spins forever. Materializing them keeps the
      egg-reaching subtree alive.

    HEAVY + env-gated: the egg sits behind a symbolic strlen loop, so it takes
    several hundred block-granular steps to reach (active grows to ~70 before
    the egg appears; it does NOT explode/segfault because block-granular
    observability lets the loop break first). Skipped unless
    ``ANGR_RUN_SLOW_CADET=1`` to keep the default suite fast and OOM-safe,
    mirroring the buffer-overflow-only CADET tests above.
    """

    def test_cadet_easter_egg_step_loop_converges(self):
        if os.environ.get("ANGR_RUN_SLOW_CADET") != "1":
            pytest.skip("heavy CADET egg-hunt — set ANGR_RUN_SLOW_CADET=1 to run")

        examples_dir = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
        cadet = os.path.join(examples_dir, "CADET_00001", "CADET_00001")
        if not os.path.exists(cadet):
            pytest.skip(f"CADET_00001 binary not found at {cadet}")

        egg = 0x804833E
        proj = angr.Project(cadet, auto_load_libs=False)
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
        mgr.set_block_granular(True)
        mgr.set_materialize_unconstrained_forks(True)

        found = None
        for _ in range(1500):
            mgr.step(n=1)
            active = mgr.active
            hits = [s for s in active if s.addr == egg]
            if hits:
                found = hits[0]
                break
            # Without the materialize flag the active stash collapses to empty
            # (the bug this test guards against); with it, it never should while
            # the egg is still reachable.
            assert len(active) > 0, f"active stash collapsed before reaching egg; counts={mgr.stash_counts()}"

        assert found is not None, f"step-loop never reached egg 0x804833E; counts={mgr.stash_counts()}"
        crashing_input = bytes(found.posix.dumps(0))
        assert len(crashing_input) > 0, "egg state posix.dumps(0) returned empty bytes"


class TestDirectCallEntryMainResolution:
    """Regression for angr-027h: a "thin" entry that is a direct ``call main``
    (no ``__libc_start_main`` trampoline, e.g. DECREE/CGC binaries) has no
    ``main`` symbol and no ``rdi``/``edi`` PUT for ``_resolve_main_address`` to
    parse. Before the fix it returned ``None``, so ``_step_python_to_main``'s
    ``main_addr is None`` heuristic waited ``step > 10`` and grabbed an
    arbitrary mid-init address (inside ``__libc_csu_init``), starting the Rust
    exploration off the real CFG (the imported state began at 0x804840c, ran a
    ``pop;pop;pop;pop;ret`` gadget, and went unconstrained immediately).

    The fix: ``_resolve_main_address`` falls back to the entry block's direct
    call target when it lands on real code in the main object (a SimProcedure
    target means the ``__libc_start_main`` case, handled by the rdi parse).
    """

    def test_resolve_main_from_direct_call_entry(self):
        examples_dir = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
        cadet = os.path.join(examples_dir, "CADET_00001", "CADET_00001")
        if not os.path.exists(cadet):
            pytest.skip(f"CADET_00001 binary not found at {cadet}")

        proj = angr.Project(cadet, auto_load_libs=False)
        # The CGC binary has no 'main' symbol and its entry is `call 0x8048080`.
        assert proj.loader.find_symbol("main") is None
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
        resolved = mgr._resolve_main_address()
        assert resolved == 0x8048080, (
            f"expected main resolved to 0x8048080 from direct-call entry, "
            f"got {resolved if resolved is None else hex(resolved)}"
        )

    def test_imported_entry_state_starts_at_main_not_mid_init(self):
        examples_dir = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
        cadet = os.path.join(examples_dir, "CADET_00001", "CADET_00001")
        if not os.path.exists(cadet):
            pytest.skip(f"CADET_00001 binary not found at {cadet}")

        proj = angr.Project(cadet, auto_load_libs=False)
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
        active = mgr.active
        assert len(active) == 1
        # Post-fix the init-to-main lands at main (0x8048080). The pre-fix bug
        # imported the state at 0x804840c (mid __libc_csu_init).
        assert active[0].addr == 0x8048080, (
            f"imported entry state should start at main 0x8048080, got {hex(active[0].addr)} (mid-init regression)"
        )


class TestRustSimOptionMatrixConsistency:
    """Cross-check the SimOption coverage matrix in
    ``docs/advanced-topics/rust_engine.rst`` against the live
    ``_RAISE_OPTION_NAMES`` / ``_REJECTED_OPTION_NAMES`` frozensets.

    rust_engine.rst is the declared single source of truth for SimOption
    compatibility, but the doc table and the Python sets drift independently
    (angr-tfic implemented AVOID_MULTIVALUED but the row still said
    "implement"; angr-fkvt demoted TRACK_ACTION_HISTORY but the row still
    said "raise"). This mirrors ``TestRustInspectAllowlistConsistency``:
    parse the doc, classify every row, and assert the classifications match
    the code. angr-fgs3.
    """

    @staticmethod
    def _parse_matrix():
        """Return (raise_opts, reject_opts) parsed from the rst matrix.

        Each is a set of option names whose status cell is classified
        "(c) raise NotImplementedError" or "(b) explicitly reject".
        """
        import pathlib

        rst = pathlib.Path(__file__).resolve().parents[3] / "docs" / "advanced-topics" / "rust_engine.rst"
        lines = rst.read_text().splitlines()

        raise_opts: set[str] = set()
        reject_opts: set[str] = set()
        # Walk list-table rows: a row begins with '* - ' (option cell);
        # subsequent '- ' lines are the description / status cells.
        cur: list[str] | None = None

        def flush(row):
            if not row:
                return
            opts = re.findall(r"``([A-Z_]+)``", row[0])
            if not opts:
                return
            status = " ".join(row[2:]) if len(row) >= 3 else ""
            if "raise NotImplementedError" in status:
                raise_opts.update(opts)
            elif "explicitly reject" in status:
                reject_opts.update(opts)

        for ln in lines:
            if re.match(r"\s*\* - ", ln):
                flush(cur)
                cur = [ln.strip()[4:].strip()]
            elif re.match(r"\s+- ", ln) and cur is not None:
                cur.append(ln.strip()[2:].strip())
            elif cur is not None and cur:
                cur[-1] += " " + ln.strip()
        flush(cur)
        return raise_opts, reject_opts

    def test_doc_raise_rows_match_raise_set(self):
        """Every option the matrix classifies "(c) raise" must be in
        ``_RAISE_OPTION_NAMES`` and vice-versa — an exact match. A drift in
        either direction means a user reading the doc would predict the
        wrong behavior."""
        from angr.exploration.rust_manager import _RAISE_OPTION_NAMES

        doc_raise, _ = self._parse_matrix()
        assert doc_raise == set(_RAISE_OPTION_NAMES), (
            "SimOption matrix 'raise' rows disagree with _RAISE_OPTION_NAMES.\n"
            f"  doc-only:  {sorted(doc_raise - set(_RAISE_OPTION_NAMES))}\n"
            f"  code-only: {sorted(set(_RAISE_OPTION_NAMES) - doc_raise)}"
        )

    def test_doc_reject_rows_cover_reject_set(self):
        """Every member of ``_REJECTED_OPTION_NAMES`` must appear as a
        "(b) explicitly reject" row. The doc may legitimately mark a few
        extra options reject — those handled via warn-on-read rather than
        an add()-time reject because they ship in the default ``symbolic``
        bundle — but never the reverse."""
        from angr import sim_options as o
        from angr.exploration.rust_manager import _REJECTED_OPTION_NAMES

        _, doc_reject = self._parse_matrix()

        missing = set(_REJECTED_OPTION_NAMES) - doc_reject
        assert not missing, f"_REJECTED_OPTION_NAMES not documented as reject: {sorted(missing)}"

        # Extra doc-reject rows must be justified: they ship in the default
        # `symbolic` mode bundle, so they cannot be add()-time rejected
        # without warning every entry_state() — they warn on read instead.
        symbolic_bundle = o.modes["symbolic"]
        for name in doc_reject - set(_REJECTED_OPTION_NAMES):
            assert getattr(o, name, object()) in symbolic_bundle, (
                f"matrix marks {name!r} reject but it is neither in "
                f"_REJECTED_OPTION_NAMES nor in the default symbolic bundle — "
                f"the matrix has drifted from the code"
            )


# Dumper body for :class:`TestForeignProcessSnapshot` — runs in a subprocess so
# the snapshot it writes carries a FOREIGN symbol-id space (angr-euw28).
_FOREIGN_SNAPSHOT_DUMPER = r"""
import resource
import sys

resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))

import angr
from angr.exploration.rust_manager import RustExplorationManager

proj = angr.Project(sys.argv[1], auto_load_libs=False)
mgr = RustExplorationManager(proj, [proj.factory.entry_state()])
mgr.step(n=10)
mgr.dump_snapshot(sys.argv[2])
"""


class TestForeignProcessSnapshot:
    """angr-euw28 / angr-wuyo9: a snapshot written by a DIFFERENT process
    carries symbol ids minted by that process's allocator, which also started
    at 0 — so without the id rebase its restored leaves alias ids this process
    already handed to its own seed state, and every id-keyed lookup (claripy
    export registry, stored_conditions, symbol table) resolves a restored leaf
    to a stranger's symbol. The Rust-level guard is
    ``stash_tests.rs::test_foreign_envelope_rebases_symbol_ids``; this is the
    end-to-end Python half.

    NOTE on what is asserted. The authoritative constraint set of a restored
    state lives in Rust and is read through
    ``_rust_mgr.export_state_constraints(state_id)``. It is NOT the claripy
    list on the exported ``SimState`` mirror: mirrors materialized through the
    full-export path start from ``project.factory.blank_state()`` and never
    receive a claripy copy of the Rust constraints — by design, since
    ``_attach_rust_solver_fallback`` routes eval/min/max/satisfiable into the
    Rust solver and a claripy re-import would be expensive and identity-lossy
    (see ``pre-pinning-dangerous``). An earlier version of this test asserted
    on ``mirror.solver.constraints``, read the empty list as a snapshot bug,
    and was pulled; assert on the Rust export instead.
    """

    def test_foreign_snapshot_restores_constraints_over_a_seeded_manager(self, fauxware_project, tmp_path):
        """Load a subprocess-written snapshot into a manager that has already
        stepped (so its own allocator has handed out ids in the same low range
        the foreign envelope uses) and assert the restored states still carry
        their full, correctly-named constraint set."""
        import subprocess
        import sys

        snapshot_path = tmp_path / "foreign.snap"
        subprocess.run(
            [sys.executable, "-c", _FOREIGN_SNAPSHOT_DUMPER, fauxware_project.filename, str(snapshot_path)],
            check=True,
            timeout=300,
        )
        assert snapshot_path.stat().st_size > 0, "subprocess wrote an empty snapshot"

        mgr = RustExplorationManager(fauxware_project, [fauxware_project.factory.entry_state()])
        mgr.step(n=3)
        seed_ids = set(mgr._rust_mgr.get_state_ids("active"))
        assert seed_ids, "seed manager must have active states before the load"

        mgr.load_snapshot(str(snapshot_path))

        restored_ids = mgr._rust_mgr.get_state_ids("active")
        assert restored_ids, "load_snapshot produced no active states"

        exported = 0
        variables = set()
        for sid in restored_ids:
            constraints = mgr._rust_mgr.export_state_constraints(sid)
            exported += len(constraints)
            for c in constraints:
                variables |= set(c.variables)

        assert exported > 0, (
            f"restored states {restored_ids} exported zero constraints — the foreign "
            f"envelope's symbol ids likely aliased the seed manager's ({sorted(seed_ids)})"
        )
        # The rebase shifts ids, never names: the restored leaves must still be
        # the stdin symbols fauxware's `read()` minted in the other process.
        assert any(v.startswith("stdin_") for v in variables), (
            f"restored constraints reference no stdin symbol: {sorted(variables)}"
        )

    def test_foreign_snapshot_resumes_to_the_find_address(self, fauxware_project, tmp_path):
        """Exploration continues normally on top of a foreign envelope loaded
        over a seeded manager — the restored lineage still reaches the
        authenticated branch, and its stdin model is non-trivial."""
        import subprocess
        import sys

        snapshot_path = tmp_path / "foreign.snap"
        subprocess.run(
            [sys.executable, "-c", _FOREIGN_SNAPSHOT_DUMPER, fauxware_project.filename, str(snapshot_path)],
            check=True,
            timeout=300,
        )

        mgr = RustExplorationManager(fauxware_project, [fauxware_project.factory.entry_state()])
        mgr.step(n=3)
        mgr.load_snapshot(str(snapshot_path))

        mgr.explore(find=0x4006ED, num_find=1)
        assert len(mgr.found) > 0, "resumed foreign-snapshot manager never reached the find address"
        # Matches the same-process round-trip assertion above: 0x4006ed is
        # reachable with either the backdoor or a matching password, so the
        # contract is a non-empty model, not a specific one.
        assert len(bytes(mgr.found[0].posix.dumps(0))) > 0, "restored lineage solved to an empty stdin model"
