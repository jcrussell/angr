"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations


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
    RustSimState,
    _RustExplorationManager,
)

# All tests in this module require the Rust extension; skip the whole module
# when it is unavailable.
pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


class TestSolverProxyTimeout:
    """Tests that state.solver.timeout = N propagates to the Rust solver.

    Regression test (angr-cmfn): the assignment must not land on the proxy
    and silently disappear, leaving Rust at the default 30s timeout.
    """

    def test_timeout_setter_propagates_to_state_solver(self, amd64_mgr):
        """Assignment updates the underlying RustSimState's solver context,
        not just a proxy field, so future forks inherit the value."""
        from angr.exploration.rust_state_proxy import RustStateProxy

        sid = amd64_mgr.create_state("active")
        proxy = RustStateProxy(amd64_mgr, sid)

        proxy.solver.timeout = 1234
        # Read back via the manager (verifies state-level propagation)
        assert amd64_mgr.get_state_solver_timeout(sid) == 1234
        # Read back via the proxy (verifies getter shape)
        assert proxy.solver.timeout == 1234

    def test_timeout_setter_bounds_satisfiable_walltime(self, amd64_mgr):
        """A tight per-state timeout must bound `state.solver.satisfiable()`
        wall-clock — the path users actually hit when they write
        `state.solver.timeout = N` before evaluating.
        """
        import time

        from angr import claripy
        from angr.exploration.rust_state_proxy import RustStateProxy

        sid = amd64_mgr.create_state("active")
        proxy = RustStateProxy(amd64_mgr, sid)

        proxy.solver.timeout = 50  # ms

        x = claripy.BVS("x_factor", 128)
        y = claripy.BVS("y_factor", 128)
        n = 0xC0DEBABE_DEADBEEF_FEEDFACE_CAFEF00D
        proxy.solver.add(x * y == n)
        proxy.solver.add(x > (1 << 60))
        proxy.solver.add(y > (1 << 60))
        proxy.solver.add(x < (1 << 100))
        proxy.solver.add(y < (1 << 100))

        start = time.monotonic()
        _ = proxy.solver.satisfiable()
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, (
            f"satisfiable() took {elapsed:.2f}s with a 50ms per-state timeout "
            f"— state.solver.timeout did not flow to the Rust solver."
        )


class TestProxyLiskovGaps:
    """Tests for previously-unimplemented load shapes on RustStateProxy
    (angr-1w75): register load by integer offset and memory load with a
    symbolic address. Both used to raise NotImplementedError despite being
    standard angr SimState idioms.
    """

    def test_register_load_by_offset_amd64(self, fauxware_project):
        """``regs.load(offset)`` resolves through ``arch.register_size_names``
        and returns the same value as the named accessor."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        # rax sits at offset 16 on AMD64.
        offset, size = state.arch.registers["rax"]
        assert (offset, size) == (16, 8)

        v_off = proxy.regs.load(offset)  # default size -> arch.bytes (8)
        v_name = proxy.regs.rax
        assert v_off.size() == v_name.size() == 64
        # Cached BVV objects are reference-equal; otherwise compare concrete
        # values. The proxy may return either depending on whether __getattr__
        # cached the read.
        assert v_off.concrete == v_name.concrete
        if v_off.concrete:
            assert v_off.concrete_value == v_name.concrete_value

    def test_register_load_subreg_via_size(self, fauxware_project):
        """``regs.load(offset, size=4)`` resolves to ``eax`` for AMD64 and
        returns a 32-bit value (the low half of rax)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        v_eax = proxy.regs.load(16, 4)
        assert v_eax.size() == 32
        # The proxy returns whatever Rust holds for "eax" — verify it matches
        # the named accessor (the resolution path is what we're testing).
        assert v_eax.concrete == proxy.regs.eax.concrete
        if v_eax.concrete:
            assert v_eax.concrete_value == proxy.regs.eax.concrete_value

    def test_register_load_bad_offset(self, fauxware_project):
        """Unknown (offset, size) pairs raise NotImplementedError with a
        message that identifies the arch — easier to diagnose than a bare
        KeyError."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        with pytest.raises(NotImplementedError, match="AMD64"):
            proxy.regs.load(7777, 8)

    def test_solver_proxy_min_max_width_gt128_falls_back(self, fauxware_project):
        """angr-fjhk9: ``RustSolverProxy.min``/``max`` (the ``RustStateProxy``
        solver, distinct from the callback ``RustSolverProxyPlugin``) must NOT
        mislabel a satisfiable >128-bit extremum as unsat. A wide bound the
        Rust ``u128`` solver can't hold falls back to claripy's big-int solver
        over the exported constraints (shared ``_resolve_none_extremum`` helper,
        same disambiguation as the ``RustSolverFallback`` twin, angr-8iv6j)."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        wide = claripy.BVS("stateproxy_wide_gt128", 160)
        proxy.solver._ensure_solver()
        assert proxy.solver._solver_ctx.min(wide) is None
        assert proxy.solver.min(wide) == 0
        assert proxy.solver.max(wide) == (1 << 160) - 1

    def test_solver_proxy_min_max_width_gt128_honors_added_constraint(self, fauxware_project):
        """angr-hv4lt.5: a constraint added via ``proxy.solver.add()`` on the
        standalone ``RustSolverProxy`` must NOT be dropped by the width>128
        big-int fallback. The standalone proxy's ``add`` only mutates the
        forked context (not the state), so ``_resolve_none_extremum`` — which
        used to re-derive constraints from ``export_state_constraints`` (state
        only) — silently omitted it. The fallback now unions the proxy's tracked
        adds, matching what the primary Rust min/max path on the forked context
        saw."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        wide = claripy.BVS("stateproxy_wide_constrained", 160)
        lower = 1 << 130  # exceeds u128, so Rust min still returns None
        upper = 1 << 155
        proxy.solver.add(wide.UGE(lower))
        proxy.solver.add(wide.ULE(upper))
        proxy.solver._ensure_solver()

        # The forked Rust context sees the constraints, but the extremum
        # exceeds u128 so the primary path returns None -> big-int fallback.
        assert proxy.solver._solver_ctx.min(wide) is None
        assert proxy.solver._solver_ctx.max(wide) is None

        # Fallback must honor the in-proxy adds (pre-fix: min=0, max=2**160-1).
        assert proxy.solver.min(wide) == lower
        assert proxy.solver.max(wide) == upper

    def test_solver_proxy_min_unsat_still_raises(self, fauxware_project):
        """angr-fjhk9: a genuinely unsat ``RustSolverProxy`` ctx still raises
        ``claripy.errors.UnsatError`` (proxy-wide exception-type convention)."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        u = claripy.BVS("stateproxy_unsat", 32)
        proxy.solver.add(u == 1)
        proxy.solver.add(u == 2)  # contradictory -> unsat
        with pytest.raises(claripy.errors.UnsatError):
            proxy.solver.min(u)
        with pytest.raises(claripy.errors.UnsatError):
            proxy.solver.max(u)

    def test_register_load_wrong_type(self, fauxware_project):
        """A float/None arg is a programming error, not an offset miss."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        with pytest.raises(TypeError):
            proxy.regs.load(1.5)

    def test_memory_load_symbolic_addr(self, fauxware_project):
        """``memory.load(sym_addr)`` evaluates the address under the state's
        constraints and falls through to the concrete read path."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        # Constrain a symbolic addr to a concrete location inside the loaded
        # binary so the concrete read returns deterministic bytes.
        addr_sym = claripy.BVS("addr_load_sym", 64)
        target = fauxware_project.entry
        state.solver.add(addr_sym == target)
        mgr = RustExplorationManager(fauxware_project, [state])

        proxy = mgr.proxy.active[0]
        loaded = proxy.memory.load(addr_sym, 4)

        # Reference: angr SimState memory.load at the concrete addr (big-endian
        # default — same convention as RustMemoryProxy).
        expected = state.solver.eval(state.memory.load(target, 4))
        assert loaded.concrete_value == expected

    def test_memory_load_unsat_symbolic_addr(self, fauxware_project):
        """An unsatisfiable symbolic addr raises UnsatError rather than
        silently reading zero bytes."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        addr_sym = claripy.BVS("addr_unsat", 64)
        state.solver.add(addr_sym == 0x1000)
        state.solver.add(addr_sym == 0x2000)  # contradictory
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        with pytest.raises(claripy.errors.UnsatError):
            proxy.memory.load(addr_sym, 4)

    def test_memory_store_symbolic_addr_single_solution_routes_multi(self, fauxware_project):
        """angr-4scu step 2: a symbolic-address write that resolves to a
        single concrete solution under the state's constraints lands in
        Rust memory and is visible through the proxy's concrete-addr load
        path. The Multi-cell entry point short-circuits single solutions
        to the eager concrete store — verify the round-trip.
        """
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        target = fauxware_project.entry
        sym_addr = claripy.BVS("write_addr_single", 64)
        state.solver.add(sym_addr == target)
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        payload = b"\x42\x43\x44\x45"
        proxy.memory.store(sym_addr, payload)

        # Round-trip through the concrete-addr load path; default endness
        # is Iend_BE for both store() and load() so the bytes survive
        # without a byte-swap.
        loaded = proxy.memory.load(target, len(payload))
        assert loaded.concrete_value == int.from_bytes(payload, "big")

    def test_memory_store_symbolic_addr_with_int_value(self, fauxware_project):
        """The symbolic-address store path coerces ``int`` values into a
        claripy BVV using the same endness logic as the concrete-addr path
        before handing them to the Multi-cell entry point.
        """
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        target = fauxware_project.entry
        sym_addr = claripy.BVS("write_addr_int", 64)
        state.solver.add(sym_addr == target)
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        proxy.memory.store(sym_addr, 0xDEADBEEF, size=4, endness="Iend_LE")

        loaded = proxy.memory.load(target, 4, endness="Iend_LE")
        assert loaded.concrete_value == 0xDEADBEEF

    def test_memory_store_symbolic_addr_with_symbolic_value(self, fauxware_project):
        """A symbolic-AST value at a single-solution symbolic address
        registers the symbol in the shared cache (the Multi-cell path runs
        ``claripy_to_rustbv`` on the data AST), so a subsequent solver
        eval through the proxy returns the expected concrete bits.

        The store uses angr's default ``Iend_BE`` endness, so a default
        (BE) load round-trips the value and an ``Iend_LE`` load sees it
        byte-swapped — stock ``SimMemory`` semantics (angr-c8mda).
        """
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        target = fauxware_project.entry
        sym_addr = claripy.BVS("write_addr_symdata", 64)
        sym_val = claripy.BVS("write_val", 32)
        state.solver.add(sym_addr == target)
        state.solver.add(sym_val == 0xCAFEBABE)
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        proxy.memory.store(sym_addr, sym_val)

        # The proxy returns the symbolic AST (angr-8dop.1), so we eval
        # through the proxy solver to observe the constrained value (a
        # constrained BVS is not the same object as a BVV with that value).
        assert proxy.solver.eval(proxy.memory.load(target, 4)) == 0xCAFEBABE
        assert proxy.solver.eval(proxy.memory.load(target, 4, endness="Iend_LE")) == 0xBEBAFECA

    @pytest.mark.parametrize("endness", [None, "Iend_BE", "Iend_LE"])
    def test_memory_store_symbolic_value_round_trips_with_endness(self, fauxware_project, endness):
        """angr-c8mda: a symbolic AST stored at a concrete address must read
        back unchanged with the same endness, and byte-reversed with the
        opposite one. The proxy used to forward the AST verbatim (an LE
        layout) while ``load`` reverses a BE read, so a default
        ``load``/``store`` copy landed byte-reversed."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        target = fauxware_project.entry
        val = claripy.BVS("rt_val", 32, explicit_name=True)
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        kw = {} if endness is None else {"endness": endness}
        proxy.memory.store(target, val, **kw)
        same = proxy.memory.load(target, 4, **kw)
        other = proxy.memory.load(target, 4, endness="Iend_BE" if endness == "Iend_LE" else "Iend_LE")
        assert not claripy.Solver().satisfiable(extra_constraints=[same != val])
        assert not claripy.Solver().satisfiable(extra_constraints=[other != claripy.Reverse(val)])

    def test_memory_copy_default_endness_preserves_bytes(self, fauxware_project):
        """angr-c8mda: ``store(dst, load(src, n))`` with default endness is a
        byte copy (sokohashv2's ``do_repmovsd`` hook). The copied region must
        hold the source bytes in order, not reversed."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        src = fauxware_project.entry
        dst = src + 0x40
        words = [claripy.BVS(f"copy_w{i}", 32, explicit_name=True) for i in range(4)]
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]
        for i, w in enumerate(words):
            proxy.memory.store(src + 4 * i, w, endness="Iend_LE")

        proxy.memory.store(dst, proxy.memory.load(src, 16))

        for i, w in enumerate(words):
            got = proxy.memory.load(dst + 4 * i, 4, endness="Iend_LE")
            assert not claripy.Solver().satisfiable(extra_constraints=[got != w]), f"word {i} not copied in order"

    def test_memory_store_unbounded_symbolic_addr_does_not_raise(self, fauxware_project):
        """angr-p1s02: a write through an unbounded/unconstrained symbolic
        address (e.g. an unconstrained ``operator new`` return pointer) has
        too many satisfying solutions for the lazy Multi-cell path. It used to
        raise ``NotImplementedError``, killing the SimProcedure callback state
        and dropping the corpus find under the callback-memory-proxy gate
        (angr-ijwp0). It must now fall
        back to angr's Max write-concretization (single concrete address,
        garbage high pages silently dropped) so the state survives.
        """
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        # High fixed prefix + unconstrained low bits — mirrors an unconstrained
        # ``operator new`` pointer (0x7fff.. garbage). Its Max concretization
        # lands on an unmapped high page, exercising the silent-drop fallback.
        unbounded = claripy.BVV(0x7FFF00000000, 64) + (claripy.BVS("unbounded_write_lo", 64) & 0xFFFF)
        before = mgr._stats_proxy_mem_symbolic_addr_fallback

        # Must not raise.
        proxy.memory.store(unbounded, claripy.BVV(0, 32))

        assert mgr._stats_proxy_mem_symbolic_addr_fallback == before + 1


class TestProxyMemoryFind:
    """angr-4scu step 1: RustMemoryProxy.find() supports the concrete-needle
    search surface used by libc SimProcs (memchr / strstr against literals).
    Symbolic needles / conditions / wide-char search raise NotImplementedError
    and route callers to the SimProcedure-hook path.
    """

    def test_find_concrete_byte_in_concrete_buffer(self, fauxware_project):
        """Write a concrete buffer through the proxy, then find a byte in it."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        addr = fauxware_project.entry
        proxy.memory.store(addr, b"hello\x00world\x00")

        a, constraints, indices = proxy.memory.find(addr, b"world", 16)
        assert a.concrete_value == addr + 6
        assert indices == [6]
        assert constraints == []

    def test_find_no_match_returns_default(self, fauxware_project):
        """When the needle is absent the default address (BVV) is returned
        and indices is empty."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        addr = fauxware_project.entry
        proxy.memory.store(addr, b"abcdefghij")

        a, constraints, indices = proxy.memory.find(addr, b"ZZ", 10, default=0)
        assert a.concrete_value == 0
        assert indices == []
        assert constraints == []

    def test_find_int_needle_treated_as_single_byte(self, fauxware_project):
        """memchr-style: passing an ``int`` needle treats it as a single byte
        (low 8 bits), matching what SimProcs pass after ``c[7:0]``."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        addr = fauxware_project.entry
        proxy.memory.store(addr, b"AAAAXBBBB")

        a, _, indices = proxy.memory.find(addr, ord("X"), 16, default=0)
        assert a.concrete_value == addr + 4
        assert indices == [4]

    def test_find_symbolic_needle_raises(self, fauxware_project):
        """Symbolic needles are out of scope for this proxy step and must
        raise NotImplementedError with a route to the SimProcedure hook."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        sym = claripy.BVS("needle_sym", 8)
        with pytest.raises(NotImplementedError, match="symbolic needles"):
            proxy.memory.find(fauxware_project.entry, sym, 16)

    def test_find_wide_char_raises(self, fauxware_project):
        """``char_size > 1`` (wide-char) belongs in SimMemory.find(); the
        proxy refuses with a clear pointer."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        with pytest.raises(NotImplementedError, match="char_size"):
            proxy.memory.find(fauxware_project.entry, b"x", 16, char_size=2)


class TestCallbackMemoryProxyInstall:
    """angr-4scu step 3: ``RustMemoryProxy`` is installed as ``state.memory``
    on every SimProcedure callback state, so loads/stores route directly to
    Rust.
    """

    def test_proxy_install_swaps_state_memory(self, fauxware_project):
        """``_install_callback_memory_proxy`` replaces
        ``state.memory`` with a ``RustMemoryProxy``. Drives the helper
        directly to keep the test scope at the swap mechanism (the SimProc
        callback wiring is exercised via separate fauxware tests)."""
        from angr.exploration.rust_state_proxy import RustMemoryProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        # Build a throwaway SimState matching what
        # ``_create_state_for_callback`` would hand to the helper.
        # Bind the proxy to the live seed state id so concrete-store / load
        # FFI lands in an existing Rust state.
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_memory_proxy(cb_state, seed_id)
        assert isinstance(cb_state.memory, RustMemoryProxy)
        assert cb_state.memory.id == "mem"
        assert cb_state.memory.category == "mem"

    def test_proxy_install_roundtrip(self, fauxware_project):
        """A ``state.memory.store`` followed by ``state.memory.load`` on the
        installed proxy lands in Rust and reads back via the same path.
        Verifies the plugin shim doesn't break the existing concrete-store
        / concrete-load fast path the proxy already supported."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_memory_proxy(cb_state, seed_id)

        addr = fauxware_project.entry
        cb_state.memory.store(addr, b"\xaa\xbb\xcc\xdd")
        loaded = cb_state.memory.load(addr, 4)
        # Default endness Iend_BE — bytes survive without a byte-swap.
        assert loaded.concrete_value == 0xAABBCCDD

    def test_proxy_install_copy_returns_proxy(self, fauxware_project):
        """The plugin shim's ``copy()`` returns a fresh ``RustMemoryProxy``
        bound to the same Rust state — preserves ``SimState.copy()``
        semantics on a callback frame that has the proxy installed."""
        from angr.exploration.rust_state_proxy import RustMemoryProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_memory_proxy(cb_state, seed_id)
        clone = cb_state.memory.copy()
        assert isinstance(clone, RustMemoryProxy)
        assert clone.id == "mem"
        assert clone._state_id == cb_state.memory._state_id
        # Fresh proxy returned from ``copy()`` is unbound (per plugin
        # protocol — ``set_state`` is invoked by the new state's
        # ``register_plugin``).
        assert clone.state is None

    def test_proxy_load_size_zero_returns_empty_bv(self, fauxware_project):
        """``proxy.memory.load(addr, 0)`` mirrors stock SimMemory and returns
        a 0-width BV without invoking the Rust FFI (a zero-width load would
        panic inside Z3). Regression for the
        ``posix/open.py`` crash discovered in step 4 parity validation when
        ``strlen.max_null_index == 0`` (null at offset 0).
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_memory_proxy(cb_state, seed_id)
        result = cb_state.memory.load(fauxware_project.entry, 0)
        assert result.length == 0

    def test_proxy_load_symbolic_returns_ast(self, fauxware_project):
        """angr-8dop.1: a load of memory that holds a symbolic BVS must
        return the symbolic AST, not a solver witness. Stock SimMemory
        returns the BVS verbatim; the proxy used to concretize the value
        via ``get_state_memory``, which caused
        ``strlen``/``strcmp``/``strchr``/``memchr`` to build comparison
        chains against a single witness instead of the actual symbolic
        bytes.

        We write through the proxy itself (which routes
        ``set_state_memory_ast`` into the Rust state) to keep the test
        independent of the Python→Rust symbolic-page import path.
        """
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_memory_proxy(cb_state, seed_id)

        addr = fauxware_project.entry
        sym = claripy.BVS("proxy_sym_byte", 8)
        cb_state.memory.store(addr, sym)

        loaded = cb_state.memory.load(addr, 1)
        assert loaded.symbolic, (
            "proxy.load must return the symbolic AST, not a concretized "
            "witness — strlen/strchr/memchr depend on this for symbolic "
            "byte-by-byte ITE construction"
        )

    def test_proxy_find_symbolic_haystack_multi_match(self, fauxware_project):
        """angr-8dop.1: when the haystack contains symbolic bytes, find()
        must emit one ITE case per SAT-able candidate index. Stock
        SmartFindMixin behaviour — strlen relies on the ``match_indices``
        list reflecting every possible null position, not just the first
        concrete one (which under concretization could be offset 0 even
        when the symbolic byte is constrained non-zero).

        Symbolic bytes are written via the proxy itself (write-through)
        to bypass the Python→Rust symbolic-page import path, keeping
        the test focused on the find() symbolic-haystack handling.
        """
        from angr import claripy

        state = fauxware_project.factory.entry_state(
            add_options={
                angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            },
        )
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_memory_proxy(cb_state, seed_id)

        addr = fauxware_project.entry
        # 3-byte symbolic store: two unconstrained bytes followed by a
        # concrete null at offset 2. A single AST store keeps the
        # underlying RustBV as one symbolic object (avoids the per-byte
        # symbolic-merge path which is a separate limitation). The find
        # iteration must emit cases for indices 0, 1, AND 2 — the
        # concrete null is_true case is popped into the default per
        # SmartFindMixin convention.
        sym_b0 = claripy.BVS("hb0", 8)
        sym_b1 = claripy.BVS("hb1", 8)
        # Build a 24-bit AST: byte 0 of memory at LSB (Iend_LE layout),
        # so concat MSB-first: [null, sym_b1, sym_b0] gives memory bytes
        # 0 = sym_b0, 1 = sym_b1, 2 = null at addr+0, +1, +2.
        triple = claripy.Concat(claripy.BVV(0, 8), sym_b1, sym_b0)
        cb_state.memory.store(addr, triple, endness="Iend_LE")

        _, _, indices = cb_state.memory.find(addr, claripy.BVV(0, 8), 3)
        # The concrete null at offset 2 is is_true → break and pop into
        # default. Indices 0 and 1 contribute symbolic == 0 cases.
        assert indices == [0, 1, 2]


class TestCallbackMemoryProxyReentryGuards:
    """angr-hcok: synchronous Rust→Python callbacks (`_cb_memory_load`,
    `_cb_memory_store`, `_cb_fetch_page`, batch variants, symbolic-full
    variants) fire from *inside* ``_rust_mgr.run()`` which holds
    ``&mut self`` on the manager PyCell. If the per-fork state's
    ``memory`` plugin is a ``RustMemoryProxy`` (proxy gate post-callback
    cache pollution), naive ``state.memory.load/store`` would re-enter
    the manager via FFI and raise "Already mutably borrowed". The
    callbacks must detect this and short-circuit.
    """

    def _mgr_with_proxy_in_cache(self, fauxware_project):
        """Construct a manager and seed the per-fork state cache with a
        SimState whose ``memory`` plugin is a ``RustMemoryProxy``. Mimics
        the post-callback cache state that triggers the re-entry.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_memory_proxy(cb_state, seed_id)
        mgr._state_cache[seed_id] = cb_state
        # Steer ``_get_per_fork_state`` to return ``cb_state``.
        mgr._get_stepping_state_id = lambda sid=seed_id: sid
        return mgr, cb_state, seed_id

    def test_cb_memory_load_bypasses_proxy(self, fauxware_project):
        """``_cb_memory_load`` must not call ``state.memory.load`` when the
        plugin is the proxy. Returns a fresh BVS so Rust can store and
        avoid re-entry. The handle must be registered for round-trip.
        """
        mgr, _cb_state, _seed_id = self._mgr_with_proxy_in_cache(fauxware_project)
        concrete, is_sym, ast = mgr._cb_memory_load(0x28, 8)
        assert ast is not None
        assert ast.symbolic
        assert ast.length == 64
        assert is_sym is True
        # Filler — concrete bytes are zeros; downstream evaluator returns
        # whatever the solver picks under no constraint.
        assert concrete == bytes(8)

    def test_cb_memory_store_bypasses_proxy(self, fauxware_project):
        """``_cb_memory_store`` is a no-op under the gate — Rust already
        performed the store before invoking the callback, so the proxy's
        ``store`` is never re-entered."""
        mgr, cb_state, _seed_id = self._mgr_with_proxy_in_cache(fauxware_project)
        calls = []
        cb_state.memory.store = lambda *a, **k: calls.append((a, k))
        mgr._cb_memory_store(0x7FFFFFFEFFF0, b"\x01\x02\x03\x04")
        # Bypass returns early — the proxy memory is left untouched.
        assert calls == []

    def test_cb_memory_load_batch_bypasses_proxy(self, fauxware_project):
        """Batch variant: one filler BVS per load entry."""
        mgr, _cb_state, _seed_id = self._mgr_with_proxy_in_cache(fauxware_project)
        results = mgr._cb_memory_load_batch([(0x28, 8), (0x100, 4)])
        assert len(results) == 2
        for _concrete, is_sym, ast in results:
            assert is_sym is True
            assert ast is not None and ast.symbolic

    def test_cb_memory_store_batch_bypasses_proxy(self, fauxware_project):
        """Batch store variant: no-op — proxy ``store`` never re-entered."""
        mgr, cb_state, _seed_id = self._mgr_with_proxy_in_cache(fauxware_project)
        calls = []
        cb_state.memory.store = lambda *a, **k: calls.append((a, k))
        mgr._cb_memory_store_batch([(0x100, b"\x01\x02"), (0x200, b"\x03\x04")])
        assert calls == []

    def test_cb_fetch_page_bypasses_proxy(self, fauxware_project):
        """``_cb_fetch_page`` returns empty/inaccessible under the gate so
        Rust falls back to its own page source."""
        mgr, _cb_state, _seed_id = self._mgr_with_proxy_in_cache(fauxware_project)
        page, perms, ok = mgr._cb_fetch_page(0x400000)
        assert page == bytes(4096)
        assert perms == 0
        assert ok is False

    def test_cb_batch_fetch_pages_bypasses_proxy(self, fauxware_project):
        """Batch fetch_pages: per-page (empty, perms=0, mapped=False)."""
        mgr, _cb_state, _seed_id = self._mgr_with_proxy_in_cache(fauxware_project)
        results = mgr._cb_batch_fetch_pages([0x400000, 0x401000])
        assert len(results) == 2
        for page, perms, mapped in results:
            assert page == bytes(4096)
            assert perms == 0
            assert mapped is False


class TestBatchFetchPagesFallbackDeclines:
    """angr-hv4lt.2: ``_cb_batch_fetch_pages`` must return is_mapped=False in
    its None-state and per-page-error fallback arms (matching the single-page
    ``_cb_fetch_page`` siblings), so a transient failure is retryable rather
    than permanently baked into Rust memory as an all-zero page.
    """

    def test_none_state_declines(self, fauxware_project):
        """When no callback/default state is available, decline every page
        (is_mapped=False) instead of mapping a permanent all-zero page."""
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr._get_callback_state = lambda: None
        mgr._get_default_state = lambda: None
        results = mgr._cb_batch_fetch_pages([0x400000, 0x401000])
        assert len(results) == 2
        for page, perms, mapped in results:
            assert page == bytes(4096)
            assert perms == 0
            assert mapped is False

    def test_load_error_declines(self, fauxware_project):
        """A per-page SimError during load must decline that page
        (is_mapped=False), keeping the retry path alive."""
        from angr.errors import SimError

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        def _boom(*_a, **_k):
            raise SimError("transient")

        state.memory.load = _boom
        mgr._get_callback_state = lambda: state
        mgr._get_default_state = lambda: state
        # Avoid the UltraPage fast path so the load error arm is exercised.
        mgr._fetch_page_from_ultrapage = lambda *_a, **_k: None
        results = mgr._cb_batch_fetch_pages([0x400000])
        assert results == [(bytes(4096), 0, False)]


class TestRustMemoryProxyPluginGapStubs:
    """angr-8dop.2: ``RustMemoryProxy`` exposes minimal stubs for the
    SimMemory plugin methods ``permissions`` / ``merge`` / ``widen`` /
    ``compare`` so callers like ``mprotect`` / ``is_bad_ptr`` /
    ``VirtualProtect`` and the cross-state plugin walk don't AttributeError
    when the callback-install gate is on.
    """

    def _install_proxy(self, fauxware_project):

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_memory_proxy(cb_state, seed_id)
        return cb_state

    def test_permissions_read_returns_default_rwx(self, fauxware_project):
        """``permissions(addr)`` (no second arg) returns a 3-bit BVV
        encoding RWX. Lets ``is_bad_ptr`` / ``mprotect`` proceed without
        crashing under the proxy."""
        from angr import claripy

        cb_state = self._install_proxy(fauxware_project)
        result = cb_state.memory.permissions(fauxware_project.entry)
        assert isinstance(result, claripy.ast.Base)
        assert result.length == 3
        assert result.concrete_value == 7

    def test_permissions_write_is_noop(self, fauxware_project):
        """``permissions(addr, perms)`` accepts and returns the prior
        permissive default. Permission bits are owned by Rust's memory
        model; the proxy doesn't track them on the Python side."""
        cb_state = self._install_proxy(fauxware_project)
        result = cb_state.memory.permissions(fauxware_project.entry, 5)
        assert result.concrete_value == 7

    def test_map_region_inside_rust_heap_is_accepted(self, fauxware_project):
        """A bounced heap growth (``SimHeapBase._ensure_mapped``) maps inside
        the Rust lazy heap region, which Rust already covers (angr-vmi2l.314)."""
        from angr import claripy
        from angr.rustylib.vex_engine import RustSimState

        start, size = RustSimState.HEAP_REGION
        cb_state = self._install_proxy(fauxware_project)
        cb_state.memory.map_region(start, 0x1000, 3)
        cb_state.memory.map_region(claripy.BVV(start + size - 0x1000, 64), claripy.BVV(0x1000, 64), 3)

    def test_map_region_outside_rust_heap_raises(self, fauxware_project):
        """Outside the heap region the mapping (e.g. a bounced anonymous
        ``mmap``'s zero fill) would be dropped, so the stub fails loud rather
        than letting later reads see the wrong bytes."""
        from angr import claripy
        from angr.rustylib.vex_engine import RustSimState

        start, size = RustSimState.HEAP_REGION
        cb_state = self._install_proxy(fauxware_project)
        with pytest.raises(NotImplementedError, match="outside the Rust heap region"):
            cb_state.memory.map_region(0x7F00_0000_0000, 0x1000, 3)
        with pytest.raises(NotImplementedError, match="outside the Rust heap region"):
            cb_state.memory.map_region(start + size - 0x1000, 0x2000, 3)
        with pytest.raises(NotImplementedError, match="symbolic range"):
            cb_state.memory.map_region(claripy.BVS("a", 64), 0x1000, 3)

    def test_merge_returns_false(self, fauxware_project):
        """``merge`` reports 'no merge happened' (False) — matches the
        register / solver proxy stubs."""
        cb_state = self._install_proxy(fauxware_project)
        assert cb_state.memory.merge([], [], None) is False

    def test_widen_returns_false(self, fauxware_project):
        cb_state = self._install_proxy(fauxware_project)
        assert cb_state.memory.widen([]) is False

    def test_compare_returns_true(self, fauxware_project):
        """``compare`` is the 'memories equal' predicate; with no
        Python-side state to diff the stub conservatively answers True."""
        cb_state = self._install_proxy(fauxware_project)
        assert cb_state.memory.compare(cb_state.memory) is True


class TestCallbackSolverProxyGate:
    """angr-8oiw: ``RustSolverProxyPlugin`` is installed as ``state.solver``
    on every SimProcedure callback state; ``state.solver.add`` routes
    directly to Rust by ``state_id`` and ``state.solver.constraints`` reads
    through Rust.
    """

    def test_proxy_install_swaps_state_solver(self, fauxware_project):
        """``_install_callback_solver_proxy`` replaces
        ``state.solver`` with a ``RustSolverProxyPlugin``."""
        from angr.exploration.rust_state_proxy import RustSolverProxyPlugin

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_solver_proxy(cb_state, seed_id)
        assert isinstance(cb_state.solver, RustSolverProxyPlugin)
        assert cb_state.solver.id == "solver"
        assert cb_state.solver.category == "solver"
        assert cb_state.solver.state is cb_state

    def test_proxy_install_copy_returns_proxy(self, fauxware_project):
        """``copy()`` returns a fresh ``RustSolverProxyPlugin`` bound to the
        same Rust state."""
        from angr.exploration.rust_state_proxy import RustSolverProxyPlugin

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_solver_proxy(cb_state, seed_id)
        clone = cb_state.solver.copy()
        assert isinstance(clone, RustSolverProxyPlugin)
        assert clone._state_id == cb_state.solver._state_id
        assert clone.state is None

    def test_proxy_add_write_through_to_rust(self, fauxware_project):
        """``state.solver.add(c)`` routes the constraint into the underlying
        Rust state's solver. ``state.solver.constraints`` reads it back."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_solver_proxy(cb_state, seed_id)

        before = len(cb_state.solver.constraints)
        sym = claripy.BVS("solver_proxy_test_byte", 8)
        cb_state.solver.add(sym == 0x42)
        after = len(cb_state.solver.constraints)
        assert after > before, (
            "RustSolverProxyPlugin.add must increase the underlying Rust "
            "state's constraint count (write-through, not fork-only)"
        )

    def test_proxy_satisfiable_and_eval(self, fauxware_project):
        """``state.solver.eval`` / ``satisfiable`` route through Rust."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_solver_proxy(cb_state, seed_id)

        # Concrete fast path
        assert cb_state.solver.eval(claripy.BVV(0xAA, 8)) == 0xAA
        # Symbolic with extra_constraints
        sym = claripy.BVS("solver_proxy_eval_byte", 8)
        cb_state.solver.add(sym == 0x33)
        assert cb_state.solver.satisfiable() is True
        assert cb_state.solver.eval(sym) == 0x33

    def test_proxy_min_max_width_gt128_falls_back(self, fauxware_project):
        """angr-fjhk9: ``RustSolverProxyPlugin.min``/``max`` must NOT mislabel a
        satisfiable >128-bit extremum as unsat.

        ``RustSolverContext.min``/``max`` return ``None`` for BOTH unsat and
        ``width > 128`` (solving_ops.rs). Pre-fix the plugin raised
        ``UnsatError`` on any ``None``, killing a valid state. Post-fix a
        satisfiable wide bound falls back to claripy's big-int solver over the
        exported constraints, the same disambiguation as the ``RustSolverFallback``
        twin (angr-8iv6j), which keeps the four solver-shim sites in sync.
        """
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_solver_proxy(cb_state, seed_id)

        wide = claripy.BVS("plugin_wide_gt128", 160)
        # Trigger is genuine: the forked Rust ctx can't bound a >128-bit symbol.
        assert cb_state.solver._get_rust_ctx().min(wide) is None
        # Post-fix: claripy big-int fallback yields the true bounds, not UnsatError.
        assert cb_state.solver.min(wide) == 0
        assert cb_state.solver.max(wide) == (1 << 160) - 1

    def test_proxy_min_unsat_still_raises(self, fauxware_project):
        """angr-fjhk9: a genuinely unsat plugin ctx still raises. The plugin
        stands in for ``state.solver``, so it raises ``SimUnsatError`` as
        ``SimSolver`` does (angr-mx00p.61)."""
        from angr import claripy
        from angr.errors import SimUnsatError

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_solver_proxy(cb_state, seed_id)

        u = claripy.BVS("plugin_unsat", 32)
        cb_state.solver.add(u == 1)
        cb_state.solver.add(u == 2)  # contradictory -> unsat
        with pytest.raises(SimUnsatError):
            cb_state.solver.min(u)
        with pytest.raises(SimUnsatError):
            cb_state.solver.max(u)
        with pytest.raises(SimUnsatError):
            cb_state.solver.eval(u)

    def test_proxy_unsat_extra_constraints_raise(self, fauxware_project):
        """Infeasible ``extra_constraints`` raise ``SimUnsatError`` from
        ``eval`` and ``eval_upto`` rather than an empty answer, and stay
        scoped to the query (angr-vmi2l.293)."""
        from angr import claripy
        from angr.errors import SimUnsatError

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_solver_proxy(cb_state, seed_id)

        x = claripy.BVS("plugin_extra_x", 32)
        cb_state.solver.add(x.ULT(5))
        with pytest.raises(SimUnsatError):
            cb_state.solver.eval(x, extra_constraints=[x == 7])
        with pytest.raises(SimUnsatError):
            cb_state.solver.eval_upto(x, 3, extra_constraints=[x == 7])
        assert cb_state.solver.satisfiable(extra_constraints=[x == 7]) is False
        assert sorted(cb_state.solver.eval_upto(x, 10)) == [0, 1, 2, 3, 4]

    def test_proxy_eval_accepts_python_values(self, fauxware_project):
        """``SimSolver.eval`` returns a plain Python value unchanged; heap
        allocators eval an int size. The plugin answered unsat for it, so a
        bounced calloc/malloc errored the state (angr-mx00p.61)."""
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_solver_proxy(cb_state, seed_id)

        assert cb_state.solver.eval(0x30) == 0x30
        assert cb_state.solver.eval_upto(0x30, 2) == [0x30]
        assert cb_state.heap.allocate(0x30) >= 0

    def test_proxy_bvs_delegates_to_claripy(self, fauxware_project):
        """``BVS`` / ``BVV`` delegate to claripy and accept the tracking key."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_solver_proxy(cb_state, seed_id)

        sym = cb_state.solver.BVS("proxy_plugin_bvs", 32, key=("test", "k1"), eternal=True)
        assert isinstance(sym, claripy.ast.Base)
        assert sym.length == 32
        assert dict(cb_state.solver.get_variables("test"))[("test", "k1")] is sym


class TestCallbackProxyRebind:
    """A materialized state copied from a cached callback frame must not keep
    that frame's proxy binding: once Rust retires the frame's id (the parallel
    scheduler does), reads raise ``state N not found`` (angr-mx00p.86)."""

    def test_bind_export_points_every_proxy_at_one_binding(self, fauxware_project):
        from angr.exploration.rust_state_proxy import RustRegisterProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        target = fauxware_project.factory.entry_state()
        regs = RustRegisterProxy(mgr._rust_mgr, 10**9, target.arch, python_mgr=mgr)
        regs._cache["rax"] = 1
        target.register_plugin("registers", regs)
        mgr._bind_export(target, sid)
        bindings = {id(target.plugins[n]._binding) for n in ("memory", "registers", "solver", "callstack")}
        assert len(bindings) == 1
        assert regs._state_id == sid and not regs._cache

    def test_copy_sees_the_target_and_restores_a_shared_source(self, fauxware_project):
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        source = fauxware_project.factory.entry_state()
        mgr._bind_export(source, sid)
        old = source.memory._binding
        old._released = "retired by the scheduler"
        copied = mgr._copy_bound_to(source, sid)
        assert source.memory._binding is old, "a frame shared by sibling ids keeps its binding"
        # The copy is the materialized state for ``sid`` itself, not a
        # copy-on-write child of it (angr-mx00p.112).
        for name in ("memory", "solver", "callstack"):
            binding = copied.plugins[name]._binding
            assert not binding.is_virtual and binding.raw_id == sid
        assert copied.solver._state_id == sid


class TestCallbackCallStackProxy:
    """angr-6o9p write-through .4: every SimProcedure callback state gets a
    ``RustCallStackProxyPlugin`` as ``state.callstack``, so iteration /
    top-frame attribute access reads the Rust frames by ``state_id`` via
    ``get_state_call_stack``.
    """

    def test_install_swaps_state_callstack(self, fauxware_project):
        """``_install_callback_callstack_proxy`` replaces ``state.callstack``
        with a ``RustCallStackProxyPlugin``."""
        from angr.exploration.rust_state_proxy import RustCallStackProxyPlugin

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_callstack_proxy(cb_state, seed_id)
        assert isinstance(cb_state.callstack, RustCallStackProxyPlugin)
        assert cb_state.callstack.id == "callstack"
        assert cb_state.callstack.category == "callstack"
        assert cb_state.callstack.state is cb_state

    def test_install_copy_returns_proxy(self, fauxware_project):
        """``copy()`` returns a fresh ``RustCallStackProxyPlugin`` bound
        to the same Rust state."""
        from angr.exploration.rust_state_proxy import RustCallStackProxyPlugin

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_callstack_proxy(cb_state, seed_id)
        cb_state.callstack.procedure_data = "marker"
        cb_state.callstack.locals["k"] = 1
        clone = cb_state.callstack.copy()
        assert isinstance(clone, RustCallStackProxyPlugin)
        assert clone._state_id == cb_state.callstack._state_id
        assert clone.state is None
        # Per-frame Python-only metadata copied verbatim.
        assert clone.procedure_data == "marker"
        assert clone.locals == {"k": 1}

    def test_read_through_empty_stack(self, fauxware_project):
        """Empty Rust call stack → proxy reports len 0 and 0-valued
        top-frame attrs (matches the existing read-only
        ``RustCallStackProxy`` shape). A blank state at main skips the init
        prefix, so its Rust stack is empty (an entry state carries main's
        frame)."""

        state = fauxware_project.factory.blank_state(addr=0x40071D)
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_callstack_proxy(cb_state, seed_id)
        proxy = cb_state.callstack
        assert len(proxy) == 0
        assert list(proxy) == []
        assert proxy.func_addr == 0
        assert proxy.ret_addr == 0
        assert proxy.stack_ptr == 0
        assert proxy.call_site_addr == 0
        assert proxy.next is None
        # CallStack.top returns self by convention.
        assert proxy.top is proxy

    def test_read_through_after_explore(self, fauxware_project):
        """After exploration, the proxy reflects Rust's current call
        frames (matches the read-only ``RustCallStackProxy``'s
        ``test_callstack_proxy_after_explore`` shape)."""
        from angr.exploration.rust_state_proxy import RustCallStackFrameProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED)
        assert mgr.found, "explore(find=0x4006ed) must reach target on fauxware — find regression"
        sid = mgr._rust_mgr.get_state_ids("found")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_callstack_proxy(cb_state, sid)
        proxy = cb_state.callstack
        raw = mgr._rust_mgr.get_state_call_stack(sid)
        assert len(proxy) == len(raw)
        # 0x4006ed sits inside main (reached via __libc_start_main from the
        # ELF entry stub), so the call stack must be non-empty at the find.
        # An empty stack here is a real export regression, not a no-op.
        assert raw, "exported call stack must be non-empty at find=0x4006ed"
        # Top frame attrs match raw[-1] (raw is push-order; top is last).
        expected = raw[-1]
        assert proxy.call_site_addr == expected[0]
        assert proxy.func_addr == expected[1]
        assert proxy.ret_addr == expected[2]
        assert proxy.stack_ptr == expected[3]
        # ``.next`` returns a frame proxy for frame 1 when there are
        # >= 2 frames, else None.
        if len(raw) >= 2:
            assert isinstance(proxy.next, RustCallStackFrameProxy)
        else:
            assert proxy.next is None

    def test_static_frame_walk_via_indexing(self):
        """Direct unit test of indexing + ``.next`` walk semantics with
        a synthetic frame list (no Rust runtime needed)."""
        from angr.exploration.rust_state_proxy import (
            RustCallStackFrameProxy,
            RustCallStackProxyPlugin,
        )

        proxy = RustCallStackProxyPlugin.__new__(RustCallStackProxyPlugin)
        proxy._mgr = None
        proxy._state_id = None
        # Pre-populate the read-through path: monkey-patch _frames to
        # return a static list (mirrors the read-only proxy's existing
        # _frames_cache pattern).
        snapshot = [
            (0x250, 0x300, 0x255, 0x7000),  # top
            (0x150, 0x200, 0x155, 0x7100),
            (0x050, 0x100, 0x055, 0x7200),  # bottom
        ]
        # Save the original class-level ``_frames`` property and restore
        # it in ``finally`` — ``del type(proxy)._frames`` would remove
        # the production property entirely (it's defined directly on
        # the class, not a per-instance override), breaking subsequent
        # tests in the same process. Caught by
        # ``TestExportCallStackProxyGate::test_sync_on_empty_rust_stack_installs_proxy``
        # when the export-gate tests ran after this one.
        original_frames = RustCallStackProxyPlugin._frames
        type(proxy)._frames = property(lambda self: snapshot)
        try:
            assert len(proxy) == 3
            top = proxy[0]
            assert top is proxy  # top-frame is the plugin itself
            assert top.func_addr == 0x300
            mid = proxy[1]
            assert isinstance(mid, RustCallStackFrameProxy)
            assert mid.func_addr == 0x200
            # .next walks frame[0] -> frame[1] -> frame[2] -> None
            assert proxy.next.func_addr == 0x200
            assert proxy.next.next.func_addr == 0x100
            assert proxy.next.next.next is None
            assert proxy[-1].func_addr == 0x100
            # iteration yields self then frame proxies in order.
            walked = list(proxy)
            assert walked[0] is proxy
            assert [f.func_addr for f in walked] == [0x300, 0x200, 0x100]
        finally:
            # Restore the original class property so subsequent tests
            # using the real FFI-backed ``_frames`` keep working.
            RustCallStackProxyPlugin._frames = original_frames

    def test_merge_widen_return_false(self, fauxware_project):
        """``merge`` / ``widen`` return False on the proxy plugin —
        cross-state_id callstack merge is out of scope for the proxy."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_callstack_proxy(cb_state, seed_id)
        assert cb_state.callstack.merge([], [], None) is False
        assert cb_state.callstack.widen([]) is False

    def test_push_pop_are_noop_stubs(self, fauxware_project):
        """``push`` / ``pop`` are gap stubs: return ``self`` and do not
        crash. SimProcedures don't manually push or pop frames in
        practice."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        cb_state = fauxware_project.factory.entry_state()
        mgr._install_callback_callstack_proxy(cb_state, seed_id)
        proxy = cb_state.callstack
        assert proxy.push(None) is proxy
        assert proxy.pop() is proxy
        assert proxy.call(0, 0) is proxy
        assert proxy.ret() is proxy

    def test_self_call_continuation_runs(self, fauxware_project):
        """A SimProcedure's ``self.call()`` continuation must resume with its
        saved locals (angr-mx00p.61).

        Two ways the proxy lost the continuation's ``procedure_data``: it was
        installed after ``_init_callback_callstack`` restored the data, so it
        replaced the plugin holding it and angr refused the continuation
        ("inapplicable stack frame"); and ``_capture_continuation_data``
        tested the plugin's truthiness, which is its Rust frame count, so a
        state with no Rust frame never stashed the caller's locals.
        """
        from angr import claripy

        proj = fauxware_project
        hook, callee, done = 0x4008C0, 0x4008D0, 0x4008E0
        seen = []

        class Caller(angr.SimProcedure):
            local_vars = ("marker",)

            def run(self):
                self.marker = 0x1234
                self.call(callee, [], "after", prototype="void x()")

            def after(self):
                seen.append(self.marker)
                self.jump(done)

        class Callee(angr.SimProcedure):
            def run(self):
                return 7

        class Done(angr.SimProcedure):
            NO_RET = True

            def run(self):
                seen.append("done")
                self.exit(0)

        try:
            proj.hook(hook, Caller(), replace=True)
            proj.hook(callee, Callee(), replace=True)
            proj.hook(done, Done(), replace=True)
            state = proj.factory.blank_state(
                addr=hook,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            state.regs.rsp = claripy.BVV(0x7FFF0000, 64)
            mgr = RustExplorationManager(proj, [state])
            mgr.run(max_steps=20)
        finally:
            for addr in (hook, callee, done):
                proj.unhook(addr)

        assert seen == [0x1234, "done"], f"continuation did not resume with its locals: {seen}"
        assert mgr.stash_counts()["errored"] == 0


class TestExportProxies:
    """An exported state's memory and call stack are live views of its Rust
    state (angr-yk2g, angr-ul4k, angr-mx00p.61); the lifetime and copy
    contract is pinned in ``test_export_live_views.py``."""

    def test_found_state_reads_rust_through_proxies(self, fauxware_project):
        from angr.exploration.rust_state_proxy import RustCallStackProxyPlugin, RustMemoryProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED)
        assert mgr.found, "explore(find=0x4006ed) must reach target on fauxware — find regression"
        found = mgr.found[0]
        sid = mgr._rust_mgr.get_state_ids("found")[0]
        assert isinstance(found.memory, RustMemoryProxy)
        assert found.memory._state_id == sid
        assert isinstance(found.callstack, RustCallStackProxyPlugin)
        raw = mgr._rust_mgr.get_state_call_stack(sid)
        # find=0x4006ed is inside main, so the stack must be non-empty;
        # an empty exported stack is a real regression.
        assert raw, "exported call stack must be non-empty at find=0x4006ed"
        assert len(found.callstack) == len(raw)
        assert found.callstack.func_addr == raw[-1][1]
        assert found.callstack.ret_addr == raw[-1][2]


class TestSimProcForkViaRustGate:
    """angr-t3mr write-through boundary: the Rust fork API behind additional
    SimProcedure successors. Every extra successor is finished from its own
    copy-on-write fork (angr-mx00p.112); the per-successor behaviour is
    pinned against the Python engine in ``test_callback_copy_on_write.py``.
    """

    def test_fork_state_to_stash_pending(self, fauxware_project):
        """Rust API: ``fork_state_to_stash(parent_id, 'active')`` returns a
        fresh state ID and lands a new entry in the active stash."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        before_count = len(mgr._rust_mgr.get_state_ids("active"))
        new_id = mgr._rust_mgr.fork_state_to_stash(seed_id, "active")
        # Fresh monotonic ID.
        assert new_id != seed_id
        # State count grew by one.
        after_ids = mgr._rust_mgr.get_state_ids("active")
        assert len(after_ids) == before_count + 1
        assert new_id in after_ids

    def test_fork_state_to_stash_inherits_pc(self, fauxware_project):
        """The fork carries the parent's PC."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        seed_id = mgr._rust_mgr.get_state_ids("active")[0]
        parent_pc = mgr._rust_mgr.get_state_pc_by_id(seed_id)
        new_id = mgr._rust_mgr.fork_state_to_stash(seed_id, "active")
        assert mgr._rust_mgr.get_state_pc_by_id(new_id) == parent_pc

    def test_fork_state_to_stash_unknown_state(self, fauxware_project):
        """Unknown parent ID raises ValueError."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        with pytest.raises(ValueError, match=r"state .* not found"):
            mgr._rust_mgr.fork_state_to_stash(99999, "active")


class TestProxyWriteThrough:
    """angr-j28e: RustStateProxy register and memory writes must write through
    to the Rust state (Rust is the single source of truth — no Python-side
    shadow store). Reads after a write observe the new value, both via the
    proxy and via the underlying RustSimState.
    """

    def test_register_write_concrete_int(self, fauxware_project):
        """proxy.regs.<name> = <int> writes through to Rust and re-reads
        return the BVV-wrapped value."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        proxy.regs.rax = 0xDEADBEEF
        rax = proxy.regs.rax
        assert rax.size() == 64
        assert rax.concrete and rax.concrete_value == 0xDEADBEEF

        # Sanity: the underlying Rust state actually holds the new value
        # (not a Python-only stash on the proxy cache).
        sid = proxy.state_id
        assert mgr._rust_mgr.get_state_register(sid, "rax") == 0xDEADBEEF

    def test_register_write_concrete_bvv(self, fauxware_project):
        """proxy.regs.<name> = <BVV> writes through (BVV path, no int wrap)."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        proxy.regs.rbx = claripy.BVV(0x4141414141414141, 64)
        rbx = proxy.regs.rbx
        assert rbx.concrete and rbx.concrete_value == 0x4141414141414141
        sid = proxy.state_id
        assert mgr._rust_mgr.get_state_register(sid, "rbx") == 0x4141414141414141

    def test_register_write_symbolic_round_trip(self, fauxware_project):
        """proxy.regs.<name> = <BVS> registers the symbol in the shared
        cache so the subsequent constraint-add path narrows the live
        symbol: the read-path symbol-identity invariant, carried over to the
        write path (angr-4pm1)."""
        from angr import claripy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        sym = claripy.BVS("write_sym_rcx", 64)
        proxy.regs.rcx = sym
        rcx = proxy.regs.rcx
        assert rcx.symbolic, "rcx must read back as symbolic after a BVS write"
        # Constrain the symbol and evaluate — proves the AST identity is
        # preserved end-to-end (write → cache → solver fork).
        proxy.solver.add(rcx == 0x55)
        assert proxy.solver.eval(rcx) == 0x55

    def test_memory_write_concrete_bytes_round_trips(self, fauxware_project):
        """proxy.memory.store(int_addr, bytes) writes through and the next
        proxy.memory.load returns the same bytes."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        # Pick a writable mapped address inside the binary's text/data
        # range that the entry_state already touches — fauxware's entry
        # point sits well inside the .text page.
        addr = fauxware_project.entry
        proxy.memory.store(addr, b"\xab\xcd\xef\x12")
        loaded = proxy.memory.load(addr, 4)
        assert loaded.concrete_value == 0xABCDEF12

    def test_memory_write_int_value_defaults_to_arch_word(self, fauxware_project):
        """proxy.memory.store(addr, int) without size= defaults the width to
        the arch word size, mirroring angr's SimMemory.store (e.g. asprintf
        writes a malloc'd pointer back via
        ``memory.store(strp, dst, endness=memory_endness)`` with no size). The
        callback-memory-proxy gate routes such stores through this path, so a
        missing size must be inferred rather than rejected (angr-92e17)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        addr = fauxware_project.entry
        word = fauxware_project.arch.bytes  # 8 on AMD64
        proxy.memory.store(addr, 0xCAFEBABE)
        # BE default: value zero-extended to the full arch word, high bytes 0.
        assert proxy.memory.load(addr, word).concrete_value == 0xCAFEBABE

    def test_memory_write_int_value_with_size(self, fauxware_project):
        """proxy.memory.store(addr, int, size=N) writes N bytes big-endian
        by default; little-endian via endness='Iend_LE'."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        proxy = mgr.proxy.active[0]

        addr = fauxware_project.entry
        proxy.memory.store(addr, 0xCAFEBABE, size=4)
        # BE default: high byte 0xCA at the lowest address.
        assert proxy.memory.load(addr, 4).concrete_value == 0xCAFEBABE
        # Now overwrite with little-endian: 0x12345678 LE = 78 56 34 12
        proxy.memory.store(addr, 0x12345678, size=4, endness="Iend_LE")
        # memory.load default is BE so we should read the byte-swapped value
        assert proxy.memory.load(addr, 4).concrete_value == 0x78563412


class TestCallbackMemoryProxySymbolicStore:
    """Coverage for the interpreter's symbolic-address store path under the
    callback memory proxy (angr-5rjbq).

    Not a red-without-the-fix guard: this single-block shape survives even
    when the store reaches Rust memory only via the registered-handle /
    symbolic-page side channel, which is exactly what made the flareon2015_5
    drop so hard to catch. The real regression signal for this drop is
    flareon2015_5 (a bench, not a unit test) — see the
    MEDIUM-suite proxy differential (angr-ijwp0).
    """

    @staticmethod
    def _run_sym_addr_store():
        """Execute ``mov [ecx], al`` with a symbolic (single-solution) address
        and a symbolic value, then read the byte back from the stepped state.
        """
        from angr import claripy

        # mov [ecx], al ; jmp $
        proj = angr.load_shellcode(b"\x88\x01\xeb\xfe", arch="X86", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)

        addr_sym = claripy.BVS("dst", 32)
        state.add_constraints(addr_sym == 0x2000)
        state.regs.ecx = addr_sym
        state.regs.eax = claripy.BVS("val", 32)

        mgr = RustExplorationManager(proj, [state])
        mgr.step(n=1)
        stepped = mgr.active[0]
        return stepped, stepped.memory.load(0x2000, 1)

    def test_symbolic_addr_store_survives_readback(self):
        """The stored byte reads back as the symbolic value, not concrete 0."""
        stepped, loaded = self._run_sym_addr_store()
        assert loaded.symbolic, f"store dropped — read back {loaded!r}"
        assert any(v.startswith("val") for v in loaded.variables)
        # Genuinely unconstrained (the low byte of eax), not a pinned witness.
        assert len(stepped.solver.eval_upto(loaded, 2)) == 2
