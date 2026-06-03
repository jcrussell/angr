"""Smoke tests for the project hook surface under the Rust engine.

The Rust engine treats user hooks identically to the Python engine: hooks
are looked up on the ``Project`` at dispatch time and run as Python
callbacks. These tests pin the four high-level entry points that users
reach for — :meth:`Project.hook` with a SimProcedure (``replace=True``),
:meth:`Project.hook_symbol` to swap a libc binding, the zero-length
``hook(addr, hook=cb, length=0)`` advance hook, and :meth:`Project.unhook`
to revert.

These surfaces are already exercised implicitly by 6+ benchmarks
(``flareon2015_5``, ``flareon2015_10``, ``whitehatvn_re400`` and others
recorded in the ``bench-simprocedure-fallback-distribution`` memory),
but no explicit smoke test had pinned the contract end-to-end. Per
``angr-3bz1`` this file makes the surface explicit so a regression here
fails loudly rather than as a benchmark slowdown.
"""

from __future__ import annotations

__package__ = __package__ or "tests.engines"  # pylint:disable=redefined-builtin

import os

import pytest

import angr


try:
    from angr.exploration import RustExplorationManager
    RUST_EXPLORATION_AVAILABLE = True
except ImportError:
    RUST_EXPLORATION_AVAILABLE = False


_candidates = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "binaries"),
    os.path.expanduser("~/repos/angr-examples/examples/fauxware"),
    os.path.expanduser("~/angr-examples/examples/fauxware"),
]
TEST_BINARIES_DIR = next((d for d in _candidates if os.path.isdir(d)), _candidates[0])


@pytest.fixture
def fauxware_project():
    """Fresh fauxware project per test — hooks mutate project state."""
    binary_path = os.path.join(TEST_BINARIES_DIR, "fauxware")
    if not os.path.exists(binary_path):
        pytest.skip("fauxware binary not found")
    return angr.Project(binary_path, auto_load_libs=False)


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestHooksRustSmoke:
    """End-to-end smoke tests for ``Project.hook`` / ``hook_symbol`` /
    ``unhook`` under :class:`RustExplorationManager`."""

    def test_project_hook_simprocedure_replace_true(self, fauxware_project):
        """``proj.hook(addr, simproc, replace=True)`` swaps the existing
        auto-installed libc SimProcedure with a user-supplied one.

        fauxware's ``strcmp`` PLT slot is hooked at load time by
        ``angr.SIM_PROCEDURES['libc']['strcmp']``. We override it with a
        deterministic stub that returns a tagged constant and confirm the
        stub actually ran (it sets a sentinel) during a Rust-engine run.
        """
        proj = fauxware_project
        sym = proj.loader.find_symbol("strcmp")
        assert sym is not None, "fauxware should auto-resolve strcmp"
        assert proj.is_symbol_hooked("strcmp"), \
            "strcmp must already be hooked for replace=True to be meaningful"

        fired = {"count": 0}

        class TaggedStrcmp(angr.SimProcedure):
            def run(self_proc, s1, s2):  # noqa: N805
                fired["count"] += 1
                return 0xC0DE

        proj.hook(sym.rebased_addr, TaggedStrcmp(), replace=True)

        assert proj.hooked_by(sym.rebased_addr).__class__ is TaggedStrcmp, \
            "replace=True should have swapped the SimProcedure class"

        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=400)

        assert fired["count"] >= 1, (
            f"TaggedStrcmp never fired during the run — replace=True did not "
            f"engage. Stash counts: {mgr.stash_counts()}"
        )

    def test_hook_symbol_swaps_libc_binding(self, fauxware_project):
        """``proj.hook_symbol('strcmp', proc, replace=True)`` resolves the
        symbol to an address and installs the SimProcedure there.

        Equivalent to ``proj.hook(loader.find_symbol(name).rebased_addr,
        proc, replace=True)`` but covers the symbol-resolution path so the
        smoke test is true to the README-level usage pattern.
        """
        proj = fauxware_project
        assert proj.is_symbol_hooked("strcmp")

        fired = {"count": 0}

        class HookSymbolStrcmp(angr.SimProcedure):
            def run(self_proc, s1, s2):  # noqa: N805
                fired["count"] += 1
                return 1  # non-zero — diverges from the auto-installed strcmp

        addr = proj.hook_symbol("strcmp", HookSymbolStrcmp(), replace=True)
        assert addr is not None, "hook_symbol should return the hooked address"
        assert proj.hooked_by(addr).__class__ is HookSymbolStrcmp, (
            "hook_symbol(replace=True) should install our SimProcedure at "
            "the symbol's address"
        )

        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=400)

        assert fired["count"] >= 1, (
            "hook_symbol-installed SimProcedure never fired — dispatch did "
            f"not reach the symbol address. Stash counts: {mgr.stash_counts()}"
        )

    def test_zero_length_user_hook_fires_once_per_visit(self, fauxware_project):
        """``proj.hook(addr, hook=callback, length=0)`` runs the callback
        and resumes at the original PC (no instructions skipped).

        Distinct from ``length>0`` advance hooks, which would replace the
        instruction. A zero-length hook must NOT re-fire in a tight loop
        at the same PC — the dispatcher emits ``Ijk_NoHook`` after the
        callback so the next step lifts the real instruction.

        We hook the first byte of ``main`` and verify the callback fires
        at least once and a bounded number of times across the small set
        of paths fauxware explores.
        """
        proj = fauxware_project
        main_sym = proj.loader.find_symbol("main")
        assert main_sym is not None
        hook_addr = main_sym.rebased_addr

        fire_addrs = []

        def cb(state):
            fire_addrs.append(state.addr)

        proj.hook(hook_addr, hook=cb, length=0)
        assert proj.is_hooked(hook_addr)

        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=400)

        assert fire_addrs, "zero-length hook never fired at main"
        # Must terminate, i.e. not re-fire in an infinite loop at hook_addr.
        # fauxware explores a small handful of paths; a runaway re-execute
        # would balloon this list well past any reasonable bound.
        assert len(fire_addrs) < 50, (
            f"zero-length hook fired {len(fire_addrs)} times — likely "
            f"re-executing at hook_addr instead of advancing to the real "
            f"instruction (Ijk_NoHook regression)."
        )
        # All fires should be at the hooked address.
        assert all(a == hook_addr for a in fire_addrs), (
            f"unexpected fire address(es): {[hex(a) for a in fire_addrs]}"
        )

    def test_unhook_removes_callback(self, fauxware_project):
        """``proj.unhook(addr)`` removes the hook so the callback no
        longer fires on subsequent runs.

        Set up a callback, exercise the path under the Rust engine,
        unhook, then exercise the same path on a fresh state — the
        callback must not fire on the second run.
        """
        proj = fauxware_project
        main_sym = proj.loader.find_symbol("main")
        assert main_sym is not None
        hook_addr = main_sym.rebased_addr

        run1_fires = []
        run2_fires = []
        active_log = run1_fires

        def cb(state):
            active_log.append(state.addr)

        proj.hook(hook_addr, hook=cb, length=0)
        assert proj.is_hooked(hook_addr)

        # First run: hook is live and must fire at least once.
        state1 = proj.factory.entry_state()
        mgr1 = RustExplorationManager(proj, [state1])
        mgr1.run(max_steps=400)
        assert run1_fires, "hook didn't fire on the pre-unhook run"

        proj.unhook(hook_addr)
        assert not proj.is_hooked(hook_addr), \
            "unhook(addr) should clear is_hooked(addr)"

        # Second run: hook is gone; callback must not fire.
        active_log = run2_fires
        state2 = proj.factory.entry_state()
        mgr2 = RustExplorationManager(proj, [state2])
        mgr2.run(max_steps=400)
        assert not run2_fires, (
            f"callback fired {len(run2_fires)} times after unhook(addr) — "
            f"the hook was not actually removed."
        )
