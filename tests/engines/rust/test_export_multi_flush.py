"""Multi-cell visibility through the state-export FFI (angr-9ke6b.101).

A symbolic-address store whose address resolves to *multiple* candidates is
routed through ``install_multi_for_candidates_safe``, which parks the value in
``SymbolicMemory::multi_objects`` rather than in the page's ``symbolic_bitmap``.
``RustSimState::export_full`` walks ``MemoryPage::symbolic_offsets``, which
reads only the symbolic bitmap — so an *unflushed* export silently reports the
stale concrete backing bytes for every Multi-covered address.

``mgr.found_states`` (the primary ``explore(find=...)`` result API) and
``mgr.get_state_by_id`` both go through that export path, so the flush has to
happen on the Rust side of the boundary. These tests pin that it does.
"""

from __future__ import annotations

import claripy
import pytest

from tests.engines.conftest import (
    RUST_EXPLORATION_AVAILABLE,
    RustExplorationManager,
)

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

# Two mapped candidates for the symbolic store address. Both live on the
# initial stack page, so neither is in a lazy region (which would make the
# safe installer defer instead of installing Multi cells).
_PAYLOAD = 0xCAFEBABE


def _multi_store_manager(project, sym_name):
    """Build a manager whose seed state has a Multi-cell store pending.

    Returns ``(mgr, seed_id, addr_sym, candidates)``.
    """
    state = project.factory.entry_state()
    sp = state.solver.eval(state.regs.sp)
    candidates = (sp - 0x200, sp - 0x100)

    addr_sym = claripy.BVS(sym_name, 64)
    state.solver.add(claripy.Or(addr_sym == candidates[0], addr_sym == candidates[1]))

    mgr = RustExplorationManager(project, [state])
    seed_id = mgr._rust_mgr.get_state_ids("active")[0]
    proxy = mgr.proxy.active[0]
    proxy.memory.store(addr_sym, claripy.BVV(_PAYLOAD, 32), endness="Iend_LE")
    return mgr, seed_id, addr_sym, candidates


def _assert_multi_visible(angr_state, addr_sym, candidates):
    """Under ``addr_sym == candidates[0]`` the exported state must read back
    the stored payload — not the stale zero byte the page holds."""
    assert angr_state is not None
    loaded = angr_state.memory.load(candidates[0], 4, endness="Iend_LE")
    assert loaded.symbolic, f"Multi-cell byte exported as concrete {loaded!r} — flush was skipped"

    probe = angr_state.copy()
    probe.solver.add(addr_sym == candidates[0])
    assert probe.solver.eval(loaded) == _PAYLOAD


class TestExportFlushesMultiCells:
    def test_get_state_by_id_sees_multi_cell_data(self, fauxware_project):
        """``get_state_by_id`` -> ``export_state`` must flush Multi cells."""
        mgr, seed_id, addr_sym, candidates = _multi_store_manager(fauxware_project, "multi_by_id")
        _assert_multi_visible(mgr.get_state_by_id(seed_id), addr_sym, candidates)

    def test_export_stash_sees_multi_cell_data(self, fauxware_project):
        """``found_states`` -> ``export_found_states`` -> ``_export_stash``
        must flush too. Move the seed into ``found`` directly rather than
        running an exploration so the test stays fast and deterministic.
        """
        mgr, seed_id, addr_sym, candidates = _multi_store_manager(fauxware_project, "multi_stash")
        mgr._rust_mgr.move_state(seed_id, "active", "found")

        found = mgr.found_states
        assert len(found) == 1
        _assert_multi_visible(found[0], addr_sym, candidates)

    def test_step_state_sees_multi_cell_data(self, fauxware_project):
        """``step_state()`` (``_step_state`` on the Rust side) must flush
        Multi cells on every returned successor too — a fourth export site
        that the original ``angr-9ke6b.101`` fix missed. Used directly by
        ``RustSimulationManagerProxy.step_state()`` and, through it, by
        Veritesting/Tracer/Slicecutor.
        """
        mgr, seed_id, addr_sym, candidates = _multi_store_manager(fauxware_project, "multi_step")
        seed_proxy = mgr.proxy.active[0]
        assert seed_proxy._state_id == seed_id

        buckets = mgr.proxy.step_state(seed_proxy)
        successors = buckets[None]
        assert successors, "expected at least one live successor from stepping the entry block"
        for succ in successors:
            _assert_multi_visible(succ, addr_sym, candidates)
