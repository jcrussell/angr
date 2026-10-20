# pylint: disable=missing-class-docstring,no-self-use
"""A warm disk init-cache hit must restore the call stack below ``main`` (angr-mx00p.92).

``rust_disk_cache.py`` pickles the init state's frames and pushes them back
onto the rebuilt blank state. The restore called ``CallStack.call`` with a
keyword it does not take and pushed every frame onto the original stack, so a
warm run handed Rust a sentinel-only stack and found states lacked ``main``'s
frame.
"""

from __future__ import annotations

from unittest import mock

import angr
from angr.exploration.rust_manager import RustExplorationManager
from tests.engines.conftest import FAUXWARE_PATH

_CACHE_KEY = "test_angr_mx00p92_callstack"


def _frames(state):
    return [(f.call_site_addr, f.func_addr, f.ret_addr) for f in state.callstack]


class TestWarmDiskCacheKeepsCallStack:
    def test_frames_round_trip(self, tmp_path):
        project = angr.Project(FAUXWARE_PATH, auto_load_libs=False)
        cold = project.factory.entry_state()
        cold.callstack.call(0x4005A4, 0x400540, retn_target=0x4005A9, stack_pointer=0x7FFFFFFEFF38)
        cold.callstack.call(0x801053, 0x40071D, retn_target=0x801058, stack_pointer=0x7FFFFFFEFF28)
        assert len(_frames(cold)) == 3

        with mock.patch.object(RustExplorationManager, "_disk_cache_dir", staticmethod(lambda: str(tmp_path))):
            mgr = RustExplorationManager(project, [project.factory.entry_state()])
            mgr._save_init_to_disk_cache(_CACHE_KEY, cold)
            warm, _mem_cache = mgr._load_init_from_disk_cache(_CACHE_KEY)

        assert warm is not None
        assert _frames(warm) == _frames(cold)

    def test_warm_run_keeps_main_frame(self, tmp_path):
        """A run whose init comes from the disk cache hands Rust the same call
        stack as a cold one, so its found state still has ``main``'s frame
        (the broken restore left only the sentinel)."""
        project = angr.Project(FAUXWARE_PATH, auto_load_libs=False)
        found = []
        with mock.patch.object(RustExplorationManager, "_disk_cache_dir", staticmethod(lambda: str(tmp_path))):
            for _run in ("cold", "warm"):
                RustExplorationManager._init_cache.clear()
                mgr = RustExplorationManager(project, [project.factory.entry_state()])
                mgr.explore(find=0x4006ED)
                found.append([(f.func_addr, f.ret_addr) for f in mgr.found[0].callstack])
        cold, warm = found
        assert warm == cold
        assert (0x40071D, 0x801058) in cold, "main's frame"
