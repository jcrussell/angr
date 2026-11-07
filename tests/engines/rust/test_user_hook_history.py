"""The history entry a Python user hook leaves under the Rust engine (angr-vmi2l.21).

angr's ``UserHook`` leaves through ``Ijk_NoHook`` (zero-length) or
``Ijk_Boring`` (sized). The Rust resume used to record every Python bounce
as a procedure's ``Ijk_Ret`` whose source was the new pc, so a state exported
from the run re-fired a zero-length hook when stepped in Python.
"""

from __future__ import annotations

import angr
from tests.engines.conftest import RustExplorationManager

BASE = 0x400000
HOOK = 0x400004
# HistoryEntry jumpkind bytes, as get_state_detailed_history returns them.
JK_BORING = 0
JK_NOHOOK = 6


def _run_one_step(length):
    proj = angr.load_shellcode(b"\x90" * 32, arch="amd64", load_address=BASE)
    calls = []
    proj.hook(HOOK, lambda st: calls.append(st.addr), length=length)
    mgr = RustExplorationManager(proj, [proj.factory.blank_state(addr=BASE)])
    mgr.run(max_steps=1)
    (sid,) = mgr._rust_mgr.get_state_ids("active")
    return proj, mgr, sid, calls


def test_zero_length_hook_records_ijk_nohook():
    _, mgr, sid, calls = _run_one_step(0)
    assert calls == [HOOK]
    assert mgr._rust_mgr.get_state_detailed_history(sid)[-1] == (HOOK, JK_NOHOOK, HOOK)


def test_sized_hook_records_ijk_boring_past_the_hook():
    _, mgr, sid, calls = _run_one_step(2)
    assert calls == [HOOK]
    assert mgr._rust_mgr.get_state_detailed_history(sid)[-1] == (HOOK, JK_BORING, HOOK + 2)


def test_exported_state_does_not_refire_a_zero_length_hook():
    proj, mgr, sid, calls = _run_one_step(0)
    # Force the snapshot export path, not a cached mirror of the callback state.
    mgr._state_cache.clear()
    mgr._state_roots.clear()
    mgr._current_callback_state_id = None
    state = mgr._materialize_single_state(sid)
    assert state.history.jumpkind == "Ijk_NoHook"
    proj.factory.successors(state.copy())
    assert calls == [HOOK], "the Python step re-ran the hook the Rust run already ran"
