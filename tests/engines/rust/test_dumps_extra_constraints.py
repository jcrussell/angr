"""``posix.dumps(0)`` on a Rust export solves stdin as it stands when called (angr-vmi2l.334).

An export laid stdin out as the bytes each symbol evaluated to at export
time, so ``dumps(0, extra_constraints=...)``, and ``dumps(0)`` after a
caller's own ``add_constraints``, returned the export-time model. Each case
runs amd64 shellcode that reads stdin and branches on its first byte (the
helpers of ``test_merge_std_streams``), explores to the common exit on both
engines and compares what ``dumps(0)`` gives under constraints added after
the run. The natively merged case is
``test_merge_std_streams.test_equal_std_streams_still_merge_natively``.
"""

from __future__ import annotations

import angr
from tests.engines.conftest import RustExplorationManager
from tests.engines.rust.test_merge_std_streams import (  # noqa: F401
    BUF,
    LOAD,
    MSG_A,
    _branchy,
    _read,
    _write,
    pytestmark,
)


def _found(engine: str, code: bytes):
    proj = angr.load_shellcode(code, arch="amd64", simos="linux", load_address=LOAD)
    st = proj.factory.entry_state(
        addr=LOAD,
        add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS, angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY},
    )
    st.memory.store(BUF, b"\0" * 8)
    st.memory.store(MSG_A, b"XY")
    mgr = RustExplorationManager(proj, [st]) if engine == "rust" else proj.factory.simulation_manager(st)
    mgr.explore(find=LOAD + len(code) - 2, num_find=2)
    found = list(mgr.found)
    assert len(found) == 2
    return found


def _dumps_under(state, values: bytes):
    """``dumps(0)``'s first ``len(values)`` bytes with byte ``i`` pinned to ``values[i]``, or ``None`` when unsat.

    The bytes past them are whatever the model picks, so they are not compared.
    """
    cons = [state.memory.load(BUF + i, 1) == v for i, v in enumerate(values)]
    if not state.solver.satisfiable(extra_constraints=cons):
        return None
    return state.posix.dumps(0, extra_constraints=cons)[: len(values)]


# Both arms read one byte; the else arm reads a second one into BUF + 1.
CODE = _branchy(_write(MSG_A, 1), _read(BUF + 1) + _write(MSG_A, 1))


def _extra_constraint_dumps(engine: str):
    return sorted(
        ((_dumps_under(s, b"A"), _dumps_under(s, b"B"), _dumps_under(s, b"BC")) for s in _found(engine, CODE)), key=repr
    )


def test_dumps_honours_extra_constraints_like_python():
    py = _extra_constraint_dumps("py")
    rust = _extra_constraint_dumps("rust")
    assert py == [(None, b"B", b"BC"), (b"A", None, None)]
    assert rust == py


def _added_constraint_dumps(engine: str):
    out = []
    for state in _found(engine, CODE):
        first = state.memory.load(BUF, 1)
        if state.solver.satisfiable(extra_constraints=[first == 0x5A]):
            state.add_constraints(first == 0x5A)
            out.append(state.posix.dumps(0)[:1])
    return out


def test_dumps_sees_constraints_added_after_the_run():
    assert _added_constraint_dumps("rust") == _added_constraint_dumps("py") == [b"Z"]
