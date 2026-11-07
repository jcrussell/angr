"""A Rust bench run that built no Rust manager fails (angr-vmi2l.312).

``engine_patch`` hands a manager built with a non-empty ``techniques=`` to the
Python ``SimulationManager``. Before, ``run_one`` reported such a run ok, so
the correctness and timing passes checked the Python engine. These run a
synthetic one-step bench through ``run_regression.run_one`` both ways.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import run_regression as rr

_FAUXWARE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "binaries", "tests", "x86_64", "fauxware")

_SOLVE = """\
import angr


def main():
    p = angr.Project({binary!r}, auto_load_libs=False)
    simgr = p.factory.simulation_manager(p.factory.entry_state(){extra})
    simgr.step()
    print("stepped")


def test():
    main()


if __name__ == "__main__":
    main()
"""


def _write_bench(tmp_path, name, extra):
    bench = tmp_path / name
    bench.mkdir()
    (bench / "solve.py").write_text(_SOLVE.format(binary=os.path.abspath(_FAUXWARE), extra=extra))


@pytest.mark.skipif(not os.path.exists(_FAUXWARE), reason="binaries repo not checked out")
@pytest.mark.parametrize("mode", ["run", "test"])
def test_run_one_fails_a_rust_run_with_no_rust_manager(tmp_path, monkeypatch, mode):
    _write_bench(tmp_path, "py_routed", ", techniques=[angr.exploration_techniques.DFS()]")
    _write_bench(tmp_path, "rust_routed", "")
    monkeypatch.setattr(rr, "EXAMPLES_DIR", str(tmp_path))

    routed_away = rr.run_one("py_routed", "rust", 120, 4096, mode=mode)
    assert not routed_away["ok"]
    assert "no RustExplorationManager was built" in routed_away["error"]

    on_rust = rr.run_one("rust_routed", "rust", 120, 4096, mode=mode)
    assert on_rust["ok"], on_rust.get("error")
    assert on_rust["rust_managers_built"] == 1

    # The Python engine never builds one and is not held to it.
    assert rr.run_one("py_routed", "python", 120, 4096, mode=mode)["ok"]
