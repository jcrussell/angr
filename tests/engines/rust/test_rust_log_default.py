"""Rust ``log::warn!`` reaches stderr under default configuration (angr-vmi2l.247).

The extension installs a ``warn`` filter at import (``DEFAULT_LOG_SPEC`` in
``native/angr/src/engine.rs``). Without it, every ``silent_default!(cat_c, ..)``
fallback was dropped unless ``RUST_LOG`` was set. Each case runs in a subprocess
with ``RUST_LOG`` / ``ANGR_RUST_LOG`` scrubbed, because the logger filter is
process-global.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from angr.exploration.rust_manager import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration extension not built")

_PRELUDE = """
import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager
p = angr.load_shellcode(b"\\x90\\xc3", arch="amd64")
s = p.factory.blank_state(add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
m = RustExplorationManager(p, [s])
"""

# A 1 ms Z3 timeout on a 64-bit factoring query makes Z3 answer Unknown, which
# `add_constraints_to_state` collapses to False through `silent_default!(cat_c, ..)`.
_CAT_C_UNKNOWN = """
from angr.rustylib.vex_engine import set_z3_global_param
sid = m._rust_mgr.get_state_ids("active")[0]
x = claripy.BVS("x", 64)
y = claripy.BVS("y", 64)
set_z3_global_param("timeout", "1")
assert m._rust_mgr.add_constraints_to_state(
    sid,
    [x * y == 0x66B1D2F4C1A3B5D7, claripy.ULT(1, x), claripy.ULT(1, y),
     claripy.ULT(x, 0xFFFFFFFF), claripy.ULT(y, 0xFFFFFFFF)],
) is False
"""

_NEW_STASH = """
m.move("active", "actve")
"""


def _stderr(body: str, **env_overrides: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("RUST_LOG", "ANGR_RUST_LOG")}
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PRELUDE + body)],
        env=env,
        capture_output=True,
        timeout=300,
        check=False,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, stderr[-4000:]
    return stderr


def test_cat_c_fallback_warns_with_no_env():
    stderr = _stderr(_CAT_C_UNKNOWN)
    assert "[rust:WARN]" in stderr
    assert "add_constraints_to_state: Z3 returned Unknown" in stderr


def test_plain_warn_reaches_stderr_with_no_env():
    stderr = _stderr(_NEW_STASH)
    assert "[rust:WARN] rustylib::stash:" in stderr
    assert "'actve'" in stderr


def test_default_filter_does_not_spam_bvv_bytes_warns():
    # Python's paged memory calls BVV(bytes, size) on every page load; the
    # default filter holds that claripy target at error.
    stderr = _stderr(_NEW_STASH)
    assert "rustylib::claripy::ast::bv" not in stderr
    assert "Unmapped VEX operation" not in stderr


def test_rust_log_off_overrides_default():
    stderr = _stderr(_CAT_C_UNKNOWN + _NEW_STASH, RUST_LOG="off")
    assert "[rust:" not in stderr


def test_set_rust_log_level_off_overrides_default():
    stderr = _stderr(
        "from angr.exploration.rust_manager import set_rust_log_level\nset_rust_log_level('off')\n" + _NEW_STASH
    )
    assert "[rust:" not in stderr
