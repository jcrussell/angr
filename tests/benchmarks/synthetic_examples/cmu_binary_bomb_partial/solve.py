"""cmu_binary_bomb wrapper that runs the FAST-tier subset of phases.

Upstream ``solve.py`` (``angr-examples/examples/cmu_binary_bomb/solve.py``)
has two stale entry points, one path-explosion bottleneck, and one
silently-unsupported angr feature:

* ``solve_flag_2``: ``state.stack_push`` of a 32-bit BVS raises
  ``SimMemoryError 'Not enough data for store'`` in both engines
  (wants a word-sized push).
* ``solve_flag_3``: builds the simulation manager with
  ``veritesting=True``. The Rust manager does not yet honor the
  veritesting flag and silently no-ops the explore loop, returning an
  empty solution list. Including it would pad Rust timing with a 0-work
  call vs Python's real exploration.
* ``solve_flag_5``: uses ``proj.kb.obj.get_symbol(...)``; the ``obj``
  knowledge-base plugin was renamed years ago and ``proj.kb.obj``
  now raises ``AttributeError``.
* ``solve_flag_6``: enumerates every reachable path in the linked-list
  phase. Takes ~73 s on Python, which would push the bench past the
  MEDIUM-tier 60 s envelope.

The crash in flag_2 aborts upstream's ``main()`` before any timing can
be captured. We wrap the upstream module here and run only the FAST
subset (flags 1, 4, secret) so ``run_single.py`` and
``run_regression.py`` can exercise the printf/scanf-heavy CMU teaching
binary without depending on a fixed upstream. See bd memory
``bench-cmu-binary-bomb-broken`` for the flag details and ``angr-w5op``
for the bench-add task.

The binary itself lives upstream (``cmu_binary_bomb/bomb``), not in this
synthetic-examples directory, so we ``chdir`` to the upstream example
dir before importing and let the upstream functions resolve ``./bomb``
relative to that cwd.
"""

from __future__ import annotations

import importlib.util
import os

_EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
_UPSTREAM_DIR = os.path.join(_EXAMPLES_DIR, "cmu_binary_bomb")
_UPSTREAM_SOLVE = os.path.join(_UPSTREAM_DIR, "solve.py")

if not os.path.exists(_UPSTREAM_SOLVE):
    raise FileNotFoundError(
        f"cmu_binary_bomb upstream solve.py not found at {_UPSTREAM_SOLVE}. "
        "Set ANGR_EXAMPLES_DIR or clone angr/angr-examples."
    )

os.chdir(_UPSTREAM_DIR)

_spec = importlib.util.spec_from_file_location("cmu_binary_bomb_upstream", _UPSTREAM_SOLVE)
_upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_upstream)


def main():
    print("Flag    1: " + _upstream.solve_flag_1())
    # Skip flag 2 — upstream stack_push bug.
    # Skip flag 3 — Rust manager silently no-ops veritesting=True.
    print("Flag    4: " + _upstream.solve_flag_4())
    # Skip flag 5 — upstream uses obsolete proj.kb.obj API.
    # Skip flag 6 — ~73s path explosion would blow past MEDIUM tier.
    print("Secret   : " + _upstream.solve_secret())


if __name__ == "__main__":
    main()
