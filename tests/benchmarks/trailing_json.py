"""Shared extraction of ``run_single.py``'s trailing ``--counters-json`` object.

``run_single.py`` emits the ``--counters-json`` blob as the TRAILING JSON
object (printed by ``json.dumps(indent=2)`` on its own line at end-of-stdout).
We locate it via the last ``\\n{`` rather than the first ``{`` because the
solve.py output-preview lines (``  > ...``) can themselves contain braces (e.g.
a recovered flag) — a first-``{`` heuristic (see ``bench_diff.load_counters``)
would grab those. Only the top-level object opens at column 0 after a newline,
so ``rfind("\\n{")`` is unambiguous.

All four gate scripts (run_findall_gate, run_zeropy_gate,
run_steal_fraction_gate, run_parallel_overhead_gate) share this one helper so a
future change to run_single's output format is re-verified in a single place,
with consistent error handling: a missing OR malformed trailing object both
yield ``None``, and each caller decides whether that is fatal (raise /
SystemExit) or soft (skip this rep).
"""

from __future__ import annotations

import json


def trailing_json(out: str) -> dict | None:
    """Return run_single's trailing counters object, or ``None``.

    ``None`` is returned both when no top-level ``{`` opens a line (no JSON
    emitted at all) and when the located payload fails to parse (truncated /
    corrupt output). Callers decide whether ``None`` is fatal or soft.
    """
    brace = out.rfind("\n{")
    if brace == -1:
        if out.startswith("{"):
            brace = 0
        else:
            return None
    try:
        return json.loads(out[brace:])
    except json.JSONDecodeError:
        return None
