#!/usr/bin/env python3
"""S10 (angr-op0dn.8) — scored parity census for the default-engine flip.

Moonshot 6 asks: *what is the minimal set of work required to flip
``default_engine="rust"`` without regressing any Python-engine user?*

The scoring axis is **loudness**, not feature count. A divergence the engine
can detect *before* it diverges (a raise at manager construction, a raise at
``state.inspect.b()`` registration, an unsupported-arch check) is not a flip
blocker at all: the tri-state dispatcher (angr-op0dn.14.5.2) routes that
workload to the Python engine and the user never notices. Only a divergence
that is **silent** — Rust quietly does something different from Python, mid-run,
with no raise, no warning, and no log record — can regress a user after a flip.

So this census sorts every known Rust/Python divergence into:

* ``raise``  — refuses the state up front. Dispatcher-handleable. NOT a blocker.
* ``warn``   — accepted, but a warn-once names it. Degraded, but attributable.
* ``silent`` — accepted, diverges, says nothing. **These are the flip gate.**

and then ranks the ``silent`` ones by blast radius, where the blast radius that
matters for a *default* flip is "does this fire for a user who wrote plain
``proj.factory.entry_state()`` and never opted into anything?"

Inputs are live code (``_RAISE_OPTION_NAMES`` / ``_REJECTED_OPTION_NAMES`` /
``_INSPECT_EVENT_SPECS``) plus the M6.5a fallback census artifact
(``fallback_census.json``, angr-op0dn.14.1.1), so the ledger cannot drift from
the implementation the way a hand-maintained docs table can.

Usage::

    python tests/benchmarks/parity_census.py            # print the scored table
    python tests/benchmarks/parity_census.py --json     # emit/refresh the artifact
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = Path(__file__).with_name("parity_census.json")
FALLBACK_CENSUS = Path(__file__).with_name("fallback_census.json")

# Modules whose references to an option do not count as blast radius: the
# option's own definition, and the Rust bridge itself (which references an
# option precisely in order to reject it).
_BLAST_EXCLUDE = ("sim_options.py", "exploration/rust_")

# Divergences that are neither a SimOption nor an inspect event nor a
# SimProcedure fallback. Each is a known behavioral gap with a bead; the
# `tier` is the loudness the engine actually gives the user today.
_STRUCTURAL_GAPS = [
    {
        "item": "state.history.actions (SimAction recording)",
        "tier": "raise",
        "why": "TRACK_*_ACTIONS raise at construction (angr-xghv); Rust never emits SimActions.",
        "bead": "angr-op0dn.14.3",
    },
    {
        "item": "solver unsat_core surfacing",
        # NOT counted in the ranked silent surface: its gating option
        # CONSTRAINT_TRACKING_IN_SOLVER is already enumerated there as a silent
        # opt-in option, and double-counting it would inflate the gate.
        "tier": "silent-via-option",
        "why": "CONSTRAINT_TRACKING_IN_SOLVER is silent (neither raise nor warn), so a user who "
        "opts into constraint tracking gets an empty unsat core with no diagnostic. Promoting the "
        "option to raise closes the flip gate; wiring the core through RustSolverProxy is then an "
        "independent feature decision.",
        "bead": "angr-op0dn.14.2",
    },
    {
        "item": "inspect 'constraints' event on native fork-guard adds",
        "tier": "silent",
        "why": "Event IS registerable and IS honored for Python-dispatched adds, but the Rust "
        "engine adds fork-guard constraints natively and does not fire it. A user who "
        "registered the breakpoint sees a subset of the constraints, with no error.",
        "bead": "angr-op0dn.14.4.1",
    },
    {
        "item": "inspect 'vex_lift' event on the native libVEX lift path",
        "tier": "silent",
        "why": "Fires from _cb_lift_block (the Python lift callback). The feature-gated native "
        "in-process lift bypasses that callback and fires nothing. Off by default today, so "
        "this only becomes a live silent gap if/when E2 (angr-z087y) ships enabled.",
        "bead": "angr-op0dn.14.4.2",
    },
    {
        "item": "engine dispatch (tri-state use_rust_engine)",
        "tier": "raise",
        "why": "factory.simulation_manager raises on Rust-ineligible kwargs/options rather than "
        "falling back. Correct today (opt-in), fatal after a flip: the raise IS the regression.",
        "bead": "angr-op0dn.14.5.2",
    },
]


# Options the docs matrix still files under "Ignored — divergence-risk" but
# which the bridge has since been shown NOT to diverge on. The matrix was
# written before these exonerations landed and is stale in the conservative
# direction; each entry below cites the live comment that clears it. They are
# subtracted from the silent surface — otherwise the flip gate counts work that
# does not exist.
_EXONERATED = {
    "BYPASS_UNSUPPORTED_IROP": "Honored transparently: the Rust interpreter's UnsupportedFeature "
    "FallbackStrategy defers the op to Python, where the bypass fires normally "
    "(see the _RAISE_OPTION_NAMES comment block, rust_manager.py).",
    "BYPASS_UNSUPPORTED_IRDIRTY": "Same UnsupportedFeature -> Python fallback path.",
    "BYPASS_UNSUPPORTED_IRCCALL": "Same UnsupportedFeature -> Python fallback path.",
    "BYPASS_UNSUPPORTED_SYSCALL": "Same UnsupportedFeature -> Python fallback path.",
    "UNSUPPORTED_BYPASS_ZERO_DEFAULT": "Modifier only — selects which value Python substitutes when "
    "its bypass fires, and that bypass runs in Python. Honored transparently.",
    "UNSUPPORTED_FORCE_CONCRETIZE": "Modifier only — same as UNSUPPORTED_BYPASS_ZERO_DEFAULT.",
    "SYMBOL_FILL_UNCONSTRAINED_MEMORY": "Vacuously honored: Rust's load_concrete_lazy "
    "(native/angr/src/memory/load.rs) already falls back to a fresh symbolic BVS when "
    "zero_fill_unconstrained is unset, so symbol-fill IS Rust's default for memory. "
    "(The REGISTERS sibling is NOT exonerated — Rust zero-fills registers.)",
    "CONCRETIZE_SYMBOLIC_WRITE_SIZES": "Honored transparently: its only in-tree consumer is "
    "SimFileBase._prep_generic (storage/file.py), and every native write path "
    "(syscalls/write.rs, CGC transmit in syscalls/cgc.rs) falls back to Python on a symbolic "
    "count, so the option fires in Python wherever it can fire at all. (angr-op0dn.14.8)",
    "CGC_NON_BLOCKING_FDS": "Honored in both directions since angr-op0dn.14.8: the native fdwait "
    "(syscalls/cgc.rs NativeFdwaitSyscall) implements exactly the option-is-set behavior and now "
    "falls back to Python when the option is unset, where the Python proc's unconstrained ready "
    "bits are produced.",
    "TRACK_MEMORY_MAPPING": "Vestigial: nothing in angr (or its ecosystem deps) ever *reads* it. "
    "The only in-tree reference adds it to an option set (analyses/identifier/runner.py, "
    "Runner.__init__); no code path branches on its presence, so ignoring it cannot change an "
    "answer. Listed as divergence-risk in the matrix only by association with the TRACK_* family. "
    "(angr-op0dn.14.7)",
    "TRACK_ACTION_HISTORY": "Demoted from raise in angr-fkvt: unlike its TRACK_*_ACTIONS siblings it "
    "does not gate action recording. Its only in-tree consumer (state_plugins/preconstrainer.py) "
    "uses it as a metadata flag whose clear/restore is a vacuous no-op under Rust.",
}


# Divergence-risk options that ship in a *default* mode bundle — so they can
# neither raise nor warn at option-set time without firing for a user who opted
# into nothing — but whose one observable effect is attributable at its
# *consumption* site instead. Counted as "warn" (degraded but attributable), not
# silent: the user does get a diagnostic, just at read time rather than
# construction time.
_CONSUMPTION_WARNED = {
    "TRACK_CONSTRAINT_ACTIONS": "Its only observable effect is the SimActionConstraint stream "
    "(state_plugins/solver.py SimSolver.add), and the Rust engine emits no SimActions. Reading "
    "`state.history.actions` on a Rust-owned state warns once per process via "
    "`_RustOwnedSimStateHistory` (angr/exploration/rust_state_export.py) — so the empty stream is "
    "attributable at the point the user actually consumes it. Ships in the `symbolic` bundle, "
    "hence excluded from _RAISE_OPTION_NAMES / _REJECTED_OPTION_NAMES by construction. "
    "(angr-op0dn.14.7)",
}


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _option_tiers() -> tuple[frozenset[str], frozenset[str]]:
    """Live raise / warn option sets from the bridge.

    The raise tier is both polarities of the hard gate: ``_RAISE_OPTION_NAMES``
    (refused when set) plus ``_REQUIRED_OPTION_NAMES`` (refused when *unset* —
    the inverse-polarity gate, angr-op0dn.14.7). Either way the manager raises
    and the auto-dispatcher routes to Python, so neither is a silent divergence.

    The warn tier is ``_REJECTED_OPTION_NAMES`` (warn-once at manager
    construction) plus ``_CONSUMPTION_WARNED`` (warn-once at the read site).
    """
    from angr.exploration.rust_manager import (
        _RAISE_OPTION_NAMES,
        _REJECTED_OPTION_NAMES,
        _REQUIRED_OPTION_NAMES,
    )

    return (
        frozenset(_RAISE_OPTION_NAMES | _REQUIRED_OPTION_NAMES),
        frozenset(_REJECTED_OPTION_NAMES) | frozenset(_CONSUMPTION_WARNED),
    )


def _docs_matrix() -> dict[str, set[str]]:
    """Parse the SimOption coverage matrix out of ``docs/.../rust_engine.rst``.

    The matrix is the authoritative *semantic* classification (honored /
    inherited / divergence-risk / no-op) — the live ``_RAISE`` / ``_REJECTED``
    sets only tell us how *loud* Rust is about an option, not whether ignoring
    it actually changes an answer. Both halves are needed: an option is a flip
    blocker only when it is divergence-risk AND silent.

    Sections are delimited by their RST headings, and rows are the
    ``   * - ``NAME``,`` cells of each list-table. The honored table's rows spill
    past the heading offset we key on, so honored/inherited names are subtracted
    from the divergence set rather than trusted positionally.
    """
    heads = [
        ("honored", "Honored options"),
        ("inherited", "Inherited (option works because the code path runs in Python)"),
        ("divergence", "Ignored — divergence-risk"),
        ("noop", "Ignored — no-op"),
        ("_end", "Provenance"),
    ]
    lines = _read(REPO / "docs/advanced-topics/rust_engine.rst").splitlines()
    starts = {}
    for key, title in heads:
        for i, line in enumerate(lines):
            if line.strip() == title and i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"~", "-"}:
                starts[key] = i
                break
        if key not in starts:
            raise SystemExit(f"docs drift: SimOption matrix heading {title!r} not found")

    # A row's first cell can name several options at once, wrapped over
    # continuation lines: ``* - ``TRACK_MEMORY_ACTIONS``, ``TRACK_TMP_ACTIONS``,``
    # / ``  ``TRACK_JMP_ACTIONS``,``. So a row is its ``* - `` line plus every
    # following line that is nothing but backticked names — the next ``- ``
    # cell (the prose column) ends it.
    row = re.compile(r"\s*\* - ``[A-Z][A-Z0-9_]+``")
    cont = re.compile(r"^\s+(``[A-Z][A-Z0-9_]+``,?\s*)+$")
    name = re.compile(r"``([A-Z][A-Z0-9_]+)``")
    out: dict[str, set[str]] = {}
    order = [k for k, _ in heads]
    for key, nxt in zip(order, order[1:]):
        chunk = lines[starts[key] : starts[nxt]]
        found: set[str] = set()
        i = 0
        while i < len(chunk):
            if row.match(chunk[i]):
                found.update(name.findall(chunk[i]))
                i += 1
                while i < len(chunk) and cont.match(chunk[i]):
                    found.update(name.findall(chunk[i]))
                    i += 1
                continue
            i += 1
        out[key] = found
    # Positional spill: a name classified honored/inherited wins over a stray
    # re-listing inside the divergence table.
    out["divergence"] -= out["honored"] | out["inherited"]
    out.pop("_end", None)
    return out


def _blast_radius() -> dict[str, int]:
    """Count angr/ modules that read each option name (excluding defn + bridge)."""
    counts: dict[str, int] = defaultdict(int)
    word = re.compile(r"\b([A-Z][A-Z0-9_]{3,})\b")
    for py in (REPO / "angr").rglob("*.py"):
        rel = py.relative_to(REPO).as_posix()
        if any(x in rel for x in _BLAST_EXCLUDE):
            continue
        seen = set(word.findall(_read(py)))
        for name in seen:
            counts[name] += 1
    return counts


def collect() -> dict:
    import angr.sim_options as so
    from angr.exploration.rust_state_proxy import _INSPECT_EVENT_SPECS
    from angr.state_plugins.inspect import EventType

    raise_set, warn_set = _option_tiers()
    blast = _blast_radius()
    matrix = _docs_matrix()

    # Both halves: option *values* (what lives in state.options) and the module
    # attribute names, since a few options are aliases whose attr name differs
    # from their value (COW_STATES = COPY_STATES) and the docs cite the alias.
    all_options = {v for k, v in vars(so).items() if k.isupper() and isinstance(v, str)}
    all_options |= {k for k, v in vars(so).items() if k.isupper() and isinstance(v, str)}
    stale = sorted({n for names in matrix.values() for n in names} - all_options)
    if stale:
        raise SystemExit(f"docs drift: matrix names no longer in sim_options: {stale}")
    ungated = sorted((raise_set | warn_set) - matrix["divergence"])
    if ungated:
        raise SystemExit(f"docs drift: raise/warn options missing from the divergence table: {ungated}")

    # Which mode bundles carry each option. `symbolic` is what entry_state()
    # ships by default, so a silent divergence-risk option living there fires
    # for every single user after a flip — that is the top of the ranking.
    bundles = {name: set(opts) for name, opts in so.modes.items()}

    bogus = sorted(set(_EXONERATED) - matrix["divergence"])
    if bogus:
        raise SystemExit(f"docs drift: exonerated options no longer in the divergence table: {bogus}")

    options = []
    for name in sorted(matrix["divergence"]):
        if name in raise_set:
            tier = "raise"
        elif name in warn_set:
            tier = "warn"
        elif name in _EXONERATED:
            tier = "exonerated"
        else:
            tier = "silent"
        options.append(
            {
                "option": name,
                "tier": tier,
                "in_default_modes": sorted(m for m, opts in bundles.items() if name in opts),
                "py_consumer_modules": blast.get(name, 0),
            }
        )

    # An option only diverges if some Python code path actually reads it. Zero
    # consumers => vestigial (defined, never consulted) => harmless whatever the
    # matrix says. Of the rest, the flip-critical ones are those that ride in on
    # a *default* mode bundle: the user never opted in, so "just don't set it"
    # is not advice they can act on.
    silent_live = [o for o in options if o["tier"] == "silent" and o["py_consumer_modules"] > 0]
    silent_default = [o for o in silent_live if o["in_default_modes"]]
    silent_opted = [o for o in silent_live if not o["in_default_modes"]]
    # `symbolic` is the bundle `factory.entry_state()` hands out with no
    # argument, so a silent option living *there* is the true 100%-of-users
    # case. The other bundles (fastpath / static / tracing) are reached only by
    # passing mode=, which is itself an opt-in.
    silent_entry_state = [o for o in silent_default if "symbolic" in o["in_default_modes"]]

    supported_events = set(_INSPECT_EVENT_SPECS)
    all_events = {e.value if hasattr(e, "value") else str(e) for e in EventType}
    unsupported_events = sorted(e for e in all_events if e not in supported_events)

    fallbacks = {}
    if FALLBACK_CENSUS.exists():
        fc = json.loads(_read(FALLBACK_CENSUS))
        summary = fc.get("summary", {})
        names = fc.get("names", {})
        fallbacks = {
            "name_classes": summary.get("name_classes", {}),
            "site_classes": summary.get("site_classes", {}),
            "unlogged_degraded_sites": len(summary.get("unlogged_degraded_sites", [])),
            "flip_blocking_names": sorted(n for n, v in names.items() if v.get("semantics") == "flip-blocking"),
        }

    return {
        "bead": "angr-op0dn.8",
        "options": options,
        "option_tier_counts": {
            t: sum(1 for o in options if o["tier"] == t) for t in ("raise", "warn", "exonerated", "silent")
        },
        "exonerated_options": _EXONERATED,
        "consumption_warned_options": _CONSUMPTION_WARNED,
        "silent_live_options": [o["option"] for o in silent_live],
        "silent_default_mode_options": [o["option"] for o in silent_default],
        "silent_entry_state_options": [o["option"] for o in silent_entry_state],
        "silent_opt_in_options": [o["option"] for o in silent_opted],
        "matrix_counts": {k: len(v) for k, v in matrix.items()},
        "inspect": {
            "supported": sorted(supported_events),
            "unsupported_raises": unsupported_events,
        },
        "fallbacks": fallbacks,
        "structural_gaps": _STRUCTURAL_GAPS,
    }


def _rank(census: dict) -> list[dict]:
    """The flip gate: every silent divergence, worst blast radius first.

    Order is blast radius, not effort: a divergence-risk option that ships in a
    default mode bundle fires for a user who opted into nothing, so it outranks
    one the user had to go out of their way to set.
    """
    by_name = {o["option"]: o for o in census["options"]}
    ranked = []
    for name in census["silent_default_mode_options"]:
        modes = ", ".join(by_name[name]["in_default_modes"])
        ranked.append(
            {
                "item": f"SimOption {name}",
                "fires_for": f"every user of the `{modes}` mode bundle — no opt-in required",
                "tier": "silent",
            }
        )
    for name in census["silent_opt_in_options"]:
        ranked.append(
            {
                "item": f"SimOption {name}",
                "fires_for": "users who explicitly set it (opt-in; a warn-once would make it attributable)",
                "tier": "silent",
            }
        )
    fb = census.get("fallbacks", {})
    for name in fb.get("flip_blocking_names", []):
        ranked.append(
            {
                "item": f"SimProcedure/syscall fallback {name}",
                "fires_for": "any binary that calls it",
                "tier": "silent",
            }
        )
    flip_sites = fb.get("site_classes", {}).get("flip-blocking", 0)
    if flip_sites:
        ranked.append(
            {
                "item": f"{flip_sites} flip-blocking bridge sites (M6.5a site census)",
                "fires_for": "whichever bench trips them — unattributable without a log record",
                "tier": "silent",
            }
        )
    for gap in census["structural_gaps"]:
        if gap["tier"] == "silent":
            ranked.append({"item": gap["item"], "fires_for": gap["bead"], "tier": "silent"})
    return ranked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="write the JSON artifact")
    args = ap.parse_args()

    census = collect()
    counts = census["option_tier_counts"]

    print("=== SimOption divergence-risk rows, by loudness ===")
    print(f"  matrix: {census['matrix_counts']}")
    print(f"  raise (dispatcher-handleable, NOT a blocker): {counts['raise']}")
    print(f"  warn-once (degraded but attributable):        {counts['warn']}")
    print(f"  exonerated (matrix stale; does not diverge):  {counts['exonerated']}")
    print(f"  silent:                                       {counts['silent']}")
    print(f"    live (>=1 Python consumer):     {len(census['silent_live_options'])}")
    print(f"    of those, in a default bundle:  {census['silent_default_mode_options']}")
    print(f"    of those, in `symbolic` (= plain entry_state()): {census['silent_entry_state_options']}")
    print(f"    of those, opt-in only:          {census['silent_opt_in_options']}")
    print()
    print("=== state.inspect ===")
    print(f"  honored: {len(census['inspect']['supported'])}")
    print(f"  raises on registration (dispatcher-handleable): {len(census['inspect']['unsupported_raises'])}")
    print()
    fb = census.get("fallbacks", {})
    if fb:
        print("=== SimProcedure/syscall fallbacks (M6.5a census) ===")
        print(f"  name classes: {fb['name_classes']}")
        print(f"  site classes: {fb['site_classes']}")
        print(f"  cat-(b) sites degrading with no log record: {fb['unlogged_degraded_sites']}")
    print()
    print("=== FLIP GATE: the silent surface (ranked) ===")
    ranked = _rank(census)
    for i, row in enumerate(ranked, 1):
        print(f"  {i:2d}. {row['item']}")
        print(f"      fires for: {row['fires_for']}")
    print()
    print(f"  silent surface size: {len(ranked)}  (flip gate is: 0)")

    if args.json:
        census["flip_gate_ranked"] = ranked
        ARTIFACT.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {ARTIFACT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
