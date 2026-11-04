#!/usr/bin/env python3
"""Per-step cost of building a stash export SimState (angr-mx00p.136.4).

A SimulationManager backend whose stashes hold export SimStates would build
one per successor per step. This measures that build against the step it
follows: for each example, step a ``RustExplorationManager`` one block at a
time and, after each step, export every active state the way
``_materialize_single_state``'s last-resort path does (``export_state`` FFI,
then ``_snapshot_to_angr``), timing each phase of ``_snapshot_to_angr``
separately. It also times ``copy()`` of each plugin ``_restore_plugins_to_state``
copies from the template, which bounds what a lazy fill on first
``get_plugin`` could save. ``_export_blank_state`` is the copy of the
manager's cached blank template that replaced ``blank_state``
(angr-mx00p.136.28).

Each example runs in a child process under a 4 GB ``RLIMIT_AS``.

Usage::

    python tests/benchmarks/profile_export_cost.py [--steps N] [--max-states K] [example ...]
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
import os
import resource
import statistics
import sys
import time

EXAMPLES_DIR = os.path.expanduser("~/repos/angr-examples/examples")
DEFAULT_EXAMPLES = ("fauxware", "ais3_crackme", "defcamp_r100")
BINARIES = {"fauxware": "fauxware", "ais3_crackme": "ais3_crackme", "defcamp_r100": "r100"}

# The phases of ``_snapshot_to_angr``, by the manager method that runs each.
PHASE_METHODS = (
    "_export_blank_state",
    "_load_snapshot_registers",
    "_restore_plugins_to_state",
    "_inject_rust_stdout",
    "_inject_rust_stdin",
    "_inject_rust_fds",
    "_inject_rust_cwd",
    "_inject_rust_getopt",
    "_inject_rust_closed_files",
    "_install_rust_history_chain",
    "_bind_export",
)
TEMPLATE_PLUGINS = ("posix", "libc", "heap", "fs", "log")


def _initial_state(proj, example):
    from angr import claripy

    if example == "ais3_crackme":
        return proj.factory.entry_state(args=[proj.filename, claripy.BVS("argv1", 100 * 8)])
    return proj.factory.entry_state()


def _child(out_q, example, steps, max_states):
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, resource.getrlimit(resource.RLIMIT_AS)[1]))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import angr
    from angr.exploration import RustExplorationManager

    proj = angr.Project(os.path.join(EXAMPLES_DIR, example, BINARIES[example]), auto_load_libs=False)
    mgr = RustExplorationManager(proj, [_initial_state(proj, example)])

    times: dict[str, list[int]] = {name: [] for name in ("step", "export_state", *PHASE_METHODS)}
    times["export_total"] = []
    plugin_copy: dict[str, list[int]] = {name: [] for name in TEMPLATE_PLUGINS}

    def timed(name, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                times[name].append(time.perf_counter_ns() - t0)

        return wrapper

    # Instance attributes shadow the mixin methods, so only this manager's
    # calls are timed.
    for name in PHASE_METHODS:
        setattr(mgr, name, timed(name, getattr(mgr, name)))

    exports_per_step = []
    for _ in range(steps):
        t0 = time.perf_counter_ns()
        mgr.step(n=1)
        times["step"].append(time.perf_counter_ns() - t0)
        ids = list(mgr._rust_mgr.get_state_ids("active"))[:max_states]
        if not ids:
            break
        exports_per_step.append(len(ids))
        for sid in ids:
            t0 = time.perf_counter_ns()
            snapshot = mgr._rust_mgr.export_state(sid)
            t1 = time.perf_counter_ns()
            mgr._snapshot_to_angr(snapshot, mirror=False)
            t2 = time.perf_counter_ns()
            times["export_state"].append(t1 - t0)
            times["export_total"].append(t2 - t0)
            template = mgr._find_plugin_template_state(sid, snapshot.parent_id)
            if template is None:
                continue
            for plugin_name in TEMPLATE_PLUGINS:
                if template.has_plugin(plugin_name):
                    plugin = getattr(template, plugin_name)
                    t0 = time.perf_counter_ns()
                    plugin.copy({})
                    plugin_copy[plugin_name].append(time.perf_counter_ns() - t0)

    def summary(samples):
        if not samples:
            return None
        return {"n": len(samples), "median_us": statistics.median(samples) / 1e3, "total_ms": sum(samples) / 1e6}

    out_q.put(
        {
            "example": example,
            "exports_per_step": exports_per_step,
            "phases": {name: summary(samples) for name, samples in times.items()},
            "plugin_copy": {name: summary(samples) for name, samples in plugin_copy.items()},
        }
    )


def _run(example, steps, max_states):
    ctx = multiprocessing.get_context("spawn")
    out_q = ctx.Queue()
    proc = ctx.Process(target=_child, args=(out_q, example, steps, max_states))
    proc.start()
    proc.join(600)
    if proc.is_alive():
        proc.kill()
        return {"example": example, "error": "timeout"}
    if out_q.empty():
        return {"example": example, "error": f"child exited {proc.exitcode}"}
    return out_q.get()


def _print(result):
    print(f"== {result['example']}")
    if "error" in result:
        print(f"   error: {result['error']}")
        return
    print(f"   active states exported per step: {result['exports_per_step']}")
    print(f"   {'phase':32} {'n':>5} {'median_us':>11} {'total_ms':>10}")
    for group in ("phases", "plugin_copy"):
        for name, s in result[group].items():
            if s is not None:
                label = name if group == "phases" else f"copy:{name}"
                print(f"   {label:32} {s['n']:>5} {s['median_us']:>11.1f} {s['total_ms']:>10.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("examples", nargs="*", default=list(DEFAULT_EXAMPLES))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-states", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = [_run(example, args.steps, args.max_states) for example in args.examples]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            _print(result)


if __name__ == "__main__":
    main()
