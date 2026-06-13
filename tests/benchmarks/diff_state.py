"""Differential state snapshot/diff utilities for run_single.py --diff-state.

The harness wraps a SimulationManager / RustExplorationManager so that after
every Nth step() call a snapshot of every stash is recorded. Two runs (Python
and Rust) are then compared step-by-step. Divergence in registers, address,
or constraint count fails before the example reaches its final solve.

Snapshots are intentionally cheap: registers concretized via the solver,
constraint count, satisfiability. Per-byte memory diffing would force a full
SimState export every step on the Rust side and balloon the snapshot file —
the tradeoff here is to catch register/PC drift quickly and let the existing
final-correctness check guard memory.
"""

from __future__ import annotations

import json
import types

# Architecturally-meaningful registers we attempt to capture per state.
# Missing names are silently skipped (e.g. amd64 regs on x86).
_DIFF_REG_NAMES = (
    # amd64
    "rip",
    "rsp",
    "rbp",
    "rax",
    "rbx",
    "rcx",
    "rdx",
    "rsi",
    "rdi",
    "r8",
    "r9",
    "r10",
    "r11",
    "r12",
    "r13",
    "r14",
    "r15",
    # x86
    "eip",
    "esp",
    "ebp",
    "eax",
    "ebx",
    "ecx",
    "edx",
    "esi",
    "edi",
    # arm
    "pc",
    "sp",
    "lr",
    "r0",
    "r1",
    "r2",
    "r3",
    "r4",
    "r5",
    "r6",
    "r7",
)

_STASH_NAMES = ("active", "found", "deadended", "avoid", "errored", "unconstrained")


def _capture_state(state) -> dict:
    """Collect a cheap signature of a single SimState."""
    out: dict = {}
    try:
        ip = state.regs.ip
        if ip.symbolic:
            out["addr"] = "<sym>"
        else:
            out["addr"] = state.solver.eval(ip)
    except Exception as e:
        out["addr"] = f"<err:{type(e).__name__}>"

    regs = {}
    for name in _DIFF_REG_NAMES:
        if not hasattr(state.regs, name):
            continue
        try:
            val = getattr(state.regs, name)
        except Exception:
            continue
        try:
            if val.symbolic:
                regs[name] = "<sym>"
            else:
                regs[name] = state.solver.eval(val)
        except Exception:
            regs[name] = "<err>"
    out["regs"] = regs

    try:
        out["constraints_count"] = len(state.solver.constraints)
    except Exception:
        out["constraints_count"] = -1

    try:
        out["satisfiable"] = bool(state.satisfiable())
    except Exception:
        out["satisfiable"] = None

    try:
        out["history_depth"] = state.history.depth
    except Exception:
        pass

    return out


def _capture_manager(manager, step_idx: int) -> dict:
    """Snapshot every stash of the manager."""
    snapshot = {"step": step_idx, "stashes": {}}
    for stash in _STASH_NAMES:
        try:
            states = list(getattr(manager, stash, []) or [])
        except Exception:
            states = []
        snapshot["stashes"][stash] = [_capture_state(s) for s in states]
    return snapshot


def install_snapshotter(manager, snapshots: list, interval: int = 1, max_snapshots: int = 200):
    """Wrap manager.step so each invocation appends a snapshot.

    snapshots: list to append to (mutated in place)
    interval: take a snapshot every Nth step call
    max_snapshots: stop snapshotting after this many entries (avoids OOM on
        long runs)

    For RustExplorationManager, .explore() and .run() normally batch through
    Rust without going via .step(); we replace them with a step(1) loop so
    snapshots fire per-step at the same granularity as the Python SimulationManager.
    """
    original_step = manager.step
    counter = [0]
    # Take a baseline snapshot before any stepping so we can diff initial state.
    snapshots.append(_capture_manager(manager, 0))

    def _patched_step(self, *args, **kwargs):
        result = original_step(*args, **kwargs)
        counter[0] += 1
        if (counter[0] % interval) == 0 and len(snapshots) < max_snapshots:
            snapshots.append(_capture_manager(self, counter[0]))
        return result

    # Bind as a method so HookSet.install_hooks (used by exploration techniques
    # like Explorer) can find self.func.__self__ on the resulting HookedMethod.
    manager.step = types.MethodType(_patched_step, manager)

    if hasattr(manager, "_rust_mgr"):
        _patch_rust_explore_to_single_step(manager, max_snapshots)


def _patch_rust_explore_to_single_step(manager, max_snapshots: int):
    """Override RustExplorationManager.{explore,run} with step(1) loops so
    the snapshotter on .step() fires per-step. Honors find/avoid addresses
    by routing them through the existing rust_mgr setters."""
    import time

    def _drive(stash, num_find, until, timeout, max_steps):
        start = time.time()
        steps = 0
        # Hard cap stepping at max_snapshots so a long-running benchmark
        # can't loop forever in diff mode.
        hard_cap = max_steps if max_steps is not None else max_snapshots * 4
        while steps < hard_cap:
            if timeout is not None and (time.time() - start) > timeout:
                break
            if not manager._rust_mgr.has_active_states():
                break
            try:
                if manager._found_count() >= num_find:
                    break
            except Exception:
                pass
            if until is not None and until(manager):
                break
            manager.step(1)
            steps += 1
            # Mirror _explore_with_predicates: callable find/avoid
            # predicates aren't observable from inside Rust, so evaluate
            # them between step(1) calls and route matches into
            # found/avoid stashes manually. Without this, csgames2018
            # and sym-write never see their stdout-based find predicate
            # fire in --diff-state mode.
            if (
                getattr(manager, "_find_predicate", None) is not None
                or getattr(manager, "_avoid_predicate", None) is not None
            ):
                try:
                    manager._evaluate_predicates_on_active()
                except Exception:
                    pass
        return manager

    def diff_explore(find=None, avoid=None, num_find=1, until=None, timeout=None, max_steps=None, **kwargs):
        if not hasattr(manager, "_find_predicate"):
            manager._find_predicate = None
        if not hasattr(manager, "_avoid_predicate"):
            manager._avoid_predicate = None
        if find is not None:
            find_addrs = manager._extract_addrs(find)
            manager._rust_mgr.set_find_addrs(find_addrs)
            manager._rust_mgr.set_find_needs_python(callable(find))
            manager._find_predicate = find if callable(find) else None
        if avoid is not None:
            avoid_addrs = manager._extract_addrs(avoid)
            manager._rust_mgr.set_avoid_addrs(avoid_addrs)
            manager._rust_mgr.set_avoid_needs_python(callable(avoid))
            manager._avoid_predicate = avoid if callable(avoid) else None
        manager._rust_mgr.set_num_find(num_find)
        return _drive("active", num_find, until, timeout, max_steps)

    def diff_run(**kwargs):
        n = kwargs.pop("n", None)
        until = kwargs.pop("until", None)
        timeout = kwargs.pop("timeout", None)
        step_func = kwargs.pop("step_func", None)
        # In diff mode we treat run like a step loop. Honor step_func by
        # calling it after each step.
        original_step = manager.step
        if step_func is not None:

            def _step_with_cb(self, *a, **kw):
                r = original_step(*a, **kw)
                step_func(self)
                return r

            manager.step = types.MethodType(_step_with_cb, manager)
        try:
            return _drive("active", float("inf"), until, timeout, n)
        finally:
            if step_func is not None:
                manager.step = original_step

    manager.explore = diff_explore
    manager.run = diff_run


def _state_signature(state: dict) -> str:
    """Stable string used to match states across engines within a stash.

    We deliberately use only the program counter — `history_depth` differs
    between engines because the Rust manager auto-skips _start->main during
    init, so depth at main is 0 in Rust vs ~17 in Python. Using addr alone
    lets us still match same-PC states across engines.
    """
    return f"{state.get('addr')}"


def _diff_stash(stash_name: str, py_states: list, rs_states: list) -> list:
    """Return human-readable lines describing differences in this stash."""
    diffs: list = []
    if len(py_states) != len(rs_states):
        diffs.append(f"  [{stash_name}] state count mismatch: py={len(py_states)} rust={len(rs_states)}")

    py_by_sig: dict = {}
    rs_by_sig: dict = {}
    for s in py_states:
        py_by_sig.setdefault(_state_signature(s), []).append(s)
    for s in rs_states:
        rs_by_sig.setdefault(_state_signature(s), []).append(s)

    only_py = sorted(set(py_by_sig) - set(rs_by_sig))
    only_rs = sorted(set(rs_by_sig) - set(py_by_sig))
    for sig in only_py:
        diffs.append(f"  [{stash_name}] state only in python: {sig}")
    for sig in only_rs:
        diffs.append(f"  [{stash_name}] state only in rust:   {sig}")

    for sig in sorted(set(py_by_sig) & set(rs_by_sig)):
        py_list = py_by_sig[sig]
        rs_list = rs_by_sig[sig]
        # Pair up by index in their original order.
        for py_s, rs_s in zip(py_list, rs_list):
            for key in ("constraints_count", "satisfiable"):
                if py_s.get(key) != rs_s.get(key):
                    diffs.append(f"  [{stash_name}] sig={sig} {key}: py={py_s.get(key)} rust={rs_s.get(key)}")
            py_regs = py_s.get("regs", {})
            rs_regs = rs_s.get("regs", {})
            common_keys = sorted(set(py_regs) & set(rs_regs))
            for rk in common_keys:
                pv = py_regs[rk]
                rv = rs_regs[rk]
                if pv != rv:
                    pv_s = hex(pv) if isinstance(pv, int) else str(pv)
                    rv_s = hex(rv) if isinstance(rv, int) else str(rv)
                    diffs.append(f"  [{stash_name}] sig={sig} reg {rk}: py={pv_s} rust={rv_s}")

    return diffs


def _align_snapshots(py_snaps: list, rs_snaps: list) -> tuple:
    """Skip past leading snapshots where the engines are not at a comparable
    point. The Rust manager auto-steps from _start through __libc_start_main
    to main during initialization (rust_manager._step_python_to_main), so its
    snapshot 0 is typically at main while Python's snapshot 0 is at _start.
    We advance Python past its initialization until its first active state's
    address matches Rust's first active state address, then begin diffing.
    Returns (py_offset, rs_offset, alignment_note)."""
    if not py_snaps or not rs_snaps:
        return 0, 0, None

    def _first_active_addr(snap):
        actives = snap.get("stashes", {}).get("active", [])
        if not actives:
            return None
        return actives[0].get("addr")

    rs_target = _first_active_addr(rs_snaps[0])
    if rs_target is None:
        return 0, 0, None
    if _first_active_addr(py_snaps[0]) == rs_target:
        return 0, 0, None
    # Search Python snapshots for a state at rs_target.
    for idx, snap in enumerate(py_snaps):
        actives = snap.get("stashes", {}).get("active", [])
        if any(s.get("addr") == rs_target for s in actives):
            return (
                idx,
                0,
                (
                    f"aligned: python advanced past {idx} init step(s) to reach "
                    f"rust starting addr "
                    f"{hex(rs_target) if isinstance(rs_target, int) else rs_target}"
                ),
            )
    # Try the symmetric case: maybe Python is ahead of Rust.
    py_target = _first_active_addr(py_snaps[0])
    if py_target is not None and py_target != rs_target:
        for idx, snap in enumerate(rs_snaps):
            actives = snap.get("stashes", {}).get("active", [])
            if any(s.get("addr") == py_target for s in actives):
                return (
                    0,
                    idx,
                    (
                        f"aligned: rust advanced past {idx} init step(s) to reach "
                        f"python starting addr "
                        f"{hex(py_target) if isinstance(py_target, int) else py_target}"
                    ),
                )
    return 0, 0, "no common starting address found; comparing from index 0"


def compare_snapshots(py_snapshots: list, rs_snapshots: list, max_report: int = 5) -> dict:
    """Diff two snapshot lists step-by-step.

    Returns a dict:
      ok: bool — True if no divergence detected
      first_divergent_step: int | None
      summary: list[str] — human-readable lines
      step_count: tuple(py_steps, rs_steps)
    """
    summary: list = []
    py_offset, rs_offset, align_note = _align_snapshots(py_snapshots, rs_snapshots)
    if align_note:
        summary.append(align_note)
    py_snapshots = py_snapshots[py_offset:]
    rs_snapshots = rs_snapshots[rs_offset:]
    common = min(len(py_snapshots), len(rs_snapshots))
    if len(py_snapshots) != len(rs_snapshots):
        summary.append(
            f"step count mismatch: python took {len(py_snapshots)} snapshots "
            f"(after alignment), rust took {len(rs_snapshots)}; comparing first {common}"
        )

    first_div = None
    diverged_steps = 0
    for i in range(common):
        py_snap = py_snapshots[i]
        rs_snap = rs_snapshots[i]
        step_diffs: list = []
        for stash in _STASH_NAMES:
            py_states = py_snap["stashes"].get(stash, [])
            rs_states = rs_snap["stashes"].get(stash, [])
            if not py_states and not rs_states:
                continue
            step_diffs.extend(_diff_stash(stash, py_states, rs_states))
        if step_diffs:
            if first_div is None:
                first_div = py_snap["step"]
            diverged_steps += 1
            if diverged_steps <= max_report:
                summary.append(f"step {py_snap['step']}: divergence:")
                summary.extend(step_diffs)

    if diverged_steps > max_report:
        summary.append(f"... plus {diverged_steps - max_report} more divergent steps suppressed")

    return {
        "ok": first_div is None,
        "first_divergent_step": first_div,
        "diverged_steps": diverged_steps,
        "summary": summary,
        "step_count": (len(py_snapshots), len(rs_snapshots)),
    }


def snapshots_to_json(snapshots: list) -> str:
    return json.dumps(snapshots, default=str)


def snapshots_from_json(s: str) -> list:
    return json.loads(s)
