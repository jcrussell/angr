"""ExplorationTechnique integration for the Rust exploration manager.

Provides use_technique(), remove_technique(), and the per-step filter/complete
callback dispatching that routes through RustStateProxy objects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from angr.misc.hookset import HookSet

if TYPE_CHECKING:
    from angr.exploration.rust_manager import RustExplorationManager

l = logging.getLogger(name=__name__)


_NATIVE_STEP_TECH_NAMES = {
    # Techniques whose step()/successors() effect is provided natively by
    # the Rust manager. Their Python step() impls should NOT be dispatched
    # again or we'd double-count effects.
    "DFS",
    "DepthFirst",
    "BFS",
    "BreadthFirst",
    "Explorer",
    "LengthLimiter",
    "Timeout",
    "CheckUniqueness",
}


# Techniques whose Python filter() maintains a monotonic internal set, so
# their filter() must be evaluated AT MOST ONCE per state id. Re-running them
# on a state-signature change would corrupt that set (e.g. CheckUniqueness's
# seen-key set prunes a state the second time it is presented). All other
# techniques' filter()s are re-run when a state's (addr, stdout_len) signature
# changes, matching SimulationManager's per-step filter contract (angr-j1ue).
# Note: CheckUniqueness is normally handled natively in Rust (_native_uniqueness)
# and skipped in the Python loop entirely; it only reaches the Python filter
# path as a fallback when register detection fails — this guard covers that.
_MONOTONIC_FILTER_TECHNIQUES = frozenset({"CheckUniqueness"})


def _has_dispatched_step_hook(tech) -> bool:
    """Return True iff `tech` overrides step() AND isn't natively handled."""
    tech_name = type(tech).__name__
    if tech_name in _NATIVE_STEP_TECH_NAMES:
        return False
    return tech._is_overridden("step")


def manager_has_step_hooks(mgr: RustExplorationManager) -> bool:
    """True iff any active technique has a non-native step() hook to dispatch."""
    return any(_has_dispatched_step_hook(t) for t in mgr._active_techniques)


def dispatch_step_with_hooks(mgr: RustExplorationManager, batch_size, stash="active"):
    """Run one step batch under ExplorationTechnique step() hook composition.

    Builds a fresh RustSimulationManagerProxy per dispatch, hooks each
    technique's overridden step() onto the proxy via HookSet (LIFO compose,
    matching the Python SimulationManager), and lets the proxy dispatch into
    the Rust engine through `_step_callback`.

    Returns the ExplorationEvent produced by the innermost Rust run() call,
    or None if no technique ever invoked simgr.step() (the technique consumed
    the batch without delegating — a possibility for stash-only techniques
    like StubStasher).
    """
    from angr.exploration.rust_state_proxy import RustSimulationManagerProxy

    captured_event = []

    proxy = RustSimulationManagerProxy(
        mgr._rust_mgr,
        project=mgr._project,
        stdin_vars=getattr(mgr, "_stdin_vars", None),
        stdout_tracker=getattr(mgr, "_stdout_tracker", {}),
        python_mgr=mgr,
    )

    def base_step(stash="active", **kwargs):
        event = mgr._rust_mgr.run(batch_size)
        captured_event.append(event)

    proxy._step_callback = base_step

    # Install step hooks in registration order. HookSet.install_hooks
    # appends to `pending`, and HookedMethod pops the LAST pending hook
    # first — so the last-registered tech wraps the earlier ones (matches
    # SimulationManager composition).
    for tech in mgr._active_techniques:
        if _has_dispatched_step_hook(tech):
            try:
                HookSet.install_hooks(proxy, step=tech.step)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: hook install failed; skip this
                # tech's step() and continue with the others.
                l.warning(
                    "Failed to install step() hook for %s: %s",
                    type(tech).__name__,
                    e,
                )

    try:
        proxy.step(stash=stash)
    except Exception as e:
        # cat-(b) FALLBACK WITH LOSS: a technique's step() raised. We still
        # need to advance Rust so the run loop makes progress; fall back to a
        # direct run unless the base impl already fired.
        l.warning(
            "Technique step() hook raised %s: %s; falling back to direct run.",
            type(e).__name__,
            e,
        )
        if not captured_event:
            captured_event.append(mgr._rust_mgr.run(batch_size))

    return captured_event[0] if captured_event else None


def use_technique(mgr: RustExplorationManager, technique, **kwargs):
    """Apply an exploration technique to the Rust manager.

    Techniques are tracked and their setup methods are called.
    Common techniques like DFS, BFS, and LoopSeer have basic support.

    Args:
        mgr: The RustExplorationManager instance.
        technique: An ExplorationTechnique instance.

    Returns:
        The technique, for chaining.
    """
    tech_name = type(technique).__name__

    # Track the technique
    mgr._active_techniques.append(technique)

    # Call setup if available
    try:
        if hasattr(technique, "setup"):
            technique.setup(mgr)
            l.debug(f"Called setup() on technique {tech_name}")
    except Exception as e:
        # cat-(b) FALLBACK WITH LOSS: technique setup failed; we keep going
        # with a partially-initialized technique. Warn so the user sees it.
        l.warning(f"Technique {tech_name} setup failed: {e}")

    # Handle specific technique types
    # DFS: Use depth-first state selection (LIFO)
    if tech_name == "DFS" or tech_name == "DepthFirst":
        try:
            mgr._rust_mgr.set_state_selection_lifo()
            l.debug("Enabled DFS (LIFO) state selection")
        except AttributeError:
            # cat-(a) EXPECTED CONTROL FLOW: probing for optional Rust API.
            l.debug("DFS technique registered (LIFO not natively supported)")

    # BFS: Use breadth-first state selection (FIFO) - default behavior
    elif tech_name == "BFS" or tech_name == "BreadthFirst":
        try:
            mgr._rust_mgr.set_state_selection_fifo()
            l.debug("Enabled BFS (FIFO) state selection")
        except AttributeError:
            # cat-(a) EXPECTED CONTROL FLOW: probing for optional Rust API.
            l.debug("BFS technique registered (default FIFO selection)")

    # LoopSeer / LocalLoopSeer: bound-only loop limiting via the native
    # LoopBound technique. The Rust run loop never dispatches the Python
    # successors()/step_state() hooks these techniques rely on, so without
    # this translation a registered bound silently does nothing. We map the
    # bound onto register_loop_bound, which moves over-bound states to the
    # technique's discard_stash using a back-edge heuristic (any block
    # repeated more than `bound` times in a state's history) rather than a
    # CFG-derived trip counter.
    elif tech_name in ("LoopSeer", "LocalLoopSeer"):
        bound = getattr(technique, "bound", None)
        discard_stash = getattr(technique, "discard_stash", "spinning") or "spinning"
        if bound is None:
            # Bound-less LoopSeer only records trip counts (no enforcement);
            # the native loop limiter has nothing to enforce.
            l.debug("%s registered without a bound; no native limiter applied", tech_name)
        elif getattr(technique, "bound_reached", None) is not None:
            # A bound_reached callback can't be invoked from the Rust loop, so
            # honoring the bound natively would silently skip the user's hook.
            l.warning(
                "%s has a bound_reached callback, which the Rust engine cannot "
                "invoke; native loop bound not applied (states will not be limited)",
                tech_name,
            )
        else:
            try:
                mgr._rust_mgr.register_loop_bound(int(bound), discard_stash)
                l.debug(
                    "Registered native loop bound=%d, discard_stash=%r for %s",
                    bound,
                    discard_stash,
                    tech_name,
                )
            except AttributeError:
                # cat-(a) EXPECTED CONTROL FLOW: probing for optional Rust API.
                l.debug("%s registered (native loop bound not supported)", tech_name)

    # Explorer: Extract find/avoid addresses
    elif tech_name == "Explorer":
        find_addrs = []
        avoid_addrs = []

        # Extract find addresses directly from the technique
        raw_find = getattr(technique, "find", None)
        if raw_find is not None:
            if isinstance(raw_find, int):
                find_addrs = [raw_find]
            elif isinstance(raw_find, (list, tuple, set)):
                find_addrs = [a for a in raw_find if isinstance(a, int)]
            # Callable find predicates can't be turned into addresses

        # Extract avoid addresses directly from the technique
        raw_avoid = getattr(technique, "avoid", None)
        if raw_avoid is not None:
            if isinstance(raw_avoid, int):
                avoid_addrs = [raw_avoid]
            elif isinstance(raw_avoid, (list, tuple, set)):
                avoid_addrs = [a for a in raw_avoid if isinstance(a, int)]

        # Fallback: try _extra_stop_points with mock state for callable find/avoid
        if not find_addrs and not avoid_addrs:
            find_func = getattr(technique, "find", None)
            avoid_func = getattr(technique, "avoid", None)
            stop_points = getattr(technique, "_extra_stop_points", set())
            if stop_points and callable(find_func) and callable(avoid_func):

                class _MockState:
                    def __init__(self, addr):
                        self.addr = addr
                        self._ip = addr
                        self.regs = type("regs", (), {"ip": addr})()

                    def block(self, *a, **kw):
                        return type("block", (), {"size": 1})()

                for addr in stop_points:
                    mock = _MockState(addr)
                    try:
                        if find_func(mock):
                            find_addrs.append(addr)
                            continue
                    except Exception:
                        # cat-(a) EXPECTED CONTROL FLOW: user-supplied predicate
                        # may reject mock state; fall through to avoid probe.
                        pass
                    try:
                        if avoid_func(mock):
                            avoid_addrs.append(addr)
                    except Exception:
                        # cat-(a) EXPECTED CONTROL FLOW: user-supplied predicate
                        # may reject mock state; addr is simply not classified.
                        pass

        if find_addrs:
            mgr._rust_mgr.set_find_addrs(find_addrs)
        if avoid_addrs:
            mgr._rust_mgr.set_avoid_addrs(avoid_addrs)
            mgr._has_technique_avoids = True
        num_find = getattr(technique, "num_find", 1)
        mgr._rust_mgr.set_num_find(num_find)
        l.debug(f"Explorer technique: find={[hex(a) for a in find_addrs]}, avoid={len(avoid_addrs)} addrs")

    # CheckUniqueness: Native register uniqueness filter in Rust
    elif tech_name == "CheckUniqueness":
        # Detect register list from the technique's filter method
        # The common pattern checks specific register names
        regs = getattr(technique, "_register_names", None)
        if regs is None:
            # Try to detect from source inspection — common grub pattern
            # uses ('eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp', 'eip')
            arch = mgr._project.arch if mgr._project else None
            if arch and arch.name in ("X86",):
                regs = ["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp", "eip"]
            elif arch and arch.name in ("AMD64", "X86_64"):
                regs = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "rip"]
            else:
                regs = None

        if regs:
            try:
                mgr._rust_mgr.register_uniqueness_filter(regs)
                # Mark as natively handled so filter() is skipped in Python
                technique._native_uniqueness = True
                l.debug(f"CheckUniqueness registered natively with {len(regs)} registers")
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: native registration failed; the
                # Python technique.filter() path still runs (just slower).
                l.debug(f"Failed to register native uniqueness: {e}")
        else:
            l.debug("CheckUniqueness registered (Python fallback)")

    # LengthLimiter: Limit path length (block count) — native Rust implementation
    elif tech_name == "LengthLimiter":
        max_length = getattr(technique, "_max_length", None)
        drop = getattr(technique, "_drop", False)
        if max_length is not None:
            try:
                mgr._rust_mgr.register_length_limiter(max_length, drop)
                technique._native_length_limiter = True
                l.debug(f"LengthLimiter registered natively (max_length={max_length}, drop={drop})")
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: native registration failed; the
                # Python technique._filter() path still runs (just slower).
                l.debug(f"LengthLimiter native registration failed: {e}, using Python fallback")
        else:
            l.debug("LengthLimiter registered (no max_length found)")

    # Timeout: Wall-clock timeout — native Rust implementation
    elif tech_name == "Timeout":
        timeout_val = getattr(technique, "timeout", None)
        if timeout_val is not None:
            try:
                mgr._rust_mgr.register_timeout(float(timeout_val))
                technique._native_timeout = True
                l.debug(f"Timeout registered natively ({timeout_val}s)")
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: native registration failed; the
                # Python check_technique_complete() path still runs.
                l.debug(f"Timeout native registration failed: {e}, using Python fallback")
        else:
            l.debug("Timeout technique registered (no timeout value)")

    # Other techniques
    else:
        l.debug(f"Technique {tech_name} registered (limited support)")

    # Surface hook coverage at registration time:
    #   step()       — now dispatched per-batch via dispatch_step_with_hooks()
    #   successors() — still no-op (would require Python-side re-run; see
    #                  RustSimulationManagerProxy.successors() docstring)
    #   step_state() — still no-op (ditto)
    if tech_name not in _NATIVE_STEP_TECH_NAMES:
        if technique._is_overridden("step"):
            l.debug(
                "Technique %s overrides step() — will be dispatched via RustSimulationManagerProxy.step() each batch.",
                tech_name,
            )
        if technique._is_overridden("successors"):
            l.warning(
                "Technique %s overrides successors(), which is not dispatched by "
                "the Rust manager. The per-successor hook will silently no-op. "
                "Drop to use_rust_engine=False if you need it.",
                tech_name,
            )
        if technique._is_overridden("step_state"):
            l.warning(
                "Technique %s overrides step_state(), which is not dispatched by "
                "the Rust manager. The per-state hook will silently no-op. "
                "Drop to use_rust_engine=False if you need it.",
                tech_name,
            )

    return technique


def remove_technique(mgr: RustExplorationManager, technique) -> bool:
    """Remove an exploration technique.

    Args:
        mgr: The RustExplorationManager instance.
        technique: The technique to remove.

    Returns:
        True if removed, False if not found.
    """
    try:
        mgr._active_techniques.remove(technique)
        return True
    except ValueError:
        # cat-(a) EXPECTED CONTROL FLOW: list.remove raises ValueError when the
        # technique was never registered; returning False is the documented contract.
        return False


def apply_technique_filters(mgr: RustExplorationManager):
    """Apply ExplorationTechnique filter() callbacks via proxy.

    After each step, iterate states through each technique's filter()
    method. If filter() returns a stash name other than 'active', move the
    state to that stash in Rust.

    Re-filter semantics (angr-j1ue): Rust state ids persist across steps for
    non-forking states, so a once-per-id skip would evaluate a filter() that
    depends on evolving state (addr, stdout) exactly once and then ignore it —
    silent drift from SimulationManager's per-step filter contract. Instead we
    cache each state's (addr, stdout_len) signature and re-run filter()s when
    the signature changes. Techniques in _MONOTONIC_FILTER_TECHNIQUES (e.g.
    CheckUniqueness, whose internal set must not see a state twice) are still
    evaluated at most once per id, even across signature changes.
    """
    from angr.exploration.rust_state_proxy import RustSimulationManagerProxy, RustStateProxy

    if not mgr._active_techniques:
        return

    # Cache: state_id -> (addr, stdout_len) signature at last filter sweep.
    # A state is re-filtered (for non-monotonic techniques) when its signature
    # changes; unchanged signatures are skipped. Presence in the dict means
    # "seen at least once" — used to gate monotonic techniques to one eval.
    if not hasattr(mgr, "_filter_eval_sigs"):
        mgr._filter_eval_sigs = {}
        mgr._filtered_cleanup_counter = 0

    # Bulk-collect (addr, stdout_len) signatures for change detection. Cheaper
    # than per-state FFI; a missing sig (stash query failed) falls back to
    # always re-running non-monotonic techniques for that state.
    state_sigs = {}  # state_id -> (addr, stdout_len)
    for stash in ("active", "errored", "deadended"):
        try:
            for sid, addr, stdout_len in mgr._rust_mgr.get_state_predicate_info(stash):
                state_sigs[sid] = (addr, stdout_len)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: no signatures for this stash means
            # change detection is disabled here; non-monotonic filters re-run
            # every sweep (correct, just less efficient).
            l.debug("get_state_predicate_info(stash=%s) failed: %s: %s", stash, type(e).__name__, e)

    simgr_proxy = RustSimulationManagerProxy(
        mgr._rust_mgr,
        project=mgr._project,
        stdin_vars=getattr(mgr, "_stdin_vars", None),
        stdout_tracker=getattr(mgr, "_stdout_tracker", {}),
    )

    # Apply filters to active, errored, and deadended states.
    # Some techniques (e.g., SearchForNull) catch states at invalid
    # addresses (like addr 0) that the Rust engine moved to errored/deadended.
    for stash in ("active", "errored", "deadended"):
        state_ids = list(mgr._rust_mgr.get_state_ids(stash))
        for sid in state_ids:
            prev_sig = mgr._filter_eval_sigs.get(sid)
            first_seen = sid not in mgr._filter_eval_sigs
            cur_sig = state_sigs.get(sid)
            # Skip only when we've seen this state before AND its signature is
            # available and unchanged. If the signature is unavailable we cannot
            # prove it is unchanged, so we fall through and re-run (monotonic
            # techniques are still gated below).
            if not first_seen and cur_sig is not None and cur_sig == prev_sig:
                continue

            state_proxy = RustStateProxy(
                mgr._rust_mgr,
                sid,
                project=mgr._project,
                python_mgr=mgr,
            )
            # Prefetch common registers in one FFI call for technique filter efficiency
            arch = mgr._project.arch if mgr._project else None
            if arch and hasattr(state_proxy.regs, "prefetch"):
                if arch.name in ("AMD64", "X86_64"):
                    state_proxy.regs.prefetch(["rip", "rsp", "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp"])
                elif arch.name == "X86":
                    state_proxy.regs.prefetch(["eip", "esp", "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp"])
            # Run through each technique's filter in order
            goto = None
            for tech in mgr._active_techniques:
                # Skip techniques handled natively in Rust
                if getattr(tech, "_native_uniqueness", False):
                    continue
                if getattr(tech, "_native_length_limiter", False):
                    continue
                if getattr(tech, "_native_timeout", False):
                    continue

                tech_name = type(tech).__name__

                # Monotonic techniques (CheckUniqueness) must see each state id
                # at most once — skip them on re-evaluation (signature change).
                if not first_seen and tech_name in _MONOTONIC_FILTER_TECHNIQUES:
                    continue

                # LengthLimiter: check _filter (uses step() pattern, not filter())
                if tech_name == "LengthLimiter" and hasattr(tech, "_filter"):
                    try:
                        if tech._filter(state_proxy):
                            drop = getattr(tech, "_drop", False)
                            goto = "_DROP" if drop else "cut"
                            break
                    except Exception as e:
                        # cat-(b) FALLBACK WITH LOSS: user technique raised; we
                        # treat the state as not-cut and let other techniques try.
                        l.debug(f"LengthLimiter._filter() error: {e}")
                    continue

                if hasattr(tech, "filter"):
                    try:
                        result = tech.filter(simgr_proxy, state_proxy)
                        if result is not None and result != stash:
                            goto = result
                            break
                    except Exception as e:
                        # cat-(b) FALLBACK WITH LOSS: user technique raised; we
                        # leave the state in its current stash. (User code bug.)
                        l.debug(f"Technique {tech_name}.filter() error: {e}")

            # Record this sweep's signature so an unchanged state is skipped
            # next time and a changed one re-runs non-monotonic filters.
            mgr._filter_eval_sigs[sid] = cur_sig

            if goto is not None and goto != stash:
                try:
                    mgr._rust_mgr.move_state(sid, stash, goto)
                    # Ensure the moved state can be reconstructed later by
                    # tracking its root state for plugin/constraint restoration
                    if sid not in mgr._state_cache:
                        try:
                            root = mgr._rust_mgr.get_state_root(sid)
                            if root is not None:
                                mgr._state_roots[sid] = root
                        except Exception:
                            # cat-(a) EXPECTED CONTROL FLOW: state may not have
                            # a Rust-tracked root; that's OK, the move succeeded.
                            pass
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: move_state failed; the state
                    # stays in its current stash and the next sweep will retry.
                    l.debug(f"Failed to move state {sid} from {stash} to {goto}: {e}")

    # Periodic cleanup: drop dead state IDs from _filter_eval_sigs
    # to prevent unbounded growth in long-running explorations.
    mgr._filtered_cleanup_counter = getattr(mgr, "_filtered_cleanup_counter", 0) + 1
    if mgr._filtered_cleanup_counter >= 100:
        mgr._filtered_cleanup_counter = 0
        try:
            live = set()
            for stash_name in ("active", "found", "errored", "deadended", "avoid"):
                live.update(mgr._rust_mgr.get_state_ids(stash_name))
            mgr._filter_eval_sigs = {sid: sig for sid, sig in mgr._filter_eval_sigs.items() if sid in live}
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: best-effort cache GC; if we can't
            # enumerate stashes the set just grows until next sweep.
            pass


def check_technique_complete(mgr: RustExplorationManager) -> bool:
    """Check ExplorationTechnique complete() callbacks.

    Returns True if any technique says exploration is complete.
    """
    from angr.exploration.rust_state_proxy import RustSimulationManagerProxy

    if not mgr._active_techniques:
        return False

    simgr_proxy = RustSimulationManagerProxy(
        mgr._rust_mgr,
        project=mgr._project,
        stdin_vars=getattr(mgr, "_stdin_vars", None),
        stdout_tracker=getattr(mgr, "_stdout_tracker", {}),
    )

    import time

    for tech in mgr._active_techniques:
        tech_name = type(tech).__name__

        # Timeout: check wall-clock and move all active states to "timeout"
        # Skip if handled natively in Rust
        if tech_name == "Timeout":
            if getattr(tech, "_native_timeout", False):
                continue  # Handled in Rust apply_native_techniques()
            timeout_val = getattr(tech, "timeout", None)
            if timeout_val is not None:
                if tech.start_time is None:
                    tech.start_time = time.time()
                if time.time() - tech.start_time > timeout_val:
                    try:
                        mgr._rust_mgr.move_states("active", "timeout", None)
                    except Exception as e:
                        # cat-(b) FALLBACK WITH LOSS: timeout-stash move failed;
                        # active states remain in 'active' but caller still gets
                        # the timeout signal via the True return below.
                        l.debug(f"Timeout move_states failed: {e}")
                    l.warning(f"exploration timeout in {timeout_val} seconds!")
                    return True

        if hasattr(tech, "complete"):
            try:
                if tech.complete(simgr_proxy):
                    l.debug(f"Technique {tech_name}.complete() returned True")
                    return True
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: user technique raised; we treat
                # the technique as not-complete and continue exploration.
                l.debug(f"Technique {tech_name}.complete() error: {e}")

    return False
