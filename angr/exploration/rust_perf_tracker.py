"""Performance counter encapsulation for the Rust exploration manager.

Centralizes the performance counters that were previously a bare dict
(``self._perf_stats``) accessed from many sites across rust_manager.py and
rust_callback_dispatch.py. Writers go through explicit named methods so the
set of fields and their update semantics live in one place.

Read access via ``__getitem__`` and ``.get()`` is preserved so existing
report builders and debug scripts can keep treating the tracker as a
read-only mapping.
"""

from __future__ import annotations

from collections.abc import Iterator


class PerformanceTracker:
    """Centralized performance counters for the Rust exploration manager."""

    _FIELDS = (
        # Init phases
        "init_total_ns",
        "init_setup_callbacks_ns",
        "init_load_binary_ns",
        "init_register_simprocedures_ns",
        "init_python_run_ns",
        "init_add_rust_state_ns",
        "init_memory_sync_ns",
        "init_register_sync_ns",
        "init_hook_register_ns",
        # Callback counts and times
        "callback_simprocedure_count",
        "callback_simprocedure_total_ns",
        "callback_simprocedure_state_create_ns",
        "callback_simprocedure_execute_ns",
        "callback_simprocedure_sync_back_ns",
        "callback_simprocedure_state_copy_ns",
        "callback_memory_load_count",
        "callback_memory_load_total_ns",
        "callback_fetch_page_count",
        "callback_fetch_page_total_ns",
        "callback_lift_block_count",
        "callback_lift_block_total_ns",
        "callback_syscall_count",
        "callback_syscall_total_ns",
        "callback_find_predicate_count",
        "callback_find_predicate_total_ns",
        "callback_avoid_predicate_count",
        "callback_avoid_predicate_total_ns",
        "callback_symbolic_branch_count",
        "callback_symbolic_branch_total_ns",
        "callback_vex_fallback_count",
        "callback_vex_fallback_total_ns",
        # angr-afbx: Python posix-plugin callback (stdin/stdout injection).
        # Enables angr-6zxx's DEFER->GO trigger (>5% wall in posix) to be
        # checked from any --counters-json capture.
        "callback_posix_count",
        "callback_posix_total_ns",
    )

    def __init__(self) -> None:
        self._stats: dict[str, int] = dict.fromkeys(self._FIELDS, 0)

    # ---- Init phase recorders ----

    def set_init_phase(self, phase: str, ns: int) -> None:
        self._stats[f"init_{phase}_ns"] = ns

    def add_init_phase(self, phase: str, ns: int) -> None:
        self._stats[f"init_{phase}_ns"] += ns

    # ---- Callback recorders ----

    def record_simprocedure_call(self, total_ns: int) -> None:
        self._stats["callback_simprocedure_count"] += 1
        self._stats["callback_simprocedure_total_ns"] += total_ns

    def increment_simprocedure_count(self) -> None:
        """Bump only the simprocedure count (used by Python VEX fallback)."""
        self._stats["callback_simprocedure_count"] += 1

    def add_simprocedure_phase(self, phase: str, ns: int) -> None:
        """phase: state_create, execute, sync_back, state_copy."""
        self._stats[f"callback_simprocedure_{phase}_ns"] += ns

    def record_memory_load(self, ns: int) -> None:
        self._stats["callback_memory_load_count"] += 1
        self._stats["callback_memory_load_total_ns"] += ns

    def record_fetch_page(self, ns: int) -> None:
        self._stats["callback_fetch_page_count"] += 1
        self._stats["callback_fetch_page_total_ns"] += ns

    def record_lift_block(self, ns: int) -> None:
        self._stats["callback_lift_block_count"] += 1
        self._stats["callback_lift_block_total_ns"] += ns

    def record_syscall_call(self, total_ns: int) -> None:
        self._stats["callback_syscall_count"] += 1
        self._stats["callback_syscall_total_ns"] += total_ns

    def record_find_predicate_call(self, total_ns: int) -> None:
        self._stats["callback_find_predicate_count"] += 1
        self._stats["callback_find_predicate_total_ns"] += total_ns

    def record_avoid_predicate_call(self, total_ns: int) -> None:
        self._stats["callback_avoid_predicate_count"] += 1
        self._stats["callback_avoid_predicate_total_ns"] += total_ns

    def record_symbolic_branch_call(self, total_ns: int) -> None:
        self._stats["callback_symbolic_branch_count"] += 1
        self._stats["callback_symbolic_branch_total_ns"] += total_ns

    def record_vex_fallback_call(self, total_ns: int) -> None:
        self._stats["callback_vex_fallback_count"] += 1
        self._stats["callback_vex_fallback_total_ns"] += total_ns

    def record_posix_call(self, total_ns: int) -> None:
        """angr-afbx: one Python posix-plugin callback (stdin or stdout inject)."""
        self._stats["callback_posix_count"] += 1
        self._stats["callback_posix_total_ns"] += total_ns

    # ---- Read access (mapping-like) ----

    def __getitem__(self, key: str) -> int:
        return self._stats[key]

    def __contains__(self, key: str) -> bool:
        return key in self._stats

    def __iter__(self) -> Iterator[str]:
        return iter(self._stats)

    def get(self, key: str, default: int = 0) -> int:
        return self._stats.get(key, default)

    def as_dict(self) -> dict[str, int]:
        return dict(self._stats)
