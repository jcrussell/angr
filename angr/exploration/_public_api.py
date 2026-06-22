"""Authoritative inventory of the public API surface of ``angr.exploration``.

This file is the source of truth for what counts as ``public`` on the Rust
exploration engine. A sibling task (angr-9cps .3) wires a snapshot test to it
so unintended additions or renames break CI.

Conventions
-----------

- ``MODULE_EXPORTS`` mirrors ``angr.exploration.__all__`` — anything reachable
  from ``from angr.exploration import *``.
- ``CLASS_PUBLIC_ATTRS`` maps each public class to the tuple of public
  attribute names (methods + properties) that callers may rely on. Underscore
  names (``_foo``) are intentionally excluded. Dunder names are excluded
  unless they are part of the documented protocol (``__len__``,
  ``__getattr__`` on classes designed for delegation, etc.).
- This inventory does **not** record signatures. The .2 policy doc covers
  signature-stability rules; the .3 snapshot test only checks attribute-name
  drift.

Updating
--------

If you intentionally add, rename, or remove a public attribute on one of the
listed classes, update the corresponding tuple here in the same PR. The
snapshot test will fail otherwise — that is the point.
"""

from __future__ import annotations

MODULE_EXPORTS: tuple[str, ...] = (
    "RustErrorRecord",
    "RustExecutionError",
    "RustExplorationManager",
    "RustMalformedIRSBError",
    "RustOomError",
    "RustUnsupportedSyscallError",
    "RustUnsupportedVexOpError",
    "RustZ3Error",
)


TYPED_EXCEPTIONS: tuple[str, ...] = (
    "RustExecutionError",
    "RustMalformedIRSBError",
    "RustOomError",
    "RustUnsupportedSyscallError",
    "RustUnsupportedVexOpError",
    "RustZ3Error",
)


CLASS_PUBLIC_ATTRS: dict[str, tuple[str, ...]] = {
    "RustErrorRecord": (
        "addr",
        "constraint_count",
        "error",
        "error_class",
        "last_statements",
        "registers",
        "reraise",
        "state",
    ),
    "RustExplorationManager": (
        "active",
        "active_proxies",
        "avoid",
        "avoid_proxies",
        "cleanup",
        "copy",
        "deadended",
        "deadended_proxies",
        "disable_profiling",
        "disable_uniqueness_filter",
        "drop",
        "drop_copy",
        "dump_snapshot",
        "enable_profiling",
        "errored",
        "eval_memory",
        "eval_register",
        "explore",
        "filter",
        "fork_state_for_copy",
        "found",
        "found_proxies",
        "found_states",
        "get_exploration_summary",
        "get_solver_stats",
        "get_state_by_id",
        "get_state_globals_py",
        "get_state_options_py",
        "is_satisfiable",
        "load_from_disk",
        "load_snapshot",
        "merge",
        "move",
        "one_active",
        "one_found",
        "one_found_state",
        "perf_report",
        "proxy",
        "prune",
        "pruned",
        "remove_technique",
        "register_uniqueness_filter",
        "reset_solver_stats",
        "run",
        "set_block_granular",
        "set_exploration_strategy",
        "set_materialize_unconstrained_forks",
        "set_progress_callback",
        "split",
        "stash",
        "stash_counts",
        "stashes",
        "stats",
        "step",
        "unconstrained",
        "unconstrained_proxies",
        "uniqueness_filter_enabled",
        "uniqueness_set_size",
        "unstash",
        "use_technique",
    ),
    "RustStateProxy": (
        "add_constraints",
        "addr",
        "arch",
        "callstack",
        "copy",
        "globals",
        "heap",
        "history",
        "inspect",
        "ip",
        "mem",
        "memory",
        "options",
        "posix",
        "project",
        "regs",
        "registers",
        "satisfiable",
        "scratch",
        "se",
        "solver",
        "state_id",
    ),
    "RustSolverProxy": (
        "add",
        "constraints",
        "eval",
        "eval_atleast",
        "eval_exact",
        "eval_one",
        "eval_upto",
        "is_false",
        "is_true",
        "max",
        "min",
        "satisfiable",
        "solution",
        "symbolic",
        "timeout",
    ),
    "RustRegisterProxy": (
        # SimMemory plugin protocol (installed as state.registers under the
        # write-through gate, angr-qj30) + the proxy's own read/prefetch API.
        "STRONGREF_STATE",
        "SUPPORTS_CONCRETE_LOAD",
        "category",
        "copy",
        "init_state",
        "load",
        "merge",
        "prefetch",
        "set_state",
        "set_strongref_state",
        "store",
        "widen",
    ),
    "RustMemoryProxy": (
        # SimMemory plugin protocol (installed as state.memory under the
        # write-through gate, angr-qj30) + the proxy's own load/store API.
        "STRONGREF_STATE",
        "SUPPORTS_CONCRETE_LOAD",
        "category",
        "compare",
        "copy",
        "find",
        "init_state",
        "load",
        "merge",
        "permissions",
        "set_state",
        "set_strongref_state",
        "store",
        "widen",
    ),
    "RustHeapProxy": (
        "allocations",
        "freed",
        "mmap_base",
    ),
    "RustScratchProxy": (
        "bbl_addr",
        "ins_addr",
        "irsb",
        "jumpkind",
        "sim_procedure",
        "stmt_idx",
        "temps",
        "tyenv",
    ),
    "RustHistoryProxy": (
        "bbl_addrs",
        "block_count",
        "recent_bbl_addrs",
    ),
    "RustPosixProxy": ("dumps",),
    "RustCallStackProxy": (
        "call_site_addr",
        "current_function_address",
        "current_return_target",
        "current_stack_pointer",
        "func_addr",
        "ret_addr",
        "stack_ptr",
        "top",
    ),
    "RustCallStackFrameProxy": (
        "call_site_addr",
        "current_function_address",
        "current_return_target",
        "current_stack_pointer",
        "func_addr",
        "jumpkind",
        "next",
        "ret_addr",
        "stack_ptr",
    ),
    "RustInspectProxy": (
        "SUPPORTED_EVENTS",
        "action",
        "add_breakpoint",
        "b",
        "make_breakpoint",
        "remove_breakpoint",
        "set_state",
    ),
    "RustSimulationManagerProxy": (
        "active",
        "avoid",
        "deadended",
        "errored",
        "filter",
        "found",
        "move",
        "one_active",
        "one_found",
        "stashes",
        "step",
        "step_state",
        "successors",
    ),
}


__all__ = [
    "CLASS_PUBLIC_ATTRS",
    "MODULE_EXPORTS",
    "TYPED_EXCEPTIONS",
]
