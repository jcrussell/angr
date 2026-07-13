"""Differential parity assertions for the fallback census (angr-op0dn.14.1.2).

The M6.5a census (``tests/benchmarks/fallback_census.json``) enumerated every
SimProcedure name that bounces out of the Rust engine into a Python proc across
the bench corpus, and classified each one *by inspection*. This module turns
that classification into evidence: for each classifiable name we run the same
one-call scenario twice — once on ``RustExplorationManager`` (which either
dispatches natively or bounces to the Python proc) and once on the pure-Python
engine — and assert the two engines agree on every observable the proc touches
(rax, the memory it wrote, stdout, the posix fd table).

Census rows that get a *documented divergence* row instead of an assertion here
— UserHook, per-solve.py hooks, and the C++ stdlib names — are listed in
``docs/extending-angr/native_coverage_matrix.rst`` under "Documented
divergences"; they have no native counterpart, so angr's Python proc *is* the
reference implementation and there is nothing to diff against.
"""

from __future__ import annotations

import claripy
import pytest

import angr
from tests.engines.conftest import (
    RUST_EXPLORATION_AVAILABLE,
    RustExplorationManager,
)

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

# Same mapped-but-unexecuted addresses test_procedures.py uses: the hook has to
# live inside fauxware's loaded image or the Rust manager runs Python init to
# main instead of the hooked call.
HOOK_ADDR = 0x4008C0
RET_ADDR = 0x4008B0
BUF_ADDR = 0x601100
FILE_ADDR = 0x601400  # fake FILE struct
FD_OFFSET = 112  # io_file_data_for_arch(AMD64)["fd"]


def _blank_state(proj):
    state = proj.factory.blank_state(
        addr=HOOK_ADDR,
        add_options={
            angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
        },
    )
    state.memory.store(state.regs.rsp, claripy.BVV(RET_ADDR, 64), endness="Iend_LE")
    return state


def _run_rust(proj, state):
    mgr = RustExplorationManager(proj, [state])
    mgr.run(max_steps=1)
    states = mgr.active + mgr.deadended
    assert len(states) == 1, f"rust: expected one post-call state, got {mgr.stash_counts()}"
    return states[0], mgr


def _run_python(proj, state):
    simgr = proj.factory.simulation_manager(state)
    simgr.step()
    states = simgr.active + simgr.deadended
    assert len(states) == 1, f"python: expected one post-call state, got {simgr}"
    return states[0]


def _rax_range(state):
    """(min, max) of rax — comparable across engines even when rax is symbolic."""
    return state.solver.min(state.regs.rax), state.solver.max(state.regs.rax)


def _mem(state, addr, size):
    return state.solver.eval(state.memory.load(addr, size), cast_to=bytes)


def _fallback_names(mgr):
    return mgr.stats.get("simprocedure_fallback_by_name", {}) or {}


class Scenario:
    """One census name: how to build the call, and what to diff afterwards."""

    def __init__(self, proc, setup, observe, bounces=False):
        self.proc = proc  # () -> SimProcedure instance
        self.setup = setup  # (state) -> None
        self.observe = observe  # (state) -> comparable value
        self.bounces = bounces  # Rust has no native handler: must fall back


def _libc(name):
    return lambda: angr.SIM_PROCEDURES["libc"][name]()


def _setup_strcmp(state):
    state.memory.store(BUF_ADDR, b"hello\x00")
    state.memory.store(BUF_ADDR + 0x10, b"hello\x00")
    state.regs.rdi = BUF_ADDR
    state.regs.rsi = BUF_ADDR + 0x10


def _setup_strncmp(state):
    state.memory.store(BUF_ADDR, b"abcX\x00")
    state.memory.store(BUF_ADDR + 0x10, b"abcY\x00")
    state.regs.rdi = BUF_ADDR
    state.regs.rsi = BUF_ADDR + 0x10
    state.regs.rdx = 3


def _setup_strncpy(state):
    state.memory.store(BUF_ADDR + 0x10, b"source\x00")
    state.regs.rdi = BUF_ADDR
    state.regs.rsi = BUF_ADDR + 0x10
    state.regs.rdx = 7


def _setup_memmove(state):
    state.memory.store(BUF_ADDR + 0x10, b"movable!")
    state.regs.rdi = BUF_ADDR
    state.regs.rsi = BUF_ADDR + 0x10
    state.regs.rdx = 8


def _setup_none(state):
    del state


def _setup_fake_file(state, fd=1):
    """A FILE* whose fd field names `fd` — what fwrite/fputc/fprintf resolve."""
    state.memory.store(FILE_ADDR + FD_OFFSET, claripy.BVV(fd, 32), endness=state.arch.memory_endness)


def _setup_fwrite(state):
    _setup_fake_file(state)
    state.memory.store(BUF_ADDR, b"STREAM")
    state.regs.rdi = BUF_ADDR
    state.regs.rsi = 1
    state.regs.rdx = 6
    state.regs.rcx = FILE_ADDR


def _setup_fputc(state):
    _setup_fake_file(state)
    state.regs.rdi = ord("Z")
    state.regs.rsi = FILE_ADDR


def _setup_fprintf(state):
    _setup_fake_file(state)
    state.memory.store(BUF_ADDR, b"n=%d\n\x00")
    state.regs.rdi = FILE_ADDR
    state.regs.rsi = BUF_ADDR
    state.regs.rdx = 42


def _setup_malloc(state):
    state.regs.rdi = 0x30


def _setup_calloc(state):
    state.regs.rdi = 4
    state.regs.rsi = 8


def _setup_open(state):
    state.memory.store(BUF_ADDR, b"/tmp/parity.txt\x00")
    state.regs.rdi = BUF_ADDR
    state.regs.rsi = 0  # O_RDONLY
    state.regs.rdx = 0


def _obs_rax(state):
    return _rax_range(state)


def _obs_rax_and_buf(state):
    return _rax_range(state), _mem(state, BUF_ADDR, 8)


def _obs_stdout(state):
    return state.posix.dumps(1)


def _obs_rax_and_stdout(state):
    return _rax_range(state), state.posix.dumps(1)


def _obs_calloc(state):
    lo, hi = _rax_range(state)
    assert lo == hi, "calloc returned a symbolic pointer"
    return (lo, hi), _mem(state, lo, 32)


# Every census name that has a Python proc we can drive in isolation. Names not
# here are covered by the documented-divergence table in the coverage matrix.
SCENARIOS = {
    # --- silent-equivalent: memory/register only -------------------------
    "strcmp": Scenario(_libc("strcmp"), _setup_strcmp, _obs_rax),
    "strncmp": Scenario(_libc("strncmp"), _setup_strncmp, _obs_rax),
    "strncpy": Scenario(_libc("strncpy"), _setup_strncpy, _obs_rax_and_buf),
    "memmove": Scenario(_libc("memmove"), _setup_memmove, _obs_rax_and_buf),
    "time": Scenario(_libc("time"), _setup_none, _obs_rax, bounces=True),
    "getenv": Scenario(
        lambda: angr.SIM_PROCEDURES["posix"]["getenv"](),
        _setup_open,  # reuses the name buffer in rdi
        _obs_rax,
        bounces=True,
    ),
    "pthread_mutex_lock": Scenario(
        lambda: angr.SIM_PROCEDURES["posix"]["pthread_mutex_lock"](),
        _setup_none,
        _obs_rax,
        bounces=True,
    ),
    "pthread_mutex_unlock": Scenario(
        lambda: angr.SIM_PROCEDURES["posix"]["pthread_mutex_unlock"](),
        _setup_none,
        _obs_rax,
        bounces=True,
    ),
    "pthread_once": Scenario(
        lambda: angr.SIM_PROCEDURES["posix"]["pthread_once"](),
        _setup_none,
        _obs_rax,
        bounces=True,
    ),
    "__ctype_b_loc": Scenario(
        lambda: angr.SIM_PROCEDURES["glibc"]["__ctype_b_loc"](),
        _setup_none,
        _obs_rax,
        bounces=True,
    ),
    # --- was flip-blocking: heap bump must survive the bounce (.14.1.3) ---
    "malloc": Scenario(_libc("malloc"), _setup_malloc, _obs_rax),
    "calloc": Scenario(_libc("calloc"), _setup_calloc, _obs_calloc),
    # --- was flip-blocking: stream writes must reach Rust's fd (.14.1.4) --
    "fwrite": Scenario(_libc("fwrite"), _setup_fwrite, _obs_rax_and_stdout, bounces=True),
    "fputc": Scenario(_libc("fputc"), _setup_fputc, _obs_rax_and_stdout, bounces=True),
    "fprintf": Scenario(_libc("fprintf"), _setup_fprintf, _obs_stdout, bounces=True),
    # --- was flip-blocking: bounced open must reach Rust's fd table (.14.1.5/6)
    "open": Scenario(
        lambda: angr.SIM_PROCEDURES["posix"]["open"](),
        _setup_open,
        _obs_rax,
        bounces=True,
    ),
}


# Census names with no differential assertion, and why. Each has a row in
# docs/extending-angr/native_coverage_matrix.rst ("Documented divergences").
DOCUMENTED_DIVERGENCES = {
    # No native counterpart exists, so angr's Python proc *is* the reference
    # implementation — there is no second engine to diff it against.
    "UserHook": "user hook — Python by definition; gets a full state.copy() snapshot",
    "my_scanf": "solve.py UserHook (defcon2016quals_baby-re)",
    "get_flag": "solve.py UserHook (whitehatvn2015_re400)",
    "readline_hook": "solve.py UserHook (cmu_binary_bomb_partial)",
    "strtol_hook": "solve.py UserHook (cmu_binary_bomb_partial)",
    # C++ stdlib symbols: angr resolves these to Python procs (or
    # ReturnUnconstrained). new/delete route to malloc/free, so their heap
    # round-trip is the malloc/calloc rows above.
    "operator new(unsigned long)": "routes to angr's malloc proc — parity is the malloc row",
    "operator delete(void*)": "routes to angr's free proc — no heap-bump mutation to lose",
    "std::allocator<char>::allocator()": "C++ stdlib — no native counterpart",
    "std::allocator<char>::~allocator()": "C++ stdlib — no native counterpart",
    "std::basic_ostream<char, std::char_traits<char>>& std::operator<<<std::char_traits<char>>"
    "(std::basic_ostream<char, std::char_traits<char>>&, char const*)": "C++ stdlib — no native counterpart",
    "std::basic_string<char, std::char_traits<char>, std::allocator<char>>::basic_string"
    "(char const*, std::allocator<char> const&)": "C++ stdlib — no native counterpart",
    "std::basic_string<char, std::char_traits<char>, std::allocator<char>>::basic_string"
    "(std::string const&)": "C++ stdlib — no native counterpart",
    "std::basic_string<char, std::char_traits<char>, std::allocator<char>>::~basic_string()": (
        "C++ stdlib — no native counterpart"
    ),
    "std::string::length() const": "C++ stdlib — no native counterpart",
}


class TestCensusCoverage:
    """The census artifact and this module must not drift apart.

    ``tests/benchmarks/fallback_census.json`` is regenerated from a bench
    sweep; a newly-observed fallback name lands there first. This test is what
    makes that arrival loud: every row needs either a parity scenario above or
    an explicit documented-divergence entry.
    """

    @staticmethod
    def _census():
        import json
        import pathlib

        path = pathlib.Path(__file__).parents[2] / "benchmarks" / "fallback_census.json"
        if not path.exists():
            pytest.skip("fallback census artifact not present")
        return json.loads(path.read_text())

    def test_every_census_name_is_covered(self):
        names = set(self._census()["names"])
        uncovered = names - set(SCENARIOS) - set(DOCUMENTED_DIVERGENCES)
        assert not uncovered, (
            f"census names with neither a parity scenario nor a documented-divergence row: {sorted(uncovered)}"
        )

    def test_no_census_row_is_still_flip_blocking(self):
        """Every flip-blocking row the census opened has been closed
        (angr-op0dn.14.1.3 through .14.1.6, plus the syscall resume fix
        angr-89w70). A new one appearing here blocks the M6 default flip."""
        census = self._census()
        blocking = [n for n, row in census["names"].items() if row["semantics"] == "flip-blocking"]
        blocking += [f"syscall {n}" for n, row in census["syscalls"].items() if row["semantics"] == "flip-blocking"]
        assert not blocking, f"flip-blocking fallbacks (gate the default flip): {sorted(blocking)}"


class TestFallbackDifferentialParity:
    """Rust (native *or* bounced) must agree with the pure-Python engine.

    A row here is the evidence the census classified the name on: if the two
    engines' observables match, the fallback is observationally silent and the
    name is not flip-blocking.
    """

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_parity(self, fauxware_project, name):
        scen = SCENARIOS[name]
        proj = fauxware_project
        try:
            proj.hook(HOOK_ADDR, scen.proc(), replace=True)

            rust_state = _blank_state(proj)
            scen.setup(rust_state)
            rust_result, mgr = _run_rust(proj, rust_state)
            rust_obs = scen.observe(rust_result)

            py_state = _blank_state(proj)
            scen.setup(py_state)
            py_obs = scen.observe(_run_python(proj, py_state))
        finally:
            proj.unhook(HOOK_ADDR)

        assert rust_obs == py_obs, f"{name}: rust {rust_obs!r} != python {py_obs!r}"

        if scen.bounces:
            # Guards against a vacuous pass: if Rust silently grew a native
            # handler for this name the diff above stops proving anything
            # about the *fallback* path, so make the census row's premise
            # explicit. Flip the flag (and the census) when that happens.
            assert name in _fallback_names(mgr), (
                f"{name}: census says this name bounces to Python, but the Rust run "
                f"recorded no fallback for it ({_fallback_names(mgr)})"
            )
