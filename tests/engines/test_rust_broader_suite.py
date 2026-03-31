"""Broader test suite for RustExplorationManager against angr-examples.

Tests the Rust exploration engine against additional CTF examples beyond
the original 10 used in benchmarking. Includes both "smoke tests" that
directly construct RustExplorationManager and full solve.py execution tests
that monkey-patch the angr factory.
"""
import importlib.util
import io
import os
import sys

import pytest

import angr

# ---------------------------------------------------------------------------
# Rust availability check
# ---------------------------------------------------------------------------

try:
    from angr.exploration import RustExplorationManager
    from angr.exploration.rust_manager import RUST_EXPLORATION_AVAILABLE
except ImportError:
    RUST_EXPLORATION_AVAILABLE = False

EXAMPLES_DIR = "/home/ubuntu/repos/angr-examples/examples"

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class BufferedStringIO(io.StringIO):
    """StringIO with a buffer attribute for compatibility with stdout.buffer."""

    def __init__(self):
        super().__init__()
        self._buffer = io.BytesIO()

    @property
    def buffer(self):
        return self._buffer

    def getvalue(self) -> str:
        text_output = super().getvalue()
        binary_output = self._buffer.getvalue()
        if binary_output:
            try:
                text_output += binary_output.decode("utf-8", errors="replace")
            except Exception:
                pass
        return text_output


def _binary_path(*parts: str) -> str:
    """Build a path under the examples directory and return it."""
    return os.path.join(EXAMPLES_DIR, *parts)


def _skip_if_missing(*parts: str):
    """Skip the test if the binary does not exist."""
    path = _binary_path(*parts)
    if not os.path.isfile(path):
        pytest.skip(f"Binary not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Fixture: monkey-patch angr factory to use RustExplorationManager
# ---------------------------------------------------------------------------

@pytest.fixture()
def rust_factory_patch():
    """Monkey-patch angr so that simulation_manager / simgr returns a
    RustExplorationManager.  Restores the originals on teardown."""

    original_simulation_manager = angr.factory.AngrObjectFactory.simulation_manager
    original_simgr = angr.factory.AngrObjectFactory.simgr

    def patched_simulation_manager(factory_self, thing=None, **kwargs):
        if thing is None:
            states = [factory_self.entry_state()]
        elif isinstance(thing, (list, tuple)):
            states = list(thing)
        else:
            states = [thing]
        return RustExplorationManager(factory_self.project, states)

    angr.factory.AngrObjectFactory.simulation_manager = patched_simulation_manager
    angr.factory.AngrObjectFactory.simgr = patched_simulation_manager

    yield

    angr.factory.AngrObjectFactory.simulation_manager = original_simulation_manager
    angr.factory.AngrObjectFactory.simgr = original_simgr


def _exec_solve_script(example_dir: str) -> str:
    """Execute a solve.py inside *example_dir*, capturing stdout.

    Returns the captured text output.
    """
    solve_path = os.path.join(example_dir, "solve.py")
    if not os.path.isfile(solve_path):
        pytest.skip(f"solve.py not found: {solve_path}")

    original_dir = os.getcwd()
    original_path = sys.path[:]
    captured = BufferedStringIO()
    original_stdout = sys.stdout

    try:
        os.chdir(example_dir)
        if example_dir not in sys.path:
            sys.path.insert(0, example_dir)

        spec = importlib.util.spec_from_file_location("__main__", solve_path)
        module = importlib.util.module_from_spec(spec)

        sys.stdout = captured
        try:
            spec.loader.exec_module(module)
        finally:
            sys.stdout = original_stdout
    finally:
        os.chdir(original_dir)
        sys.path[:] = original_path

    return captured.getvalue()


# ===========================================================================
# Smoke tests -- directly construct RustExplorationManager
# ===========================================================================

class TestRustSmokeTests:
    """Lightweight tests that directly build a RustExplorationManager and
    call explore() with known find/avoid addresses."""

    # -----------------------------------------------------------------------
    # defcamp_r100 (direct)
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(120)
    def test_defcamp_r100_direct(self):
        binary = _skip_if_missing("defcamp_r100", "r100")

        proj = angr.Project(binary, auto_load_libs=False)
        state = proj.factory.full_init_state()
        mgr = RustExplorationManager(proj, [state])

        mgr.explore(find=0x400844, avoid=0x400855)

        assert len(mgr.found) > 0, "No found states for defcamp_r100"
        solution = mgr.found[0].posix.dumps(0).strip(b"\0\n")
        assert solution.startswith(b"Code_Talkers"), (
            f"Unexpected solution: {solution!r}"
        )

    # -----------------------------------------------------------------------
    # asisctffinals2015_fake (direct)
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(120)
    def test_asisctffinals2015_fake_direct(self):
        binary = _skip_if_missing("asisctffinals2015_fake", "fake")

        proj = angr.Project(binary, auto_load_libs=False)
        import claripy

        state = proj.factory.blank_state(addr=0x4004AC)
        inp = claripy.BVS("inp", 8 * 8)
        state.regs.rax = inp

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x400684)

        assert len(mgr.found) > 0, "No found states for asisctffinals2015_fake"

    # -----------------------------------------------------------------------
    # CSCI-4968-MBE crackme0x00a (direct)
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(120)
    def test_crackme0x00a_direct(self):
        binary = _skip_if_missing(
            "CSCI-4968-MBE", "challenges", "crackme0x00a", "crackme0x00a"
        )

        proj = angr.Project(binary, load_options={"auto_load_libs": False})
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])

        find_addr = 0x08048533  # "Congrats!"
        avoid_addr = 0x08048554  # "Wrong!"

        mgr.explore(find=find_addr, avoid=avoid_addr)

        assert len(mgr.found) > 0, "No found states for crackme0x00a"
        solution = mgr.found[0].posix.dumps(0).split(b"\0")[0]
        assert solution == b"g00dJ0B!", f"Unexpected solution: {solution!r}"

    # -----------------------------------------------------------------------
    # flareon2015_2 (direct)
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(120)
    def test_flareon2015_2_direct(self):
        binary = _skip_if_missing("flareon2015_2", "very_success")

        import claripy

        proj = angr.Project(binary, load_options={"auto_load_libs": False})
        state = proj.factory.blank_state(addr=0x401084)

        # Set up the stack as the solve.py does
        state.memory.store(state.regs.esp + 12, claripy.BVV(40, state.arch.bits))
        state.mem[state.regs.esp + 8 :].dword = 0x402159
        state.mem[state.regs.esp + 4 :].dword = 0x4010E4
        state.mem[state.regs.esp :].dword = 0x401064

        state.memory.store(0x402159, claripy.BVS("ans", 8 * 40))

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x40106B, avoid=0x401072)

        assert len(mgr.found) > 0, "No found states for flareon2015_2"
        found_state = mgr.found[0]
        result = found_state.solver.eval(
            found_state.memory.load(0x402159, 40), cast_to=bytes
        ).strip(b"\0")
        assert b"@flare-on.com" in result, f"Unexpected solution: {result!r}"

    # -----------------------------------------------------------------------
    # sharif7_rev50 (direct)
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(120)
    def test_sharif7_rev50_direct(self):
        binary = _skip_if_missing("sharif7_rev50", "getit")

        proj = angr.Project(binary, auto_load_libs=False)
        state = proj.factory.entry_state(args=[proj.filename])
        mgr = RustExplorationManager(proj, [state])

        mgr.explore(find=0x4008C8, max_steps=5000)

        assert len(mgr.found) > 0, "No found states for sharif7_rev50"

    # -----------------------------------------------------------------------
    # defcon2016quals_baby-re (direct)
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(120)
    def test_defcon_baby_re_direct(self):
        binary = _skip_if_missing("defcon2016quals_baby-re", "baby-re")

        proj = angr.Project(binary, auto_load_libs=False)
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])

        mgr.explore(find=0x4028E9, avoid=0x402941, max_steps=10000)

        assert len(mgr.found) > 0, "No found states for defcon_baby_re"

    # -----------------------------------------------------------------------
    # fauxware (direct — uses run(until=...))
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(60)
    def test_fauxware_direct(self):
        binary = _skip_if_missing("fauxware", "fauxware")

        proj = angr.Project(binary, auto_load_libs=False)
        mgr = RustExplorationManager(proj, [proj.factory.entry_state()])

        mgr.run(until=lambda sm_: len(sm_.active) > 1)

        assert len(mgr.active) >= 2, (
            f"Expected >=2 active states, got {len(mgr.active)}"
        )
        inp0 = mgr.active[0].posix.dumps(0)
        inp1 = mgr.active[1].posix.dumps(0)
        assert b"SOSNEAKY" in inp0 or b"SOSNEAKY" in inp1, (
            f"SOSNEAKY not found in inputs: {inp0!r}, {inp1!r}"
        )

    # -----------------------------------------------------------------------
    # google2016_unbreakable_1 (direct — blank_state + symbolic memory)
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(60)
    def test_google2016_unbreakable_1_direct(self):
        binary = _skip_if_missing("google2016_unbreakable_1", "unbreakable")

        import claripy

        proj = angr.Project(binary, auto_load_libs=False)
        state = proj.factory.blank_state(addr=0x4005BD)
        flag = claripy.BVS("flag", 8 * 51)
        state.memory.store(0x6042C0, flag)
        for i in range(51):
            b = flag.get_byte(i)
            state.solver.add(b >= 0x20)
            state.solver.add(b <= 0x7E)
        state.options.add(angr.options.LAZY_SOLVES)

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x4005BD, avoid=0x400850, max_steps=10000)

        assert len(mgr.found) > 0, "No found states for google2016_unbreakable_1"

    # -----------------------------------------------------------------------
    # codegate_2017-angrybird (direct — manual register setup)
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(120)
    def test_codegate_angrybird_direct(self):
        binary = _skip_if_missing("codegate_2017-angrybird", "angrybird")

        proj = angr.Project(binary, auto_load_libs=False)
        state = proj.factory.entry_state(addr=0x4007C2)
        state.regs.rbp = state.regs.rsp
        state.mem[state.regs.rbp - 0x74].int = 0x40
        state.mem[state.regs.rbp - 0x70].long = 0x1000
        state.mem[state.regs.rbp - 0x68].long = 0x1008
        state.mem[state.regs.rbp - 0x60].long = 0x1010
        state.mem[state.regs.rbp - 0x58].long = 0x1018
        state.options.add(angr.options.LAZY_SOLVES)

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x404FAB, max_steps=10000)

        assert len(mgr.found) > 0, "No found states for codegate_angrybird"


# ===========================================================================
# Full solve.py execution tests (monkey-patched factory)
# ===========================================================================

class TestRustSolvePyExecution:
    """Execute each example's solve.py with the angr factory monkey-patched
    to return a RustExplorationManager, then verify the expected answer
    appears in stdout."""

    # -----------------------------------------------------------------------
    # defcamp_r100
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(180)
    def test_defcamp_r100(self, rust_factory_patch):
        example_dir = _binary_path("defcamp_r100")
        _skip_if_missing("defcamp_r100", "r100")

        output = _exec_solve_script(example_dir)
        assert "Code_Talkers" in output, (
            f"Expected 'Code_Talkers' in output, got: {output!r}"
        )

    # -----------------------------------------------------------------------
    # google2016_unbreakable_0
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(300)
    @pytest.mark.slow
    def test_google2016_unbreakable_0(self, rust_factory_patch):
        example_dir = _binary_path("google2016_unbreakable_0")
        _skip_if_missing(
            "google2016_unbreakable_0",
            "unbreakable-enterprise-product-activation",
        )

        output = _exec_solve_script(example_dir)
        assert "CTF{" in output, (
            f"Expected 'CTF{{' in output, got: {output!r}"
        )

    # -----------------------------------------------------------------------
    # google2016_unbreakable_1
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(300)
    @pytest.mark.slow
    def test_google2016_unbreakable_1(self, rust_factory_patch):
        example_dir = _binary_path("google2016_unbreakable_1")
        _skip_if_missing("google2016_unbreakable_1", "unbreakable")

        output = _exec_solve_script(example_dir)
        assert "CTF{" in output, (
            f"Expected 'CTF{{' in output, got: {output!r}"
        )

    # -----------------------------------------------------------------------
    # codegate_2017-angrybird
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(300)
    @pytest.mark.slow
    def test_codegate_angrybird(self, rust_factory_patch):
        example_dir = _binary_path("codegate_2017-angrybird")
        _skip_if_missing("codegate_2017-angrybird", "angrybird")

        output = _exec_solve_script(example_dir)
        assert "Im_so_cute&pretty_:)" in output, (
            f"Expected 'Im_so_cute&pretty_:)' in output, got: {output!r}"
        )

    # -----------------------------------------------------------------------
    # CADET_00001
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(300)
    @pytest.mark.slow
    def test_cadet_00001(self, rust_factory_patch):
        example_dir = _binary_path("CADET_00001")
        _skip_if_missing("CADET_00001", "CADET_00001")

        output = _exec_solve_script(example_dir)
        assert "EASTER EGG" in output, (
            f"Expected 'EASTER EGG' in output, got: {output!r}"
        )

    # -----------------------------------------------------------------------
    # flareon2015_2
    # -----------------------------------------------------------------------
    @pytest.mark.timeout(180)
    def test_flareon2015_2(self, rust_factory_patch):
        example_dir = _binary_path("flareon2015_2")
        _skip_if_missing("flareon2015_2", "very_success")

        output = _exec_solve_script(example_dir)
        assert "@flare-on.com" in output, (
            f"Expected '@flare-on.com' in output, got: {output!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
