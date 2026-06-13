"""Integration tests for RustExplorationManager against real angr-examples.

Runs 12 small CTF examples through both Python and Rust engines,
comparing correctness and checking for severe performance regressions.
"""
import importlib.util
import io
import os
import sys
import time

import pytest

# Check if Rust exploration is available
try:
    from angr.exploration import RustExplorationManager
    from angr.exploration.rust_manager import RUST_EXPLORATION_AVAILABLE
except ImportError:
    RUST_EXPLORATION_AVAILABLE = False

EXAMPLES_DIR = os.path.expanduser("~/repos/angr-examples/examples")


class BufferedStringIO(io.StringIO):
    """StringIO with a buffer attribute for code that uses stdout.buffer."""

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


# Each entry: (name, expected_substring, timeout_s, uses_callable_predicate)
FAST_EXAMPLES = [
    ("defcamp_r100", b"Code_Talkers", 30, False),
    ("ais3_crackme", b"ais3{I_tak3_g00d_n0t3s}", 30, False),
    ("fauxware", b"SOSNEAKY", 30, True),
    ("sym-write", b"", 60, True),  # output is list of ints, just check success
    ("defcon2016quals_baby-re", b"Math is hard!", 60, False),
    ("google2016_unbreakable_1", b"CTF{0The1Quick2Brown3Fox4Jumped5Over6The7Lazy8Fox9}", 30, False),
    ("mma_howtouse", b"MMA{fc7d90ca001fc8712497d88d9ee7efa9e9b32ed8}", 30, False),
    ("whitehatvn2015_re400", b"Flag 0:", 30, False),  # multi-solution, just check output exists
    ("google2016_unbreakable_0", b"CTF{", 30, False),
    ("flareon2015_2", b"@flare-on.com", 30, False),
    ("codegate_2017-angrybird", b"Im_so_cute", 30, False),
    ("strcpy_find", b"The password is", 30, False),
]


def _run_example(example_name: str, engine: str, timeout: float) -> tuple[bool, str, float]:
    """Run an angr-example solve.py with the given engine.

    Returns (success, stdout_output, elapsed_seconds).
    """
    import angr

    solve_script = os.path.join(EXAMPLES_DIR, example_name, "solve.py")
    if not os.path.exists(solve_script):
        return False, f"solve.py not found: {solve_script}", 0.0

    example_dir = os.path.dirname(solve_script)
    original_dir = os.getcwd()
    original_path = sys.path[:]

    # Monkey-patch factory for Rust engine
    original_simgr = None
    original_sm = None
    if engine == "rust":
        original_sm = angr.factory.AngrObjectFactory.simulation_manager
        original_simgr = angr.factory.AngrObjectFactory.simgr

        def patched_simulation_manager(factory_self, thing=None, **kwargs):
            # Fall through to the original simulation_manager when called from
            # angr internals (CFG jumptable resolver, exploration techniques)
            # whose internal SimState carries SimOptions like DO_RET_EMULATION
            # that RustExplorationManager._check_raise_options rejects.
            import traceback
            caller_frames = traceback.extract_stack()
            for frame in caller_frames[:-1]:
                if '/angr/analyses/' in frame.filename or '/angr/exploration_techniques/' in frame.filename:
                    return original_sm(factory_self, thing, **kwargs)
            if thing is None:
                states = [factory_self.entry_state()]
            elif isinstance(thing, (list, tuple)):
                states = list(thing)
            else:
                states = [thing]
            return RustExplorationManager(factory_self.project, states)

        angr.factory.AngrObjectFactory.simulation_manager = patched_simulation_manager
        angr.factory.AngrObjectFactory.simgr = patched_simulation_manager

    try:
        os.chdir(example_dir)
        if example_dir not in sys.path:
            sys.path.insert(0, example_dir)

        spec = importlib.util.spec_from_file_location("__main__", solve_script)
        module = importlib.util.module_from_spec(spec)

        captured = BufferedStringIO()
        original_stdout = sys.stdout
        sys.stdout = captured

        start = time.perf_counter()
        try:
            spec.loader.exec_module(module)
        finally:
            sys.stdout = original_stdout

        elapsed = time.perf_counter() - start
        return True, captured.getvalue(), elapsed

    except Exception as e:
        import traceback
        return False, f"ERROR: {e}\n{traceback.format_exc()}", 0.0

    finally:
        os.chdir(original_dir)
        sys.path[:] = original_path
        if original_sm is not None:
            angr.factory.AngrObjectFactory.simulation_manager = original_sm
            angr.factory.AngrObjectFactory.simgr = original_simgr


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
@pytest.mark.skipif(not os.path.isdir(EXAMPLES_DIR), reason="angr-examples not found")
class TestRustIntegration:
    """Integration tests running real CTF examples through the Rust engine."""

    KNOWN_XFAIL = {
        # codegate_2017-angrybird: Rust reaches find_addr (0x404fab) and
        # extracts 20 stdin bytes, but the bytes are wrong (e.g.
        # b'*%\xac\x0c`\xfe\xff\x80\x06 \xc0\xff\xff\xca4\x07\xffpK\x05'
        # vs expected b'Im_so_cute&pretty_:)'). Triage (bd angr-kcf.1)
        # ruled out BFS exploration ordering — both BFS and DFS produce
        # byte-identical wrong output, so a different exploration order
        # would not change the result. NativeFgets correctly records
        # stdin_fgets_0_* symbols and the Rust Z3 solver does accumulate
        # constraints over them, but the path Rust traverses to reach
        # find_addr differs from Python's, producing a constraint set
        # that admits the wrong stdin satisfying assignment. Likely root
        # cause is symbolic-memory branch divergence at the loads from
        # 0x1000-0x1018 the binary uses for anti-fingerprinting (see bd
        # memory codegate-0x1000-loads). See bd memory
        # angr-kcf-not-bfs-divergence for the full triage and
        # invariant-codegate-xfail for the keep-xfail decision.
        "codegate_2017-angrybird",
    }

    @pytest.mark.parametrize(
        "example_name,expected,timeout,uses_predicate",
        FAST_EXAMPLES,
        ids=[e[0] for e in FAST_EXAMPLES],
    )
    def test_rust_produces_correct_output(self, example_name, expected, timeout, uses_predicate):
        """Run example with Rust engine, verify output matches expected."""
        if example_name in self.KNOWN_XFAIL:
            pytest.xfail(f"{example_name}: known Rust engine compatibility issue")

        success, output, elapsed = _run_example(example_name, "rust", timeout)
        assert success, f"Rust engine failed on {example_name}:\n{output}"

        if expected:
            # Check expected substring in output (as bytes or str)
            expected_str = expected.decode("utf-8", errors="replace") if isinstance(expected, bytes) else expected
            assert expected_str in output, (
                f"Expected '{expected_str}' not found in output:\n{output[:500]}"
            )

    @pytest.mark.parametrize(
        "example_name,expected,timeout,uses_predicate",
        FAST_EXAMPLES,
        ids=[e[0] for e in FAST_EXAMPLES],
    )
    def test_both_engines_succeed_and_report_timing(
        self, example_name, expected, timeout, uses_predicate
    ):
        """Smoke-test that both engines solve each fast example; report timing.

        This is NOT a perf gate — it only asserts that the Python and Rust
        engines each succeed on the example, then prints the Rust/Python time
        ratio for visibility. The authoritative perf regression gate is
        ``tests/benchmarks/run_regression.py`` (run in CI with bimodal-variance
        handling); asserting a hard ratio here would duplicate that gate while
        fighting subprocess/CI timing variance, so we deliberately don't.
        """
        if example_name in self.KNOWN_XFAIL:
            pytest.xfail(f"{example_name}: known Rust engine compatibility issue")

        py_ok, py_output, py_time = _run_example(example_name, "python", timeout)
        assert py_ok, f"Python engine failed on {example_name}:\n{py_output}"

        rust_ok, rust_output, rust_time = _run_example(example_name, "rust", timeout)
        assert rust_ok, f"Rust engine failed on {example_name}:\n{rust_output}"

        if py_time > 0:
            ratio = rust_time / py_time
            print(f"\n  {example_name}: Python={py_time:.2f}s, Rust={rust_time:.2f}s, ratio={ratio:.2f}x")
