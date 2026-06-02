#!/usr/bin/env python3
"""One-off spike (angr-rbnk): dump Z3 assertions from hackcon final-solve.

Runs hackcon2016_angry-reverser under Rust then Python engines, captures the
final-solve assertion stack from each, writes them to /tmp/hackcon_*.smt, and
prints quick structural stats.

Usage:
    python tools/dump_hackcon_smtlib.py
"""
import os
import resource
import sys
import time

# Ensure repo root on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Memory cap
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, resource.RLIM_INFINITY))

EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
HACKCON_DIR = os.path.join(EXAMPLES_DIR, "hackcon2016_angry-reverser")


def run_hackcon(engine):
    """Run hackcon, return (found_state, rust_mgr_or_None, elapsed)."""
    import angr
    import claripy

    sys.path.insert(0, HACKCON_DIR)
    os.chdir(HACKCON_DIR)

    rust_mgr_holder = {}

    if engine == "rust":
        from angr.exploration import RustExplorationManager
        original_sm = angr.factory.AngrObjectFactory.simulation_manager

        def patched_sm(factory_self, thing=None, **kwargs):
            if thing is None:
                states = [factory_self.entry_state()]
            elif isinstance(thing, (list, tuple)):
                states = list(thing)
            else:
                states = [thing]
            mgr = RustExplorationManager(factory_self.project, states)
            rust_mgr_holder["mgr"] = mgr
            return mgr

        angr.factory.AngrObjectFactory.simulation_manager = patched_sm
        angr.factory.AngrObjectFactory.simgr = patched_sm

    # Inlined solve.py logic (avoid re-exec hassles)
    flag = claripy.BVS('flag', 20 * 8, explicit_name=True)
    buf = 0x606000
    crazy = 0x400646
    find = 0x405a6e
    avoids = [0x402c3c, 0x402eaf, 0x40311c, 0x40338b, 0x4035f8, 0x403868,
              0x403ad5, 0x403d47, 0x403fb9, 0x404227, 0x404496, 0x40470a,
              0x404978, 0x404bec, 0x404e59, 0x4050c7, 0x405338, 0x4055a9,
              0x4057f4, 0x405a2b]

    proj = angr.Project('./yolomolo', auto_load_libs=False)
    state = proj.factory.blank_state(addr=crazy, add_options={angr.options.LAZY_SOLVES})
    state.memory.store(buf, flag, endness='Iend_BE')
    state.regs.rdi = buf
    for i in range(19):
        state.solver.add(flag.get_byte(i) >= 0x30)
        state.solver.add(flag.get_byte(i) <= 0x7f)

    simgr = proj.factory.simulation_manager(state)
    t0 = time.perf_counter()
    simgr.explore(find=find, avoid=avoids)
    explore_time = time.perf_counter() - t0
    found = simgr.found[0]
    return found, flag, rust_mgr_holder.get("mgr"), explore_time


def dump_rust(out_path):
    found, flag, mgr, explore_time = run_hackcon("rust")
    t0 = time.perf_counter()
    state_id = found.scratch.rust_found_state_id
    rust_ctx = mgr._rust_mgr.fork_state_solver(state_id)
    assertions = rust_ctx.get_all_constraints_str()
    # Final eval to make the timing realistic
    val = found.solver.eval(flag, cast_to=bytes)
    eval_time = time.perf_counter() - t0
    with open(out_path, "w") as f:
        for i, a in enumerate(assertions):
            f.write(f";; assertion {i}\n{a}\n\n")
    print(f"[rust] explore={explore_time:.2f}s eval={eval_time:.2f}s "
          f"assertions={len(assertions)} -> {out_path}")
    print(f"[rust] flag={val!r}")
    return assertions


def dump_python(out_path):
    found, flag, _, explore_time = run_hackcon("python")
    import claripy
    z3backend = claripy.backends.z3
    t0 = time.perf_counter()
    assertions = []
    for c in found.solver.constraints:
        z3_ast = z3backend.convert(c)
        assertions.append(z3_ast.sexpr())
    val = found.solver.eval(flag, cast_to=bytes)
    eval_time = time.perf_counter() - t0
    with open(out_path, "w") as f:
        for i, a in enumerate(assertions):
            f.write(f";; assertion {i}\n{a}\n\n")
    print(f"[python] explore={explore_time:.2f}s eval={eval_time:.2f}s "
          f"assertions={len(assertions)} -> {out_path}")
    print(f"[python] flag={val!r}")
    return assertions


def summarize(label, assertions):
    """Quick structural summary."""
    lens = [len(a) for a in assertions]
    total = sum(lens)
    max_l = max(lens) if lens else 0
    # Cheap depth proxy: max paren nesting level
    def max_depth(s):
        d = 0
        m = 0
        for ch in s:
            if ch == '(':
                d += 1
                if d > m:
                    m = d
            elif ch == ')':
                d -= 1
        return m
    depths = [max_depth(a) for a in assertions]
    # Count concat tokens & extract tokens (proxies for AST shape)
    concats = sum(a.count("concat") for a in assertions)
    extracts = sum(a.count("extract") for a in assertions)
    print(f"  {label}: n={len(assertions)} total_chars={total} "
          f"max_chars={max_l} max_paren_depth={max(depths) if depths else 0} "
          f"sum_depth={sum(depths)} concats={concats} extracts={extracts}")


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "both"
    if engine in ("rust", "both"):
        rs = dump_rust("/tmp/hackcon_rust.smt")
        summarize("rust", rs)
    if engine in ("python", "both"):
        # Re-init clean — angr state is process-local but the explore
        # mutates working dir; reset.
        py = dump_python("/tmp/hackcon_python.smt")
        summarize("python", py)


if __name__ == "__main__":
    main()
