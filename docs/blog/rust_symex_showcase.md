# A Rust symbolic-execution engine for angr: now practical, and a little bit new

*Draft — angr-4n26m.11. Every number below was freshly measured with the
checked-in demo scripts under `tests/benchmarks/show_*.py` (run them yourself
with `python tests/benchmarks/show_all.py`). Nothing here is copied from
`baseline_timings.json`, which is a regression-gate file and is stale by
design.*

---

angr has always been able to symbolically execute real software. The honest
problem was never *can it* — it was *how long you waited*. We've been building a
Rust execution engine that plugs into the same angr you already use, driven from
Python through a thin `RustExplorationManager` wrapper, sharing one Z3 context
across the FFI boundary. This post walks through what that buys you, measured on
real binaries, with the wins and the non-wins both on the table.

Four angles: **raw speed**, **multi-architecture breadth**, **bug finding**, and
**checkpoint/resume** — plus a bonus look *inside* the solver.

## The setup

The Rust engine is opt-in and drop-in. You hand it the same project and the same
initial state you'd hand `simgr`:

```python
import angr

proj = angr.Project(binary, auto_load_libs=False)
state = proj.factory.entry_state()

mgr = angr.exploration.RustExplorationManager(proj, [state])
mgr.explore(find=success_addr)

found = mgr.found[0]
print(found.posix.dumps(0))   # the stdin that reaches success_addr
```

Under the hood: a Rust VEX interpreter, a lazy copy-on-write memory model, and
native SimProcedures for the hot libc paths — all sharing Python's Z3 so ASTs
pass across the boundary without re-translation.

## 1. Raw speed: minutes become seconds

This is the headline, and it's a *practicality* claim, not a *possibility* one.
Pure-Python angr could already solve every target below; the Rust engine just
turns "go get a coffee" into "blink."

Each row is the **same workload** run end-to-end under both engines, median of
**N=5**, with the observed min–max range. (`tests/benchmarks/show_raw_speed.py`.)

| Target | Python (s) | Rust (s) | Speedup |
|---|---|---|---|
| `ekopartyctf2016_rev250` | 33.48 (32.87–33.76) | 2.28 (2.25–2.28) | **14.7×** |
| `flareon2015_5` | 25.69 (25.39–26.14) | 3.48 (3.23–3.50) | **7.4×** |
| `sharif7_rev50` | 3.48 (3.43–3.52) | 0.87 (0.86–0.88) | **4.0×** |
| `ais3_crackme` | 2.63 (2.60–2.67) | 0.86 (0.83–0.89) | **3.1×** |
| `defcamp_r100` | 1.06 (1.05–1.10) | 0.28 (0.27–0.29) | **3.8×** |

The largest wins are on the workloads where you spend real wall-clock — a 33s
solve dropping to 2.3s is the difference between an interactive tool and a batch
job.

**The non-win, on the table:** a static `busybox` smoke target runs in **0.46s**
under Python and **1.39s** under Rust — i.e. **0.33×**, a *loss*. When a run is
dominated by one-time state-export and engine-init overhead and barely touches
the solver, Python's head start wins. The Rust engine pays its setup tax up
front and only amortizes it once there's real symbolic work to do. We keep this
row in the demo on purpose: the speedup is real where it's real, and we're not
going to hide where it isn't.

**A note on the benches we *don't* quote as wins:** four targets in our corpus
are bimodal under Z3 (`google2016_unbreakable_1`, `securityfest_fairlight`,
`ekopartyctf2016_sokohashv2`, `hackcon2016_angry-reverser`) — their runtime is
dominated by an irreducible Z3 `check()`/`eval` floor, and Rust is at or below
parity there (0.3×–0.7×). Those are documented architectural limits, not
headline material. See `docs/advanced-topics/rust_bimodal_variance.rst`.

## 2. Multi-architecture: breadth and parity

The Rust engine isn't x86-only. The same logical crackme — *symbolic input must
equal 42 to reach the success block* — runs correctly across **six**
architectures spanning 32/64-bit and little/big-endian
(`tests/benchmarks/show_multiarch.py`):

| Arch | Endian | Found | Solved input |
|---|---|---|---|
| AMD64 | LE | ✓ | 42 |
| AARCH64 | LE | ✓ | 42 |
| ARMEB | **BE** | ✓ | 42 |
| X86 | LE | ✓ | 42 |
| MIPS32 | LE | ✓ | 42 |
| MIPS64 | LE | ✓ | 42 |

This is a **parity / correctness** claim, deliberately **not** a speed claim. We
don't have a Python baseline for a real non-x86 binary to compare against — the
firmware corpus (`../binaries`) isn't present here, and the one real ARM ELF
bench we do have (`android_arm_license_validation`) is actually ~0.8× under Rust.
So the story is "it runs everywhere, and it gets the right answer everywhere,"
which for a symbolic engine is the load-bearing part. Every target above is
built in-memory from byte encodings lifted verbatim from the CI-verified
`tests/engines/rust/test_multiarch.py`, so the demo reproduces from a clean
checkout with no cross-compiler.

## 3. Finding a bug: guided reachability

Speed only matters if it's pointed at something. Here's the engine used the way
you'd actually use it in a triage loop — drive execution *toward* a dangerous
sink and *away* from the safe path (`tests/benchmarks/show_vuln_finding.py`, on
the `strcpy_find` example):

```python
mgr = angr.exploration.RustExplorationManager(proj, [state])
mgr.explore(find=sink_addr, avoid=safe_addr)

if mgr.found:
    print(mgr.found[0].posix.dumps(0))   # input that drives to the strcpy sink
```

The engine reaches the `strcpy` sink at `0x4001e0` in **0.04s**, producing a
concrete stdin (`"Totally not the password..."`) that drives execution to the
vulnerable call while avoiding the safe branch — a single accumulated path
constraint. That's the whole point of symbolic reachability: you describe the
condition (*reach this address*), and you get back an input that satisfies it.

## 4. Checkpoint/resume: pick the search back up

This one is genuinely **new** — but we're going to be precise about what it does,
because the precise version is still useful and the over-claimed version is a
lie.

You can snapshot a live exploration to disk, exit the process entirely, and in a
*fresh* process resume the **search** from where you left off
(`tests/benchmarks/show_checkpoint_resume.py`, on `fauxware`):

```python
# Process A: explore partway, then snapshot
mgr.run(max_steps=14)
mgr.dump_to_disk("snapshot.bin")     # 1.66 MB
# ... process A exits ...

# Process B (fresh interpreter): resume and finish
mgr = RustExplorationManager.load_from_disk(proj, "snapshot.bin")
mgr.explore(find=0x4006ed)
```

What's **guaranteed**: the search frontier is restored *exactly*. We fingerprint
the structural frontier — stash counts plus the sorted set of active block
addresses — immediately before the dump and immediately after the load:

```
before dump:  {active: 2, deadended: 1}  @ [0x4006df, 0x400713]
after load:   {active: 2, deadended: 1}  @ [0x4006df, 0x400713]   ✓ identical
```

The snapshot survives a real process boundary (write file → exit → fresh
`python` → read file), and the resumed search goes on to find the target at
`0x4006ed`.

What is **not** guaranteed — and where the honesty framing matters: this is
"resume the **search**," not "restore the **run** bit-for-bit." Model equality
across a restore is not guaranteed even with `deterministic=True`; the resumed
solver may hand you a *different* satisfying input for the same path, because the
find address is reachable by more than one input. Concretely, two artifacts
show the two halves of this: the headline checkpoint/resume run
(`show_checkpoint_resume.py`) recovers a *different* satisfying input after the
restore than it would have pre-dump — exactly the "not bit-for-bit" behavior
above. A separate deterministic-mode probe (`show_resume_boundary.py`) *does*
land on a stable `SOSNEAKY`-shaped value across the boundary, but treat that as
an **observed bonus**, not a contract — it's a different run with determinism
pinned, not the headline demo. If you need binary-identical replay, that's a
separate guarantee we don't make here. See
`docs/advanced-topics/rust_engine.rst` (snapshot limits) and the determinism
boundary probe in `show_resume_boundary.py`.

## Bonus: looking inside the solver

The Rust engine ships per-call-site Z3 instrumentation that's hard to get out of
stock Python angr. Bracket a search with `reset_solver_stats()` /
`get_solver_stats()` and you can see exactly where the solver budget went
(`tests/benchmarks/show_solver_instrumentation.py`, on `defcamp_r100`, solution
`Code_Talkers`):

- **25** `check()` calls, **43.8 ms** total — about **35%** of the whole 0.127s
  search.
- Of that check time, the *satisfiable* call site alone is **98%**; the
  branch-true/branch-false sites are rounding error.
- **13** lazy-fork materializations (`z3_materialize_count`) — forks that only
  paid for a Z3 clone once they actually diverged.
- **11 / 1** branch model hit/miss — most branch directions resolved from a
  cached model without a fresh solve.

That breakdown — "your search spent a third of its time in Z3, and almost all of
*that* in one call site" — is the kind of thing that turns a vague "it's slow"
into an actionable lead. (Heads-up if you wire this up yourself:
`get_solver_stats()` returns the *full* counter dict, not just the `z3_*` keys —
filter for what you want.)

## Try it

Everything above is a checked-in script, and a single runner verifies the whole
set reproduces:

```bash
python tests/benchmarks/show_all.py            # all demos, PASS/SKIP/FAIL
python tests/benchmarks/show_all.py --quick    # fast subset
python tests/benchmarks/show_raw_speed.py      # just the speed table
```

The honest summary: the Rust engine makes real-software symbolic execution
**practical** (single-digit-to-15× on solver-bound workloads, at parity or worse
on init-bound ones), runs **correctly across six architectures**, and adds
**resumable search** and **solver introspection** that Python angr doesn't have.
It is not a universal speedup and it is not a bit-exact time machine — and now
you have the scripts to check every one of those claims yourself.
