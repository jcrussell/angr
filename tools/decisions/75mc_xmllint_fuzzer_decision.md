# Decision: angr-75mc xmllint bench unblock — fuzzer feature vs. alternative solve

## Status

Bead: `angr-75mc` ("Bench .1: add xmllint (real-world libc-heavy
utility binary)") — P2, parent epic `angr-vx8p`.

Blocked since 2026-06-03 (iter 85) on the same decision: the
in-tree `angr-examples/examples/xmllint/solve.py` imports
`angr.rustylib.fuzzer.{Fuzzer, InMemoryCorpus, ClientStats}`,
which is gated behind the `fuzzer` Cargo feature. Default
features are `["vex-engine", "vex-engine-z3", "automaton"]`
(see `native/angr/Cargo.toml:12`); fuzzer is off by default.

The 2026-06-05 dreamy-frog review (U1) said: pick a path,
apply, update notes — **do not file another tracking bead**.
This brief is that pick.

vx8p siblings already closed as won't-fix on adjacent feature-
gate / data-gap problems: `angr-528r` (no binary checked in),
`angr-86c4` (TRACK_ACTION_HISTORY + OOM). 75mc is the only
bead the epic still blocks on.

## Context

xmllint is the only real-world libc-heavy utility in
angr-examples. The vx8p epic premise was that adding it would
exercise unregistered-syscall fallback paths that the CTF
corpus does not — validating round-1 syscall beads
(`angr-8j16`, `angr-6009`, `angr-aig2`, `angr-ecut`) have a
measurement surface.

The in-tree solve.py uses the **fuzzer** to mutate a seed
corpus until it hits the XML-entity-resolution code path. The
seed differs from a positive case by a single byte flip — this
is a property of the fuzzer harness, not the symbolic-execution
engine. Replacing the fuzzer with vanilla `simgr.explore()`
changes what xmllint exercises in the engine.

## Options

### (a) Enable `fuzzer` Cargo feature in default build

Change `native/angr/Cargo.toml:12` from

    default = ["vex-engine", "vex-engine-z3", "automaton"]

to

    default = ["vex-engine", "vex-engine-z3", "automaton", "fuzzer"]

**What we can verify offline:** `cargo check
--manifest-path native/angr/Cargo.toml --release` resolves the
deps (icicle-fuzzing / icicle-vm / pcode from git rev
`4d7ed93254a20b7e5c16bd7b0c6b46db49e1c72e`, libafl /
libafl_bolts `=0.15.3`). Cargo has the git deps cached at
`~/.cargo/git/db/icicle-emu-5f59d77c9fb950c7`.

**What we cannot verify offline:**

- Wheel build (`pip wheel .`) under `cibuildwheel` — git deps
  with pinned rev work in local cargo but the wheel-build
  sandbox may have stricter networking constraints.
- Build-time inflation for downstream contributors. The
  libafl + icicle-emu transitive set adds substantial
  compilation time (estimated 30-60s warm cargo check, far
  more for cold).
- That CI's macOS / Windows jobs (if any) still pass — libafl
  has historically had platform-specific build constraints.
- Whether 75mc would actually pass under run_single.py:
  fuzzer runs are nondeterministic and may not produce stable
  baseline_timings.json entries.

**Side effects:**

- Every contributor's local rebuild pulls icicle-emu + libafl
  even if they never touch fuzzer code.
- Cargo.lock churn — fuzzer deps move from optional/unused to
  always-compiled, increasing the resolved set.
- Cargo workspace size grows.

### (b) Write an alternative non-fuzzer xmllint solve.py

Replace the in-tree `angr-examples/examples/xmllint/solve.py`
with a vanilla symbolic-execution harness:

- `entry_state(args=[target, "--noout", "--nonet", "--recover",
  "--noent", "-"])` with `ZERO_FILL_UNCONSTRAINED_*`
- Symbolic stdin of a fixed length (e.g. 80 bytes)
- `simgr.explore(find=<addr>)` where `<addr>` is one of:
  - A specific syscall callsite (read/open/stat-family) reached
    on the entity-resolution path
  - A library-symbol callsite identifiable from the dynamic
    symbol table (xmllint is stripped but PLT entries survive)
  - The return address of `xmlSAX2Reference()` if symbols
    leak through libxml2 (would need a non-stripped libxml2
    in the load path)

**What we can verify offline:** that the solve.py imports
cleanly without `angr.rustylib.fuzzer`.

**What we cannot verify offline:**

- The chosen `find=` target is actually reachable in a sane
  step budget. xmllint has deep call graphs; vanilla
  exploration may not converge.
- Wall time is bounded enough to land in FAST or even MEDIUM
  tier — risk of joining the "MEDIUM/SLOW catalog at
  `run_single.py:74`" with no useful gate.
- The new harness validates the *same* round-1 syscall beads.
  The fuzzer hits entity-resolution; a `find=puts@plt` smoke
  test does not.

**Side effects:**

- Cross-repo: solve.py lives in `angr-examples`, not this
  repo. Path (b) requires a separate angr-examples PR
  (similar boat to `angr-b3sc` archinfo).
- Existing fuzzer-based solve.py is the only end-to-end angr
  fuzzer integration test in angr-examples. Replacing it
  removes that coverage; an additive
  `solve_symex.py` next to `solve.py` would preserve it.

## Recommendation

**Defer to human decision, with a lean toward (b) additive.**

Reasoning:

1. Path (a) has at least three categories of risk we cannot
   verify from this box: wheel-CI, platform-CI, baseline
   stability. Flipping a default feature for the *only*
   blocker on a P2 bench should not be done without the
   ability to test those paths. The loop pattern in
   `OFFLINE_WORKFLOWS.md` (and in `tools/decisions/README.md`,
   sibling of this file) says: when neither path is safe to
   apply autonomously, file the brief and surface the
   decision.

2. Path (b) is safer to *prepare* offline but the cross-repo
   landing is real friction. The cleanest version is
   **additive**: file `solve_symex.py` next to the existing
   `solve.py` in angr-examples, leave the fuzzer harness in
   place, and have run_single.py prefer the
   non-fuzzer script when the fuzzer feature is off (existing
   pattern: harness auto-discovery). This preserves the
   fuzzer integration coverage and unblocks the bench.

3. If the human picks (a), the unblock is mechanical and the
   risk-reduction work is post-landing (watch wheel CI for a
   release cycle, revert if it breaks). If the human picks
   (b), the next slice is a cross-repo angr-examples PR + a
   run_single.py harness-discovery tweak.

**If forced to pick autonomously** the recommendation is
(b) additive: lower blast radius, additive change is easier
to revert, no wheel-CI risk.

## Implementation sketch — path (b) additive

Three artifacts, one offline (here), two needing the next
human-in-the-loop iter:

1. **(offline, this brief)** Decision documented; bead notes
   updated to point here.
2. **(cross-repo, online)** Open angr-examples PR adding
   `examples/xmllint/solve_symex.py`. The new script:
   - Imports angr + claripy only (no `angr.rustylib.fuzzer`).
   - Builds the entry state per the spec above.
   - Picks `find=<addr>` after `objdump -d xmllint_bin |
     grep '<plt>'` (or `nm -D`) identifies a deterministic
     library-call address reachable on the parse path. First
     candidate: the address of `getenv@plt` (xmllint reads
     `XML_DEBUG_CATALOG` early; small step budget).
   - Returns `(found_state, len(simgr.found))` for the
     harness.
3. **(in-tree, online)** Update
   `tests/benchmarks/run_single.py` xmllint catalog entry
   (line ~74 family) to prefer `solve_symex.py` when present.
   Capture rust+python baseline, add to
   `baseline_timings.json` per
   `invariant-benchmark-suite-tuple-format`.

The cross-repo angr-examples PR is the same offline-prepare
pattern as `angr-b3sc` (archinfo) — draft the patch + PR body
here, file the actual PR online.

## Implementation sketch — path (a) if chosen

One artifact, one verification, one watch:

1. **(offline, here)** Edit `native/angr/Cargo.toml:12` to
   include `"fuzzer"` in `default`. Run `cargo check
   --manifest-path native/angr/Cargo.toml --release` to
   confirm the deps resolve.
2. **(online)** Watch the next wheel-build CI run. If it
   passes on all platforms, the path holds; if it breaks,
   revert and switch to path (b).
3. **(online)** Run `run_single.py xmllint --both`, capture
   timings, add to `baseline_timings.json`. Confirm the run
   is deterministic enough for a regression gate (fuzzer
   seed is currently hardcoded to 12751 in solve.py, which
   helps).

## Offline validation of path (b) tractability (2026-06-19, iter 11)

The brief's primary unverifiable risk for path (b) was: "the chosen
`find=` target is actually reachable in a sane step budget … vanilla
exploration may not converge" and "wall time bounded enough to land in
FAST/MEDIUM tier." An offline probe (`/tmp/xmllint_probe.py`, Python
engine, symbolic 16-byte stdin, `use_sim_procedures=True`, isolated 4G
systemd scope, hard step caps) resolves this empirically:

- **No state explosion.** Through 2000 steps / ~10.5s the run stays
  **single-state** (active==1, 0 deadended, 0 errored). The startup +
  CLI option-parsing path is fully deterministic — no symbolic branching
  — so the "deep call graphs may diverge / OOM" fear does **not**
  materialize in this regime. Wall time grows ~linearly (~0.4s per 100
  steps after a ~3s project-load + first-step warmup).
- **`find=getenv` is trivially reachable: 185 steps, ~3.8s.** getenv is
  hit on the deterministic startup path (xmllint reads
  `XML_DEBUG_CATALOG`/`XML_CATALOG_FILES` early), well inside FAST tier.
  Viable as a *mechanics smoke* bench, but it fires **before** symbolic
  stdin is consumed, so it does NOT exercise the entity-resolution /
  syscall-fallback paths the vx8p epic's secondary criterion wants.
- **The read/parse callsites are NOT reached in 2000 steps / 10.5s.**
  Targeting `fread`/`fgets`/`xmlReadFd`/`xmlReadMemory` (PLT) the run is
  still churning deterministic CLI-option processing at the step-2000 cap
  (cycling addrs 0x406xxx binary / 0x7b31c0 libc). Reaching the actual
  parser needs either a much larger step budget or a `call_state`/
  `blank_state` seeded closer to `xmlReadFd` to skip the long
  deterministic startup. Whichever is chosen, exploration stays bounded.

**Implication for the decision:** path (b) is *mechanically viable and
safe* (bounded, no OOM, cheap). The remaining design choice is the
`find=` target — a shallow `getenv` smoke vs. a parser-path target that
needs a deeper budget or a seeded start state. This does NOT change the
a/b recommendation (still lean (b) additive), and does NOT itself unblock
75mc: the cross-repo angr-examples `solve_symex.py` PR + the
`baseline_timings.json` add still require the human-in-the-loop /
online iter. It removes the "won't converge / will explode" risk from
the (b) column.

## Resolved

(Pending — leave for the human-in-the-loop iter that applies
the chosen path. Format expected: `Resolved: applied path
<a|b> on <YYYY-MM-DD>, commit <hash>.`)

## Iter 30 addendum — path-b real-glibc blocker removed

Empirical follow-up to iter11. iter11 probed with
`use_sim_procedures=True` (libc stubbed) and found exploration
tractable. iter30 probed the **`use_sim_procedures=False`** path —
the one that actually exercises the real syscall-fallback surface
the vx8p epic premise wanted — and found it died at step 3 on
`operation error: unmapped VEX opcode: Iop_GetMSBs8x16` (SSE
PMOVMSKB inside glibc's SSE strlen/memchr during startup).

Implemented the missing op (commit `af98545ea`, `VGetMSBs`:
`Iop_GetMSBs8x{8,16}`). Post-fix, vanilla xmllint advances past the
wall (errored 1->0, reaches step 5 then deadends). This is a real,
generally-useful engine improvement (benefits any real-glibc binary),
but does **not** by itself land the bench: full-glibc symex without
sim procedures will hit a chain of further missing ops/syscalls before
reaching the parser. The a/b decision still stands; if (b) is chosen,
the pragmatic sub-choice is `use_sim_procedures=True` (iter11's bounded
path) vs. continuing to grind out real-glibc op blockers.

Diagnostic harness: `tools/xmllint_probe.py` (bounded steps, RLIMIT_AS,
stash-watching). bd memory: `xmllint-vanilla-symex-probe`.

## Iter 31 addendum — path-b dead end is uninitialized glibc, not an op chain

iter30 predicted a "chain of further missing ops/syscalls" past the
`VGetMSBs` wall. iter31 traced it (PC-by-PC, both engines, via the
extended `tools/xmllint_probe.py` `PROBE_ENGINE=rust|python`) and the
prediction was **wrong**: there is no further missing-op chain. The
vanilla `use_sim_procedures=False` path runs into **unmapped/garbage
memory** within a handful of blocks and dies — on *both* engines:

- **Rust:** deadends at PC `0x6` by step 5 (Rust chains many VEX blocks
  per `step`, so "step 5" is deep). No errored/unmapped-op; just a
  garbage jump target.
- **Python:** errors at unmapped `0x1043554` at step 22 (one basic block
  per step).

Root cause is **not** an engine bug — CLE prints
`invalid tls_data_size. Skip TLS loading` at load, so glibc startup
(`__libc_start_main`) executes against uninitialized TLS / unrelocated
init structs and computes garbage jump targets. Verified this is *not*
a Rust memory/relocation defect: the `call qword ptr [0x413fc0]` pointer
in `_start` reads identically as `0x72a200` (`__libc_start_main`) under
CLE ground truth, the Python state, **and** the Rust state. The
step-by-step PC divergence between engines is purely block-chaining
granularity (Rust chains; Python single-steps), not a control-flow
correctness gap.

**Implication for the a/b decision:** path-b "grind out real-glibc op
blockers" is a mirage — the blocker is TLS/glibc-init modeling, not the
VEX op surface. The only viable path-b sub-choice is iter11's
`use_sim_procedures=True` bounded harness (stubs glibc init, sidesteps
the uninitialized-TLS wall). path-(a) (fuzzer Cargo feature) remains the
other option. bd memory: `xmllint-path-b-glibc-init-wall`.

## iter32 addendum — measured `use_sim_procedures=True` cross-engine run (angr-6d3l)

iter31 established that the *vanilla* (`use_sim_procedures=False`) path
is dead on both engines (uninitialized-TLS wall). iter32 measured the
**viable** path-b config — `use_sim_procedures=True` — under both engines
via `tools/xmllint_probe.py PROBE_SIM_PROCS=1 PROBE_ENGINE=rust|python`
(new toggle; same ZERO_FILL entry state, 80-step budget, 4G nested scope).

**Result — stubbing glibc init clears the iter31 wall, but the engines
DIVERGE in control flow:**

| | Rust | Python |
|---|---|---|
| First observed active PC | `0x406820` (main region) | `0x408a80` (`_start`) |
| Behaviour | exits the `0x406883`↔`0x406896` main loop after ~1 iter, jumps to **stack** PC `0x7ffffffeffd0`, **deadends at step 20** | grinds the `0x406896` loop 50+ steps, still active at step 80 (BUDGET EXHAUSTED) |
| Stash | single state throughout (no explosion) | single state throughout |
| Peak RSS | ~370 MB | ~244 MB |
| Wall | 3.7 s | 3.4 s |

So **resource cost is comparable and bounded on both engines** (no
explosion, no OOM) — but **control flow is NOT identical**. Python iterates
the main-region loop many times; Rust exits early and deadends by jumping
into the stack (`No bytes in memory for block starting at 0x7ffffffeffd0`).

The divergence already shows at the very first observed PC: Rust's state
is at `main` (`0x406820`) while Python is still at `_start` (`0x408a80`),
pointing at a difference in the **`__libc_start_main` SimProcedure /
entry→main stack-arg setup** rather than a mid-program op. This is a
**candidate Rust correctness divergence on a real binary** and warrants a
dedicated investigation bead (filed; see bd). It does NOT block the a/b
decision — it is a 6d3l characterization finding. bd memory:
`xmllint-simprocs-cross-engine-divergence`.

## 2026-06-21 addendum — path (b) DEEP target empirically NOT viable (post-aca6y)

Decision was: apply path (b) and invest in a deep, parser-reaching target
(`xmlReadFd@plt`/`fread@plt`/`xmlParseDocument@plt`) instead of the shallow
`getenv` smoke. Drove it with `simgr.explore(find=<rebased plt addr>)` — the
mechanism the bench would actually use (ad-hoc probes: `use_sim_procedures=True`,
ZERO_FILL entry state, symbolic vs concrete stdin, 3.5 GB RLIMIT_AS; pattern
reusable from `tools/xmllint_probe.py`). Result:
**neither engine reaches the parser; both blow up well before it, with concrete
XML stdin (no symbolic input at all):**

- **Python engine:** `z3.z3types.Z3Exception: out of memory` inside a `satisfiable()`
  check — constraint explosion. (Consistent with iter-11: getenv reachable at 185
  steps, but the parser callsites never reached.)
- **Rust engine:** `TypeError: unsupported operand for -: NoneType - NoneType` in
  `concretization_strategies/range.py` (`mx - mn`, both `None`) via the Python
  address-concretization fallback — crashes **before even `getenv@plt`**, within
  ~60 s while RSS climbs past 1.2 GB, states forking to 22 active / 187 deadended
  (many deadends at PC `0x0`).

**Root cause:** with `use_sim_procedures=True`, stubbed libc returns *unconstrained
symbolic* values that propagate into pointers and branch conditions → state +
constraint explosion → Z3 OOM (Python) / symbolic-pointer concretization that
returns `None` bounds (Rust fallback). This is independent of stdin: concrete XML
on stdin crashes identically. The only thing that has ever driven xmllint's parser
is the **fuzzer (concrete execution, no symbolic branching)** — i.e. path (a).

**Implications:**
1. A deep *symbolic* xmllint bench is not achievable on either engine without first
   taming the SimProcedure-injected-symbolic explosion (large, open-ended work).
2. Even a *shallow* symbolic bench is not viable on the **Rust** engine — it crashes
   before `getenv`. So the iter-11 "getenv smoke" fallback only ever worked under
   Python.
3. The Rust `mx - mn None` concretization-fallback crash is a genuine robustness
   bug surfaced by a real binary — worth its own bead regardless of the bench.

Net: the path-(b) deep-target route is a dead end as plain symex. Realistic routes
to a deep xmllint workload are (a) the fuzzer feature (concrete execution) or a
pivot to a different/smaller real binary for the vx8p syscall-surface goal.
(Reproduce with an `explore(find=rebased_plt)` probe à la `tools/xmllint_probe.py`,
symbolic or concrete stdin — both explode.)

**Rust crash (item 3) FIXED — angr-8iv6j.** Root cause: the callback solver shim
`_rust_min`/`_rust_max` (`rust_callback_dispatch.py`, and the sibling
`RustSolverFallback` in `rust_state_export.py`) only fell back to claripy on an
*exception*, but `RustSolverContext.min`/`max` *return* `None` (Option) for an
unsat ctx or `width > 128` (solving_ops.rs). The `None` reached `range.py`'s
`mx - mn` → `TypeError`. Fix disambiguates the two `None` causes instead of blindly
falling back: **width>128** → fall back to claripy (its big-int solver can bound it);
**unsat** → raise `SimUnsatError` directly (the state is dead and claripy would only
re-derive the same unsat — round-tripping to Python is wasted work; the sat result is
cached so the check is free). Regression tests:
`tests/engines/rust/test_procedures.py::TestCallbackSolverConcretizationFallback`
(width>128 fallback + unsat-raises-without-Python-round-trip). This only removes the
Rust hard-crash; xmllint plain symex still explodes (Z3 OOM) on both engines, so it
remains unviable as a symbolic bench.
