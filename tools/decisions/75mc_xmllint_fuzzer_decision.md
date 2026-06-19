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
