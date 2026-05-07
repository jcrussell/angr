# Loop session notes (2026-05-07, 102nd loop session)

## Task: angr-b6og — Audit native/angr for allocator hotspots via flamegraph (CLOSED)

Bead closed (commit 03d825fa3).

### What changed

Committed `tests/benchmarks/profile_rust_bench.sh` (the previous session
created it but left it untracked) and fixed two issues that prevented it
from producing readable profiles on this host:

1. **strip override.** Workspace `[profile.release]` sets `strip = "symbols"`
   and the bench profile inherits that. perf reports were showing only raw
   `0xNNN` addresses. Script now passes
   `--config 'profile.bench.strip=false' --config 'profile.bench.debug=true'`
   to all `cargo bench` invocations.
2. **z3 header fallback.** When `Z3_SYS_Z3_HEADER` is unset and
   `.venv/.../z3/include/z3.h` is missing, the script now auto-falls-back to
   `/usr/include/z3.h`. The same workaround that's needed for `cargo check`.

Also pointed `TARGET_DIR` at the workspace root target (`$REPO_ROOT/target`)
since cargo writes there, not into the package's own `target/`.

### Audit findings

Validated end-to-end with `perf record` (after `sudo sysctl -w
kernel.perf_event_paranoid=2` — the loop-agent host has passwordless sudo).
Captured profiles for four representative groups; raw artifacts at
`target/profile/{rustbv_symbolic,symcontext_fork,memory_,state_fork}/`.

Top hotspots:
- **rustbv_symbolic** — `drop_in_place<RustBV>` + `Arc::drop_slow` ~36%,
  malloc/cfree ~14.6%, Z3 ref-count ~6%, op methods 17.5%. Arc-tree teardown
  beats the math.
- **symcontext_fork** — `Z3_dec_ref` + `Z3_inc_ref` 22.4%, fork-time vec
  clone (`Vec::extend_trusted`) 10.75%. Each cloned constraint pays 2 FFI
  ref-count ops.
- **state_fork** — three `HashMap::clone` instances total ~9%; suggests at
  least one map field is still doing a deep clone instead of structural
  sharing.
- **memory_** — `load_concrete` 13.97%, `BuildHasher::hash_one` 4.26%; the
  default SipHasher dominates concrete page lookups.

### Follow-ups filed

- `angr-6n56` (P2) — Arc/arena experiment for transient RustBV results.
- `angr-w6nq` (P2) — Share Z3 assertion vec by Arc instead of cloning.
- `angr-ar8r` (P2) — Audit which `RustSimState` maps still deep-clone.
- `angr-75y3` (P3) — Try `FxHasher` for the `SymbolicMemory` page map.

### Memories saved

- `b6og-perf-hotspots` — full per-group hotspot summary with paths to the
  perf data so future audits start from this baseline.
- `invariant-profile-rust-bench-script` — the two non-obvious traps the
  script now handles (strip override, Z3 header fallback) plus the runtime
  prereqs (perf paranoid, sudo on this host, no inferno locally).

### Verification

- `cargo check --release` clean (with `Z3_SYS_Z3_HEADER=/usr/include/z3.h`).
- `bash -n` of the script: clean.
- `perf report` on captured profiles resolves Rust symbols correctly.
- No production code touched, so the test suite was not re-run; the change
  is purely dev-tooling.

## Suggested next slices

- `angr-w6nq` (Z3 ref-count Arc-share) — direct, well-scoped optimisation;
  bench is already wired for the before/after measurement.
- `angr-ar8r` (HashMap clone audit) — needs a quick read of
  `RustSimState::fork` + per-field profile pass to localise the clone.
- `angr-pufm` lazy guarded-entries (still open from prior sessions).
- `angr-fk0m` mixin unification.
