## Session log: 2026-05-08, 168th loop session

### Task: angr-4j5u.1 (closed) — Extract ProfilingCollector struct

First child of the angr-4j5u decomposition epic. Move three profiling-related
fields off `RustExplorationManager` and into a single `ProfilingCollector`
struct so the manager can track an enable flag + per-step stats + native-proc
counters via one delegated field.

### What changed

**New: native/angr/src/exploration/profiling.rs (34 lines)**
- `ProfilingCollector` struct with three pub(crate) fields:
  `profiling_enabled: bool`, `accumulated_stats: ExecutionStats`,
  `native_proc_stats: NativeProcStats`. `#[derive(Default)]`.
- `NativeProcStats` moved here from mod.rs; the inline `impl Default`
  collapsed to a derive (HashMap::default == HashMap::new for the
  `call_counts` field).

**native/angr/src/exploration/mod.rs**
- New `mod profiling;` + `use self::profiling::ProfilingCollector;`.
- Removed inline `NativeProcStats` definition (~17 lines).
- Removed the three field declarations on `RustExplorationManager`,
  replaced with a single `pub(crate) profiling: ProfilingCollector`.
- Constructor: three `field: default()` lines collapse to
  `profiling: ProfilingCollector::default()`.
- All ~25 callsites updated mechanically:
  `self.profiling_enabled` → `self.profiling.profiling_enabled`,
  `self.accumulated_stats.<x>` → `self.profiling.accumulated_stats.<x>`,
  `self.native_proc_stats.<x>` → `self.profiling.native_proc_stats.<x>`.

**native/angr/src/exploration/stepping.rs**
- Same mechanical replacement for the ~20 callsites here.
- No structural changes; uses `super::*` so it picks up `ProfilingCollector`
  via the parent module.

### Field count on the manager: 42 → 40 (net -2)

The bead description said "95-field god struct" but the actual count is in
the low 40s (per `avoid-deferred-4j5u` memory). Three fields collapsed to
one means 42 → 40.

### Build hiccup (recurring)

The CLAUDE.md `pip install -e . --no-build-isolation --no-deps` path is still
broken — `.venv/bin/pip` is missing entirely (just `python` + `python3.12`).
Used the cargo + cp fallback per `avoid-broken-venv-pip-fallback-cargo-build`:
1. `cargo build --manifest-path native/angr/Cargo.toml --release`
2. `cp -f target/release/librustylib.so angr/rustylib.cpython-312-x86_64-linux-gnu.so`

### Final state

- 342/342 tests passing on tests/engines/test_rust_exploration.py (19.6s)
- fauxware: OK rust 0.34s peak_mem=188MB
- ais3_crackme: OK rust 0.96s peak_mem=387MB
- Profiling counters in benchmark output unchanged (block_exec, lift,
  z3 stats, etc. — all populated as before)
- Commit: `c3af9aa80` on rust-symex

### Memories saved

- `invariant-profiling-collector-direct-fields` — pub(crate) direct field
  access by design; do not "improve" it with methods as a separate cleanup
  (parent angr-4j5u was deferred multiple times for cosmetic gain)

### Next children in the angr-4j5u sequence

- 4j5u.2 — Extract ConstraintTracker + ConstraintSolver fields
- 4j5u.3 — Extract MemoryConfiguration struct
- 4j5u.4 — Extract ExecutionEnvironment struct
- 4j5u.5 — Consolidate orchestrator (final pass)

Each is sized to one session per the round-3 audit notes on the parent.
