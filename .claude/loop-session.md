## Session log: 2026-05-14 — Two phases closed (angr-n082, angr-aija)

### Closed tasks

**1. angr-n082** — Phase 1.2: Multi-cell collapse in load_concrete_lazy_inner.
Commit `5030e0989`.

**2. angr-aija** — Phase 1.3: store_concrete_multi /
store_symbolic_unified_multi helpers. Commit `405f4c3db`.

### What landed

#### Phase 1.2 (load side)
- `native/angr/src/memory/load.rs`:
  * `load_concrete_lazy_inner` now detects any Multi byte in the touched
    range (cheap `multi_objects.is_empty()` short-circuit). On hit:
    perm-check, then dispatch to `assemble_load_with_multi`.
  * `assemble_load_with_multi`: per-byte walk; Multi byte right-folds
    `payload.alternatives()` into an ITE chain with the page's concrete
    byte as the default else. Plain-Symbolic and Concrete bytes handled
    too. `record_mem_ite_depth(payload.len())` per Multi byte.
  * `extract_byte_lane` promoted to `pub(super)` for store-side reuse.
- 4 new tests (single-byte LE / mixed LE / mixed BE / two-multi-byte LE).

#### Phase 1.3 (store side)
- `native/angr/src/memory/store.rs`:
  * `install_multi_for_candidates` (private core): per candidate per
    byte, builds `(addr==cand, byte_b)` alts and merges with any
    existing payload via `set_multi_alternatives`.
  * `store_concrete_multi` (test-rig wrapper, no concretizer).
  * `store_symbolic_unified_multi` (production API): mirrors
    `store_symbolic_unified` but routes Multiple/Strided to the
    Multi-cell path. Single short-circuits to eager. TooLarge/Failed
    surface the same errors.
- 5 new tests (LE round-trip / BE round-trip / fork independence /
  unified-multi-multiple end-to-end / concrete-address fast path).

### Validation

- `cargo test --release --lib`: 756 / 756 (up from 747 baseline).
- `pytest tests/engines/test_rust_exploration.py`: 396 / 396.
- Production paths unchanged — Phase 1.3 entry points are not yet
  wired into any default path.

### Memories saved this session

- `invariant-multi-load-collapse` — right-fold construction details,
  page-byte default else, endianness concat pattern.
- `invariant-multi-load-not-in-load_concrete` — Phase 1.2 did NOT patch
  `load_concrete`; future production wiring (Phase 1.4 / Phase 2)
  needs a parallel detection block there once stores emit Multi.
- `invariant-multi-store-merge` — install_multi_for_candidates MERGES
  rather than replaces; safe due to disjoint conds.
- `gotcha-stride-detection-two-addrs` — concretizer with default
  settings classifies two regularly-spaced solutions as Strided, not
  Multiple. Tests need `matches!(.., Multiple(_) | Strided{..})`.

### Next ready: angr-5zw8 — Phase 1.4 strchr SimProcedure wiring

NOT picked up this session. It touches the Python side (SimProcedure
dispatch + MultiwriteAnnotation preservation through the claripy
bridge) and requires profiling work — a significantly larger surface
than the two Rust-only sub-steps. Recommend starting fresh.

Open questions to answer before starting Phase 1.4:
- Does `memory_store_symbolic_full` callback path bypass the Rust
  store entirely, or does it call back into Rust after Python work?
  Check `angr/exploration/rust_state_sync.py` and the bridge.
- Is `MultiwriteAnnotation` already round-tripped or stripped at the
  Rust boundary? Grep `MultiwriteAnnotation` in the Python tree first.
- Need a baseline of `mem_ite_depth_max` on a sym-write or strchr-heavy
  benchmark before wiring, so we can measure the win.
