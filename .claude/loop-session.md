## Session log: 2026-05-17 — angr-apre closed (SYMBOL_FILL_UNCONSTRAINED_REGISTERS raise)

### Status: CLOSED

### Task

**angr-apre (P3, bug)** — Promote SYMBOL_FILL_UNCONSTRAINED_REGISTERS to
raise NotImplementedError under Rust. Pattern matches angr-n129 / cf9h /
gmrc / csmm.

### Root cause

Rust's `RegisterFile` in `native/angr/src/arch/mod.rs:178-185` initializes
storage with `vec![0; size]` and has no "uninitialized" marker. Reads of
never-written registers always return concrete zero from `data[]`.

Python's `_fill` in `light_registers.py:140-164` and `_default_value` in
`default_filler_mixin.py:45-49` create a fresh symbolic BVS when
`ZERO_FILL_UNCONSTRAINED_REGISTERS` is absent. `SYMBOL_FILL_UNCONSTRAINED_REGISTERS`
explicitly opts into symbolic-fill (and suppresses the warning).

The Rust engine silently ignored the option, returning concrete-zero
registers. A user who explicitly opted into symbolic-fill would never
see the divergence — paths driven by unconstrained initial register
values would simply not be explored.

### Resolution

- Added `SYMBOL_FILL_UNCONSTRAINED_REGISTERS` to `_RAISE_OPTION_NAMES`
  in `angr/exploration/rust_manager.py` with rationale block.
- The MEMORY variant `SYMBOL_FILL_UNCONSTRAINED_MEMORY` is NOT promoted
  — Rust's `load_concrete_lazy` (`native/angr/src/memory/load.rs:333-339`)
  already defaults to symbolic-fill (`unc_mem_*` BVS) when
  `zero_fill_unconstrained` is unset.
- Also generalized the `_check_raise_options` error message: the old
  wording claimed offending options "require SimAction/SimEvent records",
  which has not been accurate since CONCRETIZE / CONSERVATIVE_WRITE_STRATEGY
  / DO_RET_EMULATION / CALLLESS / EFFICIENT_STATE_MERGING / now SYMBOL_FILL
  joined the set for non-action reasons.

### Files modified

- `angr/exploration/rust_manager.py` — _RAISE_OPTION_NAMES + comment +
  generalized error message.
- `tests/engines/test_rust_exploration.py` — 2 new tests
  (test_symbol_fill_unconstrained_registers_option_raises_at_construction,
  test_symbol_fill_unconstrained_memory_option_does_not_raise). Updated 2
  existing tests (test_filler_materialised_multibyte_symbolic_preserved,
  test_uninitialized_annotation_on_bvs_in_memory) that opted into the
  now-raising option — they exercise memory symbolicity, only the MEMORY
  variant was needed.
- `docs/advanced-topics/rust_engine.rst` — split joint
  SYMBOL_FILL_UNCONSTRAINED_{MEMORY,REGISTERS} row, added
  "Followup (angr-apre, 2026-05-17)" callout.

### Verification

- `cargo check --release`: clean (no Rust changes).
- `tests/engines/test_rust_exploration.py`: 450/450 pass (was 448, +2 new).

### Memories saved

- `apre-root-cause`: RegisterFile vec![0; size] no uninitialized marker.
- `invariant-symbol-fill-memory-matches`: SYMBOL_FILL_UNCONSTRAINED_MEMORY
  is silently accepted but NOT a divergence — Rust defaults match.
- `invariant-rust-honored-simoptions`: updated to include SYMBOL_FILL.

### Followup ideas

The previous-session followup remains valid: candidates for
SimOption coverage are documented in `docs/advanced-topics/rust_engine.rst`.
The remaining (a)-implement and (b)-reject entries in the matrix are:
- `AVOID_MULTIVALUED_READS` / `AVOID_MULTIVALUED_WRITES` — Rust always
  enumerates within strategy limits.
- `CONCRETIZE_SYMBOLIC_WRITE_SIZES` — Rust concretizes addresses but
  not sizes the same way.
- `PRODUCE_ZERODIV_SUCCESSORS` — Rust treats div-by-zero as a single
  state (no second successor with `divisor == 0`).
- `CONSERVATIVE_READ_STRATEGY` — currently in `_REJECTED_OPTION_NAMES`
  (warn-only); the WRITE variant raises. Asymmetry tracks ticket scope
  per the doc; promoting to raise is a parallel small change if desired.
- `ZERO_FILL_UNCONSTRAINED_REGISTERS` — actually a "matches by default"
  case under Rust (always zero-fills regardless), so no action needed.

For PRODUCE_ZERODIV_SUCCESSORS especially: it ships in the `tracing`
mode bundle (sim_options.py:430) but tracing already pulls UNICORN
which Rust rejects, so users hitting this are already directed away
from Rust. Probably safe to leave silent.
