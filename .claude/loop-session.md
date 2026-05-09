## Session log: 2026-05-09 — angr-pe5t CLOSED (185th loop session)

### Task: Document SimOption coverage matrix

Doc-only audit of which `sim_options` flags the Rust engine honors,
which inherit through Python code paths, and which are silently
ignored. Source-of-truth read from `angr/sim_options.py` and the four
read sites in `angr/exploration/rust_manager.py` (lines 697–706,
1751–1756, 1962–1966, 2320–2325).

### Files modified

- docs/RUST_SIMOPTION_COVERAGE.md (new, ~150 lines) — full matrix
  with four sections: Honored, Inherited, Ignored — divergence-risk,
  Ignored — no-op. Each silent-ignore row tagged with future fix
  (a) implement, (b) explicitly reject, (c) accept-but-document.
- CLAUDE.md (+9) — added SimOption Coverage stub between Architecture
  Support Matrix and Rust Symbolic Execution sections, linking to the
  new doc.

### Honored set (5)

LAZY_SOLVES, ZERO_FILL_UNCONSTRAINED_MEMORY, APPROXIMATE_MEMORY_INDICES,
SYMBOLIC_WRITE_ADDRESSES, STRICT_PAGE_ACCESS.

### Notable divergence-risk findings

The matrix flags KEEP_IP_SYMBOLIC, NO_IP_CONCRETIZATION, ENABLE_NX,
NO_SYMBOLIC_JUMP_RESOLUTION, NO_SYMBOLIC_SYSCALL_RESOLUTION,
AVOID_MULTIVALUED_*, CONCRETIZE_SYMBOLIC_WRITE_SIZES, the BYPASS_*
resilience set, the TRACK_*_ACTIONS set, DO_RET_EMULATION,
PRODUCE_ZERODIV_SUCCESSORS as silent-divergence options. None are
fixed in this commit; the doc tags each with a recommended remediation
(implement vs explicit-reject).

### Smoke test

7 option-related tests pass (test_zero_fill / test_strict_page /
test_lazy / test_options).
