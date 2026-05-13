## Session log: 2026-05-13 — angr-ipd0 (Fix u32 underflow in LoadG truncation) — CLOSED

### Task

**angr-ipd0** (P3, CLOSED) — Swap argument order in
`apply_loadg_conversion`'s truncation branch:
`extract(0, target_bits)` → `extract(target_bits - 1, 0)`.

The branch is currently latent (LoadG always widens) but was a
correctness trap waiting for a future refactor. With the wrong
arg order `extract` would have asserted in debug or underflowed
`result_width = 0 - target_bits + 1` to ~4B in release.

### Root cause

`RustBV::extract(high, low, ctx)` takes INCLUSIVE bit indices,
asserts `high >= low`, and computes `result_width = high - low + 1`.
To keep the low N bits, the correct call is `extract(N - 1, 0)`.
Old call had high=0, low=target_bits — backwards.

### Files touched

- `native/angr/src/interpreter_cb/expressions.rs:711` — fixed
  arg order with explanatory comment.
- `native/angr/src/interpreter_cb/expressions.rs` (test module) —
  added `apply_loadg_conversion_truncates_when_src_wider`
  regression test (0xdead_beef @ 32 bits truncates to 0xbeef @ 16).

### Verification

- `cargo check --release` ✓
- `cargo test --release --lib apply_loadg_conversion` — 5/5 pass ✓
- `tools/rebuild-rust.sh --cargo-only` rebuild ✓ (venv pip is
  still broken — `avoid-broken-venv-pip-rebuild`).
- `pytest tests/engines/test_rust_exploration.py` — 389 pass,
  same 3 pre-existing failures as last session
  (`dcas_cmpxchg16b_no_match_keeps_memory`,
  `pipe_native_dispatch_creates_two_fds`,
  `dup2_native_dispatch_redirects_stdin`). Unrelated to this fix.

### Commit

`566be309f` fix(rust-symex): correct extract arg order in LoadG truncation branch (angr-ipd0)

### Memories saved / updated

- NEW `invariant-rustbv-extract-args` — `RustBV::extract` API
  semantics + how to construct low/high N-bit extracts correctly.
- UPDATED `apply-loadg-truncation-bug` — marked FIXED with commit
  hash and regression-test name.

### Still-open followup from prior session (unchanged)

Three pre-existing test failures (`dcas`, `pipe native`, `dup2
native`) remain. They reproduce after a `tools/rebuild-rust.sh
--cargo-only` rebuild. No existing bd issue covers them. The prior
session deferred filing because the cargo-only rebuild path can
differ subtly from `pip install -e .` and the venv's pip is still
broken so we can't re-check the conventional way. Worth filing as
a P2 bug after `avoid-broken-venv-pip-rebuild` is resolved.
