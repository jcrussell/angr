# Loop session notes (2026-05-06, eighty-seventh loop session — DONE)

## Status: COMPLETE — angr-88mp closed

Made symbolic-store fallback policy consistent across the three sites
flagged by the audit in interpreter_cb/statements.rs.

## Changes

`native/angr/src/interpreter_cb/statements.rs`:

1. **StoreG sym-guard, non-Single addr (lines ~360-380)** —
   replaced silent `log::warn!("Store may be lost, but continuing
   execution.")` with `Err(CbExecutionError::Unsupported(...))`.
   Still delegates via `memory_store_symbolic_full` when registered.

2. **StoreG concrete-guard sym-data, Multiple addrs (lines ~422-440)** —
   replaced silent `call_memory_store_symbolic_value(addrs[0], ...)` (which
   dropped the other candidates) with the canonical pattern: prefer
   full callback; else `build_ite_store_from_callbacks` for ≤16 addrs;
   else return Unsupported.

3. **CAS sym-addr sym-data (lines ~702-715)** — was hard-erroring
   without trying any delegation; now tries
   `memory_store_symbolic_full` first.

## Verification

- `cargo check --release`: clean.
- `pytest tests/engines/test_rust_exploration.py`: 243/243 pass.
- `tests/benchmarks/run_regression.py`: 11/12 pass (csgames2018 is
  pre-existing per loop-session 2026-05-06).
- Smoke: `run_single.py {fauxware, csaw_wyvern, codegate_2017-angrybird}`
  all pass with rust engine.

## Memories saved (bd remember)

- `invariant-symbolic-store-fallback-policy` — the canonical policy and
  where to apply it in statements.rs.
- `memory-store-symbolic-full-not-registered` — `memory_store_symbolic_full`
  callback is plumbed in Rust but no Python code wires it up; `has_*()`
  is always false.
- `build-z3-headers-not-in-venv` — venv pip is broken; manual build via
  `cargo build --release` + `cp target/release/librustylib.so
  angr/rustylib.cpython-312-x86_64-linux-gnu.so`. Use
  `Z3_SYS_Z3_HEADER=/usr/include/z3.h`.

## Files modified

- `native/angr/src/interpreter_cb/statements.rs` (+36, -19)
