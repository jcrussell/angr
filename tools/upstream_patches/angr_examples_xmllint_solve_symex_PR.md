# [angr-examples] xmllint: add solve_symex.py (symbolic-execution harness)

## Summary

Adds `examples/xmllint/solve_symex.py`, a symbolic-execution sibling of the
existing fuzzer-based `examples/xmllint/solve.py`. Both drive the same vendored
`xmllint_bin` (an x86-64 PIE built from libxml2): `solve.py` uses the angr
fuzzer (concrete execution); the new `solve_symex.py` uses symbolic execution
with `use_sim_procedures=True` and a bounded `explore(find=getenv)`.

No existing file changes — this is a single new, self-contained script.

## Motivation

Downstream (the angr Rust-symex engine) wants a real-world, libc-heavy command
line utility in its benchmark corpus rather than yet another CTF crackme.
`xmllint` is the only such binary in this repo. The fuzzer harness exercises the
deep parser but is concrete-only; the symbolic harness gives a *symbolic*
workload over real libc startup, which is what the symex benchmark gate measures.

## What it does

- `use_sim_procedures=True` so angr's libc SimProcedure stubs replace real glibc
  init (cheap startup).
- 16 symbolic stdin bytes, standard CLI options (`--noout --nonet --recover
  --noent -`), `ZERO_FILL_UNCONSTRAINED_{MEMORY,REGISTERS}`.
- `explore(find=getenv, num_find=1, n=400)` — bounded, stops at the first state
  to reach `getenv` (resolved in the loaded libc, not the main-object PLT stub).
- Stays **single-state** and reaches `getenv` in ~185 steps — no state
  explosion, well inside a few hundred MB.

## Why `getenv` and not a parser callsite

`getenv` sits on the deterministic startup path *before* the symbolic stdin is
consumed, so it's a clean mechanics smoke. Targeting a parser callsite
(`fread` / `xmlReadFd` / `xmlParseDocument`) is **not viable as plain symex on
either engine**: with `use_sim_procedures=True` the stubbed libc returns
unconstrained symbolic values that propagate into pointers and branch
conditions, producing a constraint/state explosion (Z3 OOM under the Python
engine; symbolic-pointer concretization fallout under the Rust engine). This was
verified empirically — concrete XML on stdin explodes identically, so it is not
a symbolic-input artifact. The fuzzer (`solve.py`) remains the only driver that
reaches xmllint's parser. `getenv` is therefore the deliberate, de-risked find
target for a symbolic xmllint workload.

## Test

`solve_symex.py` ships a `unittest.TestCase` (`TestXmllintSymex.test`) asserting
`getenv` is reached within the step budget. It is **not** `@unittest.skip`-ped
(unlike the fuzzer `solve.py::test`, which is disabled), so it runs in CI:

```bash
cd examples/xmllint
python -m unittest solve_symex      # single-state, ~185 steps, a few seconds
python solve_symex.py               # prints "reached getenv -> 1 found state(s)"
```

## Notes for the reviewer

- Self-contained: resolves the binary relative to `__file__` (same pattern as
  the fuzzer `solve.py`), so no env var is required.
- The downstream benchmark already vendors an in-repo equivalent
  (`tests/benchmarks/synthetic_examples/xmllint_getenv/solve.py` in the angr
  Rust-symex branch) that resolves `xmllint_bin` via `ANGR_EXAMPLES_DIR`; this
  PR upstreams the canonical copy next to the binary so the example is
  self-documenting.
