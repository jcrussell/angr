# Loop session notes (2026-05-08, 144th loop session)

## Task: angr-qlcr — declare_proc macro: registration + arg extraction + num_args in lockstep (closed)

### Status: complete; closed

### Summary
Added `declare_proc!` in `native/angr/src/procedures/macros.rs` that
emits a unit struct + `NativeSimProcedure` impl from one declarative
form. The procedure's `name()`, `num_args()`, and the per-argument
extraction prelude are all derived from the same `args = [...]` list,
so it is no longer possible to register a procedure whose declared
arity disagrees with the actual extraction count.

Migrated 16 procedures (target was ≥5):
  - strlen.rs: strlen, strnlen
  - strcmp.rs: strcmp, strncmp, strcasecmp
  - memcmp.rs: memcmp
  - strtol.rs: strtol, strtoul, atoi, atol  (also refactored
    `run_strtol` to take direct nptr/endptr/base instead of a slice)
  - ctype.rs:  isdigit, isalpha, isspace, isalnum, isupper, islower,
               isxdigit, isprint, tolower, toupper  (helpers
               refactored to take a single `&RustBV`)

Net diff: -114 lines.

### Verification
- cargo test procedures::: 185/187 passing. The 2 failures
  (`test_heap_metadata_cloned_on_fork`, `test_getenv_env_preserved_on_fork`)
  fail identically on pristine HEAD (PyO3 not auto-initialized in
  cargo test). Saved as memory `avoid-pyo3-init-test-failures`.
- Python suite: 300/300 passing.
- fauxware benchmark unchanged (~0.3s, found SOSNEAKY).

### Memories saved
- `declare-proc-macro`: macro location, two arg kinds (concrete/bv),
  no-return flag, manual registration still required, tests must
  re-import the trait inside cfg(test) blocks.
- `avoid-pyo3-init-test-failures`: pre-existing failures unrelated to
  this work.

### Files changed
- native/angr/src/procedures/macros.rs (new, +95 lines)
- native/angr/src/procedures/mod.rs (mod macros; #[macro_use])
- native/angr/src/procedures/{strlen,strcmp,memcmp,strtol,ctype}.rs
  (migrated to declare_proc!)

### Original Plan
1. Define `declare_proc!` in `procedures/mod.rs` (or new `macros.rs`).
2. Macro form:
   - `name = "..."` (procedure name string)
   - `struct = NativeFoo` (unit struct identifier)
   - `args = [name1: kind, name2: kind, ...]`
     - `kind` ∈ `concrete` (extracts u64 via `extract_concrete_arg`) or
       `bv` (clones the RustBV).
   - optional `no_return` flag
   - `call |state| { body }` — body uses bound names + `state` as
     `&mut RustSimState`.
3. Macro emits unit struct + impl NativeSimProcedure where
   `name()`, `num_args()`, and the extraction prelude are all
   derived from the same args list. Registration stays manual in
   `mod.rs` (no inventory dep available).

### Migration targets (≥5 to satisfy acceptance criteria)
- strlen — args=[addr: concrete]
- strncmp — args=[s1, s2, n: concrete]
- strcmp — args=[s1, s2: concrete]
- memcmp — args=[s1, s2, n: concrete]
- atoi — args=[nptr: concrete]
- isdigit (raw form via `bv`) — args=[c: bv] — needs ranges_predicate
  refactored to take `arg: &RustBV` instead of `args: &[RustBV]`.

### Files to touch
- native/angr/src/procedures/mod.rs (define macro)
- native/angr/src/procedures/{strlen,strcmp,memcmp,strtol,ctype}.rs (migrate)

### Verification
- cargo check --release
- cargo test (procedures::*)
- python -m pytest tests/engines/test_rust_exploration.py
