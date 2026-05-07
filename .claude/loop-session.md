# Loop session notes (2026-05-07, 118th loop session)

## Task: angr-4pkm — csgames2018 'list index out of range' regression  ✓ CLOSED

### Root cause

`Cdecl::return_register()` in `native/angr/src/arch/calling_conventions.rs:232`
returned offset **16**. Comment claimed "EAX (same offset as RAX in VEX)" but
EAX in the x86 VEX guest state is at **offset 8**; offset 16 is EDX. Native
SimProcedures returning natively for 32-bit x86 binaries thus wrote their
result to EDX while EAX kept the stale prior value (typically the buffer
address pointer that was the strlen argument).

### Why csgames2018 surfaced it

The bug was latent until commit f378ad9a0 (angr-qlh7) made strlen handle
symbolic bytes natively (instead of falling back to Python). csgames2018
calls `strlen(argv[1])` where argv[1] is a 16-byte symbolic input. Native
strlen now returns a symbolic ITE chain ranging 0..=16, but it landed in
EDX. `cmp $0x10, %eax` then compared EAX (a concrete pointer ≠ 16) against
16, jne taken concretely, state went straight down the "incorrect" branch
with no fork — eventually deadending at `exit()` with no stdout. The
callable predicate `correct(state)` checks `b'correct!' in state.posix.dumps(1)`
which never matched, so `simulation_manager.found` was empty and
`simulation_manager.found[-1]` raised IndexError.

### Bisect log (89132680b good ↔ 76aef1ca2 bad → first bad: f378ad9a0)

```
ca6557cbc good
348336891 bad
5148bfd3e good
5929d6fb9 bad
2d4e36624 good
7e1bd9645 good
f378ad9a0 bad   ← first bad commit (strlen symbolic-arg support)
```

f378ad9a0 itself is correct; it just exposed the latent Cdecl bug.

### Fix

`native/angr/src/arch/calling_conventions.rs:232` — return `8` (EAX), not `16`
(EDX/RAX). Added unit test `test_return_register_offsets_per_arch` asserting
Cdecl=8, SystemVAMD64=16, MicrosoftX64=16 to lock the offsets per arch.

### Validation

- `cargo test --release --lib calling_conventions`: 4/4 pass.
- `pytest tests/engines/test_rust_exploration.py`: 261/261 pass.
- `tests/benchmarks/run_regression.py`: 12/12 pass (csgames2018 0.97s/294MB,
  back to baseline performance).
- `tests/benchmarks/run_single.py csgames2018 --engine rust`: OK 0.98s,
  100+ keys reported.

### Why other 32-bit benchmarks were unaffected

flareon2015_2 is the only other 32-bit x86 benchmark. It doesn't exercise a
path where a native SimProcedure's return value is consumed in a way that
needs to fork — most 32-bit benches either rely on procedures that fall back
to Python (symbolic args) or compare against a value the binary doesn't
inspect. fauxware/ais3/defcamp are amd64 (correct return offset).

## Status: complete
