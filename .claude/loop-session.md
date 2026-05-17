## Session log: 2026-05-17 — angr-kcf.2 closed (xfail with memory entry)

### Status: CLOSED

### Task

**angr-kcf.2 (P2, task)** — Fix-or-doc for codegate_2017-angrybird.
Acceptance: test passes OR is xfailed with documented reason.

### Resolution: (b) keep xfail with memory entry.

Commit `24af5350f`: added `codegate_2017-angrybird` to
`KNOWN_XFAIL` in `tests/engines/test_rust_integration.py` with an
inline comment summarizing the triage.

### What I confirmed in this session

1. Initial hypothesis was wrong. I suspected
   `native/angr/src/syscalls/read.rs::NativeReadSyscall::call()` was
   missing `state.record_stdin_symbol()` (compared to
   `procedures/read.rs::NativeRead` which does record). But objdump
   shows the binary uses `fgets@plt`, not raw `read` syscall, so
   syscall read is not relevant to codegate. (The missing
   record_stdin_symbol on the syscall path is real but doesn't fire
   for this binary.)

2. NativeFgets DOES record stdin symbols. `procedures/fgets.rs:86-88`
   calls `state.record_stdin_symbol(name.clone(), 8)` for each
   created byte. Verified by running with `ANGR_RUST_LOG=debug`: the
   Rust Z3 solver's SMT-LIB output references
   `stdin_fgets_0_0, _0_1, _0_4, _0_7, _0_18` etc. in constraints.

3. The Python-side `_inject_rust_stdin`
   (`angr/exploration/rust_callback_dispatch.py:327`) correctly
   retrieves the recorded symbols, evaluates each via Rust solver
   (`_eval_stdin_symbol` at `native/angr/src/exploration/state_api.rs:757`),
   and appends concrete bytes to `posix.stdin.content`.

4. Symptom: `posix.dumps(0)[:20]` returns 20 wrong bytes
   (`b'*%\xac\x0c\`\xfe\xff\x80\x06 \xc0\xff\xff\xca4\x07\xffpK\x05'`),
   not b''. So injection IS happening — the model assignment from
   the Rust solver just doesn't match Python's. Parent angr-kcf
   description ("dumps(0) returns b''") is stale.

5. Likely root cause is constraint/path divergence, not stdin
   tracking. The path Rust takes to find_addr 0x404fab differs from
   Python's, producing a constraint set whose satisfying assignment
   to stdin is wrong. Likely involves symbolic-memory branch
   divergence at the 0x1000-0x1018 anti-fingerprint loads the
   solve.py sets up (see memory `codegate-0x1000-loads`).

### What landed

- Commit `24af5350f`: codegate added to KNOWN_XFAIL with inline doc.
- Memory `invariant-codegate-xfail`: full keep-xfail rationale.
- Memory `avoid-stdin-tracking-fix-for-codegate`: warn future agents
  off the stale "stdin tracking" framing of parent angr-kcf.
- bd issue `angr-dmqr` (P3, bug): test_rust_integration.py's patched
  simulation_manager unconditionally returns RustExplorationManager,
  which breaks strcpy_find when CFG's jumptable resolver calls
  `factory.simulation_manager(state, resilience=True)` with a state
  that has DO_RET_EMULATION. Separate test-infrastructure issue
  found while running the integration suite. Not blocking.
- `angr-kcf.2` closed with reason.

### Test state

`tests/engines/test_rust_exploration.py`: 444/444 pass (unchanged).
`tests/engines/test_rust_integration.py`: 20 pass / 2 xfail (codegate
correctness + perf) / 2 fail (strcpy_find — pre-existing,
documented in `angr-dmqr`).

### Followup work for next session

`bd ready` after this close:
- angr-prem (P2) / angr-prem.1 (P3) — MemoryLayer trait. Deferred
  multiple times; parent notes say premise is wrong.
- angr-fk0m (P2) — refactor sync/cache/export mixins. "Purely
  cosmetic"; deferred multiple times.
- angr-34w.12 (P2) — grub OOM. Big. See memory
  `grub-investigation-depth`.
- angr-dmqr (P3, NEW) — fix the strcpy_find integration test by
  having the patched factory fall back to the original
  `simulation_manager` when the state's SimOptions would be rejected
  by RustExplorationManager (cheap, well-scoped).
- Real fix for codegate path divergence — non-trivial, would need
  a concrete plan to align Rust+Python branch decisions on
  symbolic-memory loads.
