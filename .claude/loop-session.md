## Session log: 2026-05-14 — angr-q7ij (fauxware Rust empty output)

### Status: complete

### Outcome
Fixed: NativeRead now records the symbolic stdin bytes it creates, so
state export's `_inject_rust_stdin` can evaluate them and inject the
concrete bytes back into `posix.stdin.content` — fauxware's
`posix.dumps(0)` now contains the satisfying `SOSNEAKY` bytes.

### Root cause
`native/angr/src/procedures/read.rs::NativeRead::call` created
`stdin_{read_id}_{i}` symbolic bytes and stored them to the buffer,
but never called `state.record_stdin_symbol(name, 8)`. The state
export path in `_inject_rust_stdin` short-circuits on
`!self._rust_mgr.has_state_stdin_symbols(state_id)`, so Python's
`posix.stdin.content` stayed empty and `posix.dumps(0)` returned `b''`.

Other native stdin sources (fgets/fgetc/getchar/scanf) already record
symbols; only read.rs was missing this line. Probably an oversight from
when NativeRead was first added (angr-3tek.2).

### Fix
`native/angr/src/procedures/read.rs:67-76` — collect names up front,
then `state.record_stdin_symbol(name.clone(), 8)` for each. Added a
unit test `test_read_records_stdin_symbols` covering this.

### Verified
- All 395 Rust-exploration unit tests pass.
- `python tests/benchmarks/run_single.py fauxware --both`:
  python output `         SOSNEAKY `, rust output now contains
  `SOSNEAKY` (was empty before).
- ais3_crackme smoke test: still finds `b'ais3{I_tak3_g00d_n0t3s}'`.
- Property fuzzer still reports `exact-mismatch` (different padding
  between python/rust stdin layouts), but the SOSNEAKY content is
  present — the "empty Rust output" bug from angr-q7ij is resolved.
  Promoting fauxware to rust_only=False is a separate decision.

### Files changed
- `native/angr/src/procedures/read.rs` (record_stdin_symbol + test)
