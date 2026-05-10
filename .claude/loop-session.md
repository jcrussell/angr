## Session log: 2026-05-10 — angr-3uye.1 (211th loop session, COMPLETE)

### Task
Trace-only bead: localize divergence point where Rust's symex of a single
`ret` instruction concretizes the popped PC to 0 instead of routing the
state to the `unconstrained` stash (Python's behaviour). Acceptance:
divergence named down to file:line so angr-3uye.2 can take a one-shot fix.

### Repro confirmed
- `load_shellcode(b"\xc3", AMD64, 0x1000)`; `blank_state(addr=0x1000)`;
  `rsp = 0x7fff_0000`; step 2.
- Python: `{'unconstrained': 1}`, rip = `<BV64 mem_7fff0000_0_64>`.
- Rust:   `{'deadended': 1}`, addr = 0x0 (P21 generic skip log line).

### Method
Added 3 temporary `log::debug!` statements (exits.rs, expressions.rs) with
`ANGR_RUST_LOG=debug`, ran the repro in a subprocess with RLIMIT_AS=4GB,
captured the divergence chain end-to-end, then **reverted** all
instrumentation. `cargo check --release` clean, all 372 tests pass on the
clean tree.

### Root divergence (named down to file:line)
`angr/exploration/rust_state_sync.py:486-503` — `_sync_stack_page` slow
path solver.eval()s the symbolic stack page to concrete bytes (default
model = zeros) and maps them into Rust's page cache. The
`_has_user_symbolic_var` filter at lines 540-552 explicitly rejects
variable names starting with `mem_`/`reg_`/`unconstrained`, so the
auto-generated stack placeholders (`mem_7fff0000_*`) are never re-imported
as symbolic regions. Net: Rust's stack is all-zero where Python's is
symbolic.

### Downstream chain (consequence)
1. `interpreter_cb/expressions.rs:60` — `IRExpr::Load` concrete-addr fast
   path returns `Concrete(0, 64)`.
2. `interpreter_cb/exits.rs:47-52` — `eval_next_addr_concretized`
   concrete fast path returns `Single(0)` without consulting solver.
3. `interpreter_cb/exits.rs:160,189-196` — `handle_exit(0, Ijk_Ret)`:
   `is_in_binary(0)` false => `BlockResult::UnmodeledCall { addr: 0,
   return_addr: 0, … }`.
4. `exploration/stepping.rs:710,720-742` — no `resolve_function` =>
   `unmodeled_call_generic_skip` sets rax=0, pc=0, continues. State ends
   in `deadended` at 0x0.

### Python counterpart (for reference)
- `angr/engines/successors.py:308-323` — `_eval_target_brutal` enumerates
  the symbolic ip up to `max_targets+1` (256, from
  `address_concretization_mixin.py:31`); on overflow appends to
  `unconstrained_successors` (successors.py:323).
- `angr/sim_manager.py:519` maps that list to the `'unconstrained'` stash.

### Two fix axes for angr-3uye.2 (documented in bead notes)
- **A. Preserve symbolic stack bytes during sync** (root fix): relax the
  `_has_user_symbolic_var` filter for stack pages whose load is wholly
  symbolic, or sync the page itself as one BVS.
- **B. Defensive route in handle_exit**: before the `!is_in_binary`
  branch at exits.rs:189, detect Ijk_Ret to 0 with empty/mismatched call
  stack and emit `BlockResult::UnconstrainedJump`.

### Files touched
None (all instrumentation reverted). `git diff` is empty.

### Memories saved
- `3uye-stack-page-eager-materialization` — root cause + file:line refs.
- `invariant-has-user-symbolic-var-divergence-knob` — the explicit
  filter at rust_state_sync.py:540-552 that gates Python->Rust symbolic
  identity preservation.
- `avoid-concretize-fast-path-skips-solver` — concretize.rs:344-347
  unguarded fast path that would silently concretize a `Constrained` BV.

### Closed beads
- `angr-3uye.1` (this trace bead).

### Status
COMPLETE. Unblocks `angr-3uye.2` (the one-shot fix bead).
