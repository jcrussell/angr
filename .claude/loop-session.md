## Session log: 2026-05-11 — angr-sgbn (fast-path _concretize_stack_registers)

### Task
Profile `_concretize_stack_registers` (claimed 12ms on cold blank_state path
per add-state-init-breakdown-2026-05-11 memory) and decide between
optimization or rationale.

### Profile (wall-clock, fauxware, 20 iterations)

Via mgr._concretize_stack_registers (real call):
  Cold (blank_state):  median 10.04 ms (rbp branch = 9.79 ms)
  Warm (entry_state):  median 0.08 ms

Root cause: blank_state sets `regs.sp = stack_end` (concrete) but rbp is
filled with a fresh unconstrained BVS by default_filler_mixin. solver.eval
on the unconstrained BVS pays full Z3 ctx init + check + model (~10ms)
just to return 0.

Z3's model for an unconstrained rbp is arbitrary — observed `0x0` across
all iterations.

### Fix (commit 1a233418b)

Added `_eval_or_default(state, reg_val, default)` helper:
- Scan `state.solver.constraints` for variable overlap with `reg_val.variables`
- If no constraint references any var, return `default` directly (skip Z3)
- Otherwise fall back to `solver.eval`

For SP, default is `arch.initial_sp` (or 0). For BP, default is 0
(matching Z3's empirical result).

Also dropped a dead `sp_val = state.solver.eval(reg_val)` in the SP
non-symbolic else branch — value was never used.

### Results

  Cold (blank_state):  10.04 ms → 0.29 ms  (34x faster, saves ~9.7 ms per call)
  Warm (entry_state):   0.08 ms → 0.08 ms  (no regression)

Tests: 382/382 RustExploration tests still pass. 3 pre-existing failures
unchanged (verified on master: test_dcas_cmpxchg16b_no_match_keeps_memory,
test_pipe_native_dispatch_creates_two_fds, test_dup2_native_dispatch_redirects_stdin).

### Memory saved
`sgbn-unconstrained-sp-fastpath` — root cause + fix + before/after numbers.

### Status
COMPLETE — bead angr-sgbn closed.
