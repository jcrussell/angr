# Loop session notes (2026-05-02, sixteenth session)

Closed three tasks. All landed; tests 208/208 throughout.

## Closed: angr-7qll (commit 7fec7db2d)

[perf] SegmentListIter snapshots into a Vec at construction. Was
`.iter().nth(idx)` on a BTreeMap-backed RangeMap → O(n²) total
iteration. Now O(n).

Memory saved: `segmentlist-iter-snapshot`.

## Closed: angr-4knw (commit fb52689e3)

[fix] AST_CACHE in claripy_bridge.rs validates width on hit. claripy
hashes are content-addressed but a hash collision between two ASTs of
different widths could silently corrupt downstream VEX ops. On hit,
fetch ast.length (None for Bool ASTs → width 1) and compare against
the cached BV's width. Mismatch evicts and falls through.

Memory saved: `invariant-ast-cache-width-check`.

## Closed: angr-t77r (commit bf2111b74)

[feat] Symbolic DivMod in vex/ops.rs. The error path was actually
unsound: `OpError::UnsupportedVectorOp` was caught by the binop
fallback in `interpreter_cb/expressions.rs:329`, which mints a fresh
symbolic `unsup_binop_<pc>` unrelated to the dividend or divisor.

Recipe: zero-extend divisor for unsigned (sign-extend for signed),
do full-width udiv/sdiv + urem/srem, extract low N/2 of each, and
`concat(remainder, quotient)` — concat puts self in high, other in
low, matching the `(remainder<<N) | quotient` packing of the concrete
case. Z3's div/mod by zero is total and matches claripy, so no
explicit zero guard.

5 new unit tests covering 64→32 and 128→64 in both concrete and
symbolic modes.

Memories saved: `invariant-divmod-symbolic`,
`avoid-fresh-symbolic-on-unsupported-binop`.

## Carryover

- Pre-existing baseline timing variance in `run_regression.py`
  (ais3 +75%, re400 +62%). Re-record or widen tolerances.
- Other ready P2 bugs: angr-3tek (read/write SimProcedures stale
  cache — needs Rust→Python cached-state sync per memory
  `avoid-enabling-native-read`), angr-mboi (mma_howtouse 0.7x perf),
  angr-xidi (google2016_unbreakable_1 -93%).
- P3 refactors: angr-w4os, angr-2fs0, angr-1f8s, angr-cbko, angr-3ijo
  (bincode spike — needs profiling first), angr-8em4 (panic audit,
  too large for one session per its own description).
