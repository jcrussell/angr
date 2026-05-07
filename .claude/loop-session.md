# Loop session notes (2026-05-07, 109th loop session)

## Task: angr-faws — Decide on opt-out flag for Z3 FP theory if intrinsic overhead is irreducible

This bead asks for a *decision* between Option A (accept the regression)
and Option B (RUST_FP_LEGACY opt-out flag). The bead's audit notes
already reframe the work to "Add a concrete-FP fast path that skips Z3
when both operands are concrete, instead of an opt-out flag" — so this
session is an audit + decision, no Rust code change.

### Decision: Option A — accept the regression. Close as wontfix.

#### Audit of vex/ops.rs concrete fast paths (HEAD = d2752ebe0)

Every FP op already has the concrete fast path:

| Op family               | Lines       | Guard                              |
|-------------------------|-------------|------------------------------------|
| float_neg / float_abs   | 1184–1213   | bitwise xor/and (always fast)      |
| float_sqrt              | 1215        | `arg.as_u128()`                    |
| float_add/sub/mul/div   | 1241–1342   | `(left.as_u128(), right.as_u128())`|
| float_madd / float_msub | 1346–1407   | `(a, b, c).as_u128()`              |
| vec_float_scalar_op     | 1409        | `(left, right).as_u128()`          |
| vec_float_scalar_sqrt   | 1460        | `arg.as_u128()`                    |
| vec_float_scalar_max/min| 1496–1559   | `(left, right).as_u128()`          |
| vec_float_op            | 1739        | `total_width<=128 && (l, r).as_u128()` |
| vec_float_unop          | 1808        | `total_width<=128 && arg.as_u128()`|
| vec_float_minmax        | 1869        | `total_width<=128 && (l, r).as_u128()` |
| float_to_int            | 2316        | `arg.as_u128()`                    |
| float_to_float          | 2336        | `arg.as_u128()`                    |
| f{32,64}_to_f{64,32}    | 2355        | `arg.as_u128()`                    |
| f{32,64}_to_i{32,64}{s,u}| 2400+      | `arg.as_u128()` (each variant)     |
| float_to_int_rm         | 2547+       | `(rm, arg).as_u128()`              |
| float_to_float_rm       | 2568        | `(rm, arg).as_u128()`              |
| binop_with_rm           | 427         | RNE-concrete → `Self::binop` fast  |
| unop_with_rm            | 459         | RNE-concrete → `Self::unop` fast   |

#### Remaining gaps (intentionally NOT fixed)

1. **Non-RNE concrete rm + concrete operands in binop_with_rm /
   unop_with_rm**: currently routes through Z3 (via
   `build_fp_arith_rm_cached`) even when all operands are concrete.
   Filling this gap in pure Rust would need platform-specific
   rounding-mode intrinsics (no portable `f32`/`f64` op honors a
   non-default rm). Z3 already produces the correct result; the
   cost-benefit doesn't justify the complexity.

2. **AVX 256-bit packed FP (vec_float_op / vec_float_unop /
   vec_float_minmax when total_width > 128)** falls through to the
   symbolic Z3 path even when all inputs are concrete, because
   `as_u128()` can't represent >128-bit values. Future work could add
   a chunked concrete path. Not relevant to fairlight (xmm-only).

#### Why Option A is the right call

Per memory `fairlight-bottleneck-2026-05` (HEAD=fee4ea333,
run_single.py per-site profiling):

- Total fairlight wallclock: 12.9–14.0s
- Z3 ~95% of total (z3_check umbrella 12.3–13.3s / 35 calls)
- Top sites: z3_site_branch_true 9.1–10.6s / 13 calls (~700–820 ms /
  call — path-condition accumulation, not single-op cost)
- Interpreter expr_eval: 14 ms over 11,109 calls (well under 1%)

Adding a concrete fast path for non-RNE rm or for 256-bit packed
lanes would reduce single-op cost, but single-op cost is already
< 1% of total. The Z3 path-condition accumulation in
`z3_site_branch_true` is the lever that matters; that's tracked under
angr-pufm (lazy memory) and the broader Z3-perf cluster, not here.

The FP commits (769d1ee54, fcf33c14c, 37983a77b, d0377abb9) fix a
genuine correctness issue — silently dropped constraints. Reverting
via RUST_FP_LEGACY opt-out would re-introduce that correctness bug.
Per the bead's own guidance ("Do NOT silently revert the FP commits —
they fix a genuine correctness issue"), Option B is unacceptable.

### Files changed

None — close-only.

### Memories saved

- `fp-concrete-fast-path-complete` — audit anchor for FP concrete
  fast paths in vex/ops.rs as of d2752ebe0; lists the two intentional
  gaps (non-RNE concrete rm; >128-bit packed) and why they're not
  worth filling.

### Tests

None run — no code change.

### Next session

- angr-pufm (P1) is the largest unblocked. Audit notes recommend
  splitting into two children:
  (a) configurable concretization fallback strategy enum
  (b) lazy guarded entries in symbolic_objects+spans for >N solutions
  Each is its own session-sized task. Recommend creating the children
  beads before claiming.
- angr-6n56 (P2) is an investigation task on Arc-tree teardown
  hotspots — also session-sized.
