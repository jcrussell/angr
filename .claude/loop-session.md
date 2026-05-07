# Loop session notes (2026-05-07, 108th loop session)

## Task: angr-8f7d — Optionally cache lane extraction in SSE scalar float ops (CLOSED wontfix)

Bead closed as wontfix without code change.

### Decision

The bead's acceptance criterion explicitly allowed close-as-wontfix when
profiling shows lane-extract is not hot. The prerequisite bead angr-q2dk
already had that profiling and the answer was unambiguous:

- **Total fairlight wallclock:** 12.9-14.0s
- **Z3 contribution:** ~95% (z3_check umbrella 12.3-13.3s)
- **Top single cost:** z3_site_branch_true (9.1-10.6s over 13 calls,
  ~700-820ms per call — path-condition accumulation, not single-op cost)
- **Interpreter expr_eval:** 14ms over 11,109 calls (well under 1%)
- **block_exec ex-Z3:** sub-second

vec_float_scalar_lane_binop / vec_float_scalar_lane_minmax /
vec_float_scalar_sqrt do the extract(127, lane_bits) + concat that the
bead worried about, but only on the symbolic fallback path — the
concrete fast path (`as_u128()` guard) already returns early. Even if
every one of the 11109 expr_evals were one of these ops, there's still
no daylight: the entire interpreter side is far below 1% of the runtime
that fairlight cares about.

The remaining headroom for fairlight lives in the Z3 path-condition
side. That's tracked by angr-faws (now unblocked).

### Files changed

None — close-only.

### Memories saved

- `sse-lane-extract-not-hot` — anchors the wontfix decision so we
  don't re-claim or re-investigate this in a future session.

### Tests

None run — no code change. (Build state inherited from 76c76c1b3 /
ccb327850 from the prior session; benchmarks last sweep stayed at
11/12 regression-pass with csgames2018 already-broken-on-master per
existing memory.)

### Unblocked

- angr-faws (Decide on opt-out flag for Z3 FP theory if intrinsic
  overhead is irreducible) was depending on angr-8f7d. Per the bead
  notes, the right reframe is "Add a concrete-FP fast path that skips
  Z3 when both operands are concrete" — but inspection of vex/ops.rs
  shows that fast path is already present in vec_float_scalar_*
  (`as_u128` guard returns `RustBV::concrete(...)` early). A future
  session should audit whether any FP op is missing the concrete fast
  path and then close angr-faws accordingly.
