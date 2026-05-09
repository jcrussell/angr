## Session log: 2026-05-09 — angr-800o + angr-1bqa (186th loop session)

Closed two related tasks: arch coverage promotion and SimOption rejection.

### angr-800o — AArch64 + MIPS32 real-binary integration tests

Sibling/follow-up of angr-lvem (ARM landed in 32328959f). Added end-to-end
.explore() tests for both arches using cle's Blob backend with hand-assembled
instructions (7 each). Each test asserts a symbolic input is constrained
to 42 in the found state, exercising VEX lifting, register sync, branch
handling, and PC propagation end-to-end.

Resolved the dirty state from the prior session:
- Inadvertent regression in TestCallableStepFunc::test_callable_with_rust_engine.
  Original `_load_binary_regions` skipped externs/tls/kernel via
  `if obj.binary is None` — but those pseudo-objects actually have a
  synthetic `obj.binary='cle##externs'` (str, not None). They were "safely"
  skipped only because they have no executable *sections*. Adding the
  Blob-needed segments fallback exposed an executable segment at
  0x700000-0x700030 in cle##externs, which the Rust interpreter then
  tried to lift as code, splitting Callable on a symbolic condition.
  Fix: explicit `obj.binary.startswith("cle##")` skip before
  section/segment walk.

Promoted ARM64 + MIPS32 from Skeleton to Experimental in the support
matrix (CLAUDE.md).

Commit 0ceebfc8a.

### angr-1bqa — Warn-once for divergence-risk SimOptions

Follow-up of angr-pe5t's docs/RUST_SIMOPTION_COVERAGE.md. The matrix
classifies many SimOptions as `(b) explicitly reject` — they would
change Python-engine semantics but the Rust engine silently ignores
them. Implemented `_REJECTED_OPTION_NAMES` + `_warn_rejected_options()`
that emits a UserWarning the first time each option is seen on a state.

Key implementation choices:
- Warn-once per option per manager (using a per-instance set), not per
  state added. Three states with CALLLESS yield exactly one warning.
- Check fires from `__init__` on user-supplied states. Cannot fire from
  `_add_rust_state` alone because `_apply_state_metadata` strips options
  to {LAZY_SOLVES, STRICT_PAGE_ACCESS} on the cached-init-state path,
  losing user-set options before they reach `_add_rust_state`.
- Excluded `TRACK_CONSTRAINT_ACTIONS` and `TRACK_MEMORY_MAPPING` from
  the warn set even though doc-tagged (b): they ship in the default
  `symbolic` mode bundle (sim_options.py:391, 374), so every plain
  `entry_state()` would otherwise emit warnings the user did not opt
  into. Documented the exclusion in
  docs/RUST_SIMOPTION_COVERAGE.md.

3 new tests cover positive (TRACK_MEMORY_ACTIONS + DO_RET_EMULATION),
dedup (3x CALLLESS = 1 warning), and no-false-positive (default
entry_state emits no spurious warnings).

Commit 0293ea03d.

### Test status

357/357 passing (was 354). 354 → +3 from arch tests. 354 → +3 from
SimOption tests. Final 357 because removing the cle## pseudo-object
fix could otherwise have broken the existing Callable test.

### Memories saved this session

- `cle-pseudo-objects-binary-not-none` — pseudo-objects have synthetic
  `obj.binary='cle##externs'`, not None.
- `blob-loader-segment-fallback` — Blob loader exposes only segments,
  not sections; segments fallback is required.
- `invariant-apply-state-metadata-strips-options` — disk-cache hit
  path drops user-set SimOptions except LAZY_SOLVES + STRICT_PAGE_ACCESS.
- `avoid-frozenset-intersection-on-simstateoptions` — set protocol
  fails on SimStateOptions; iterate the smaller set instead.
- `invariant-default-symbolic-mode-tracks` — TRACK_CONSTRAINT_ACTIONS
  and TRACK_MEMORY_MAPPING ship in default symbolic mode; cannot be
  warned/rejected without spam.
