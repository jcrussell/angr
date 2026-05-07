# Loop session notes (2026-05-07, 126th loop session)

## Task: angr-4j5u — Decompose Rust-side RustExplorationManager god struct

### Status: AUDIT → DEFER

After audit, deferring with the same reasoning template as the prior six
architecture refactor deferrals (angr-borb / angr-ja0b / angr-x3xu /
angr-m2hf / angr-prem / angr-fk0m). The proposed decomposition is
cosmetic and would touch ~91+ callsites across two Rust files for no
behavioural gain.

### Audit findings

1. **Bead description overstates scope.**
   Bead title says "95-field god struct"; NOTES already correct to 41;
   actual count via `awk '/^pub struct RustExplorationManager/,/^}/'
   ... | grep -c "pub(crate)"` is **45 fields**. Description is off
   by ~2.1x.

2. **Existing organization already partially follows the proposed
   pattern.**
   - `ExecutionConfig` is already a sub-struct
     (`pub(crate) exec_config: ExecutionConfig`, line 360) covering
     deferred-fork execution config.
   - `NativeProcedureRegistry`, `NativeSyscallRegistry`,
     `NativeProcStats`, `StashManager` are already sub-structs.
   - File has 8 documented logical sections in the impl block
     (PyAPI, Native Proc Mgmt, Uniqueness Filter, Native Techniques,
     State Export, Run loop, Resume, Solver Profiling Stats) with
     comment dividers at lines 519, 1927, 2019, 2051, 2105, 2440,
     2973, 3623.

3. **Refactor cost is high, payoff is cosmetic.**
   Callsite counts from `grep -rc "self\.<field>"`:
   - Profiling fields (profiling_enabled / accumulated_stats /
     native_proc_stats): **47 callsites** (mod.rs:25, stepping.rs:22).
   - Hook fields (hooks / skip_hook_stack): **16 callsites**
     (mod.rs:14, stepping.rs:2).
   - Procedure dispatcher fields (native_procedures / simprocedures /
     calling_convention): **28 callsites** (mod.rs:17, stepping.rs:7,
     helpers.rs:4).
   Total: **91+ callsites** to mechanically rewrite from
   `self.field` → `self.collector.field`. No new behavior.

4. **No bug class motivates the work.**
   `bd memories god-struct` and `bd memories rust-manager-mod-rs`
   produce no incidents. 146/146 tests + 16/16 benchmarks pass. The
   only documented refactor pain in this file is the StepError enum
   (angr-ja0b, already deferred with audit memory).

5. **High invariant-breakage risk.**
   Multiple documented invariants run through this struct:
   - `invariant-stepping-decomposition` (stepping.rs has TWO
     deferred-fork code paths that share these fields)
   - `invariant-no-return-deadend` (native SimProc dispatchers in
     mod.rs:run + stepping.rs honor no_return)
   - `avoid-fixing-only-one-native-dispatch-path` (exit hook hit via
     two paths)
   Any partial decomposition risks breaking these without test signal.

### Why this matches the prior deferral pattern

Same template as angr-borb / angr-ja0b / angr-x3xu / angr-m2hf /
angr-prem / angr-fk0m:
- (a) Bead description references infrastructure that has shifted
  (struct is 45 fields, not 95; sub-struct pattern already in use).
- (b) Full scope is large (~91 callsites + multiple files for the
  "small" extractions; full refactor would touch 161 methods).
- (c) Proposed extractions = grouping by adjacent comment-section
  fields. No architectural improvement.
- (d) No bug class observed to motivate the work.
- (e) Half-measures (e.g., extracting only ProfilingCollector) still
  cost 47 callsites for purely cosmetic gain.

### Action

1. Defer angr-4j5u with this audit as the reason.
2. Save memory `avoid-deferred-4j5u-rustexploration-decomposition` so
   future sessions don't re-open without (a) a concrete bug showing
   field-coupling drift, OR (b) a refactor that genuinely changes
   responsibilities (e.g., splitting orchestration from execution),
   not just renames field groups.

### Files modified

- None — audit only; no source edits.

## Status: complete (deferred with audit + memory saved)
