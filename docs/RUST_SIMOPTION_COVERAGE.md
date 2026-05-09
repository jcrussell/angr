# Rust Engine SimOption Coverage Matrix

This file documents which `angr.sim_options` flags the Rust symbolic-execution
engine (`RustExplorationManager`) actually honors. Anything not listed as
**Honored** or **Inherited** is a **silent no-op** when running under
`use_rust_engine=True` — the option may be set on the SimState and visible to
Python introspection, but the Rust interpreter / memory model / solver does
not consult it.

**Status legend**

| Status | Meaning |
|--------|---------|
| ✅ Honored | Rust reads the option and changes behavior accordingly. |
| ↪ Inherited | Option still has its Python effect because the relevant code path runs in Python (claripy AST passthrough, Python SimProcedure callbacks, pyvex lifting). |
| ⚠ Ignored — divergence-risk | Setting the option in Python would change behavior; under Rust it does nothing, so two engines may produce different results. |
| ◌ Ignored — no-op | The Python feature this gates isn't implemented in Rust at all (unicorn, abstract memory, CGC, JAVA, action tracking, etc.). Setting it has no effect either way. |

**Future-fix legend** (for Ignored rows)

- (a) **implement** — Rust should grow support for this option.
- (b) **explicitly reject** — `add()` of this option on a state owned by
  `RustExplorationManager` should raise, so silent divergence becomes a loud
  error.
- (c) **accept-but-document** — option is benign, leave the silent-ignore
  but note it here.

The Python-side option set lives in
`RustExplorationManager._py_state_options` (per-state-id dict, see
`get_state_options_py`); writing to `state.options.add(X)` through the proxy
stores the value but does not change Rust behavior unless this matrix shows
otherwise.

## Honored options

| Option | Where Rust reads it | Effect |
|--------|---------------------|--------|
| `LAZY_SOLVES` | `rust_manager.py:698`, `:2322` (re-checked at `explore()` time) | Calls `set_lazy_solves(True)`; Rust skips per-block satisfiability checks. Also propagated through `_clone_state_metadata` on disk-cache reuse (`:1752`). |
| `ZERO_FILL_UNCONSTRAINED_MEMORY` | `rust_manager.py:701` | Calls `set_zero_fill_unconstrained(True)`; uninitialized memory reads return `BVV(0, n)` instead of fresh symbols. |
| `APPROXIMATE_MEMORY_INDICES` | `rust_manager.py:705` | Passed as `use_approx` to `configure_concretization_strategies`; Rust adds an approximation strategy ahead of full Z3 enumeration on symbolic loads. |
| `SYMBOLIC_WRITE_ADDRESSES` | `rust_manager.py:706` | Passed as `sym_write` to `configure_concretization_strategies`; Rust permits multi-valued symbolic write targets instead of forcing concretization. |
| `STRICT_PAGE_ACCESS` | `rust_manager.py:1963` (per-state, also propagated through `_clone_state_metadata` on disk-cache reuse) | Calls `RustSimState::set_enforce_permissions(True)`; loads/stores violating per-page R/W bits raise `SimSegfaultError`. Preserved across forks via `SymbolicMemory::fork`. |

## Inherited (option works because the code path runs in Python)

| Option | Why it still works |
|--------|--------------------|
| `SYMBOLIC` | Rust always operates symbolically. The option is required for claripy/Python-side SimProcedures to behave symbolically. |
| `SYMBOLIC_INITIAL_VALUES` | claripy AST creation honors this; Rust receives ASTs from Python. |
| `TRACK_CONSTRAINTS` | Constraints are stored in claripy on the Python side via the constraint sync; the Rust solver mirrors them. |
| `COMPOSITE_SOLVER` | Solver flavor is on the claripy side — Rust uses its own Z3 context but only stores constraints, not the solver topology. |
| `SUPPORT_FLOATING_POINT` | claripy gates float ops behind this. ASTs that propagate through Rust come back to Python for any non-VEX-native operation. |
| `SIMPLIFY_*` (claripy AST simplification) | Simplification happens inside claripy when building the AST (e.g. `state.solver.simplify`). Rust embeds those ASTs unchanged. |
| `USE_SYSTEM_TIMES` | Time-related SimProcedures (`gettimeofday`, `time`, `clock_gettime`) dispatch to Python; the Python procedure honors the flag. Inherited only when the procedure has not been replaced by a native variant in `native/angr/src/procedures/`. |
| `ALLOW_SEND_FAILURES`, `FILES_HAVE_EOF`, `ALL_FILES_EXIST`, `ANY_FILE_MIGHT_EXIST`, `SHORT_READS`, `CONCRETIZE_SYMBOLIC_FILE_READ_SIZES` | POSIX SimProcedures (`read`, `write`, `open`, `recv`, `send`) execute in Python and consult the angr SimState directly. Native POSIX procedures (when enabled — angr-3tek tracks this) would bypass these flags. |
| `RUN_HOOKS_AT_PLT` | Hook dispatch is orchestrated by the Python wrapper; PLT detection runs against the Python project. |
| `OPTIMIZE_IR`, `NO_CROSS_INSN_OPT` | pyvex still produces the IRSB on the Python side before handing bytes to the Rust interpreter. |

## Ignored — divergence-risk (a candidate fixes are bolded)

These options *would* change Python-engine behavior. Under Rust they are
silently dropped, which can produce different found/avoided sets vs. Python.

> **Implementation note (angr-1bqa, 2026-05-09):** The (b) classified
> options are warn-once'd via `_REJECTED_OPTION_NAMES` in
> `angr/exploration/rust_manager.py`. Two entries from the (b) category —
> `TRACK_CONSTRAINT_ACTIONS` and `TRACK_MEMORY_MAPPING` — are intentionally
> *excluded* from the warn set because they ship in the default `symbolic`
> mode bundle (`sim_options.py:391`, `:374`); warning every default-options
> `entry_state()` would generate noise the user did not consent to. They
> remain divergence-risk in this matrix, just not warn-on-add.

| Option | What Python does | Suggested fix |
|--------|------------------|---------------|
| `KEEP_IP_SYMBOLIC` | Allows IP to remain symbolic across blocks. | **(a) implement** — Rust always concretizes IP at block boundaries; symbolic-IP support is a real gap. |
| `NO_IP_CONCRETIZATION` | Aborts on symbolic IP instead of concretizing. | **(a) implement** — same gap as above, opposite direction. |
| `ENABLE_NX` | Raises on execution from non-X pages. | (a) implement — Rust does not consult page X-bits during instruction fetch. |
| `NO_SYMBOLIC_JUMP_RESOLUTION` | Suppresses symbolic-jump enumeration. | (a) implement — Rust resolves symbolic jumps differently and may diverge on heavy-symbolic targets. |
| `NO_SYMBOLIC_SYSCALL_RESOLUTION` | Same, for syscalls. | (a) implement. |
| `AVOID_MULTIVALUED_READS` / `AVOID_MULTIVALUED_WRITES` | Returns unconstrained instead of enumerating addresses. | (a) implement — Rust always enumerates within strategy limits. |
| `CONCRETIZE_SYMBOLIC_WRITE_SIZES` | Concretizes the *size* of a symbolic-sized write. | (a) implement — Rust concretizes addresses but not sizes the same way. |
| `CONSERVATIVE_WRITE_STRATEGY` / `CONSERVATIVE_READ_STRATEGY` | Refuses to concretize on range-check failure. | **(b) explicitly reject** — silent ignore can mask intended-conservative analyses. |
| `CONCRETIZE` | Eagerly concretizes every symbol introduced. | **(b) explicitly reject** — totally changes semantics; silent ignore is dangerous. |
| `ZERO_FILL_UNCONSTRAINED_REGISTERS` | Default-zero registers instead of fresh symbols. | (a) implement — pairs with the already-honored memory variant. Currently Rust always picks one or the other depending on init-state plumbing. |
| `SYMBOL_FILL_UNCONSTRAINED_MEMORY` / `SYMBOL_FILL_UNCONSTRAINED_REGISTERS` | Force symbolic fill (the opposite of ZERO_FILL_*). | (a) implement — partial overlap with ZERO_FILL behavior. |
| `TRACK_MEMORY_ACTIONS`, `TRACK_REGISTER_ACTIONS`, `TRACK_TMP_ACTIONS`, `TRACK_JMP_ACTIONS`, `TRACK_CONSTRAINT_ACTIONS`, `TRACK_OP_ACTIONS` | Populate `state.history.actions` with `SimAction*` records. | **(b) explicitly reject** — the action stream is empty under Rust regardless, so silent acceptance misleads users who rely on `state.history.actions`. |
| `TRACK_ACTION_HISTORY` | Same, across path. | (b) explicitly reject. |
| `TRACK_MEMORY_MAPPING` | Logs map/unmap into `state.history`. | (b) explicitly reject. |
| `CONSTRAINT_TRACKING_IN_SOLVER` | Required for `solver.unsat_core()`. | (a) implement — Rust solver tracks constraints internally but `unsat_core` is not surfaced. |
| `BYPASS_UNSUPPORTED_IROP`, `BYPASS_ERRORED_IROP`, `BYPASS_UNSUPPORTED_IREXPR`, `BYPASS_UNSUPPORTED_IRSTMT`, `BYPASS_UNSUPPORTED_IRDIRTY`, `BYPASS_UNSUPPORTED_IRCCALL`, `BYPASS_ERRORED_IRCCALL`, `BYPASS_UNSUPPORTED_SYSCALL`, `BYPASS_ERRORED_IRSTMT`, `BYPASS_VERITESTING_EXCEPTIONS`, `UNSUPPORTED_BYPASS_ZERO_DEFAULT`, `UNSUPPORTED_FORCE_CONCRETIZE` | Tell the Python engine to swallow / fall back to a default on unsupported VEX. | (a) implement — Rust has its own error path; the bypass set is not consulted. Resilience modes that work in Python may abort under Rust. |
| `UNINITIALIZED_ACCESS_AWARENESS`, `BEST_EFFORT_MEMORY_STORING` | Affect SimMemory error handling. | (b) explicitly reject. |
| `DO_RET_EMULATION`, `TRUE_RET_EMULATION_GUARD` | Add emulated ret-site successors. | (b) explicitly reject — Rust does not emulate. |
| `CALLLESS` | Replaces calls with unconstraining of return register. | (b) explicitly reject. |
| `SUPER_FASTPATH`, `FAST_MEMORY`, `FAST_REGISTERS`, `UNDER_CONSTRAINED_SYMEXEC` | Select alternate Python engines / memory plugins. | (b) explicitly reject — fundamentally incompatible with Rust state model. |
| `PRODUCE_ZERODIV_SUCCESSORS` | Spawns successor with `divisor == 0`. | (a) implement — Rust treats div-by-zero as a single state. |
| `EXTENDED_IROP_SUPPORT` | pyvex extended ops; Rust may not handle every op. | (a) implement / audit per-op coverage. |

## Ignored — no-op (Rust does not implement the corresponding feature)

These options gate Python-only features that Rust simply doesn't have. The
silent ignore is harmless: they have no effect under either engine in Rust
mode.

| Option | Reason |
|--------|--------|
| `ABSTRACT_MEMORY` | Rust uses `RustSimMemory` only — there is no SimAbstractMemory backend. |
| `ABSTRACT_SOLVER` | Rust does not have an abstract-domain solver. |
| `AST_DEPS`, `ACTION_DEPS`, `AUTO_REFS`, `ADD_AUTO_REFS` | Dependency tracking on SimActions; Rust produces no actions. |
| `REVERSE_MEMORY_NAME_MAP`, `REVERSE_MEMORY_HASH_MAP`, `MEMORY_SYMBOLIC_BYTES_MAP` | Python SimSymbolicMemory bookkeeping. |
| `REGION_MAPPING` | Python memory plugin region log. |
| `DO_CCALLS`, `USE_SIMPLIFIED_CCALLS` | Rust executes VEX directly without ccall helpers. |
| `SYMBOLIC_TEMPS` | Rust temps are SSA-style register internals. |
| `SPECIAL_MEMORY_FILL` | Python memory fill hook; Rust uses its own fill path. |
| `MEMORY_CHUNK_INDIVIDUAL_READS` | Python memory-bp granularity tweak. |
| `MEMORY_FIND_STRICT_SIZE_LIMIT` | Argument to Python `SimMemory.find()`. |
| `EFFICIENT_STATE_MERGING` | Rust does not yet support state merge. |
| `DOWNSIZE_Z3` | Python claripy Z3 downsize; Rust manages its own context. |
| `REPLACEMENT_SOLVER`, `CACHELESS_SOLVER`, `HYBRID_SOLVER`, `APPROXIMATE_FIRST` | Alternate claripy solver flavors. |
| `APPROXIMATE_GUARDS`, `APPROXIMATE_SATISFIABILITY`, `APPROXIMATE_MEMORY_SIZES`, `VALIDATE_APPROXIMATIONS` | claripy-side approximation; Rust has its own concretization strategy. |
| `SYMBOLIC_MEMORY_NO_SINGLEVALUE_OPTIMIZATIONS` | Tracing-mode SimMemory tweak. |
| `CPUID_SYMBOLIC` | Python ccall flag. |
| `EXCEPTION_HANDLING` | Python OS plugin segfault handler. |
| `SYNC_CLE_BACKEND_CONCRETE` | Python claripy concrete backend sync. |
| `TRACK_SOLVER_VARIABLES` | claripy variable tracking (`solver.all_variables`). |
| `COW_STATES`, `COPY_STATES` | Rust always copies on fork via `RustSimState::fork`. Functionally always-on. |
| `UNICORN`, `UNICORN_*` (~10 options) | Rust does not integrate with the unicorn engine. |
| `CGC_NO_SYMBOLIC_RECEIVE_LENGTH`, `CGC_ENFORCE_FD`, `CGC_NON_BLOCKING_FDS` | CGC-only; Rust does not run CGC binaries today. |
| `JAVA_IDENTIFY_GETTER_SETTER`, `JAVA_TRACK_ATTRIBUTES` | Java analysis only. |

## Provenance

Generated 2026-05-09 from `angr/sim_options.py` and the SimOption read sites
in `angr/exploration/rust_manager.py` (lines 697–706, 1751–1756, 1962–1966,
2320–2325). When new SimOptions land in `sim_options.py`, add a row here and
either wire detection into `rust_manager.py` (Honored) or classify the
silent-ignore reason.
