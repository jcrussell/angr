## Session log: 2026-05-09 — angr-wqao.3 (196th loop session, CLOSED)

### Task
angr-wqao.3 — Document rust_manager.py invariants. Pure docs task, sibling
of deferred refactor tasks (.1, .2, .4) which propose mechanical extraction
with no motivating bug. wqao.3 was different: documenting existing
cross-mixin invariants for future contributors.

### Outcome
**CLOSED** in commit 30e682208.

### What was done
Added a 95-line docstring block at the top of `angr/exploration/rust_manager.py`
listing 10 cross-mixin invariants. Implemented as docstring (not a separate
INVARIANTS.md) per CLAUDE.md "no .md files" preference; the bead description
explicitly allowed either location.

The 10 invariants:
  I1  Disk-cache key axes (_RUST_CACHE_VERSION/_PYTHON_METADATA_VERSION + memo)
  I2  Init pipeline phases (post angr-khth split: pickle / deserialize /
      side-effects / orchestrator)
  I3  Init-cache user-symbolic gate (_state_has_user_symbolic must gate)
  I4  _apply_state_metadata is an allow-list (only LAZY_SOLVES + STRICT_PAGE_ACCESS)
  I5  Register filter at FFI boundary (_supported_register_names in
      rust_state_sync.py)
  I6  State-cache pinning + manager-vs-mixin override of _cleanup_state_cache
  I7  Rust↔Python field sync uses max(), not overwrite
  I8  Exploration-loop termination must check Rust-native + predicate finds
      via _found_count()
  I9  push_to_active_or_drop helper enforces max_active_states
  I10 mgr.stats is a @property; mgr._rust_mgr.stats() is a method

### Decisions
- Cited test file lines (stable) but not rust_manager.py self-references
  (would rot — adding 95 lines already shifted everything below).
- Acceptance criteria asked for ≥4 invariants; delivered 10 to make the
  document genuinely useful as an entry point.

### Files modified
- angr/exploration/rust_manager.py (+113 lines, docstring only)

### Verification
- 357/357 tests passing (`pytest tests/engines/test_rust_exploration.py`)
- Docstring imports cleanly

### Memory saved
- invariant-rust-manager-docstring-canonical (pointer to the docstring
  as canonical source)

### Bead status
- angr-wqao.3 CLOSED with rationale.
- Parent angr-wqao still open. Three siblings (.1 .2 .4) still open with
  their parent's deferral memory applying. None of them block on this docs
  task — they were independent refactor proposals that the deferral
  audit found unjustified.
