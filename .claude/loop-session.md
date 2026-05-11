## Session log: 2026-05-11 — angr-glgd (claripy version-pin packaging fragility)

### Task
Resolve the unobtainable `9.2.210.dev0` pins on archinfo / claripy / cle / pyvex
in pyproject.toml that broke fresh `pip install -e .` from a clean cache.

### Root cause
PyPI never published `.dev0` versions of any angr-ecosystem package; PyPI's
release sequence jumps 9.2.209 → 9.2.211. Existing venvs only worked because
pip's local cache held 9.2.209 from earlier installs. Fresh installs hit a
ResolutionImpossible error.

### Fix
Pin all four to `==9.2.209` (also updated build-system requires for pyvex).
Chosen over `>=9.2.209,<9.3` because claripy 9.2.209 declares
`z3-solver==4.13.0.0` exactly, matching the installed venv z3 and the
version z3-sys was built against. Bumping further would risk pulling a
claripy that requires a different z3, breaking the Rust↔Python shared Z3
context (per `avoid-pip-install-deps` memory).

### Verification
- pip3 download claripy/pyvex/archinfo/cle ==9.2.209 → all four resolved
- Test suite: 385 passed, 3 pre-existing failures (dcas_cmpxchg16b_no_match,
  pipe_native_dispatch_creates_two_fds, dup2_native_dispatch_redirects_stdin)
- `python -c "import claripy"` → 9.2.209 (matches new pin)
- `import z3` → 4.13.0 (matches claripy 9.2.209's z3-solver==4.13.0.0 dep)

### Files modified
- pyproject.toml (5 occurrences of 9.2.210.dev0 → 9.2.209)
- CLAUDE.md (Bootstrap caveat rewritten as "Note" explaining new pin)

### Commit / bead
- 27a11d99f build: relax angr-ecosystem pins from 9.2.210.dev0 to 9.2.209
- angr-glgd closed.

### Memory saved
- invariant-angr-ecosystem-deps-pin — these four deps MUST stay pinned to
  ==9.2.209, not relaxed to >=, because claripy 9.2.209 requires
  z3-solver==4.13.0.0 exactly. Newer claripy may bundle different z3 →
  shared-context segfault risk.

### Status
COMPLETE — angr-glgd closed; fresh `pip install -e .` will now resolve from
PyPI without cache. No regressions.
