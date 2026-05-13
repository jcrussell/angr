## Session log: 2026-05-13 — angr-6apa (Centralize Z3 header discovery) — CLOSED

### Task

**angr-6apa** (P2, CLOSED) — Z3 header auto-detect probes the venv,
pkg-config, and standard system include paths from `setup.py`, sets
`Z3_SYS_Z3_HEADER` before cargo runs, and emits an actionable
install hint when nothing is found.

### Why setup.py and not build.rs

z3-sys's build.rs runs BEFORE our crate's build.rs (cargo runs dep
build scripts first), so our build.rs cannot influence z3-sys's
header discovery via cargo:rustc-env. The env var must be set by
the PARENT process before cargo is invoked — setup.py (called by
setuptools-rust before cargo) is the right injection point.

### Probe order (matches setup.py + tools/rebuild-rust.sh --cargo-only)

1. venv `site-packages/z3/include/z3.h` — rare (PyPI wheel ships
   no headers; only useful after manual copy).
2. `pkg-config --variable=includedir z3` + `/z3.h`.
3. `/usr/include`, `/usr/local/include`, `/opt/homebrew/include`,
   `/opt/local/include`.

If no header matches, prints apt/dnf/brew install hints to stderr
and lets z3-sys's own pkg-config probe report the build failure.

### Verification

- pkg-config returns `/usr/include`, helper resolves
  `Z3_SYS_Z3_HEADER=/usr/include/z3.h` ✓
- preset `Z3_SYS_Z3_HEADER=/sentinel` is honored ✓
- `cargo check --release` passes ✓
- `tools/rebuild-rust.sh --cargo-only` rebuild OK (venv pip remains
  broken — `avoid-broken-venv-pip-rebuild`) ✓
- `pytest tests/engines/test_rust_exploration.py` 389 pass, 3 fail
  (`dcas_cmpxchg16b_no_match_keeps_memory`,
  `pipe_native_dispatch_creates_two_fds`,
  `dup2_native_dispatch_redirects_stdin`) — these failures
  reproduce on master (stashed) — pre-existing, unrelated.

### Files touched

- `setup.py` — new `_resolve_z3_header()` helper, called at module
  load (before `setup()`).
- `native/angr/build.rs` — comment now points at the setup.py helper.
- `tools/rebuild-rust.sh` — --cargo-only fallback probes the same
  ordered list as setup.py.
- `CLAUDE.md` — Build section + Common Issues subsection refreshed.

### Commit

`d66c5d969` build(rust-symex): centralize Z3 header discovery in setup.py (angr-6apa)

### Memories saved

- `invariant-z3-sys-build-script-ordering`
- `z3-header-discovery-probe-order`

### Followups visible from session (not filed)

- The 3 pre-existing test failures (dcas, pipe native, dup2 native)
  reproduce on a stashed checkout — present before this session
  began. No existing bd issue covers them. Did not file because
  the failures appear after a `tools/rebuild-rust.sh --cargo-only`
  rebuild, which may differ subtly from `pip install -e .` (the
  normal path). Worth re-checking after the next `pip install`
  builds; if still failing, file a P2 bug.
