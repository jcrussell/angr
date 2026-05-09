## Session log: 2026-05-09, 177th loop session

### Task: angr-rte6 — clean Rust rebuild script — CLOSED

Added `tools/rebuild-rust.sh` matching the style of the existing
`tools/restore-venv.sh`. Deterministic clean rebuild of the Rust
extension.

**What it does (default mode):**
1. Remove `angr/rustylib*.so` (so a missed rebuild surfaces as
   ImportError instead of silent stale-symbol behavior)
2. Remove `build/` (setuptools intermediate)
3. `cargo clean --manifest-path Cargo.toml` (workspace-wide)
4. `.venv/bin/python -m pip install -e . --no-build-isolation --no-deps`

**Optional modes:**
- `--keep-cargo-cache` skips step 3 (faster, when only the Python
  wrapper / setuptools-rust state is suspect)
- `--cargo-only` skips pip; uses `cargo build --release` then copies
  `target/release/librustylib.so` → `angr/rustylib<EXT_SUFFIX>.so`.
  Recovery path when the venv's pip/setuptools is corrupt — the PyO3
  .so exports the same module so `import angr.rustylib` works.
  Honors `Z3_SYS_Z3_HEADER`, falls back to `/usr/include/z3.h` when the
  venv-shipped `z3/include/z3.h` is missing.

**Surprises hit during impl:**
- This `.venv/` has NO `pip` executable script in `bin/` (only
  `python*`). Initial draft used `$VENV/bin/pip` and failed with
  "No such file or directory". Switched to `python -m pip`. Saved
  as memory `venv-no-pip-executable`.
- The default pip-based mode on this venv hits the
  `pip._vendor.resolvelib` ImportError described in
  `avoid-broken-venv-pip-fallback-cargo-build`. So I verified the
  script via `--cargo-only` instead. The pip path is still the
  documented default for healthy venvs.

### Verification

- `--cargo-only --keep-cargo-cache`: rebuilt .so in ~2s (incremental),
  342/342 rust exploration tests pass.
- `--cargo-only` (full cargo clean + rebuild): 41.8s rebuild from
  scratch, .so loads, smoke tests pass.
- CLAUDE.md "Stale .so file" entry now points at the new script.

### Memories saved

- `tool-rebuild-rust-script` — what the script does and how to invoke
  each mode.
- `venv-no-pip-executable` — current loop venv lacks `bin/pip`; always
  use `python -m pip`.

### Next session

`bd ready` — many P2/P3 tasks. Several P2 items (angr-is4x, angr-8s4b,
angr-3tek) are auto-deferred after 3 dirty iterations — handle with
care, prefer smaller scoped tasks first.
