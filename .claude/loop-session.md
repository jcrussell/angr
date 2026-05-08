# Loop session notes (2026-05-08, 139th loop session)

## Task: angr-as3c — posix_brk Rust→Python sync (closed)

### Status: complete; closed

### Summary
Mirrors angr-0cnm (mmap_base) but with an extra wrinkle: defaults
between Python and Rust DON'T match for posix_brk. angr's SimUserland
loader sets `state.posix.brk = binary_last_addr + page` (≈0x602000 for
fauxware), while Rust hardcodes 0x1B00000. Naïve `max(rust, python)`
sync would clobber Python's loader-set value with Rust's stale default
on the very first export.

Fix is **bidirectional**:
- Init push (Python→Rust) in `_add_rust_state` so Rust's posix_brk
  starts at the loader-set base.
- Export sync (Rust→Python) on stash export.

### Changes
- **state.rs** — added `posix_brk` getter/setter on `PyRustSimState`
  (`#[getter]/#[setter]`).
- **rust_manager.py** — added init push of `angr_state.posix.brk` →
  `rust_state.posix_brk` in `_add_rust_state` (right after PC). Skips
  BV-valued brk (Python's set_brk wraps it after a grow).
- **exploration/mod.rs** — added `get_state_posix_brk(state_id)` and
  `set_state_posix_brk(state_id, addr)`, mirroring
  `get_state_mmap_base/set_state_mmap_base`.
- **rust_state_export.py** — added `_sync_rust_posix_brk_to_state`,
  wired into all four export paths in `_get_stash_states`. Uses
  `isinstance(py_brk, int)` to skip BV-valued brk; `max(rust, python)`
  for the int case.
- **test_rust_exploration.py** — `TestPosixBrkSync` (7 new tests):
  default getter, setter round-trip, unknown-id error, init-push
  alignment, end-to-end export sync, no-clobber direction, BV preserved.

### Verification
- `cargo check --release` clean.
- `pytest tests/engines/test_rust_exploration.py` — **276/276 passing**
  (was 269 before; +7 new posix_brk tests).
- Pre-fix: end-to-end test `test_export_path_syncs_rust_posix_brk_into_state_posix`
  failed because Rust's default 0x1B00000 > fauxware's loader-set 0x602000;
  max() returned the wrong direction.

### Memories saved
- `invariant-rust-python-default-divergence` — Rust defaults aren't
  always equal to angr loader defaults (mmap_base happened to align,
  posix_brk doesn't).
- `avoid-naive-max-merge-sync` — pattern for adding new field syncs:
  always check whether the Python loader overrides the field, and if
  so, also init-push.
- `fragile-venv-recovery` — recipe for the recurring .venv corruption
  (force-reinstall pip + setuptools<81 + semantic_version).

### .venv side trip
The .venv had stripped .py files in pip/_vendor and setuptools/_vendor
(only .pyc left). `pip install -e .` failed with FileNotFoundError.
Recovered with `pip install --force-reinstall --no-deps pip setuptools<81 semantic_version`.

### Files changed
- native/angr/src/state.rs (+15)
- native/angr/src/exploration/mod.rs (+29)
- angr/exploration/rust_manager.py (+14)
- angr/exploration/rust_state_export.py (+35)
- tests/engines/test_rust_exploration.py (+148)
