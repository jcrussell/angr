# Loop session notes (2026-05-05, fifty-second loop session — BLOCKED)

## Status: ENV BROKEN, NO WORK DONE

## What I found
- `.venv/bin/` contained only `__pycache__/` (no `python`, no `pip`, no `activate`).
- `.venv/lib/python3.12/site-packages/` contained ~80 directories, but most
  (claripy, cle, archinfo, pyvex) had only `__pycache__/` subdirs — no actual
  `.py` source files. Every angr-dep was a hollow shell.
- The pre-built `angr/rustylib.cpython-312-x86_64-linux-gnu.so` is still present
  on disk, but unusable until python deps come back.

## What I tried (all failed for one reason)
1. `python3 -m venv .venv-new` + copy `lib/python3.12/site-packages/*` over →
   the source dirs were already empty so nothing to copy.
2. `pip install setuptools setuptools-rust protobuf` → succeeded, but downstream
   imports (`from cle import SymbolType`) still fail because cle's source files
   don't exist locally.
3. `pip install -e . --no-build-isolation --no-deps` → fails at z3-sys build
   (looks for `.venv/lib/python3.12/site-packages/z3/include/z3.h` which the
   broken venv never had — z3 dir only contained `lib/`).
4. `pip install claripy==9.2.210.dev0` → not on PyPI. 9.2.210 itself doesn't
   exist (versions jump 9.2.209 → 9.2.211). The `pyproject.toml` pin is to
   an unreleased dev build.
5. Memory `avoid-pip-install-deps` explicitly forbids installing released
   claripy/pyvex/cle/archinfo because they break the z3 shared context
   (segfault) and cascade incompatible deps.

## Why the loop can't unblock itself
- Recovery would need to install released ~9.2.213 angr-deps (forbidden by
  memory) **or** locate dev0 wheels (none exist on PyPI or in pip cache;
  cache has 9.2.196 / 9.2.209 / 9.2.213 only).
- No sibling angr-deps repos are checked out under /home/ubuntu/repos/.
- Z3 headers for the Rust build can be redirected to /usr/include/z3.h
  (Z3_SYS_Z3_HEADER), but it doesn't matter while python imports fail.

## Memories saved
- `env-venv-fully-wiped-2026-05-05` (the full incident report)

## What a human needs to do
Pick one:
- **Option A (fast):** Relax `pyproject.toml` pins from `==9.2.210.dev0` to
  `>=9.2.213` for archinfo/claripy/cle/pyvex, then `pip install -e .` — accept
  the documented z3 context incompatibility risk and validate via tests.
- **Option B (safer):** Restore .venv from another machine snapshot or
  rebuild claripy/cle/archinfo/pyvex from upstream master at the commit
  matching the prior dev0 build.
- **Option C (radical):** Accept that the dev0 pins reference a transient
  build artifact that no longer exists and re-pin to the most recent stable
  release; treat any z3-context regression as a follow-up.

After whichever option, the next loop session should run:
```
export PATH="$HOME/.cargo/bin:$PATH"
source .venv/bin/activate
pip install -e . --no-build-isolation --no-deps
python -m pytest tests/engines/test_rust_exploration.py -v --tb=short
```
to confirm the world is sane before claiming any task.

## Beads
None claimed/closed. `bd list --status=in_progress` returned empty.

## Next-up (still ready, from prior session)
- angr-eygl (P1) Differential test harness
- angr-pufm (P1) Symbolic address concretization fallback when intractable
- angr-0dgj (P1) Arc-wrap symbol_table and forkable interpreter state
- angr-nwbx (P1) Cache claripy↔Z3 conversion for register sync
- angr-k9hr (P2) Arc-wrap interpreter temps to make RdTmp clones cheap
