# Upstream patches (offline-prepared)

This directory holds patches and draft PR text prepared offline
against angr-ecosystem upstream repos. Each artifact pair is a
slice of an upstream-PR bead (see `bd recall <key>`) that could
not be filed directly because github.com was unreachable from the
working environment.

For the broader pattern family (when to use this vs.
`tools/draft_patches/` vs. an aggregation-doc v1 stub) see
[`../OFFLINE_WORKFLOWS.md`](../OFFLINE_WORKFLOWS.md).

## How to file when network is available

For each `<topic>.patch` + `<topic>_PR.md` pair:

```bash
# 1. Clone the upstream repo (or fetch latest if already cloned)
git clone https://github.com/angr/<repo> /tmp/<repo>
cd /tmp/<repo>

# 2. Verify the patch still applies cleanly against current main
git checkout main
git pull
git apply --check /path/to/angr/tools/upstream_patches/<topic>.patch
# (or `patch -p1 --dry-run < ...` — patches carry the leading
# `Subject:` block plus a `diff --git a/... b/...` header so both
# work)

# 3. Apply, sanity-test, and push to a feature branch
git checkout -b <topic>
git apply /path/to/angr/tools/upstream_patches/<topic>.patch
# Run upstream tests; refresh patch if anything drifted
python -m pytest tests/ -k <relevant-tests>

# 4. Commit + push + open PR using the prepared body
git add -A && git commit
git push -u origin <topic>
gh pr create --title "<title from PR.md>" \
    --body-file /path/to/angr/tools/upstream_patches/<topic>_PR.md
```

## Current artifacts

### `archinfo_aarch64_be.patch` + `archinfo_aarch64_be_PR.md`

Target: `angr/archinfo` (prepared against 9.2.221).

Adds AArch64 big-endian support: threads
`instruction_endness=Endness.BE` to `Arch.__init__` (mirroring
ARM precedent) and registers a BE alias for the canonical Linux/qemu
spellings `aarch64eb` / `aarch64be` / `arm64eb` / `arm64be`. Without
this, `arch_from_id("aarch64eb")` silently returns an LE Arch.

Downstream blocker: `bd recall angr-ig3o.2` (ARM64 BE integration
test). Unblock sequence: file PR → upstream merge + release →
bump `archinfo==<new>` in `pyproject.toml` → close `angr-b3sc` and
`angr-ig3o.2`.

Verification (run after applying patch in an archinfo checkout):

```python
import archinfo
from archinfo.arch import Endness

for name in ("aarch64eb", "aarch64be", "arm64eb", "arm64be"):
    a = archinfo.arch_from_id(name)
    assert a.memory_endness == Endness.BE
    assert a.instruction_endness == Endness.BE

assert archinfo.arch_from_id("aarch64").memory_endness == Endness.LE
```
