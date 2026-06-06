# Runbook: ywu7 coverage fail_under

Target bd bead: `angr-ywu7` (CI: enforce coverage minimum via
`[tool.coverage.report].fail_under`).

Slice bead: `angr-zlzw` (this offline-prepared patch).

## Placeholder

The patch contains exactly one placeholder:

| Token            | Where it appears                                                                | Type    | What to substitute                                  |
|------------------|---------------------------------------------------------------------------------|---------|-----------------------------------------------------|
| `TBD-PY-FLOOR`   | `pyproject.toml` → `[tool.coverage.report] fail_under = TBD-PY-FLOOR`           | integer | `floor(measured_baseline_pct - 2)`                  |

`fail_under` accepts integer or float percentages (e.g. `78` or
`78.5`). Match the convention used by other Python projects on
the team if there is one; otherwise an integer is fine.

The patch as committed will NOT apply cleanly under `coverage`
parser if the substitution is skipped — `TBD-PY-FLOOR` is not
valid TOML. The runbook deliberately leaves it as a syntax-
error sentinel so an accidentally-applied draft fails loudly.

## Why a placeholder

The chosen threshold has to be calibrated against the current
coverage baseline. Picking it blind would either be (a) too
strict — every PR fails on noise — or (b) too lax — coverage can
silently drop without tripping the gate. The `angr-ywu7` bead's
acceptance criterion explicitly says "Baseline X first … do not
pick a number blind."

Measuring that baseline is the blocked sibling `angr-439q`. Its
two reachable paths:

1. **CI artifact route (preferred).** Read the most recent
   successful `coverage.yml` run on `master`:
   ```bash
   gh run list --workflow=coverage.yml --branch=master --status=success --limit=1
   gh run download <run-id> --pattern 'results-*' --dir /tmp/cov
   # `results-*` shards each carry coverage.xml; aggregate them:
   cd /tmp/cov
   python -m coverage combine $(find . -name '.coverage*' -print)
   python -m coverage report | tail -1   # last line is TOTAL: ... XX%
   ```

2. **Local baseline route (fallback).** Fresh venv, full pytest
   under coverage. ~30–60 min wall on this hardware once the
   venv is healthy:
   ```bash
   source .venv/bin/activate
   pytest -n auto --forked --cov=angr --cov=tests --cov-report=term tests/
   ```

Either path yields the `measured_baseline_pct` for the formula
above.

## Apply workflow

```bash
# 1. Obtain measured_baseline_pct (see above).
BASELINE_PCT=<value>
FAIL_UNDER=$(python3 -c "import math; print(math.floor($BASELINE_PCT - 2))")
echo "FAIL_UNDER=$FAIL_UNDER"

# 2. Substitute the placeholder.
sed -i "s/TBD-PY-FLOOR/$FAIL_UNDER/g" tools/draft_patches/ywu7_coverage_fail_under.patch

# 3. Verify the patch still applies cleanly (line numbers may
#    drift if pyproject.toml or coverage.yml were edited in the
#    meantime — refresh the hunks if so).
git apply --check tools/draft_patches/ywu7_coverage_fail_under.patch

# 4. Apply, smoke-test syntax of the changed files.
git apply tools/draft_patches/ywu7_coverage_fail_under.patch
python3 -c "import tomllib; tomllib.loads(open('pyproject.toml').read())"
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/coverage.yml'))"

# 5. Commit + delete the staged patch + close beads.
git rm tools/draft_patches/ywu7_coverage_fail_under.patch \
       tools/draft_patches/ywu7_coverage_fail_under_RUNBOOK.md
git add pyproject.toml .github/workflows/coverage.yml tools/draft_patches/
git commit -m "ci(coverage): enforce fail_under=$FAIL_UNDER (angr-ywu7)"
bd close angr-ywu7 --reason="fail_under=$FAIL_UNDER landed against baseline=$BASELINE_PCT"
bd close angr-439q --reason="baseline measured at $BASELINE_PCT% on YYYY-MM-DD"
bd remember "Coverage baseline measured YYYY-MM-DD: Python angr=$BASELINE_PCT%. CI gate set to $FAIL_UNDER (floor-2pp). See pyproject.toml and coverage.yml." --key baseline-coverage-pct-YYYY-MM-DD
```

## Why this shape

* **Patch lives in pyproject.toml**, not in coverage.yml's pytest
  args, because `[tool.coverage.report].fail_under` is read by
  *any* coverage invocation (developers running `coverage report`
  locally also get the gate). Keeping the value in one place
  avoids drift.
* **Enforcement runs in the Report job**, not the Test job,
  because each shard sees ~10% of the corpus and would fall
  under any non-trivial threshold on its own. The Report job is
  the only point in the pipeline where all shards are
  aggregated.
* **`coverage combine` over per-shard `.coverage*` data files**
  rather than parsing the per-shard `coverage*.xml`. The XML
  representation is lossy for combine purposes (line counts are
  already aggregated per file), so the data files are the
  reliable input.
* **Test job adds `--cov-report=`** (empty string) alongside the
  existing `--cov-report=xml` so pytest-cov writes both the XML
  report (consumed by Codecov uploads) AND the raw
  `.coverage.<host>.<pid>.<rand>` data file (consumed by the new
  combine step).
* **`./.coverage*` added to the artifact upload list** so the
  Report job's `download-artifact` step finds the data files
  alongside the XML.

## Future-proofing

If `coverage.yml` changes structure before this patch is
applied (e.g. new shard count, new artifact naming), the line-
numbered hunks will fail to apply. Refresh by regenerating the
patch against current HEAD; the conceptual shape (4 changes:
toml fail_under, pytest `--cov-report=`, artifact glob, new
combine step) does not change.
