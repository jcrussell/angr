# Makefile — canonical developer workflows for the rust-symex branch.
#
# Most targets shell out to the scripts under tools/ and tests/benchmarks/
# so the Makefile is a thin, discoverable index of "how do I do X here?"
# rather than a separate build system. Discover targets with `make help`.
#
# Conventions:
#   - PY points at the venv interpreter so `make` works without an active
#     `source .venv/bin/activate` (handy from IDE tasks).
#   - Cargo is invoked through ~/.cargo/bin/cargo if PATH does not already
#     carry it (matches the build-rust documentation in CLAUDE.md).
#   - Long-running targets (profile-bench, test-full, bench-regression) are
#     not wired into `make` (no PHONY 'all') so `make` with no args prints
#     help instead of churning the box.

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
VENV      := $(REPO_ROOT)/.venv
PY        := $(VENV)/bin/python
PIP       := $(PY) -m pip
PYTEST    := $(PY) -m pytest

CARGO     ?= cargo
MANIFEST  := $(REPO_ROOT)/native/angr/Cargo.toml

RUST_TESTS := tests/engines/rust/

# Curated subset of vanilla angr tests that exercise the Python engine's
# hot paths (loading, VEX lifter, hooks, sim-procedures, solver, posix,
# Callable, SimState merge/pickle/global-condition). Intent: catch
# regressions in vanilla Python angr caused by rust-symex changes BEFORE
# the full nightly test suite runs.
# Sized to finish under ~5 minutes on the GitHub Actions ubuntu-latest
# runner — see `make test-python-baseline` and the `python_baseline` CI
# job in .github/workflows/ci.yml. Requires angr/binaries cloned at
# $(REPO_ROOT)/../binaries (the standard layout — same as
# tests/common.py expects). Add a file here when it covers a Python-
# engine path that isn't already exercised; keep the total under the
# 5-minute soft budget.
PYTHON_BASELINE_TESTS := \
    tests/test_load_shellcode.py \
    tests/engines/test_actions.py \
    tests/engines/test_hook.py \
    tests/engines/vex/test_lifter.py \
    tests/state_plugins/solver/test_simsolver.py \
    tests/state_plugins/solver/test_symbolic.py \
    tests/state_plugins/posix/test_files.py \
    tests/procedures/test_sim_procedure.py \
    tests/factory/test_callable.py \
    tests/sim/test_state.py

export PATH := $(HOME)/.cargo/bin:$(PATH)

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help.
	@awk 'BEGIN { FS = ":.*##"; printf "Available targets:\n" } \
	      /^[a-zA-Z0-9_-]+:.*##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' \
	      $(MAKEFILE_LIST)

# --- Build ---------------------------------------------------------------

.PHONY: rebuild
rebuild:  ## Incremental: rebuild the Rust .so via pip editable install.
	$(PIP) install -e . --no-build-isolation --no-deps

.PHONY: rebuild-clean
rebuild-clean:  ## Full clean rebuild (wipes .so, build/, cargo cache).
	$(REPO_ROOT)/tools/rebuild-rust.sh

.PHONY: rebuild-cargo
rebuild-cargo:  ## Rebuild bypassing pip (cargo build + copy) — broken-venv fallback.
	$(REPO_ROOT)/tools/rebuild-rust.sh --cargo-only

.PHONY: rebuild-fast
rebuild-fast:  ## Inner-loop rebuild via [profile.release-fast] (~3s warm vs ~36s). NOT for bench gates.
	$(REPO_ROOT)/tools/rebuild-rust.sh --fast

.PHONY: check
check:  ## cargo check (release) — fast type/borrow check, no link.
	$(CARGO) check --manifest-path $(MANIFEST) --release

.PHONY: clean
clean:  ## Remove built .so files and setuptools build/ dir.
	rm -fv $(REPO_ROOT)/angr/rustylib*.so
	rm -rf $(REPO_ROOT)/build

# --- Tests ---------------------------------------------------------------

.PHONY: test
test:  ## Run the Rust-engine pytest suite (canonical fast tier, ~1-2 min).
	$(PYTEST) $(RUST_TESTS) --tb=short -q

.PHONY: test-quick
test-quick: test  ## Alias for `test`.

.PHONY: test-verbose
test-verbose:  ## Same as test, but verbose.
	$(PYTEST) $(RUST_TESTS) -v --tb=short

.PHONY: test-full
test-full:  ## Run the full angr test suite (long; use sparingly).
	$(PYTEST) tests/ --tb=short -q

.PHONY: test-python-baseline
test-python-baseline:  ## Run a fast (~5min) vanilla angr Python-engine regression subset (needs angr/binaries).
	@if [ ! -d "$(REPO_ROOT)/../binaries" ] && [ "$$CI" != "true" ]; then \
	    echo "ERROR: test-python-baseline needs angr/binaries cloned at $(REPO_ROOT)/../binaries"; \
	    echo "       git clone https://github.com/angr/binaries.git $(REPO_ROOT)/../binaries"; \
	    exit 1; \
	fi
	$(PYTEST) $(PYTHON_BASELINE_TESTS) --tb=short -q

# --- Benchmarks ----------------------------------------------------------

.PHONY: bench-regression
bench-regression:  ## Fast-tier benchmark regression check (7 examples, subprocess-bounded).
	$(PY) $(REPO_ROOT)/tests/benchmarks/run_regression.py

.PHONY: bench-single
bench-single:  ## Run one benchmark — EXAMPLE=name [ARGS="--both"] (defaults to fauxware --both). Use ARGS="--dump-counters" for the full counter table; ARGS="--counters-json" for machine-readable output.
	$(PY) $(REPO_ROOT)/tests/benchmarks/run_single.py $(or $(EXAMPLE),fauxware) $(or $(ARGS),--both)

.PHONY: profile-bench
profile-bench:  ## Profile criterion bench [FILTER=group SECS=10 TOOL=auto].
	$(REPO_ROOT)/tests/benchmarks/profile_rust_bench.sh \
	    $(if $(FILTER),--filter $(FILTER)) \
	    $(if $(SECS),--secs $(SECS)) \
	    $(if $(TOOL),--tool $(TOOL))

# --- Lint / format -------------------------------------------------------
# Python lint/format runs through pre-commit (.pre-commit-config.yaml is the
# source of truth for ruff + pyupgrade + assorted file-level checks). Rust
# uses cargo fmt directly. pre-commit must be installed on PATH; install
# via `pipx install pre-commit` (or your distro's package).

.PHONY: lint
lint:  ## Run pre-commit on all files (ruff, pyupgrade, format checks, etc.).
	pre-commit run --all-files

.PHONY: lint-changed
lint-changed:  ## Run pre-commit on files changed vs origin/master.
	pre-commit run --from-ref master --to-ref HEAD

.PHONY: fmt
fmt:  ## Apply pre-commit autofixes + cargo fmt across Python + Rust.
	-pre-commit run --all-files
	$(CARGO) fmt --manifest-path $(MANIFEST)

.PHONY: fmt-check
fmt-check:  ## Verify cargo fmt would not change anything (CI gate). For Python use `make lint`.
	$(CARGO) fmt --manifest-path $(MANIFEST) -- --check
