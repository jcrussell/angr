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

RUST_TESTS := tests/engines/test_rust_exploration.py

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

# --- Benchmarks ----------------------------------------------------------

.PHONY: bench-regression
bench-regression:  ## Fast-tier benchmark regression check (7 examples, subprocess-bounded).
	$(PY) $(REPO_ROOT)/tests/benchmarks/run_regression.py

.PHONY: bench-single
bench-single:  ## Run one benchmark — EXAMPLE=name [ARGS="--both"] (defaults to fauxware --both).
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
