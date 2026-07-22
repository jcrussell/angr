"""Tests for the bounded-cache policy on the Rust init disk cache.

Covers ``_disk_cache_budgets`` (env overrides) and ``_prune_disk_cache``
(LRU-by-mtime eviction past the byte/count budget) — see angr-728cf. These
exercise pure filesystem bookkeeping, so they need neither the Rust extension
nor a project.
"""

from __future__ import annotations

import os

import pytest

from angr.exploration.rust_disk_cache import (
    _DISK_CACHE_MAX_BYTES,
    _DISK_CACHE_MAX_FILES,
    RustDiskCacheManager,
    _disk_cache_budgets,
)


def _write(cache_dir, name: str, size: int, mtime: float) -> str:
    path = os.path.join(cache_dir, name)
    with open(path, "wb") as f:
        f.write(b"\0" * size)
    os.utime(path, (mtime, mtime))
    return path


def _names(cache_dir) -> set[str]:
    return set(os.listdir(cache_dir))


class TestDiskCacheBudgets:
    def test_defaults_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("ANGR_RUST_INIT_CACHE_MAX_BYTES", raising=False)
        monkeypatch.delenv("ANGR_RUST_INIT_CACHE_MAX_FILES", raising=False)
        assert _disk_cache_budgets() == (_DISK_CACHE_MAX_BYTES, _DISK_CACHE_MAX_FILES)

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_BYTES", "4096")
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_FILES", "3")
        assert _disk_cache_budgets() == (4096, 3)

    @pytest.mark.parametrize("bad", ["", "not-a-number", "-1"])
    def test_bad_values_fall_back_to_defaults(self, monkeypatch, bad):
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_FILES", bad)
        assert _disk_cache_budgets()[1] == _DISK_CACHE_MAX_FILES


class TestPruneDiskCache:
    def test_evicts_oldest_past_count_budget(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_FILES", "2")
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_BYTES", "0")
        d = str(tmp_path)
        for i, name in enumerate(["old.pkl", "mid.pkl", "new.pkl"]):
            _write(d, name, 10, mtime=1000.0 + i)

        assert RustDiskCacheManager._prune_disk_cache(d) == 1
        assert _names(d) == {"mid.pkl", "new.pkl"}

    def test_evicts_past_byte_budget(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_BYTES", "250")
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_FILES", "0")
        d = str(tmp_path)
        for i, name in enumerate(["a.pkl", "b.pkl", "c.pkl"]):
            _write(d, name, 100, mtime=1000.0 + i)

        assert RustDiskCacheManager._prune_disk_cache(d) == 1
        assert _names(d) == {"b.pkl", "c.pkl"}

    def test_keep_key_survives_even_when_oversized(self, tmp_path, monkeypatch):
        """The just-written entry is never the one evicted, even if it alone
        blows the byte budget — degrade to a one-entry cache, not an empty one."""
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_BYTES", "50")
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_FILES", "0")
        d = str(tmp_path)
        # `fresh` is the OLDEST by mtime, so without keep_key it would go first.
        _write(d, "fresh.pkl", 500, mtime=1000.0)
        _write(d, "other.pkl", 10, mtime=2000.0)

        assert RustDiskCacheManager._prune_disk_cache(d, keep_key="fresh") == 1
        assert _names(d) == {"fresh.pkl"}

    def test_zero_budgets_disable_pruning(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_BYTES", "0")
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_FILES", "0")
        d = str(tmp_path)
        for i in range(5):
            _write(d, f"f{i}.pkl", 1000, mtime=1000.0 + i)

        assert RustDiskCacheManager._prune_disk_cache(d) == 0
        assert len(_names(d)) == 5

    def test_ignores_non_pkl_and_missing_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_FILES", "1")
        monkeypatch.setenv("ANGR_RUST_INIT_CACHE_MAX_BYTES", "0")
        d = str(tmp_path)
        _write(d, "keep.pkl", 10, mtime=2000.0)
        _write(d, "notes.txt", 10, mtime=1000.0)

        assert RustDiskCacheManager._prune_disk_cache(d) == 0
        assert _names(d) == {"keep.pkl", "notes.txt"}
        # A nonexistent directory is a no-op, not an exception.
        assert RustDiskCacheManager._prune_disk_cache(os.path.join(d, "nope")) == 0
