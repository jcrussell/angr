"""Tests for the loader-identity axis of the Rust init disk cache key.

The cached payload embeds pages and post-relocation section patches for every
``loader.all_objects`` entry, so the shared-library set must be part of the
cache key — see angr-uv7z5 and ``_loader_identity_digest``. Pure bookkeeping
over a fake loader; needs neither the Rust extension nor a project.
"""

from __future__ import annotations

import os

from angr.exploration.rust_disk_cache import RustDiskCacheManager


class _FakeObj:
    def __init__(self, binary, mapped_base=0x400000):
        self.binary = binary
        self.mapped_base = mapped_base


class _FakeLoader:
    def __init__(self, objs):
        self.all_objects = objs


def _digest(objs) -> str:
    return RustDiskCacheManager._loader_identity_digest(_FakeLoader(objs))


def _lib(tmp_path, name: str, content: bytes = b"lib") -> str:
    path = os.path.join(tmp_path, name)
    with open(path, "wb") as f:
        f.write(content)
    return path


class TestLoaderIdentityDigest:
    def test_stable_for_identical_loaders(self, tmp_path):
        main = _lib(tmp_path, "main")
        libc = _lib(tmp_path, "libc.so.6")
        objs = [_FakeObj(main), _FakeObj(libc, 0x500000)]
        assert _digest(objs) == _digest(list(objs))

    def test_insensitive_to_object_order(self, tmp_path):
        a = _FakeObj(_lib(tmp_path, "main"))
        b = _FakeObj(_lib(tmp_path, "libc.so.6"), 0x500000)
        assert _digest([a, b]) == _digest([b, a])

    def test_extra_shared_library_changes_digest(self, tmp_path):
        main = _FakeObj(_lib(tmp_path, "main"))
        libc = _FakeObj(_lib(tmp_path, "libc.so.6"), 0x500000)
        # auto_load_libs off vs on: same main binary, different object set.
        assert _digest([main]) != _digest([main, libc])

    def test_library_content_change_changes_digest(self, tmp_path):
        main = _FakeObj(_lib(tmp_path, "main"))
        libc_path = _lib(tmp_path, "libc.so.6", b"old-libc")
        before = _digest([main, _FakeObj(libc_path, 0x500000)])
        with open(libc_path, "wb") as f:
            f.write(b"upgraded-libc-with-different-size")
        after = _digest([main, _FakeObj(libc_path, 0x500000)])
        assert before != after

    def test_base_address_change_changes_digest(self, tmp_path):
        main = _FakeObj(_lib(tmp_path, "main"))
        libc_path = _lib(tmp_path, "libc.so.6")
        assert _digest([main, _FakeObj(libc_path, 0x500000)]) != _digest([main, _FakeObj(libc_path, 0x700000)])

    def test_synthetic_object_without_file_is_tolerated(self):
        # cle##externs and friends have no readable backing file; the digest
        # must still be computable and must distinguish presence from absence.
        externs = _FakeObj(None, 0x600000)
        assert _digest([]) != _digest([externs])
        assert _digest([externs]) == _digest([_FakeObj(None, 0x600000)])

    def test_missing_all_objects_attribute(self):
        class _Bare:
            pass

        assert RustDiskCacheManager._loader_identity_digest(_Bare())


class TestDiskCacheKeyLoaderAxis:
    def test_key_differs_by_loader_digest(self, tmp_path):
        binary = _lib(tmp_path, "main", b"\x7fELF" + b"\0" * 64)
        k1 = RustDiskCacheManager._disk_cache_key(binary, "AMD64", "aaaa")
        k2 = RustDiskCacheManager._disk_cache_key(binary, "AMD64", "bbbb")
        assert k1 and k2 and k1 != k2

    def test_key_memo_is_per_loader_digest(self, tmp_path):
        binary = _lib(tmp_path, "main", b"\x7fELF" + b"\0" * 64)
        RustDiskCacheManager._disk_cache_key(binary, "AMD64", "cccc")
        assert (binary, "AMD64", "cccc") in RustDiskCacheManager._disk_key_cache
        assert (binary, "AMD64", "dddd") not in RustDiskCacheManager._disk_key_cache

    def test_unreadable_binary_still_disables_cache(self, tmp_path):
        assert RustDiskCacheManager._disk_cache_key(os.path.join(tmp_path, "nope"), "AMD64", "eeee") == ""
