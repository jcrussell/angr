# pylint: disable=missing-class-docstring,no-self-use
"""Stale-stdin-packet dropping is by provenance, not byte value (angr-p8jyz).

``_drop_stale_rust_stdin_packets`` used to identify "packets Rust injected" by a
manager-wide ``set[bytes]``, so any legitimate Python-side concrete packet whose
bytes collided with a previously-injected eval was silently stripped from
``posix.dumps(0)``. Collisions are realistic: an unconstrained materialization
evals its stdin to all-zeros, so ``b'\\x00'*N`` enters the set almost
immediately and any zero-filled concrete preseed is then dropped.

The fix tags each injected BVV with a ``_RustInjectedStdin`` annotation and drops
purely by that tag. These tests pin the annotation's survival across a copy and
the by-provenance (not by-value) drop semantics.
"""

from __future__ import annotations

import types

import claripy

from angr.exploration.rust_callback_dispatch import RustCallbackDispatchMixin, _RustInjectedStdin


def _tagged(data: bytes):
    return (claripy.BVV(data).annotate(_RustInjectedStdin()), claripy.BVV(len(data), 64))


def _plain(data: bytes):
    return (claripy.BVV(data), claripy.BVV(len(data), 64))


class TestStdinPacketProvenance:
    def test_annotation_survives_ast_copy(self):
        """The tag rides along the AST identity a SimState ``.copy()`` preserves."""
        bv = claripy.BVV(b"\x00" * 32).annotate(_RustInjectedStdin())
        # A no-op AST op returns a fresh AST; a relocatable annotation carries.
        derived = bv + claripy.BVV(0, len(bv))
        assert any(isinstance(a, _RustInjectedStdin) for a in derived.annotations)

    def test_collision_keeps_untagged_python_packet(self):
        """A Python preseed colliding by value with a prior eval is NOT dropped."""
        mixin = RustCallbackDispatchMixin.__new__(RustCallbackDispatchMixin)
        # Same all-zeros bytes: a concrete preseed (untagged) followed by a
        # Rust-injected eval (tagged). By value they are identical.
        preseed = _plain(b"\x00" * 32)
        injected = _tagged(b"\x00" * 32)
        stream = types.SimpleNamespace(content=[preseed, injected])

        mixin._drop_stale_rust_stdin_packets(stream)

        assert len(stream.content) == 1
        kept = stream.content[0]
        assert not RustCallbackDispatchMixin._is_rust_injected(kept)
        assert kept is preseed

    def test_only_tagged_packets_dropped(self):
        """Multiple tagged packets go; every untagged one stays, order kept."""
        mixin = RustCallbackDispatchMixin.__new__(RustCallbackDispatchMixin)
        a = _plain(b"AAAA")
        b = _tagged(b"BBBB")
        c = _plain(b"CCCC")
        d = _tagged(b"DDDD")
        stream = types.SimpleNamespace(content=[a, b, c, d])

        mixin._drop_stale_rust_stdin_packets(stream)

        assert stream.content == [a, c]

    def test_empty_content_is_noop(self):
        mixin = RustCallbackDispatchMixin.__new__(RustCallbackDispatchMixin)
        stream = types.SimpleNamespace(content=[])
        mixin._drop_stale_rust_stdin_packets(stream)
        assert stream.content == []
