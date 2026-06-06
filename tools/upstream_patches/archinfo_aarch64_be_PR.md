# [archinfo] Add AArch64 big-endian support: instruction_endness + aarch64eb/aarch64be/arm64eb/arm64be aliases

## Summary

`ArchAArch64` partially handles big-endian construction
(`pcode_id="AARCH64:BE:64:v8A"`, `ida_processor="armb"`) but the
runtime endness of executed instructions stays LE, and there is no
`register_arch` alias that maps the canonical Linux/qemu BE arch
names (`aarch64_be`, `aarch64eb`, `arm64_be`, `arm64eb`) to
`Endness.BE`. The result is that
`arch_from_id("aarch64eb")` silently returns an
`ArchAArch64(Endness.LE)` instance, which then mis-lifts BE
binaries.

This PR closes that gap by:

1. **Threading `instruction_endness=Endness.BE` to
   `Arch.__init__` from `ArchAArch64.__init__`** when constructed
   with `endness=Endness.BE`. The base class already flips
   `memory_endness` / `register_endness` / capstone+keystone mode
   on `Endness.BE`, but only honors `instruction_endness` when the
   subclass explicitly passes it (see `arch.py` lines around
   "if instruction_endness is not None"). `ArchARM.__init__`
   already does the same — this PR mirrors that precedent.

2. **Adding a BE-flavored `register_arch` alias** for
   `r".*aarch64eb|.*aarch64be|.*arm64eb|.*arm64be"`, registered
   BEFORE the existing `r".*arm64.*|.*aarch64*"` `Endness.ANY`
   pattern. `arch_from_id` iterates registration order and breaks
   on first match; putting the BE pattern first makes the canonical
   Linux/qemu BE name spellings route to `ArchAArch64(Endness.BE)`,
   while `aarch64` / `arm64` continue to fall through to the
   ANY-endness path.

Mirrors the ARM precedent:

```python
register_arch([r".*armeb|.*armbe"], 32, Endness.BE, ArchARM)
register_arch([r".*armel|arm.*"], 32, Endness.LE, ArchARMEL)
register_arch([r".*arm.*|.*thumb.*"], 32, Endness.ANY, ArchARM)
```

## Motivation

We hit this bug downstream in the angr Rust-symex engine while
adding an ARM64 BE integration test. With unpatched archinfo,
`arch_from_id("aarch64eb")` returns an `Endness.LE` Arch, which
means a BE ELF cannot be constructed through cle without a
downstream monkey-patch — exactly the workaround we wanted to
avoid. The same pattern was already established for 32-bit ARM,
so the AArch64 gap looks like an oversight rather than an
intentional design choice.

## Test plan

Manual repro (drop in `tests/test_aarch64.py` or run interactively):

```python
import archinfo
from archinfo.arch import Endness

# Before patch: aarch64eb returns LE (BUG).
# After patch: aarch64eb returns BE.
be = archinfo.arch_from_id("aarch64eb")
assert be.memory_endness == Endness.BE, be.memory_endness
assert be.register_endness == Endness.BE, be.register_endness
assert be.instruction_endness == Endness.BE, be.instruction_endness
assert be.pcode_id == "AARCH64:BE:64:v8A", be.pcode_id

# LE still works.
le = archinfo.arch_from_id("aarch64")
assert le.memory_endness == Endness.LE
assert le.instruction_endness == Endness.LE
assert le.pcode_id == "AARCH64:LE:64:v8A"

# All four BE alias spellings route to BE.
for name in ("aarch64eb", "aarch64be", "arm64eb", "arm64be"):
    a = archinfo.arch_from_id(name)
    assert a.memory_endness == Endness.BE, (name, a.memory_endness)
```

Smoke-tested locally against archinfo 9.2.221 with the patch
applied — all assertions pass; pre-patch reproduces the LE-when-BE-
expected behavior.

## Backwards compatibility

- The `Endness.ANY` registration is untouched and still wins for
  plain `aarch64` / `arm64`. The only routing change is for the
  four explicit BE name spellings — none of which previously
  produced a useful result.
- LE-constructed `ArchAArch64` instances are unchanged
  (`instruction_endness` was already LE at class level and still
  is).
- No public API surface added or removed.

## References

- ARM BE precedent: `archinfo/arch_arm.py::ArchARM.__init__`
  threads `instruction_endness=Endness.BE` to super; same file
  registers `.*armeb|.*armbe → Endness.BE`.
- Base `Arch.__init__` endness handling: `archinfo/arch.py` —
  flips `memory_endness` and `register_endness` on
  `Endness.BE`, but only honors a subclass-supplied
  `instruction_endness=` kwarg.
