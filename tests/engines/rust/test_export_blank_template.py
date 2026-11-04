"""An export SimState built from the cached blank template equals one built by ``blank_state`` (angr-mx00p.136.28).

``_snapshot_to_angr`` copies a per-manager blank template instead of calling
``blank_state(addr=pc)``, and ``_write_named_registers`` stores contiguous
runs of concrete registers in one store each instead of one ``setattr`` per
name.
"""

from __future__ import annotations

import re

import angr
from angr import claripy
from angr.exploration import rust_state_export
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000

# mov rax, 0x1122334455667788; mov rbx, rdi; add rbx, 3; jmp +0; then jmp $.
CODE = b"\x48\xb8\x88\x77\x66\x55\x44\x33\x22\x11" + b"\x48\x89\xfb" + b"\x48\x83\xc3\x03" + b"\xeb\x00\xeb\xfe"


def _project():
    return angr.load_shellcode(CODE, arch="amd64", load_address=LOAD)


def _register_view(state, names):
    # A register neither side wrote fills lazily with a fresh variable, so
    # symbolic values compare by variable name less the global counter.
    view = {}
    for name in names:
        value = getattr(state.regs, name)
        if value.concrete:
            view[name] = state.solver.eval(value)
        else:
            view[name] = frozenset(re.sub(r"_\d+(_\d+)$", r"\1", v) for v in value.variables)
    return view


def _export_both_ways(options):
    proj = _project()
    state = proj.factory.blank_state(addr=LOAD, add_options=options)
    state.regs.rdi = 0x40
    mgr = RustExplorationManager(proj, [state])
    mgr.step(n=1)
    (state_id,) = mgr._rust_mgr.get_state_ids("active")
    snapshot = mgr._rust_mgr.export_state(state_id)
    # The registers Rust exports; others an export cannot read through a
    # RustRegisterProxy.
    names = sorted({*snapshot.get_registers_named(), *snapshot.get_symbolic_register_names()})
    from_template = mgr._snapshot_to_angr(snapshot)
    mgr._export_blank_state = lambda pc: proj.factory.blank_state(addr=pc)
    from_blank = mgr._snapshot_to_angr(mgr._rust_mgr.export_state(state_id))
    return from_template, from_blank, names


def _assert_equal_exports(from_template, from_blank, names):
    assert set(from_template.plugins) == set(from_blank.plugins)
    assert from_template.options.tally() == from_blank.options.tally()
    assert type(from_template.registers) is type(from_blank.registers)
    assert _register_view(from_template, names) == _register_view(from_blank, names)
    assert from_template.addr == from_blank.addr
    assert from_template.scratch.ins_addr == from_blank.scratch.ins_addr == from_template.addr
    assert from_template.history.jumpkind == from_blank.history.jumpkind


def test_template_export_equals_blank_state_export():
    from_template, from_blank, names = _export_both_ways(set())
    _assert_equal_exports(from_template, from_blank, names)
    assert from_template.solver.eval(from_template.regs.rax) == 0x1122334455667788
    assert from_template.solver.eval(from_template.regs.rbx) == 0x43


def test_template_export_equals_blank_state_export_with_plain_registers():
    # Zero-filled registers leave Rust no symbolic register, so the export
    # keeps a plain register file and the bulk store is what fills it.
    from_template, from_blank, names = _export_both_ways({angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
    _assert_equal_exports(from_template, from_blank, names)
    assert from_template.solver.eval(from_template.regs.rbx) == 0x43


def test_template_is_not_shared_between_exports():
    proj = _project()
    mgr = RustExplorationManager(proj, [proj.factory.blank_state(addr=LOAD)])
    first = mgr._export_blank_state(LOAD)
    first.regs.rax = 0x99
    first.memory.store(0x1000, claripy.BVV(0xAB, 8))
    second = mgr._export_blank_state(LOAD + 0x10)
    assert second.addr == LOAD + 0x10
    assert second.scratch.bbl_addr == LOAD + 0x10
    assert second.regs.rax.symbolic
    assert second.memory.load(0x1000, 1).symbolic


def test_bulk_register_store_matches_per_name_setattr():
    proj = _project()
    named = {
        "rax": (0x1122334455667788, 64),
        "rcx": (0x10, 64),
        "rdx": (0xFFFFFFFFFFFFFFFF, 64),
        "rsp": (0x7FFF0000, 64),
        "rip": (LOAD, 64),
        "xmm0": ((1 << 128) - 2, 128),
        # Overlaps rcx: left for the per-name path.
        "ecx": (0x20, 32),
        # Not an arch register, and a width that is not the arch's.
        "not_a_register": (1, 64),
        "rbx": (5, 32),
    }
    bulk = proj.factory.blank_state(addr=LOAD)
    rest = rust_state_export._store_register_runs(bulk, named)
    assert {name for name, _ in rest} == {"ecx", "not_a_register", "rbx"}

    per_name = proj.factory.blank_state(addr=LOAD)
    for name, (value, bits) in named.items():
        if name not in {"ecx", "not_a_register", "rbx"}:
            setattr(per_name.regs, name, claripy.BVV(value, bits))
    for name in ("rax", "rcx", "rdx", "rsp", "rip", "xmm0"):
        assert bulk.solver.eval(getattr(bulk.regs, name)) == per_name.solver.eval(getattr(per_name.regs, name))
        assert bulk.solver.eval(getattr(bulk.regs, name)) == named[name][0]
