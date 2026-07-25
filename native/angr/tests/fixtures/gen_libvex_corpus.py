#!/usr/bin/env python3
"""Generate the libVEX-FFI corpus parity fixture (angr-3s5js.4).

Walks a real AMD64 binary with a lightweight recursive-descent worklist
(``project.factory.block(addr)`` + ``block.vex.constant_jump_targets`` — no
CFG, low memory) and dumps every *unique* decoded basic block as a triple:

    {"addr": int, "arch": "AMD64", "bytes": "<hex>", "pyvex_json": "<json>"}

The ``pyvex_json`` is produced by the SAME ``serialize_irsb`` the Rust engine's
``_cb_lift_block`` uses in production, and the block is lifted with the SAME
default options (``project.factory.block(addr).vex``). The cargo-side harness in
``src/vex/libvex_corpus_tests.rs`` replays each block through both the native
libVEX lifter and ``pyvex_bridge::deserialize_irsb`` and asserts structural IRSB
equality — the 100%-parity gate for libVEX-FFI Stage-1.

Run (subprocess, memory-capped per the ralph MEMORY SAFETY rules)::

    systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- \
        env PYTHONPATH=$PWD .venv/bin/python \
        native/angr/tests/fixtures/gen_libvex_corpus.py \
        --binary /path/to/fauxware --out native/angr/tests/fixtures/libvex_corpus.json
"""

from __future__ import annotations

import argparse
import json
import sys

import archinfo
import pyvex

import angr
from angr.exploration.rust_irsb_serializer import serialize_irsb

# Hand-assembled blocks covering IR shapes a recursive-descent walk of an
# ordinary binary never reaches. Real corpora are SSE-const-free, so without
# these the V128/V256 restricted-vector const marshalling is untested.
# Addresses sit well above any real image so they can never collide.
SYNTHETIC_BLOCKS: list[tuple[int, str, str]] = [
    # pcmpeqd xmm0, xmm0 ; ret  -> Ico_V128 all-ones (pattern 0xffff)
    (0x7F00_0000, "660f76c0c3", "V128 all-ones const"),
    # pxor xmm0, xmm0 ; ret     -> Ico_V128 zero
    (0x7F00_0010, "660fefc0c3", "V128 zero const"),
    # vpxor ymm0, ymm0, ymm0 ; ret -> Ico_V256 zero
    (0x7F00_0020, "c5fdefc0c3", "V256 zero const"),
    # vzeroall ; ret            -> a run of Ico_V128 zero consts
    (0x7F00_0030, "c5fc77c3", "V128 const run"),
    # --- SIMD vector-op coverage (angr-qwyti.6) -------------------------------
    # The angr-ph300 audit flagged these opcode families as "zero direct test":
    # the Interleave family (14 parse arms), VMulLo, vec_int_lane VSub/VCmpGT,
    # and SetV128lo32/64. A recursive-descent walk of an ordinary binary rarely
    # reaches SSE integer vector ops, so pin them as synthetic blocks. Each ends
    # in `ret` (c3) so the block is self-terminating.
    # punpcklbw xmm0, xmm1 ; ret -> Iop_InterleaveLO8x16
    (0x7F00_0040, "660f60c1c3", "InterleaveLO8x16 (punpcklbw)"),
    # punpckhbw xmm0, xmm1 ; ret -> Iop_InterleaveHI8x16
    (0x7F00_0050, "660f68c1c3", "InterleaveHI8x16 (punpckhbw)"),
    # punpcklwd xmm2, xmm3 ; ret -> Iop_InterleaveLO16x8
    (0x7F00_0060, "660f61d3c3", "InterleaveLO16x8 (punpcklwd)"),
    # pmulld xmm0, xmm1 ; ret    -> Iop_Mul32x4
    (0x7F00_0070, "660f3840c1c3", "Mul32x4 (pmulld)"),
    # psubd xmm4, xmm5 ; ret     -> Iop_Sub32x4
    (0x7F00_0080, "660ffae5c3", "Sub32x4 (psubd)"),
    # pcmpgtd xmm0, xmm1 ; ret   -> Iop_CmpGT32Sx4
    (0x7F00_0090, "660f66c1c3", "CmpGT32Sx4 (pcmpgtd)"),
    # movd xmm0, eax ; ret       -> Iop_SetV128lo32
    (0x7F00_00A0, "660f6ec0c3", "SetV128lo32 (movd xmm)"),
    # movq xmm0, rax ; ret       -> Iop_SetV128lo64
    (0x7F00_00B0, "66480f6ec0c3", "SetV128lo64 (movq xmm)"),
]


def synthetic_blocks() -> list[dict]:
    """Lift the hand-assembled SYNTHETIC_BLOCKS through the production path."""
    arch = archinfo.ArchAMD64()
    out: list[dict] = []
    for addr, hex_bytes, label in SYNTHETIC_BLOCKS:
        raw = bytes.fromhex(hex_bytes)
        irsb = pyvex.lift(raw, addr, arch)
        if irsb.size == 0:
            print(f"ERROR: synthetic block {label} @ 0x{addr:x} did not lift", file=sys.stderr)
            continue
        out.append(
            {
                "addr": addr,
                "arch": "AMD64",
                "bytes": raw[: irsb.size].hex(),
                "pyvex_json": serialize_irsb(irsb),
            }
        )
    return out


def collect_blocks(binary: str, max_blocks: int) -> list[dict]:
    proj = angr.Project(binary, auto_load_libs=False)
    main = proj.loader.main_object

    def in_binary(a: int) -> bool:
        return main.min_addr <= a < main.max_addr

    seen: dict[int, dict] = {}
    worklist: list[int] = [proj.entry]
    # Seed with symbol addresses too, for broader block coverage.
    for sym in main.symbols:
        if sym.is_function and sym.rebased_addr and in_binary(sym.rebased_addr):
            worklist.append(sym.rebased_addr)

    while worklist and len(seen) < max_blocks:
        addr = worklist.pop()
        if addr in seen or not in_binary(addr):
            continue
        try:
            block = proj.factory.block(addr)
            irsb = block.vex
        except Exception:
            continue
        if block.size == 0 or irsb.size == 0:
            continue
        seen[addr] = {
            "addr": addr,
            "arch": "AMD64",
            "bytes": block.bytes.hex(),
            "pyvex_json": serialize_irsb(irsb),
        }
        # Recursive descent: enqueue constant successors.
        try:
            for tgt in irsb.constant_jump_targets:
                if tgt not in seen and in_binary(tgt):
                    worklist.append(tgt)
        except Exception:
            pass

    return sorted(seen.values(), key=lambda e: e["addr"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, action="append", dest="binaries")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-blocks", type=int, default=200)
    args = ap.parse_args()

    # Merge blocks across binaries, deduped by (addr, bytes): the same address
    # in two binaries can hold different bytes, and RIP-relative constants make
    # the IRSB addr-dependent, so both the addr and the bytes are part of the key.
    merged: dict[tuple[int, str], dict] = {}
    for binary in args.binaries:
        for entry in collect_blocks(binary, args.max_blocks):
            merged[(entry["addr"], entry["bytes"])] = entry
    for entry in synthetic_blocks():
        merged[(entry["addr"], entry["bytes"])] = entry

    blocks = sorted(merged.values(), key=lambda e: (e["addr"], e["bytes"]))
    if not blocks:
        print("ERROR: no blocks collected", file=sys.stderr)
        return 1

    with open(args.out, "w") as f:
        json.dump(blocks, f, indent=1)
        f.write("\n")
    print(f"OK wrote {len(blocks)} unique blocks -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
