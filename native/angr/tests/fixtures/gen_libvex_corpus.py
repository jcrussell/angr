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

import angr
from angr.exploration.rust_irsb_serializer import serialize_irsb


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
