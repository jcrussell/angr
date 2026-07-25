//! Corpus IRSB parity harness for libVEX-FFI Stage-1 (angr-3s5js.4).
//!
//! This is the REAL gate for the whole libVEX-FFI Stage-1 effort. It replays a
//! corpus of real AMD64 basic blocks (dumped from a binary by
//! `tests/fixtures/gen_libvex_corpus.py`) through BOTH lift paths and asserts
//! structural IRSB equality:
//!
//!   * **native** — `NativeLibVEXLifter::lift(bytes)` (direct libVEX FFI), and
//!   * **pyvex** — `pyvex_bridge::deserialize_irsb(json)`, where `json` was
//!     produced by the same `serialize_irsb` the Rust engine's `_cb_lift_block`
//!     uses in production.
//!
//! GATE: 100% structural parity on the corpus block set. A single divergent
//! block fails the test and prints a per-block `{:#?}` diff so the offending
//! marshalling/lift-config mismatch is immediately visible.
//!
//! Structural equality is compared via the derived `Debug` representation:
//! `vex::ir` IRSB/IRStmt/IRExpr do not derive `PartialEq`, but both paths target
//! the exact same `vex::ir` shape, so identical trees render to identical
//! `{:#?}` strings. This also catches field-order and enum-variant divergence
//! that a hand-rolled structural walk might miss.

use serde::Deserialize;

use super::NativeLibVEXLifter;
use crate::vex::pyvex_bridge::deserialize_irsb;
use crate::vex::{VEXLifter, VexArch};

/// One corpus entry from `tests/fixtures/libvex_corpus.json`.
#[derive(Debug, Deserialize)]
struct CorpusBlock {
    addr: u64,
    arch: String,
    /// Hex-encoded decoded block bytes.
    bytes: String,
    /// `serialize_irsb` output (the production pyvex-serialized path).
    pyvex_json: String,
}

const CORPUS_JSON: &str = include_str!("../../tests/fixtures/libvex_corpus.json");

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("valid hex byte"))
        .collect()
}

/// Map a corpus `arch` string to the `VexArch` the native lifter replays it
/// through. Kept in lockstep with the strings `gen_libvex_corpus.py` emits.
fn arch_from_str(arch: &str) -> VexArch {
    match arch {
        "AMD64" => VexArch::AMD64,
        "ARM" => VexArch::ARM,
        "ARM64" => VexArch::ARM64,
        "MIPS32" => VexArch::MIPS32,
        "MIPS64" => VexArch::MIPS64,
        other => panic!("corpus block has unsupported arch {other:?}"),
    }
}

#[test]
fn test_corpus_native_vs_pyvex_structural_parity() {
    let corpus: Vec<CorpusBlock> =
        serde_json::from_str(CORPUS_JSON).expect("corpus fixture parses");
    assert!(!corpus.is_empty(), "corpus fixture is non-empty");

    let lifter = NativeLibVEXLifter::new();
    let mut mismatches: Vec<String> = Vec::new();

    for block in &corpus {
        let arch = arch_from_str(&block.arch);
        let bytes = hex_to_bytes(&block.bytes);

        let native = lifter
            .lift(&bytes, block.addr, arch)
            .unwrap_or_else(|e| panic!("native lift @ 0x{:x} failed: {e:?}", block.addr));
        let pyvex = deserialize_irsb(&block.pyvex_json)
            .unwrap_or_else(|e| panic!("pyvex deserialize @ 0x{:x} failed: {e:?}", block.addr));

        let native_dbg = format!("{native:#?}");
        let pyvex_dbg = format!("{pyvex:#?}");
        if native_dbg != pyvex_dbg {
            mismatches.push(format!(
                "block @ 0x{:x} ({} bytes) DIVERGED:\n--- native ---\n{native_dbg}\n--- pyvex ---\n{pyvex_dbg}\n",
                block.addr,
                bytes.len(),
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "corpus parity gate FAILED: {}/{} blocks diverged\n\n{}",
        mismatches.len(),
        corpus.len(),
        mismatches.join("\n"),
    );
}
