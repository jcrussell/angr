//! Property-based tests for concrete `SymbolicMemory` read/write round-trips
//! that straddle a page boundary (angr-qwyti.13, item 2 of angr-qwyti.5).
//!
//! Two adjacent pages are mapped (`0x1000..0x3000`) and a concrete value is
//! stored at an address whose byte range crosses the `0x2000` boundary. The
//! store/load split-and-merge across the page seam is the code under test; the
//! property checks the loaded bytes against an independent plain-`u64`
//! reference computed from the source value and endianness — mirroring the
//! reference-computation style of `value_ops_property_tests.rs`.
//!
//! `quickcheck` covers random (value, width, straddle-offset, endianness)
//! tuples; deterministic sweeps pin the exact boundary offsets so a regression
//! reproduces without relying on the random seed.

use super::super::*;
use quickcheck_macros::quickcheck;

/// Second-page boundary the stored range must cross.
const BOUNDARY: u64 = 0x2000;

/// Byte-widths exercised (2/4/8 bytes); all fit in `as_u64`.
const WIDTHS: [u32; 3] = [2, 4, 8];

/// Low-`n_bytes*8` mask as a `u64` (`n_bytes` in `1..=8`).
fn mask_bytes(n_bytes: u32) -> u64 {
    if n_bytes >= 8 {
        u64::MAX
    } else {
        (1u64 << (n_bytes * 8)) - 1
    }
}

/// Independent reference for the byte that little-/big-endian layout places at
/// `addr + i` when a `n_bytes`-wide value `v` is stored at the base address.
fn ref_byte(v: u64, n_bytes: u32, i: u32, big: bool) -> u64 {
    let src = if big { n_bytes - 1 - i } else { i };
    (v >> (8 * src)) & 0xff
}

/// Fresh two-page memory (`0x1000..0x3000`, both RWX).
fn two_page_mem(big: bool) -> SymbolicMemory {
    let mut mem = SymbolicMemory::new(if big { Endness::Big } else { Endness::Little });
    mem.map(0x1000, 0x2000, Permission::RWX);
    mem
}

/// Store `v` (`n_bytes` wide) straddling `BOUNDARY` and assert every readback
/// path matches the independent reference: full-width reload, per-byte loads,
/// and (for little-endian) an arbitrary sub-range load.
fn check_straddle(v: u64, n_bytes: u32, k: u64, big: bool) {
    let ctx = SymContext::new_mock();
    let v = v & mask_bytes(n_bytes);
    // addr = BOUNDARY - k, with 1 <= k <= n_bytes-1, so the range
    // [addr, addr + n_bytes) strictly crosses BOUNDARY.
    let addr = BOUNDARY - k;

    let mut mem = two_page_mem(big);
    mem.store_concrete(addr, RustBV::concrete(v as u128, n_bytes * 8))
        .expect("straddling store must map to two adjacent mapped pages");

    // Full-width reload round-trips the value exactly.
    let full = mem
        .load_concrete(addr, n_bytes, &ctx)
        .expect("full-width reload after straddling store");
    assert_eq!(
        full.as_u64(),
        Some(v),
        "full reload mismatch: v={v:#x} n_bytes={n_bytes} k={k} big={big}"
    );

    // Per-byte layout matches the endianness reference across the seam.
    for i in 0..n_bytes {
        let b = mem
            .load_concrete(addr + i as u64, 1, &ctx)
            .expect("per-byte reload");
        assert_eq!(
            b.as_u64(),
            Some(ref_byte(v, n_bytes, i, big)),
            "byte {i} mismatch (spans boundary at offset {}): v={v:#x} big={big}",
            BOUNDARY as i64 - addr as i64
        );
    }

    // Little-endian sub-range loads have a clean plain-integer reference:
    // bytes [j, j+m) of the value read back as (v >> 8j) & mask(m).
    if !big {
        for j in 0..n_bytes {
            for m in 1..=(n_bytes - j) {
                let sub = mem
                    .load_concrete(addr + j as u64, m, &ctx)
                    .expect("sub-range reload");
                let expected = (v >> (8 * j)) & mask_bytes(m);
                assert_eq!(
                    sub.as_u64(),
                    Some(expected),
                    "LE sub-range [{j},{}) mismatch: v={v:#x} n_bytes={n_bytes} k={k}",
                    j + m
                );
            }
        }
    }
}

#[quickcheck]
fn prop_straddle_roundtrip_le(v: u64, width_sel: u8, off_sel: u8) -> bool {
    let n_bytes = WIDTHS[(width_sel as usize) % WIDTHS.len()];
    let k = (off_sel as u64 % (n_bytes as u64 - 1)) + 1; // 1..=n_bytes-1
    check_straddle(v, n_bytes, k, false);
    true
}

#[quickcheck]
fn prop_straddle_roundtrip_be(v: u64, width_sel: u8, off_sel: u8) -> bool {
    let n_bytes = WIDTHS[(width_sel as usize) % WIDTHS.len()];
    let k = (off_sel as u64 % (n_bytes as u64 - 1)) + 1;
    check_straddle(v, n_bytes, k, true);
    true
}

/// Deterministic sweep: every legal straddle offset for every width and both
/// endiannesses, with a value whose bytes are all distinct so a swapped or
/// dropped byte cannot alias to the correct one.
#[test]
fn test_straddle_deterministic_all_offsets() {
    // Distinct, non-zero bytes 0x11,0x22,...,0x88 (LE: low byte 0x11).
    let v: u64 = 0x8877_6655_4433_2211;
    for &n_bytes in &WIDTHS {
        for k in 1..n_bytes as u64 {
            check_straddle(v, n_bytes, k, false);
            check_straddle(v, n_bytes, k, true);
        }
    }
}

/// A store fully inside the first page must round-trip identically to the
/// straddling case — guards against the split path leaking into same-page
/// stores (and vice-versa).
#[test]
fn test_same_page_matches_straddle_reference() {
    let ctx = SymContext::new_mock();
    let v: u64 = 0xdead_beef_cafe_babe;
    for &big in &[false, true] {
        let mut mem = two_page_mem(big);
        // Well inside the first page, no seam crossed.
        mem.store_concrete(0x1100, RustBV::concrete(v as u128, 64))
            .unwrap();
        assert_eq!(
            mem.load_concrete(0x1100, 8, &ctx).unwrap().as_u64(),
            Some(v)
        );
        for i in 0..8u32 {
            assert_eq!(
                mem.load_concrete(0x1100 + i as u64, 1, &ctx)
                    .unwrap()
                    .as_u64(),
                Some(ref_byte(v, 8, i, big)),
                "same-page byte {i} big={big}"
            );
        }
    }
}
