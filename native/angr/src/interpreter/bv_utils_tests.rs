//! Unit tests for [`super::bv_utils`] — the RustBV <-> byte conversions and
//! the structural ITE-target extractor.

use super::*;

#[test]
fn test_bytes_to_bv() {
    let bytes = vec![0x78, 0x56, 0x34, 0x12];
    let bv = bytes_to_bv(&bytes, 32);
    assert_eq!(bv.as_u64(), Some(0x12345678));
}

#[test]
fn test_bv_to_bytes() {
    let bv = RustBV::concrete(0x12345678, 32);
    let bytes = bv_to_bytes(&bv);
    assert_eq!(bytes, vec![0x78, 0x56, 0x34, 0x12]);
}

// --- >128-bit width coverage (angr-03vl4.4 / angr-03vl4.31) ---
//
// `RustBV::Concrete` legitimately carries AVX/YMM-scale widths up to 256 while
// its payload is a `u128`, so both conversions have to stop at byte 16. Before
// the fix each ran `<< (i * 8)` / `>> (i * 8)` for `i` up to 31: a shift-overflow
// abort under debug assertions, and in release a wraparound that aliased byte 16
// onto byte 0 (a silently corrupt value, not a loud failure).

#[test]
fn bv_to_bytes_of_256_bit_value_zero_fills_above_the_u128_payload() {
    let bv = RustBV::concrete(u128::MAX, 256);
    let bytes = bv_to_bytes(&bv);
    assert_eq!(bytes.len(), 32);
    assert!(bytes[..16].iter().all(|&b| b == 0xff), "{bytes:?}");
    // The wrapping bug made byte 16 alias byte 0 (0xff) instead of reading 0.
    assert!(bytes[16..].iter().all(|&b| b == 0), "{bytes:?}");
}

#[test]
fn bytes_to_bv_of_256_bit_buffer_keeps_every_byte_via_concat() {
    let bytes: Vec<u8> = (0..32u8).map(|i| i.wrapping_add(1)).collect();
    let bv = bytes_to_bv(&bytes, 256);
    assert_eq!(bv.width(), 256);

    let lo = u128::from_le_bytes(bytes[..16].try_into().unwrap());
    let hi = u128::from_le_bytes(bytes[16..].try_into().unwrap());
    // The truncating version dropped `hi` entirely; the wrapping one OR-ed
    // byte 16 (0x11) over byte 0 (0x01), giving a low lane ending in 0x11.
    assert_eq!(bv.extract_no_ctx(127, 0).as_u128(), Some(lo));
    assert_eq!(bv.extract_no_ctx(255, 128).as_u128(), Some(hi));
}

#[test]
fn bytes_to_bv_round_trips_through_bv_to_bytes_at_256_bits() {
    // `bv_to_bytes` can only serialize the u128 payload, so the round trip is
    // exact exactly for the values that fit in it — the high bytes it emits are
    // zeros and come back as zeros.
    let value = 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100u128;
    let bv = RustBV::concrete(value, 256);
    let round_tripped = bytes_to_bv(&bv_to_bytes(&bv), 256);
    assert_eq!(round_tripped.width(), 256);
    assert_eq!(round_tripped.extract_no_ctx(127, 0).as_u128(), Some(value));
    assert_eq!(round_tripped.extract_no_ctx(255, 128).as_u128(), Some(0));
}

#[test]
fn bytes_to_bv_ignores_bytes_past_the_requested_width() {
    // Buffer longer than `width` — the extra bytes are not part of the value.
    let bytes = vec![0x78, 0x56, 0x34, 0x12, 0xff, 0xff];
    assert_eq!(bytes_to_bv(&bytes, 32).as_u64(), Some(0x1234_5678));
}

// --- extract_ite_targets coverage (angr-szg45.5) ---
//
// The symbolic-IP fast-path that pulls concrete jump targets out of a nested
// ITE BV without solver queries. Integration symbolic-jump tests route through
// AddressConcretizer with BVS+Or constraints and never build a structural Ite,
// so these branches need constructed RustBV inputs.

/// Build an `Ite` expression whose true/false operands are `t`/`f`. The
/// condition operand (operands[0]) is a dummy concrete — `extract_ite_targets`
/// never inspects it.
fn ite_bv(t: RustBV, f: RustBV) -> RustBV {
    RustBV::Expression {
        id: RustBV::EXPRESSION_ID,
        width: 64,
        op: BVOp::Ite,
        operands: std::sync::Arc::from(vec![RustBV::concrete(0, 1), t, f]),
        memo: Default::default(),
    }
}

#[test]
fn test_extract_ite_targets_single_leaf() {
    // A bare concrete leaf (the keep_ip_symbolic single-target case).
    let bv = RustBV::concrete(0x400123, 64);
    assert_eq!(extract_ite_targets(&bv, 8), Some(vec![0x400123]));
}

#[test]
fn test_extract_ite_targets_nested_three() {
    // ite(_, a, ite(_, b, c)) -> all three concrete addrs collected.
    let inner = ite_bv(RustBV::concrete(0xb, 64), RustBV::concrete(0xc, 64));
    let bv = ite_bv(RustBV::concrete(0xa, 64), inner);
    let mut got = extract_ite_targets(&bv, 8).expect("three targets");
    got.sort_unstable();
    assert_eq!(got, vec![0xa, 0xb, 0xc]);
}

#[test]
fn test_extract_ite_targets_dedup() {
    // Duplicate address across branches collapses to one entry.
    let bv = ite_bv(RustBV::concrete(0x42, 64), RustBV::concrete(0x42, 64));
    assert_eq!(extract_ite_targets(&bv, 8), Some(vec![0x42]));
}

#[test]
fn test_extract_ite_targets_constrained_leaf() {
    // A Constrained leaf (symbolic with a known concrete value) is treated
    // like a concrete target.
    let leaf = RustBV::Constrained {
        id: 7,
        value: 0x555,
        width: 64,
    };
    assert_eq!(extract_ite_targets(&leaf, 8), Some(vec![0x555]));
}

#[test]
fn test_extract_ite_targets_non_ite_symbolic_aborts() {
    // An ITE wrapping a non-ITE symbolic leaf (here an Add expression) can't be
    // resolved structurally -> None.
    let add = RustBV::Expression {
        id: RustBV::EXPRESSION_ID,
        width: 64,
        op: BVOp::Add,
        operands: std::sync::Arc::from(vec![RustBV::concrete(1, 64), RustBV::concrete(2, 64)]),
        memo: Default::default(),
    };
    let bv = ite_bv(RustBV::concrete(0xa, 64), add);
    assert_eq!(extract_ite_targets(&bv, 8), None);
}

#[test]
fn test_extract_ite_targets_exceeds_max() {
    // A chain of four distinct targets with max_targets=2 aborts to None.
    let i1 = ite_bv(RustBV::concrete(0x3, 64), RustBV::concrete(0x4, 64));
    let i2 = ite_bv(RustBV::concrete(0x2, 64), i1);
    let bv = ite_bv(RustBV::concrete(0x1, 64), i2);
    assert_eq!(extract_ite_targets(&bv, 2), None);
}
