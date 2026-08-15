//! Wide (multi-byte) symbolic store/load byte ordering.
//!
//! 128-bit stores must lay bytes out MSB-first on big-endian and LSB-first on
//! little-endian, and sub-word loads must extract the matching lanes — both
//! through the exact-address path and through the wider-symbolic linear-scan
//! fallback. Regression home for the byte-reversal bug class.
//!
//! Carved out of the former monolithic `symbolic.rs` (angr-c7xno.56).

use super::super::*;

/// angr-syf4: 128-bit symbolic store to big-endian memory must lay out
/// bytes MSB-first (byte at addr+0 = MSB, byte at addr+15 = LSB) and
/// sub-word loads must extract the corresponding lanes.
///
/// Regression for the wide-symbolic-object byte-reversal class of bugs
/// described in project_endianness_bug.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_big_endian_128bit_wide_symbolic_store() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // 128-bit symbolic value pinned to a known constant so we can
    // predict every byte. MSB byte = 0x10, LSB byte = 0x1F.
    let pinned: u128 = 0x10111213_14151617_18191A1B_1C1D1E1F;
    let sym = RustBV::symbolic(&ctx, "wide128", 128);
    ctx.assume_true(&sym.eq(&RustBV::concrete(pinned, 128), &ctx));

    let addr = RustBV::concrete(0x1000, 64);
    mem.store_symbolic(addr, sym, &ctx, &concretizer)
        .expect("store_symbolic must succeed");
    assert!(ctx.is_sat(), "context must remain SAT after store");

    // Exact 16-byte load returns the full symbolic value.
    let full = mem
        .load_concrete(0x1000, 16, &ctx)
        .expect("16-byte load must succeed");
    assert_eq!(
        ctx.eval(&full),
        Some(pinned),
        "exact-width load must round-trip the pinned u128"
    );

    // Per-byte BE layout: byte at addr+i corresponds to byte position
    // (15 - i) when the value is interpreted MSB-first.
    for i in 0u64..16 {
        let byte_bv = mem
            .load_concrete(0x1000 + i, 1, &ctx)
            .expect("single-byte load must succeed");
        let expected: u128 = (pinned >> ((15 - i) * 8)) & 0xff;
        assert_eq!(
            ctx.eval(&byte_bv),
            Some(expected),
            "BE byte at offset {i} expected 0x{expected:02x}"
        );
    }

    // 4-byte load at offset 4 should return bytes [4..8) of the BE
    // layout (bits [95:64] of the original u128 = 0x14151617).
    let word = mem
        .load_concrete(0x1004, 4, &ctx)
        .expect("4-byte load must succeed");
    let expected_word: u128 = (pinned >> 64) & 0xFFFF_FFFF;
    assert_eq!(
        ctx.eval(&word),
        Some(expected_word),
        "BE 4-byte load at offset 4 expected 0x{expected_word:08x}"
    );

    // 8-byte halves: high half at offset 0, low half at offset 8.
    let qhi = mem
        .load_concrete(0x1000, 8, &ctx)
        .expect("hi qword load must succeed");
    let expected_qhi: u128 = (pinned >> 64) & 0xFFFF_FFFF_FFFF_FFFF;
    assert_eq!(
        ctx.eval(&qhi),
        Some(expected_qhi),
        "BE high qword expected 0x{expected_qhi:016x}"
    );

    let qlo = mem
        .load_concrete(0x1008, 8, &ctx)
        .expect("lo qword load must succeed");
    let expected_qlo: u128 = pinned & 0xFFFF_FFFF_FFFF_FFFF;
    assert_eq!(
        ctx.eval(&qlo),
        Some(expected_qlo),
        "BE low qword expected 0x{expected_qlo:016x}"
    );
}

/// angr-v1q2: 128-bit symbolic store to little-endian memory must lay
/// out bytes LSB-first (byte at addr+0 = LSB, byte at addr+15 = MSB)
/// and sub-word loads must extract the corresponding lanes.
///
/// Counterpart to `test_big_endian_128bit_wide_symbolic_store`. Hits
/// both partial-extract paths in `load_concrete`: the exact-address
/// path (load at 0x1000) and the symbolic_spans path (load at
/// non-zero offsets). Before the fix, the LE branch returned MSB-side
/// bytes from the wide BV instead of LSB-side bytes, so single-byte
/// and 4-byte loads at non-zero offsets gave wrong values.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_little_endian_128bit_wide_symbolic_store() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // 128-bit symbolic value pinned to a known constant. For LE, the
    // byte at addr+0 is the LSB (0x1F here) and addr+15 is the MSB.
    let pinned: u128 = 0x10111213_14151617_18191A1B_1C1D1E1F;
    let sym = RustBV::symbolic(&ctx, "wide128_le", 128);
    ctx.assume_true(&sym.eq(&RustBV::concrete(pinned, 128), &ctx));

    let addr = RustBV::concrete(0x1000, 64);
    mem.store_symbolic(addr, sym, &ctx, &concretizer)
        .expect("store_symbolic must succeed");
    assert!(ctx.is_sat(), "context must remain SAT after store");

    // Exact 16-byte load returns the full symbolic value.
    let full = mem
        .load_concrete(0x1000, 16, &ctx)
        .expect("16-byte load must succeed");
    assert_eq!(
        ctx.eval(&full),
        Some(pinned),
        "exact-width load must round-trip the pinned u128"
    );

    // Per-byte LE layout: byte at addr+i is bits [(i+1)*8-1 : i*8].
    for i in 0u64..16 {
        let byte_bv = mem
            .load_concrete(0x1000 + i, 1, &ctx)
            .expect("single-byte load must succeed");
        let expected: u128 = (pinned >> (i * 8)) & 0xff;
        assert_eq!(
            ctx.eval(&byte_bv),
            Some(expected),
            "LE byte at offset {i} expected 0x{expected:02x}"
        );
    }

    // 4-byte load at offset 4 should return bits [63:32] of the
    // pinned u128 = 0x17161514.
    let word = mem
        .load_concrete(0x1004, 4, &ctx)
        .expect("4-byte load must succeed");
    let expected_word: u128 = (pinned >> 32) & 0xFFFF_FFFF;
    assert_eq!(
        ctx.eval(&word),
        Some(expected_word),
        "LE 4-byte load at offset 4 expected 0x{expected_word:08x}"
    );

    // 8-byte halves: low half at offset 0, high half at offset 8.
    let qlo = mem
        .load_concrete(0x1000, 8, &ctx)
        .expect("lo qword load must succeed");
    let expected_qlo: u128 = pinned & 0xFFFF_FFFF_FFFF_FFFF;
    assert_eq!(
        ctx.eval(&qlo),
        Some(expected_qlo),
        "LE low qword expected 0x{expected_qlo:016x}"
    );

    let qhi = mem
        .load_concrete(0x1008, 8, &ctx)
        .expect("hi qword load must succeed");
    let expected_qhi: u128 = (pinned >> 64) & 0xFFFF_FFFF_FFFF_FFFF;
    assert_eq!(
        ctx.eval(&qhi),
        Some(expected_qhi),
        "LE high qword expected 0x{expected_qhi:016x}"
    );
}

/// angr-76mo: wider-symbolic linear-scan fallback in load_concrete
/// must honour memory endianness when extracting a sub-range.
///
/// The linear scan inside `containing_wider_sym` is reached when no per-byte
/// reconstruction succeeds but a containing wider BV exists. Direct
/// manipulation of `symbolic_objects` (without populating
/// `symbolic_spans`) is the surest way to force this path: the
/// exact-address branch needs sym.width() == size*8 (skipped), the
/// symbolic_spans branch finds nothing, per-byte concat finds
/// nothing, then the linear scan triggers.
fn wide_linear_scan_setup(endness: Endness) -> (SymContext, SymbolicMemory, u128) {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(endness);
    mem.map(0x1000, 0x1000, Permission::RWX);
    let pinned: u128 = 0x1011_1213_1415_1617_1819_1A1B_1C1D_1E1F;
    let sym = RustBV::symbolic(&ctx, "wide128_lin", 128);
    ctx.assume_true(&sym.eq(&RustBV::concrete(pinned, 128), &ctx));
    // Insert directly into symbolic_objects without populating
    // symbolic_spans, then mark each byte as symbolic on the page so
    // has_symbolic flips during the byte scan.
    mem.symbolic_objects.insert(Address(0x1000), sym);
    let page = mem.pages.get_mut(&(0x1000 >> 12)).expect("page mapped");
    page.mark_symbolic(0, 16);
    (ctx, mem, pinned)
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_wide_linear_scan_little_endian() {
    let (ctx, mem, pinned) = wide_linear_scan_setup(Endness::Little);
    // 4-byte load at offset 4 → LE bytes [4..8) = bits [63:32].
    let word = mem
        .load_concrete(0x1004, 4, &ctx)
        .expect("4-byte load must succeed");
    let expected: u128 = (pinned >> 32) & 0xFFFF_FFFF;
    assert_eq!(
        ctx.eval(&word),
        Some(expected),
        "LE linear scan expected 0x{expected:08x}"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_wide_linear_scan_big_endian() {
    let (ctx, mem, pinned) = wide_linear_scan_setup(Endness::Big);
    // 4-byte load at offset 4 → BE bytes [4..8) = bits [95:64].
    let word = mem
        .load_concrete(0x1004, 4, &ctx)
        .expect("4-byte load must succeed");
    let expected: u128 = (pinned >> 64) & 0xFFFF_FFFF;
    assert_eq!(
        ctx.eval(&word),
        Some(expected),
        "BE linear scan expected 0x{expected:08x}"
    );
}

/// angr-xloth.4: `extract_byte_lane`'s bounds check used to be spelled
/// `(byte_offset + 1) * 8 > total_bits`. Both callers form `byte_offset` as a
/// *wrapping* `Address` difference (`(byte_addr - base_addr) as u32`), so a
/// stale `symbolic_spans` entry whose base sits above the byte produces a
/// near-`u32::MAX` offset — and with release overflow-checks off that product
/// wrapped to a small value, passing the check and handing the BE arm an
/// underflowed `total_bits - byte_offset * 8 - 1`. The guard is now
/// `checked_add`/`checked_mul`, so an unrepresentable offset is refused.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_extract_byte_lane_refuses_wrapping_byte_offset() {
    let ctx = SymContext::new_mock();
    let sym = RustBV::symbolic(&ctx, "wide64", 64);
    for endness in [Endness::Little, Endness::Big] {
        // (2^29) * 8 == 2^32 → the old `(off + 1) * 8` wrapped to 0.
        let wrapped = (1u32 << 29) - 1;
        assert!(
            SymbolicMemory::extract_byte_lane(&sym, wrapped, endness, &ctx).is_none(),
            "{endness:?}: offset whose bit position overflows u32 must be refused"
        );
        assert!(
            SymbolicMemory::extract_byte_lane(&sym, u32::MAX, endness, &ctx).is_none(),
            "{endness:?}: u32::MAX offset must be refused, not wrap to lane 0"
        );
        // The in-range lanes still work.
        assert!(
            SymbolicMemory::extract_byte_lane(&sym, 7, endness, &ctx).is_some(),
            "{endness:?}: last in-range lane must still extract"
        );
        assert!(
            SymbolicMemory::extract_byte_lane(&sym, 8, endness, &ctx).is_none(),
            "{endness:?}: first out-of-range lane must still be refused"
        );
    }
}
