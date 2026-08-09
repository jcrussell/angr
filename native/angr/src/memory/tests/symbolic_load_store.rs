//! Core symbolic store/load paths: the per-byte concat fallback, the
//! `import_symbolic_value` width contract, and load-path resolution order
//! (a containing wider symbolic span winning over narrower entries).
//!
//! Carved out of the former monolithic `symbolic.rs` (angr-c7xno.56); the
//! endianness-sensitive *wide* paths live in `symbolic_wide`, overlapping
//! stores in `symbolic_overlap`.

use super::super::*;

/// angr-76mo: per-byte symbolic concat path in load_concrete must
/// honour memory endianness.
///
/// Stores four independent 8-bit symbolic BVs at consecutive byte
/// addresses, then loads 4 bytes back. This bypasses both the
/// exact-address fast path (no width-32 entry at base) and the
/// symbolic_spans path (8-bit stores have no span entries), forcing
/// the per-byte concat fallback in `load_concrete_common`.
///
/// Before the fix, the LE concat order was hardcoded for both
/// endiannesses: parts[N-1] :: ... :: parts[0]. For BE that put byte 0
/// at the LSB instead of the MSB.
fn per_byte_symbolic_setup(endness: Endness) -> (SymContext, SymbolicMemory, [u128; 4]) {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(endness);
    mem.map(0x1000, 0x1000, Permission::RWX);
    let pinned: [u128; 4] = [0xAA, 0xBB, 0xCC, 0xDD];
    for (i, &val) in pinned.iter().enumerate() {
        let sym = RustBV::symbolic(&ctx, format!("byte{i}"), 8);
        ctx.assume_true(&sym.eq(&RustBV::concrete(val, 8), &ctx));
        mem.store_concrete(0x1000 + i as u64, sym).unwrap();
    }
    assert!(
        ctx.is_sat(),
        "context must remain SAT after per-byte stores"
    );
    (ctx, mem, pinned)
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_per_byte_symbolic_concat_little_endian() {
    let (ctx, mem, pinned) = per_byte_symbolic_setup(Endness::Little);
    let word = mem
        .load_concrete(0x1000, 4, &ctx)
        .expect("4-byte load must succeed");
    // LE: byte at addr+i is at bits [(i+1)*8-1 : i*8]
    let expected: u128 = (pinned[3] << 24) | (pinned[2] << 16) | (pinned[1] << 8) | pinned[0];
    assert_eq!(
        ctx.eval(&word),
        Some(expected),
        "LE per-byte concat expected 0x{expected:08x}"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_per_byte_symbolic_concat_big_endian() {
    let (ctx, mem, pinned) = per_byte_symbolic_setup(Endness::Big);
    let word = mem
        .load_concrete(0x1000, 4, &ctx)
        .expect("4-byte load must succeed");
    // BE: byte at addr+i is at bits [(N-i)*8-1 : (N-i-1)*8]
    let expected: u128 = (pinned[0] << 24) | (pinned[1] << 16) | (pinned[2] << 8) | pinned[3];
    assert_eq!(
        ctx.eval(&word),
        Some(expected),
        "BE per-byte concat expected 0x{expected:08x}"
    );
}

/// angr-sqfj8.73: `import_symbolic_value` rejects a width that is not a
/// positive multiple of 8 instead of storing an object no load can
/// reconstruct. Before the guard, a 1-bit import truncated to `size == 0`:
/// the entry landed in `symbolic_objects` but the page bitmap stayed clear,
/// so a later byte load found `contains_key(addr)` true,
/// `bytes_all_marked_symbolic` false, and silently returned the page's
/// concrete placeholder byte. Also pins that a rejected import leaves *no*
/// trace — no side-table entry, no imported-addr mark, no auto-mapped page.
#[test]
fn test_import_symbolic_value_rejects_non_byte_multiple_width() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);

    for width in [1u32, 4, 12, 63] {
        let sym = RustBV::symbolic(&ctx, format!("sub_byte_{width}"), width);
        let err = mem
            .import_symbolic_value(0x1000, sym, None)
            .expect_err("non-byte-multiple width must be rejected");
        assert!(
            matches!(
                err,
                MemoryError::UnalignedWidth {
                    addr: 0x1000,
                    width_bits
                } if width_bits == width
            ),
            "expected UnalignedWidth for width {width}, got {err:?}"
        );
    }

    // Nothing was recorded, and the auto-map the success path performs did
    // not run either.
    assert_eq!(mem.symbolic_object_count(), 0);
    assert!(!mem.is_imported_addr(0x1000));
    assert!(!mem.pages.contains_key(&(0x1000u64 >> 12)));

    // The narrowest accepted width still works and marks its byte.
    let byte_sym = RustBV::symbolic(&ctx, "byte_sqfj8_73", 8);
    mem.import_symbolic_value(0x1000, byte_sym, None)
        .expect("byte-wide import must be accepted");
    assert_eq!(mem.symbolic_object_count(), 1);
    assert!(mem.bytes_all_marked_symbolic(Address(0x1000), 1));
}

/// angr-uwtj: `containing_wider_sym` consults `symbolic_spans` first
/// (O(1)) before falling back to a linear scan over
/// `symbolic_objects` (O(n)). After `import_symbolic_value`, the
/// spans reverse index covers offsets 1..sym_bytes; the base address
/// is matched against `symbolic_objects` directly. This test pins
/// the helper's three branches:
///   - addr strictly inside the wider sym → spans path 1
///   - addr IS the base of a wider sym for a partial read → path 2
///   - addr outside the wider sym → returns None
///   - stale spans (object deleted) → falls back to linear scan,
///     which also misses, returning None.
#[test]
fn test_containing_wider_sym_spans_first() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    let wide = RustBV::symbolic(&ctx, "wide128_uwtj", 128);
    mem.import_symbolic_value(0x1000, wide, None).unwrap();

    // Path 1: addr inside the wider sym → spans hit.
    let hit = mem
        .containing_wider_sym(Address(0x1008), 4)
        .expect("spans-first lookup must find wider sym");
    assert_eq!(hit.0, Address(0x1000));
    assert_eq!(hit.1.width(), 128);

    // Path 2: addr IS the base of a wider sym, partial read.
    let hit_base = mem
        .containing_wider_sym(Address(0x1000), 4)
        .expect("base-address lookup must find wider sym");
    assert_eq!(hit_base.0, Address(0x1000));

    // Out-of-range: addr beyond the wider sym → miss.
    assert!(
        mem.containing_wider_sym(Address(0x1020), 4).is_none(),
        "addr outside wider sym should not match"
    );

    // Load range crosses the wider sym's end → miss (full
    // containment is required).
    assert!(
        mem.containing_wider_sym(Address(0x100C), 8).is_none(),
        "load crossing wider sym's end should not match"
    );

    // Stale spans: delete the base object but leave the spans
    // entries pointing at it. The helper's path 1 sees the spans
    // entry, fails to find the base object, falls through to path
    // 2 (no entry at addr), then to the linear scan (also empty).
    // Net: None — confirming the safety net works.
    mem.symbolic_objects.remove(&Address(0x1000));
    assert!(
        mem.containing_wider_sym(Address(0x1008), 4).is_none(),
        "stale spans without backing object must not return a hit"
    );
}

/// angr-uwtj: end-to-end load_concrete via the spans-first slow path.
/// Setup forces `has_inner_overlap=true` (bypassing the
/// `symbolic_spans` fast path in `load_concrete_common`) and removes
/// a single spans entry so
/// `try_byte_merge_load` fails on that byte. The slow-path
/// reconstruction then calls `containing_wider_sym`, which finds the
/// wider sym via spans path 1 and extracts the load range
/// endianness-correctly.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_load_concrete_slow_path_spans_first_little_endian() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    let pinned: u128 = 0x1011_1213_1415_1617_1819_1A1B_1C1D_1E1F;
    let wide = RustBV::symbolic(&ctx, "wide128_slow_le", 128);
    ctx.assume_true(&wide.eq(&RustBV::concrete(pinned, 128), &ctx));
    mem.import_symbolic_value(0x1000, wide, None).unwrap();
    // Insert an unrelated symbolic_objects entry inside the load
    // range to force has_inner_overlap=true on the load below.
    let noise = RustBV::symbolic(&ctx, "noise8", 8);
    mem.symbolic_objects.insert(Address(0x1006), noise);
    // Remove the spans entry at 0x1007 so try_byte_merge_load
    // returns None and we fall through to the slow path.
    mem.symbolic_spans.remove(&Address(0x1007));
    let word = mem
        .load_concrete(0x1004, 4, &ctx)
        .expect("4-byte load must succeed via spans-first slow path");
    // LE bytes [4..8) of the wide value live at bits [63:32].
    let expected: u128 = (pinned >> 32) & 0xFFFF_FFFF;
    assert_eq!(
        ctx.eval(&word),
        Some(expected),
        "LE spans-first slow path expected 0x{expected:08x}"
    );
}
