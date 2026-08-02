// Tests for symbolic/bv_chunk.rs — the read tests moved here from
// exploration/helpers_tests.rs when angr-9ke6b.230 sank the helpers to the
// symbolic layer. `E = String` keeps them hermetic (no Python interpreter).
use super::*;

/// The chunker must walk the range in <=16-byte steps (the `u128` round-trip
/// limit), hand each chunk's own address to the source, and concatenate in
/// order — a wider single call would wrap `u128_to_le_bytes` mod 128
/// (angr-9ke6b.82 consolidated three hand-copies of this loop).
#[test]
fn load_concrete_bytes_chunked_splits_at_16_and_concatenates() {
    let mut seen = Vec::new();
    let out = load_concrete_bytes_chunked::<String, _>(0x1000, 40, |a, n| {
        seen.push((a, n));
        // Distinguishable per-chunk payload: low byte of the chunk address.
        Ok(vec![(a & 0xff) as u8; n as usize])
    })
    .expect("chunked read");

    assert_eq!(seen, vec![(0x1000, 16), (0x1010, 16), (0x1020, 8)]);
    assert_eq!(out.len(), 40);
    assert_eq!(&out[..16], &[0x00u8; 16]);
    assert_eq!(&out[16..32], &[0x10u8; 16]);
    assert_eq!(&out[32..], &[0x20u8; 8]);
}

/// A failing chunk aborts the whole read: no partial prefix is returned, and
/// no later chunk is requested. Every call site depends on this — a short
/// buffer would be consumed by length as if it were real data (angr-ph300.19).
#[test]
fn load_concrete_bytes_chunked_propagates_first_error() {
    let mut calls = 0usize;
    let res = load_concrete_bytes_chunked::<String, _>(0, 48, |_a, n| {
        calls += 1;
        if calls == 2 {
            Err("unreadable".to_string())
        } else {
            Ok(vec![0xaa; n as usize])
        }
    });
    assert!(res.is_err(), "second-chunk failure must abort the read");
    assert_eq!(calls, 2, "must not request chunks after the failure");
}

/// Zero-length reads short-circuit without touching the source, mirroring
/// `store_concrete_bytes_chunked`'s empty-slice behavior.
#[test]
fn load_concrete_bytes_chunked_zero_size_never_calls_source() {
    let mut calls = 0usize;
    let out = load_concrete_bytes_chunked::<String, _>(0x2000, 0, |_a, _n| {
        calls += 1;
        Ok(vec![0xff])
    })
    .expect("zero-size read");
    assert!(out.is_empty());
    assert_eq!(calls, 0);
}

/// Write side, same 16-byte boundary: each chunk gets its own address and a
/// width matching its byte count, and the low chunk's payload is little-endian
/// packed. A single 40-byte `RustBV::concrete` would shift-overflow (angr-5aj8).
#[test]
fn store_concrete_bytes_chunked_splits_at_16_with_le_packing() {
    let data: Vec<u8> = (0..40u8).collect();
    let mut seen: Vec<(u64, u32, Option<u128>)> = Vec::new();
    store_concrete_bytes_chunked::<String, _>(0x2000, &data, |a, bv| {
        seen.push((a, bv.width(), bv.as_u128()));
        Ok(())
    })
    .expect("chunked store");

    let widths: Vec<(u64, u32)> = seen.iter().map(|(a, w, _)| (*a, *w)).collect();
    assert_eq!(widths, vec![(0x2000, 128), (0x2010, 128), (0x2020, 64)]);
    // First chunk is bytes 0..16 packed little-endian: byte i sits at bits 8i.
    let expected: u128 = (0..16u32).fold(0u128, |acc, i| acc | ((i as u128) << (i * 8)));
    assert_eq!(seen[0].2, Some(expected));
}

/// An empty buffer stores nothing, and a failing sink aborts before the next
/// chunk (no partially-written tail beyond the failing chunk).
#[test]
fn store_concrete_bytes_chunked_empty_and_error_paths() {
    let mut calls = 0usize;
    store_concrete_bytes_chunked::<String, _>(0x3000, &[], |_a, _bv| {
        calls += 1;
        Ok(())
    })
    .expect("empty store");
    assert_eq!(calls, 0);

    let data = vec![0xabu8; 48];
    calls = 0;
    let res = store_concrete_bytes_chunked::<String, _>(0x3000, &data, |_a, _bv| {
        calls += 1;
        if calls == 2 {
            Err("unwritable".to_string())
        } else {
            Ok(())
        }
    });
    assert!(res.is_err());
    assert_eq!(calls, 2, "must not store chunks after the failure");
}

/// `u128_to_le_bytes` is the exact inverse of the pack loop for widths up to
/// `MAX_CONCRETE_CHUNK`.
#[test]
fn u128_to_le_bytes_round_trips_the_pack_loop() {
    let data: Vec<u8> = (0..MAX_CONCRETE_CHUNK as u8)
        .map(|i| i.wrapping_mul(7))
        .collect();
    let mut packed = None;
    store_concrete_bytes_chunked::<String, _>(0, &data, |_a, bv| {
        packed = bv.as_u128();
        Ok(())
    })
    .expect("store");
    let val = packed.expect("concrete chunk");
    assert_eq!(u128_to_le_bytes(val, data.len()), data);
}
