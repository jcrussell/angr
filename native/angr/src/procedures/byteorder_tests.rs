//! Tests for byteorder.rs host<->network byte-order SimProcedures.

use super::*;
use crate::procedures::NativeSimProcedure;

fn make_state() -> RustSimState {
    // amd64 is little-endian, so the swap path is exercised.
    RustSimState::new("amd64").unwrap()
}

fn call_with(proc: &dyn NativeSimProcedure, state: &mut RustSimState, v: u128) -> u64 {
    let args = [RustBV::concrete(v, 64)];
    proc.call(state, &args).unwrap().unwrap().as_u64().unwrap()
}

#[test]
fn test_htonl_swaps_low_four_bytes() {
    let mut s = make_state();
    let p = NativeHtonl;
    // 0x01020304 -> 0x04030201; upper 32 bits cleared by zero_extend.
    assert_eq!(call_with(&p, &mut s, 0x0102_0304), 0x0403_0201);
    // Garbage in the high word is dropped (only low 32 bits are converted).
    assert_eq!(call_with(&p, &mut s, 0xdead_beef_0102_0304), 0x0403_0201);
}

#[test]
fn test_htons_swaps_low_two_bytes() {
    let mut s = make_state();
    let p = NativeHtons;
    // 0x0102 -> 0x0201; only the low 16 bits are converted.
    assert_eq!(call_with(&p, &mut s, 0x0102), 0x0201);
    assert_eq!(call_with(&p, &mut s, 0xffff_ffff_0102), 0x0201);
}

#[test]
fn test_ntohl_alias_round_trips_htonl() {
    // ntohl is the same swap, so htonl(htonl(x)) restores the original 32 bits.
    let mut s = make_state();
    let p = NativeHtonl;
    let once = call_with(&p, &mut s, 0x1122_3344);
    assert_eq!(call_with(&p, &mut s, once as u128), 0x1122_3344);
}

#[test]
fn test_byteorder_registered() {
    let registry = crate::procedures::NativeProcedureRegistry::new();
    assert!(registry.has_native("htonl"));
    assert!(registry.has_native("htons"));
    // ntohl/ntohs are registered via the alias list.
    assert!(registry.has_native("ntohl"));
    assert!(registry.has_native("ntohs"));
}

#[test]
fn test_symbolic_htonl_returns_bv() {
    // A symbolic argument should produce a symbolic result, not a fallback.
    let mut s = make_state();
    let ctx = s.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "v", 64);
    drop(ctx);
    let p = NativeHtonl;
    let result = p.call(&mut s, std::slice::from_ref(&sym)).unwrap().unwrap();
    assert_eq!(result.width(), 64);
    assert!(result.as_u64().is_none());
}

#[test]
fn test_htonl_is_identity_on_big_endian_state() {
    // Regression (angr-9ke6b.1): the swap decision read
    // `Arch::is_little_endian()`, hardcoded true in every arch impl, so a
    // genuinely big-endian ARM state still byte-swapped -- corrupting a value
    // that is already in network order. Host order == network order on BE, so
    // htonl/htons must be the identity there.
    let mut s = RustSimState::new_with_endian("ARM", Some(false)).unwrap();
    assert!(!s.is_little_endian());
    // ARM is 32-bit, so the result is zero-extended back to 32 bits.
    assert_eq!(call_with(&NativeHtonl, &mut s, 0x0102_0304), 0x0102_0304);
    assert_eq!(call_with(&NativeHtons, &mut s, 0x0102), 0x0102);
}

#[test]
fn test_htonl_still_swaps_on_little_endian_arm_state() {
    // The mirror case: an ARM state left at its little-endian default (and one
    // with the override set explicitly) must still perform the swap, so the
    // fix above did not turn the conversion off wholesale.
    for little in [None, Some(true)] {
        let mut s = RustSimState::new_with_endian("ARM", little).unwrap();
        assert!(s.is_little_endian());
        assert_eq!(call_with(&NativeHtonl, &mut s, 0x0102_0304), 0x0403_0201);
        assert_eq!(call_with(&NativeHtons, &mut s, 0x0102), 0x0201);
    }
}

#[test]
fn test_htonl_is_identity_on_big_endian_mips_state() {
    // MIPS is the other bi-endian arch; big-endian MIPS32 is the common case
    // in the wild, and `MIPS32::is_little_endian()` also hardcodes true.
    let mut s = RustSimState::new_with_endian("MIPS32", Some(false)).unwrap();
    assert_eq!(call_with(&NativeHtonl, &mut s, 0xdead_beef), 0xdead_beef);
}
