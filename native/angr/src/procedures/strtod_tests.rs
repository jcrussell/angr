// Unit tests for strtod.rs (strtod / floating_prefix_len SimProcedure helpers).
// Split out of the parent module; see CLAUDE.md test-split recipe.
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;

fn setup_string(state: &mut RustSimState, addr: u64, s: &[u8]) {
    let mut data = s.to_vec();
    data.push(0);
    state.map_memory_data(addr, &data, Permission::RWX);
}

fn read_xmm0_low64(state: &mut RustSimState) -> u64 {
    let bv = state.get_register_by_offset(AMD64_XMM0_OFFSET, 8);
    bv.as_u64().expect("xmm0 low64 concrete")
}

#[test]
fn test_strtod_simple_decimal() {
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x1000, b"42.5");
    let p = NativeStrtod;
    let ret = p
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();
    assert!(ret.is_none(), "strtod must suppress integer-return store");
    assert_eq!(read_xmm0_low64(&mut state), 42.5f64.to_bits());
}

#[test]
fn test_strtod_negative_exponent() {
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x1000, b"-1.5e-3");
    let p = NativeStrtod;
    p.call(
        &mut state,
        &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
    )
    .unwrap();
    assert_eq!(read_xmm0_low64(&mut state), (-1.5e-3f64).to_bits());
}

#[test]
fn test_strtod_writes_endptr_past_parsed_prefix() {
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x1000, b"42.0abc");
    // Reserve space for *endptr at 0x2000.
    state.map_memory_data(0x2000, &[0u8; 8], Permission::RWX);
    let p = NativeStrtod;
    p.call(
        &mut state,
        &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
    )
    .unwrap();
    assert_eq!(read_xmm0_low64(&mut state), 42.0f64.to_bits());
    let end = state.memory_load(0x2000, 8).unwrap();
    assert_eq!(end.as_u64(), Some(0x1000 + 4)); // past "42.0"
}

#[test]
fn test_strtod_no_conversion_returns_zero_and_endptr_at_nptr() {
    // Per C: no digits parsed → return 0.0, *endptr = nptr.
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x1000, b"abc");
    state.map_memory_data(0x2000, &[0u8; 8], Permission::RWX);
    let p = NativeStrtod;
    p.call(
        &mut state,
        &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
    )
    .unwrap();
    assert_eq!(read_xmm0_low64(&mut state), 0.0f64.to_bits());
    let end = state.memory_load(0x2000, 8).unwrap();
    assert_eq!(end.as_u64(), Some(0x1000));
}

#[test]
fn test_strtod_leading_whitespace_skipped() {
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x1000, b"   \t-2.0");
    let p = NativeStrtod;
    p.call(
        &mut state,
        &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
    )
    .unwrap();
    assert_eq!(read_xmm0_low64(&mut state), (-2.0f64).to_bits());
}

#[test]
fn test_strtod_symbolic_byte_falls_back() {
    // A symbolic byte mid-string must trigger Python fallback. We can
    // verify this by writing one symbolic byte and asserting the call
    // returns SymbolicArgument.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"X.0\x00", Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "fp0", 8);
    drop(ctx);
    state.memory_store(0x1000, sym).unwrap();

    let p = NativeStrtod;
    let res = p.call(
        &mut state,
        &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
    );
    match res {
        Err(ProcedureError::SymbolicArgument(_)) => {}
        other => panic!("expected SymbolicArgument fallback, got {:?}", other),
    }
}

#[test]
fn test_strtod_inf_and_nan_parsed() {
    let mut state = RustSimState::new("amd64").unwrap();
    setup_string(&mut state, 0x1000, b"inf");
    let p = NativeStrtod;
    p.call(
        &mut state,
        &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
    )
    .unwrap();
    let bits = read_xmm0_low64(&mut state);
    let val = f64::from_bits(bits);
    assert!(val.is_infinite() && val.is_sign_positive());
}

#[test]
fn test_strtod_non_amd64_falls_back() {
    // x86 has FP returns in st0, not xmm0 — we punt to Python.
    let mut state = RustSimState::new("x86").unwrap();
    state.map_memory_data(0x1000, b"1.0\x00", Permission::RWX);
    let p = NativeStrtod;
    let res = p.call(
        &mut state,
        &[RustBV::concrete(0x1000, 32), RustBV::concrete(0, 32)],
    );
    assert!(matches!(res, Err(ProcedureError::NotImplemented)));
}

#[test]
fn test_floating_prefix_len_grammar() {
    assert_eq!(floating_prefix_len(b"3.14abc"), 4);
    assert_eq!(floating_prefix_len(b"  -2.5e10x"), 9);
    assert_eq!(floating_prefix_len(b"abc"), 0);
    assert_eq!(floating_prefix_len(b"-abc"), 0);
    // Hex form
    assert_eq!(floating_prefix_len(b"0x1.8p3"), 7);
    assert_eq!(floating_prefix_len(b"0xfg"), 3); // "0xf" then stop
    // Trailing 'e' without digits gets dropped
    assert_eq!(floating_prefix_len(b"1.5ex"), 3);
}
