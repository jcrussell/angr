use super::*;

#[test]
fn test_rand_returns_symbolic() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeRand.call(&mut state, &[]).unwrap();
    let bv = result.unwrap();
    // Should be symbolic (32 bits: 31-bit symbolic zero-extended)
    assert!(bv.as_u64().is_none()); // symbolic, not concrete
    assert_eq!(bv.width(), 32);
}

#[test]
fn test_rand_unique_names() {
    let mut state = RustSimState::new("amd64").unwrap();
    let r1 = NativeRand.call(&mut state, &[]).unwrap().unwrap();
    let r2 = NativeRand.call(&mut state, &[]).unwrap().unwrap();
    // Both should be symbolic but different
    assert!(r1.as_u64().is_none());
    assert!(r2.as_u64().is_none());
}

#[test]
fn test_rand_metadata() {
    assert_eq!(NativeRand.name(), "rand");
    assert_eq!(NativeRand.num_args(), 0);
    assert!(!NativeRand.no_return());
}

#[test]
fn test_srand_noop() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeSrand
        .call(&mut state, &[RustBV::concrete(42, 32)])
        .unwrap();
    assert!(result.is_none());
    assert_eq!(NativeSrand.name(), "srand");
}
