// Tests for getid.rs (NativeGetuid / NativeGeteuid / NativeGetgid / NativeGetegid).
// Extracted from the parent module; see the `#[path]` attr in getid.rs.
use super::*;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;

#[test]
fn test_getuid_returns_1000_word_sized() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeGetuid.call(&mut state, &[]).unwrap().unwrap();
    assert_eq!(result.as_u64(), Some(1000));
    assert_eq!(result.width(), 64);
}

#[test]
fn test_identity_getters_all_return_1000() {
    let mut state = RustSimState::new("amd64").unwrap();
    for proc in [
        &NativeGetuid as &dyn NativeSimProcedure,
        &NativeGeteuid as &dyn NativeSimProcedure,
        &NativeGetgid as &dyn NativeSimProcedure,
        &NativeGetegid as &dyn NativeSimProcedure,
    ] {
        let result = proc.call(&mut state, &[]).unwrap().unwrap();
        assert_eq!(result.as_u64(), Some(1000), "{} mismatch", proc.name());
    }
}

#[test]
fn test_getuid_width_tracks_arch_32bit() {
    let mut state = RustSimState::new("x86").unwrap();
    let result = NativeGetuid.call(&mut state, &[]).unwrap().unwrap();
    assert_eq!(result.as_u64(), Some(1000));
    assert_eq!(result.width(), 32);
}
