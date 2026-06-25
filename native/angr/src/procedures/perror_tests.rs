// Tests extracted from perror.rs (see #[path] attr in parent module).
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

#[test]
fn test_perror_writes_string_to_stderr() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"boom\0", Permission::RWX);

    let result = NativePerror
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();

    // Void return — no value set, stderr buffer carries the string (no errno
    // suffix, matching Python's perror).
    assert!(result.is_none());
    assert_eq!(state.fd_buffer(2), b"boom");
    // stdout untouched.
    assert_eq!(state.stdout_buffer(), b"");
}

#[test]
fn test_perror_empty_string() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\0", Permission::RWX);

    let result = NativePerror
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();

    assert!(result.is_none());
    assert_eq!(state.fd_buffer(2), b"");
}

#[test]
fn test_perror_symbolic_byte_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    // Leave 0x1000 unmapped so the scan hits a non-concrete load → error.
    let result = NativePerror.call(&mut state, &[RustBV::concrete(0x1000, 64)]);
    assert!(result.is_err());
}

#[test]
fn test_perror_registered() {
    let registry = crate::procedures::NativeProcedureRegistry::new();
    assert!(registry.has_native("perror"));
}
