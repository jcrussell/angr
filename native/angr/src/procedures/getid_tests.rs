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

// The PLT-libc-call path (`procedures/getid.rs`) and the raw-syscall path
// (`syscalls/identity.rs`) back the same four getters. Since angr-5mnx3.35 they
// share one `DEFAULT_UID_GID`, but that only pins the *const* — nothing stops a
// future edit from handing `constant_syscall!` a different value, so pin the
// agreement behaviourally by driving both dispatchers.
#[test]
fn test_procedure_and_syscall_paths_agree_on_uid_gid() {
    use crate::syscalls::{NativeSyscall, SyscallOutcome, identity};

    let pairs: [(&dyn NativeSimProcedure, &dyn NativeSyscall); 4] = [
        (&NativeGetuid, &identity::NativeGetuidSyscall),
        (&NativeGeteuid, &identity::NativeGeteuidSyscall),
        (&NativeGetgid, &identity::NativeGetgidSyscall),
        (&NativeGetegid, &identity::NativeGetegidSyscall),
    ];
    let mut state = RustSimState::new("amd64").unwrap();
    for (proc, sys) in pairs {
        assert_eq!(proc.name(), sys.name(), "name mismatch across paths");
        let from_proc = proc.call(&mut state, &[]).unwrap().unwrap();
        let SyscallOutcome::Continue { ret: from_sys } = sys.call(&mut state, &[]).unwrap() else {
            panic!("{} syscall should Continue", sys.name());
        };
        assert_eq!(
            from_proc.as_u64(),
            Some(from_sys),
            "{} disagrees between procedure and syscall paths",
            proc.name()
        );
    }
}
