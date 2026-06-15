//! Tests for [`super`] — directory / file-name syscall handlers.
//!
//! Extracted from the former inline `#[cfg(test)] mod tests` block in
//! `directory.rs` (angr-7k4r) per the mod-tests sibling-extraction pattern.

use super::*;
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::syscalls::{NativeSyscall, SyscallOutcome};

/// Sweep the 9 stub handlers across every supported arch and verify
/// they return a fresh `RustBV::symbolic` of width `arch().bits()`.
/// Successive invocations must yield distinct fresh symbols.
#[test]
fn dir_stub_handlers_return_fresh_symbolic_on_all_arches() {
    let cases: &[(&'static dyn NativeSyscall, &str, usize)] = &[
        (&NativeFchdirSyscall, "fchdir", 1),
        (&NativeMkdirSyscall, "mkdir", 2),
        (&NativeMkdiratSyscall, "mkdirat", 3),
        (&NativeRmdirSyscall, "rmdir", 1),
        (&NativeUnlinkSyscall, "unlink", 1),
        (&NativeUnlinkatSyscall, "unlinkat", 3),
        (&NativeRenameSyscall, "rename", 2),
        (&NativeRenameatSyscall, "renameat", 4),
        (&NativeRenameat2Syscall, "renameat2", 5),
    ];

    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        for &(handler, label, nargs) in cases {
            assert_eq!(handler.name(), label);
            assert_eq!(handler.num_args(), nargs, "{label} arity");

            let args: Vec<RustBV> = (0..nargs).map(|_| RustBV::concrete(0, bits)).collect();
            let outcome = handler
                .call(&mut state, &args)
                .unwrap_or_else(|e| panic!("{arch} {label}: {e:?}"));
            let ret = match outcome {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                other => {
                    panic!("{arch} {label}: expected ContinueSymbolic, got {other:?}")
                }
            };
            assert_eq!(ret.width(), bits);
            assert!(ret.as_u64().is_none(), "{arch} {label} should be symbolic");

            let outcome2 = handler.call(&mut state, &args).unwrap();
            let ret2 = match outcome2 {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                _ => unreachable!(),
            };
            let (id1, id2) = match (&ret, &ret2) {
                (RustBV::Symbolic { id: a, .. }, RustBV::Symbolic { id: b, .. }) => (*a, *b),
                _ => panic!("{arch} {label}: expected Symbolic variant"),
            };
            assert_ne!(
                id1, id2,
                "{arch} {label} successive calls must yield distinct symbol IDs"
            );
        }
    }
}

/// chdir + getcwd round-trip: store path bytes at buf, run chdir,
/// then getcwd into a second buffer and confirm the bytes match.
/// This is the integration acceptance test from the bd description.
#[test]
fn chdir_and_getcwd_round_trip() {
    let mut state = RustSimState::new("amd64").expect("state");

    // Default cwd should be b"/".
    assert_eq!(state.file_system_ref().cwd(), b"/");

    // Map a writable region covering both the chdir path buffer
    // (0x1000) and the getcwd destination buffer (0x2000).
    use crate::memory::Permission;
    state.map_memory_data(0x1000, &[0u8; 0x2000], Permission::RW);

    // Write "/tmp/foo\0" at 0x1000.
    let path = b"/tmp/foo";
    for (i, &b) in path.iter().enumerate() {
        state
            .memory_store(0x1000 + i as u64, RustBV::concrete(b as u128, 8))
            .unwrap();
    }
    state
        .memory_store(0x1000 + path.len() as u64, RustBV::concrete(0, 8))
        .unwrap();

    // chdir(0x1000) — should return 0 and set cwd to "/tmp/foo".
    let outcome = NativeChdirSyscall
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0, "chdir returns 0"),
        other => panic!("chdir: expected Continue, got {other:?}"),
    }
    assert_eq!(state.file_system_ref().cwd(), b"/tmp/foo");

    // getcwd(buf=0x2000, size=64) — should write "/tmp/foo\0" and
    // return 9.
    let outcome = NativeGetcwdSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(64, 64)],
        )
        .unwrap();
    let ret = match outcome {
        SyscallOutcome::Continue { ret } => ret,
        other => panic!("getcwd: expected Continue, got {other:?}"),
    };
    assert_eq!(ret, (path.len() + 1) as u64);

    // Read back the bytes and confirm.
    let mut readback = Vec::new();
    for i in 0..(path.len() + 1) {
        let bv = state.memory_load(0x2000 + i as u64, 1).unwrap();
        readback.push(bv.as_u64().unwrap() as u8);
    }
    let mut expected = path.to_vec();
    expected.push(0);
    assert_eq!(readback, expected, "getcwd writes cwd + NUL");
}

/// getcwd with size < len(cwd)+1 must return -ERANGE without touching
/// the buffer.
#[test]
fn getcwd_returns_erange_when_buf_too_small() {
    let mut state = RustSimState::new("amd64").expect("state");
    // Default cwd "/" + NUL = 2 bytes.
    let outcome = NativeGetcwdSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x3000, 64), RustBV::concrete(1, 64)],
        )
        .unwrap();
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_ERANGE),
        other => panic!("getcwd undersized: expected Continue, got {other:?}"),
    }
}

/// chdir on a symbolic path falls back to Python via
/// `SyscallError::SymbolicArgument`.
#[test]
fn chdir_symbolic_path_pointer_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let bits = state.arch().bits();
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "sym_path_ptr", bits)
    };
    let err = NativeChdirSyscall.call(&mut state, &[sym]).unwrap_err();
    match err {
        SyscallError::SymbolicArgument(_) => (),
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
}
