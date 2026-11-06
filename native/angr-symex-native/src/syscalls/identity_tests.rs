// Tests for syscalls/identity.rs (getpid/getppid/getuid/geteuid/getgid/getegid/gettid).
// Extracted from the inline `mod tests` block; see rust-module-root-split-recipe.

use super::*;
use crate::state::ProcessIds;

fn assert_continue(outcome: SyscallOutcome, expected: u64) {
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, expected),
        other => panic!("expected Continue({expected}), got {other:?}"),
    }
}

#[test]
fn getpid_returns_default_pid() {
    let h = NativeGetpidSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    assert_eq!(h.name(), "getpid");
    assert_eq!(h.num_args(), 0);
    let outcome = h.call(&mut state, &[]).expect("getpid never errs");
    assert_continue(outcome, 1337);
}

#[test]
fn gettid_returns_default_pid() {
    // gettid shares pid in single-threaded angr.
    let h = NativeGettidSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    assert_eq!(h.name(), "gettid");
    let outcome = h.call(&mut state, &[]).expect("gettid never errs");
    assert_continue(outcome, 1337);
}

#[test]
fn getppid_returns_default_ppid() {
    let h = NativeGetppidSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    assert_eq!(h.name(), "getppid");
    let outcome = h.call(&mut state, &[]).expect("getppid never errs");
    assert_continue(outcome, 1336);
}

#[test]
fn pid_getters_return_the_state_process_ids() {
    // A user-set `state.posix.pid` / `ppid` reaches the getters through
    // `process_ids` (angr-vmi2l.149).
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.set_process_ids(ProcessIds {
        pid: 4242,
        ppid: 4241,
    });
    let pid = NativeGetpidSyscall.call(&mut state, &[]).expect("getpid");
    assert_continue(pid, 4242);
    let tid = NativeGettidSyscall.call(&mut state, &[]).expect("gettid");
    assert_continue(tid, 4242);
    let ppid = NativeGetppidSyscall.call(&mut state, &[]).expect("getppid");
    assert_continue(ppid, 4241);
    // Forks and snapshots carry them.
    let restored = RustSimState::from_snapshot(state.fork().to_snapshot()).expect("restore");
    assert_eq!(restored.process_ids().pid, 4242);
    assert_eq!(restored.process_ids().ppid, 4241);
}

#[test]
fn setuid_setgid_return_fresh_symbolic() {
    // setuid / setgid have no Python SimProcedure; the stub returns
    // a fresh unconstrained symbol. The native handler must do the
    // same — every invocation yields a distinct symbol (distinct
    // RustBV::Symbolic.id), and the BV must be sized to the arch's
    // `long` (== arch().bits()) on every supported arch.
    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();

        for handler in [
            &NativeSetuidSyscall as &dyn NativeSyscall,
            &NativeSetgidSyscall as &dyn NativeSyscall,
        ] {
            assert_eq!(handler.num_args(), 1);
            let arg = RustBV::concrete(0, bits);
            let outcome = handler
                .call(&mut state, &[arg])
                .expect("setuid/setgid never errs");
            let ret = match outcome {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                other => panic!(
                    "{arch} {} expected ContinueSymbolic, got {other:?}",
                    handler.name()
                ),
            };
            assert_eq!(
                ret.width(),
                bits,
                "{arch} {} return width should match arch().bits()",
                handler.name(),
            );
            assert!(
                ret.as_u64().is_none(),
                "{arch} {} return must be symbolic (not concrete)",
                handler.name(),
            );
            // Fresh symbol on every call: confirm by id inequality
            // across a second invocation.
            let arg2 = RustBV::concrete(0, bits);
            let outcome2 = handler.call(&mut state, &[arg2]).unwrap();
            let ret2 = match outcome2 {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                _ => unreachable!(),
            };
            let (id1, id2) = match (ret.leaf_var(), ret2.leaf_var()) {
                (Some((a, _)), Some((b, _))) => (a, b),
                _ => panic!("{arch} {} both returns should be Symbolic", handler.name()),
            };
            assert_ne!(
                id1,
                id2,
                "{arch} {} successive calls must yield distinct fresh symbols",
                handler.name(),
            );
        }
    }
}

#[test]
fn uid_gid_getters_return_1000() {
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    for (h, label) in [
        (&NativeGetuidSyscall as &dyn NativeSyscall, "getuid"),
        (&NativeGeteuidSyscall as &dyn NativeSyscall, "geteuid"),
        (&NativeGetgidSyscall as &dyn NativeSyscall, "getgid"),
        (&NativeGetegidSyscall as &dyn NativeSyscall, "getegid"),
    ] {
        assert_eq!(h.name(), label);
        assert_eq!(h.num_args(), 0);
        let outcome = h.call(&mut state, &[]).expect("getter never errs");
        assert_continue(outcome, DEFAULT_UID_GID);
    }
}
