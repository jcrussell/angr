//! Native POSIX process-identity getters: getuid / geteuid / getgid / getegid.
//!
//! Angr's Python SimProcedures (`angr/procedures/posix/{getuid,geteuid,getgid,
//! getegid}.py`) each `return 1000` — a constant word-sized value. The native
//! side previously only had these as *syscall* handlers (`syscalls/identity.rs`,
//! `DEFAULT_UID_GID`); a PLT libc call to `getuid` dispatches through the
//! procedure registry, not the syscall path, so every such call fell back to
//! Python. These four zero-arg procedures restore native parity.
//!
//! The return value is imported from
//! [`crate::syscalls::identity::DEFAULT_UID_GID`] rather than redeclared, so
//! the two dispatch paths cannot drift apart (angr-5mnx3.35).
//!
//! The return BV is sized to `arch().bits()`, matching how angr's
//! `SimProcedure.ret(<python int>)` builds a `BVV(value, arch.bits)`.

use crate::syscalls::identity::DEFAULT_UID_GID;

crate::declare_const_proc! {
    /// ```c
    /// uid_t getuid(void);
    /// ```
    name = "getuid",
    struct = NativeGetuid,
    value = DEFAULT_UID_GID,
}

crate::declare_const_proc! {
    /// ```c
    /// uid_t geteuid(void);
    /// ```
    name = "geteuid",
    struct = NativeGeteuid,
    value = DEFAULT_UID_GID,
}

crate::declare_const_proc! {
    /// ```c
    /// gid_t getgid(void);
    /// ```
    name = "getgid",
    struct = NativeGetgid,
    value = DEFAULT_UID_GID,
}

crate::declare_const_proc! {
    /// ```c
    /// gid_t getegid(void);
    /// ```
    name = "getegid",
    struct = NativeGetegid,
    value = DEFAULT_UID_GID,
}

test_submod!("getid_tests.rs" => tests);
