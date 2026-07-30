//! Native POSIX process-identity getters: getuid / geteuid / getgid / getegid.
//!
//! Angr's Python SimProcedures (`angr/procedures/posix/{getuid,geteuid,getgid,
//! getegid}.py`) each `return 1000` — a constant word-sized value. The native
//! side previously only had these as *syscall* handlers (`syscalls/identity.rs`,
//! `DEFAULT_UID_GID`); a PLT libc call to `getuid` dispatches through the
//! procedure registry, not the syscall path, so every such call fell back to
//! Python. These four zero-arg procedures restore native parity.
//!
//! The return BV is sized to `arch().bits()`, matching how angr's
//! `SimProcedure.ret(<python int>)` builds a `BVV(value, arch.bits)`.

/// Angr's default uid/gid return value (mirrors `DEFAULT_UID_GID` in
/// `syscalls/identity.rs` and `procedures/posix/getuid.py`).
const DEFAULT_UID_GID: u128 = 1000;

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

#[cfg(test)]
#[path = "getid_tests.rs"]
mod tests;
