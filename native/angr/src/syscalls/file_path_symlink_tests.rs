//! Symlink-aware `stat` / `lstat` (angr-9ke6b.235) and `newfstatat`
//! `AT_SYMLINK_NOFOLLOW` (angr-zueuw) tests.
//!
//! Before .235, `stat`/`lstat`/`newfstatat` all gated on
//! `FileSystem::is_path_known` while `readlink` resolved through the separate
//! symlink table — a path registered ONLY as a symlink stat'd as unknown. Now
//! `lstat` reports it as `S_IFLNK` (via `stat_lookup_nofollow`) and
//! `stat`/`newfstatat` follow the link (via `stat_lookup_follow`).
//!
//! The `AT_SYMLINK_NOFOLLOW` flag used to be discarded, so `newfstatat` always
//! followed. That is wrong everywhere, but *only* reachable-as-lstat on ARM64:
//! the asm-generic ABI dropped legacy `lstat`, so glibc's `lstat()` there
//! lowers to `fstatat(AT_FDCWD, path, buf, AT_SYMLINK_NOFOLLOW)`.

use super::*;
use super::stat::AT_SYMLINK_NOFOLLOW;
use super::stat_layouts::{S_IFLNK_0777, S_IFREG_0755};
use super::file_path_tests_support::*;
use crate::memory::Permission;
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;
use crate::syscalls::NativeSyscall;


#[test]
fn lstat_symlink_reports_iflnk_and_target_len() {
    let target = b"/real/destination";
    let mut state = state_with_path(b"/link");
    state
        .file_system()
        .add_symlink("/link".to_string(), target.to_vec());
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeLstatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("lstat ok");
    assert_eq!(expect_continue(out), 0);
    // amd64 layout: st_mode at 0x18 (u32), st_size at 0x30 (u64).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFLNK_0777 as u32);
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), target.len() as u64);
}

#[test]
fn lstat_symlink_does_not_need_the_path_registered() {
    // The symlink table alone is enough — `is_path_known` is false here.
    let mut state = state_with_path(b"/link");
    state
        .file_system()
        .add_symlink("/link".to_string(), b"/t".to_vec());
    assert!(!state.file_system_ref().is_path_known("/link"));
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeLstatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("lstat ok");
    assert_eq!(expect_continue(out), 0);
}

#[test]
fn stat_follows_symlink_to_its_target() {
    let mut state = state_with_path(b"/link");
    state
        .file_system()
        .add_symlink("/link".to_string(), b"/tmp/target".to_vec());
    let _fd = state
        .file_system()
        .open_with_content("/tmp/target".into(), FdFlags::ReadOnly, vec![0u8; 11])
        .expect("fd space is not exhausted in tests");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), 0);
    // Regular-file mode + the TARGET's size, not the link's.
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 11);
}

#[test]
fn stat_dangling_symlink_returns_minus_one() {
    // Link registered, target never registered → ENOENT.
    let mut state = state_with_path(b"/link");
    state
        .file_system()
        .add_symlink("/link".to_string(), b"/tmp/missing".to_vec());
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
    assert_eq!(read_u64_le(&state, 0x4000), 0);
}

#[test]
fn stat_symlink_cycle_returns_minus_one() {
    // /link → /other → /link: `MAX_SYMLINK_HOPS` bounds the walk (ELOOP)
    // rather than spinning forever.
    let mut state = state_with_path(b"/link");
    state
        .file_system()
        .add_symlink("/link".to_string(), b"/other".to_vec());
    state
        .file_system()
        .add_symlink("/other".to_string(), b"/link".to_vec());
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn stat_follows_a_two_hop_symlink_chain() {
    let mut state = state_with_path(b"/a");
    state
        .file_system()
        .add_symlink("/a".to_string(), b"/b".to_vec());
    state
        .file_system()
        .add_symlink("/b".to_string(), b"/tmp/real".to_vec());
    let _fd = state
        .file_system()
        .open_with_content("/tmp/real".into(), FdFlags::ReadOnly, vec![0u8; 5])
        .expect("fd space is not exhausted in tests");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), 0);
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 5);
}

#[test]
fn stat_non_utf8_symlink_target_returns_minus_one() {
    // Targets are raw bytes; a non-UTF-8 one can never name a key in the
    // String-keyed known-path set, so it dangles rather than panicking.
    let mut state = state_with_path(b"/link");
    state
        .file_system()
        .add_symlink("/link".to_string(), vec![0xff, 0xfe]);
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeStatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("stat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
}

#[test]
fn newfstatat_follows_symlink_like_stat() {
    let mut state = state_with_path(b"/link");
    state
        .file_system()
        .add_symlink("/link".to_string(), b"/tmp/nf".to_vec());
    let _fd = state
        .file_system()
        .open_with_content("/tmp/nf".into(), FdFlags::ReadOnly, vec![0u8; 9])
        .expect("fd space is not exhausted in tests");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(AT_FDCWD_UNSIGNED.into(), 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x4000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("newfstatat ok");
    assert_eq!(expect_continue(out), 0);
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 9);
}

#[test]
fn newfstatat_nofollow_reports_the_link_itself() {
    let target = b"/tmp/nf-target";
    let mut state = state_with_path(b"/link");
    state
        .file_system()
        .add_symlink("/link".to_string(), target.to_vec());
    let _fd = state
        .file_system()
        .open_with_content("/tmp/nf-target".into(), FdFlags::ReadOnly, vec![0u8; 9])
        .expect("fd space is not exhausted in tests");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x4000, 64),
                RustBV::concrete(AT_SYMLINK_NOFOLLOW as u128, 64),
            ],
        )
        .expect("newfstatat ok");
    assert_eq!(expect_continue(out), 0);
    // The LINK's mode/size, not the 9-byte target's — same answer
    // `NativeLstatSyscall` gives for this scenario on amd64.
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFLNK_0777 as u32);
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), target.len() as u64);
}

#[test]
fn newfstatat_nofollow_on_aarch64_matches_lstat_semantics() {
    // The scenario the bug actually breaks: an ARM64 binary calling
    // lstat() on a registered symlink.
    let target = b"/tmp/arm-target";
    let mut state = RustSimState::new("aarch64").expect("state");
    stage_path(&mut state, 0x2000, b"/armlink");
    state
        .file_system()
        .add_symlink("/armlink".to_string(), target.to_vec());
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x4000, 64),
                RustBV::concrete(AT_SYMLINK_NOFOLLOW as u128, 64),
            ],
        )
        .expect("newfstatat ok");
    assert_eq!(expect_continue(out), 0);
    // ARM64 layout: st_mode at 0x10 (u32), st_size at 0x30 (u64).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x10), S_IFLNK_0777 as u32);
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), target.len() as u64);
}

#[test]
fn newfstatat_nofollow_honors_other_flag_bits_alongside() {
    // Only the AT_SYMLINK_NOFOLLOW bit is consulted; unrelated bits
    // (here AT_EMPTY_PATH 0x1000, still unmodeled) must not clear it.
    let target = b"/tmp/mix";
    let mut state = state_with_path(b"/link");
    state
        .file_system()
        .add_symlink("/link".to_string(), target.to_vec());
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x4000, 64),
                RustBV::concrete((AT_SYMLINK_NOFOLLOW | 0x1000) as u128, 64),
            ],
        )
        .expect("newfstatat ok");
    assert_eq!(expect_continue(out), 0);
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFLNK_0777 as u32);
}

#[test]
fn newfstatat_nofollow_on_dangling_link_still_succeeds() {
    // lstat() on a dangling symlink succeeds (it stats the link, never
    // the target) where the following variant returns -1/ENOENT.
    let mut state = state_with_path(b"/link");
    state
        .file_system()
        .add_symlink("/link".to_string(), b"/tmp/never-registered".to_vec());
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let args = |flag: u64| {
        [
            RustBV::concrete(TEST_AT_FDCWD as u128, 64),
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(0x4000, 64),
            RustBV::concrete(flag as u128, 64),
        ]
    };

    let followed = NativeNewfstatatSyscall
        .call(&mut state, &args(0))
        .expect("newfstatat ok");
    assert_eq!(expect_continue(followed), NEG_ONE);

    let nofollow = NativeNewfstatatSyscall
        .call(&mut state, &args(AT_SYMLINK_NOFOLLOW))
        .expect("newfstatat ok");
    assert_eq!(expect_continue(nofollow), 0);
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFLNK_0777 as u32);
}

#[test]
fn newfstatat_symbolic_flag_falls_back() {
    // The flag now selects between two different st_mode answers, so an
    // unconstrained flag must defer to Python rather than assume follow.
    let mut state = state_with_path(b"/tmp/x");
    let sym_flag = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "flag", 64)
    };

    let err = NativeNewfstatatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(TEST_AT_FDCWD as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x4000, 64),
                sym_flag,
            ],
        )
        .expect_err("must fall back");
    assert_symbolic_arg(err, "flag");
}
