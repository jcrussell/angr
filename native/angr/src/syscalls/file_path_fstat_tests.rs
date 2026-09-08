//! `fstat` tests, including the `statbuf`-near-`u64::MAX` wraparound sweep
//! (angr-03vl4.65).

use super::*;
use super::stat_layouts::{
    S_IFREG_0755, ST_BLKSIZE, STAT_LAYOUT_WRITERS, require_stat_arch, write_stat_for_arch,
};
use super::file_path_tests_support::*;
use crate::memory::Permission;
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;
use crate::syscalls::NativeSyscall;

#[test]
fn fstat_unknown_fd_returns_minus_one() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[RustBV::concrete(99, 64), RustBV::concrete(0x4000, 64)],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), NEG_ONE);
    // Buffer must NOT have been touched on the failure path
    // (read 0s from the freshly-mapped page).
    assert_eq!(read_u64_le(&state, 0x4000), 0);
}

#[test]
fn fstat_known_fd_writes_amd64_layout_and_returns_zero() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    // Seed a file with concrete content so content_len = 13.
    let fd = state
        .file_system()
        .open_with_content(
            "/tmp/hello".into(),
            FdFlags::ReadOnly,
            b"hello, world!".to_vec(),
        )
        .expect("fd space is not exhausted in tests");

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x4000, 64),
            ],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), 0);

    // st_size at offset 0x30
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 13);
    // st_mode at offset 0x18 (u32)
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
    // st_blksize at offset 0x38
    assert_eq!(read_u64_le(&state, 0x4000 + 0x38), ST_BLKSIZE);
    // st_dev at offset 0 — zero
    assert_eq!(read_u64_le(&state, 0x4000), 0);
    // st_ctimensec at 0x70 — zero
    assert_eq!(read_u64_le(&state, 0x4000 + 0x70), 0);
}

/// angr-0xyq2 Phase 1: an fd whose bounded symbolic content came from the
/// `file_contents` registry reports the symbolic byte count as st_size
/// (the concrete buffer is empty — `effective_size` must not read it).
#[test]
fn fstat_symbolic_content_fd_reports_symbolic_len() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let bytes: Vec<RustBV> = (0..21u64)
        .map(|i| RustBV::symbolic_with_id(0x6f000 + i, format!("fstat_sym_{i}"), 8))
        .collect();
    state
        .file_system()
        .register_file_content("/tmp/symflag", bytes);
    let fd = state
        .file_system()
        .open("/tmp/symflag".into(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x4000, 64),
            ],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), 0);

    // st_size at offset 0x30 — the symbolic byte count.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 21);
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), S_IFREG_0755 as u32);
}

#[test]
fn fstat_known_fd_writes_aarch64_layout_and_returns_zero() {
    let mut state = RustSimState::new("aarch64").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let fd = state
        .file_system()
        .open_with_content("/tmp/arm".into(), FdFlags::ReadOnly, vec![0u8; 4096])
        .expect("fd space is not exhausted in tests");

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x4000, 64),
            ],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), 0);

    // AArch64-specific: st_mode at 0x10 (NOT 0x18 like AMD64).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x10), S_IFREG_0755 as u32);
    // st_nlink at 0x14 — zero u32
    assert_eq!(read_u32_le(&state, 0x4000 + 0x14), 0);
    // st_size at 0x30
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 4096);
    // st_blksize at 0x38 is u32 here
    assert_eq!(read_u32_le(&state, 0x4000 + 0x38), ST_BLKSIZE as u32);
    // The last padding word at 0x78 — zero
    assert_eq!(read_u64_le(&state, 0x4000 + 0x78), 0);
}

#[test]
fn fstat_known_fd_writes_i386_layout_and_returns_zero() {
    // i386 fstat64 / struct stat64 layout (`write_i386_stat`), mirroring
    // angr/procedures/linux_kernel/fstat64.py::_store_i386. Offsets differ
    // from AMD64/AArch64: st_mode at 0x10, st_size at 0x2C, st_blksize at
    // 0x34. Buffers are 32-bit on X86 so the args are sized accordingly.
    let mut state = RustSimState::new("x86").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let fd = state
        .file_system()
        .open_with_content(
            "/tmp/i386".into(),
            FdFlags::ReadOnly,
            b"hello, world!".to_vec(),
        )
        .expect("fd space is not exhausted in tests");

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 32),
                RustBV::concrete(0x4000, 32),
            ],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), 0);

    // st_mode at 0x10 (u32) — concrete S_IFREG | 0o755.
    assert_eq!(read_u32_le(&state, 0x4000 + 0x10), S_IFREG_0755 as u32);
    // st_size at 0x2C (u64).
    assert_eq!(read_u64_le(&state, 0x4000 + 0x2C), 13);
    // st_blksize at 0x34 (low word of the 64-bit store).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x34), ST_BLKSIZE as u32);
    // st_dev at 0 — zero.
    assert_eq!(read_u64_le(&state, 0x4000), 0);
    // st_uid at 0x18 — zero (the overlapping st_nlink 64-bit store is 0).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), 0);
}

#[test]
fn fstat_known_fd_writes_arm_layout_and_returns_zero() {
    // ARM (32-bit EABI) fstat64 / struct stat64 layout (`write_arm_stat`),
    // mirroring angr/procedures/linux_kernel/fstat64.py::_store_arm.
    // Offsets differ from i386: st_size at 0x30 (vs 0x2C), st_blksize at
    // 0x38 (vs 0x34), no padding store. Buffers are 32-bit on ARM.
    let mut state = RustSimState::new("armel").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let fd = state
        .file_system()
        .open_with_content(
            "/tmp/arm".into(),
            FdFlags::ReadOnly,
            b"hello, world!".to_vec(),
        )
        .expect("fd space is not exhausted in tests");

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 32),
                RustBV::concrete(0x4000, 32),
            ],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), 0);

    // st_mode at 0x10 (u32) — concrete S_IFREG | 0o755.
    assert_eq!(read_u32_le(&state, 0x4000 + 0x10), S_IFREG_0755 as u32);
    // st_size at 0x30 (u64).
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 13);
    // st_blksize at 0x38 (low word of the 64-bit store).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x38), ST_BLKSIZE as u32);
    // st_dev at 0 — zero.
    assert_eq!(read_u64_le(&state, 0x4000), 0);
    // st_uid at 0x18 — zero (the overlapping st_nlink 64-bit store is 0).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x18), 0);
    // st_ino verification copy at 0x60 — zero.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x60), 0);
}

#[test]
fn fstat_known_fd_writes_mips32_layout_and_returns_zero() {
    // MIPS32 O32 fstat64 / struct stat64 layout (`write_mips32_stat`),
    // mirroring angr/procedures/linux_kernel/fstat64.py::_store_mips32.
    // Notable differences from i386/ARM: st_size at 0x30, st_blksize at
    // 0x50, and NO st_mode field is written (angr's MIPS layout omits it).
    // "mipsel" gives a little-endian MIPS32 so the read_u*_le helpers apply.
    let mut state = RustSimState::new("mipsel").expect("state");
    assert_eq!(state.arch().name(), "MIPS32");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    let fd = state
        .file_system()
        .open_with_content(
            "/tmp/mips".into(),
            FdFlags::ReadOnly,
            b"hello, world!".to_vec(),
        )
        .expect("fd space is not exhausted in tests");

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 32),
                RustBV::concrete(0x4000, 32),
            ],
        )
        .expect("fstat ok");
    assert_eq!(expect_continue(out), 0);

    // st_size at 0x30 (u64).
    assert_eq!(read_u64_le(&state, 0x4000 + 0x30), 13);
    // st_blksize at 0x50 (low word of the 64-bit store).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x50), ST_BLKSIZE as u32);
    // st_dev at 0 — zero.
    assert_eq!(read_u64_le(&state, 0x4000), 0);
    // st_ino at 0x10 — zero.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x10), 0);
    // st_blocks at 0x58 — zero.
    assert_eq!(read_u64_le(&state, 0x4000 + 0x58), 0);
    // MIPS layout omits st_mode entirely — the byte where i386/ARM put it
    // (0x10) is covered by the st_ino store and stays zero (NOT S_IFREG).
    assert_eq!(read_u32_le(&state, 0x4000 + 0x10), 0);
}

#[test]
fn fstat_unsupported_arch_falls_back() {
    // MIPS64 (legacy 64-bit struct stat, `_store_mips64` in fstat.py) has
    // no native Rust writer, so the handler intentionally errors out and
    // lets the dispatcher fall back to the Python proc. X86 (angr-11djq.5.1),
    // ARM (angr-11djq.5.2) and MIPS32 (angr-11djq.5.3, via `write_mips32_stat`)
    // ARE supported now, so they are excluded here.
    {
        let arch = "mips64";
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        let fd = state
            .file_system()
            .open("/tmp/foo".into(), FdFlags::ReadOnly)
            .expect("fd space is not exhausted in tests");

        let err = NativeFstatSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(fd as u128, bits),
                    RustBV::concrete(0x4000, bits),
                ],
            )
            .expect_err("{arch}: must surface as Other");
        match err {
            SyscallError::Other(msg) => {
                assert!(msg.contains("unsupported arch"), "{arch}: got {msg:?}",)
            }
            other => panic!("{arch}: expected Other, got {other:?}"),
        }
    }
}

#[test]
fn fstat_symbolic_fd_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_fd = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "fd", 64)
    };
    let err = NativeFstatSyscall
        .call(&mut state, &[sym_fd, RustBV::concrete(0x4000, 64)])
        .expect_err("must fall back");
    assert_symbolic_arg(err, "fd");
}

#[test]
fn fstat_symbolic_buf_falls_back() {
    let mut state = RustSimState::new("amd64").expect("state");
    let sym_buf = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "statbuf", 64)
    };
    let err = NativeFstatSyscall
        .call(&mut state, &[RustBV::concrete(0, 64), sym_buf])
        .expect_err("must fall back");
    assert_symbolic_arg(err, "statbuf");
}

#[test]
fn fstat_unmapped_buf_surfaces_error() {
    let mut state = RustSimState::new("amd64").expect("state");
    let fd = state
        .file_system()
        .open("/tmp/x".into(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");
    // Do NOT map the destination page — store should error.
    let err = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x8000, 64),
            ],
        )
        .expect_err("unmapped should error");
    // The MemoryError surfaces as SyscallError via the `?` conversion.
    assert_memory_err(err);
}
/// `statbuf` is unchecked `extract_concrete_arg` output, so `buf + off` in
/// the `store_stat_*` writers can overflow. `buf = u64::MAX - 7` keeps every
/// 8-byte-aligned field within one page (offset 0 in the top page, the rest
/// wrapped into page 0), so the whole amd64 layout is written and must wrap
/// rather than panic under CI's `release-checked` (overflow-checks = true)
/// profile (angr-03vl4.65).
#[test]
fn fstat_statbuf_near_u64_max_wraps_field_offsets() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0xFFFF_FFFF_FFFF_F000, 0x1000, Permission::RW);
    state.map_memory(0, 0x1000, Permission::RW);

    let fd = state
        .file_system()
        .open_with_content(
            "/tmp/hello".into(),
            FdFlags::ReadOnly,
            b"hello, world!".to_vec(),
        )
        .expect("fd space is not exhausted in tests");
    let buf = u64::MAX - 7;

    let out = NativeFstatSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(buf as u128, 64),
            ],
        )
        .expect("must not panic on a wrapping statbuf");
    assert_eq!(expect_continue(out), 0);

    // st_dev at offset 0 lands in the top page; every later field wraps.
    assert_eq!(read_u64_le(&state, buf), 0);
    // st_size at 0x30 -> wrapped address 0x28.
    assert_eq!(read_u64_le(&state, buf.wrapping_add(0x30)), 13);
    // st_mode at 0x18 (u32) -> wrapped address 0x10.
    assert_eq!(
        read_u32_le(&state, buf.wrapping_add(0x18)),
        S_IFREG_0755 as u32
    );
}

/// Harness 6 boundary sweep for the fix above: every value in the shared
/// `test_boundary_values` table, not just the single `u64::MAX - 7` pivot,
/// must complete without panicking, and — with the touched pages mapped —
/// every `store_stat_*` field must land at exactly `buf.wrapping_add(off)`.
#[test]
fn fstat_statbuf_boundary_sweep_never_panics_and_wraps_field_offsets() {
    // Set when some sweep value's st_size or st_mode store genuinely
    // wrapped past u64::MAX (the wrapped address is less than buf), not
    // merely landed unwrapped near the top or was rejected outright — see
    // the boundary_addresses doc comment for why the table needs dedicated
    // entries to ever hit this.
    let mut saw_genuine_wrap = false;
    for &buf in &crate::test_boundary_values::boundary_addresses() {
        let mut state = RustSimState::new("amd64").expect("state");
        // The amd64 layout (`write_amd64_stat`) spans offsets 0x00..0x90;
        // map every page a sample of those wrapped addresses could land on
        // (start/mid/near-end is enough to straddle any wraparound point).
        for off in [0x00u64, 0x30, 0x48, 0x88] {
            let page = buf.wrapping_add(off) & !0xFFFu64;
            if state.memory().page_permissions(page >> 12).is_none() {
                state.map_memory(page, 0x1000, Permission::RW);
            }
        }

        let fd = state
            .file_system()
            .open_with_content(
                "/tmp/hello".into(),
                FdFlags::ReadOnly,
                b"hello, world!".to_vec(),
            )
            .expect("fd space is not exhausted in tests");

        let out = NativeFstatSyscall.call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(buf as u128, 64),
            ],
        );
        // A legitimately unreachable neighbor page is fine (this sweep maps
        // only a start/mid/end sample, not every field's page) — the
        // property under test is "never panics", not "always succeeds", so
        // only the `Ok` side gets a further assertion.
        if let Ok(outcome) = out {
            assert_eq!(expect_continue(outcome), 0, "buf={buf:#x}");
            let size_addr = buf.wrapping_add(0x30);
            let mode_addr = buf.wrapping_add(0x18);
            // st_size (offset 0x30, u64) must land at the wrapped address.
            assert_eq!(
                read_u64_le(&state, size_addr),
                13,
                "buf={buf:#x}: st_size must land at the wrapped offset"
            );
            // st_mode (offset 0x18, u32) likewise.
            assert_eq!(
                read_u32_le(&state, mode_addr),
                S_IFREG_0755 as u32,
                "buf={buf:#x}: st_mode must land at the wrapped offset"
            );
            if size_addr < buf || mode_addr < buf {
                saw_genuine_wrap = true;
            }
        }
    }
    assert!(
        saw_genuine_wrap,
        "boundary sweep never exercised a genuine st_size/st_mode wraparound \
         (wrapped addr < buf) — the table may have regressed to only far-from-top \
         (no field wraps) or right-at-top (first field already overflows) values"
    );
}

/// `require_stat_arch` (the handlers' pre-flight guard) and
/// `write_stat_for_arch` (the dispatcher) both read `STAT_LAYOUT_WRITERS`,
/// so an arch added to that table reaches all four handlers at once —
/// the drift angr-6cp06.53 removed. Pins the agreement so a future
/// rewrite that re-hand-rolls either side reds here.
#[test]
fn require_stat_arch_agrees_with_the_writer_table() {
    let mut state = RustSimState::new("amd64").expect("state");
    state.map_memory(0x4000, 0x1000, Permission::RW);

    for (arch_name, _) in STAT_LAYOUT_WRITERS {
        require_stat_arch("fstat", arch_name, true)
            .unwrap_or_else(|e| panic!("{arch_name} is in the table but the guard rejects it: {e}"));
        write_stat_for_arch(&mut state, arch_name, 0x4000, 13, S_IFREG_0755 as u32)
            .unwrap_or_else(|e| panic!("{arch_name} is in the table but has no writer: {e}"));
    }

    // ARM64 is the one arch the legacy-`stat`-number handlers exclude by
    // design; every other table entry stays allowed with `allow_arm64 = false`.
    assert!(require_stat_arch("stat", "ARM64", false).is_err());
    for (arch_name, _) in STAT_LAYOUT_WRITERS {
        if arch_name != "ARM64" {
            assert!(require_stat_arch("stat", arch_name, false).is_ok());
        }
    }

    // Off-table arch: rejected by both, and the guard's message names the
    // allowed set rather than a hand-copied literal.
    let err = require_stat_arch("lstat", "PPC32", false).expect_err("PPC32 has no layout");
    let msg = format!("{err}");
    assert!(msg.contains("lstat: unsupported arch PPC32"), "{msg}");
    assert!(msg.contains("AMD64/X86/ARM/MIPS32"), "{msg}");
    assert!(!msg.contains("ARM64"), "{msg}");
    assert!(write_stat_for_arch(&mut state, "PPC32", 0x4000, 13, S_IFREG_0755 as u32).is_err());
}
