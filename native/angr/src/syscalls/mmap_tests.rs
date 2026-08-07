// angr-fxyi: mmap/mmap2 syscall handler unit tests, extracted out of the
// former in-file `mod tests` (~671 lines) into a sibling file to shrink
// syscalls/mmap.rs below the god-object threshold. Declared as a direct child
// of `mmap` so `use super::*` reaches the module's private items.

use super::*;
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

const DEFAULT_MMAP_BASE: u64 = 0xC100_0000;
const ANON_FD: u64 = 0xFFFF_FFFF;
const ANON_PRIVATE: u64 = MAP_ANONYMOUS | MAP_PRIVATE;
const ANON_SHARED: u64 = MAP_ANONYMOUS | MAP_SHARED;

fn fresh_state() -> RustSimState {
    RustSimState::new("amd64").expect("amd64 state")
}

fn args(addr: u64, length: u64, prot: u64, flags: u64, fd: u64, offset: u64) -> Vec<RustBV> {
    vec![
        RustBV::concrete(addr as u128, 64),
        RustBV::concrete(length as u128, 64),
        RustBV::concrete(prot as u128, 64),
        RustBV::concrete(flags as u128, 64),
        RustBV::concrete(fd as u128, 64),
        RustBV::concrete(offset as u128, 64),
    ]
}

#[test]
fn default_mmap_base_is_0xc1000000() {
    let state = fresh_state();
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
}

#[test]
fn anonymous_addr_zero_allocates_from_mmap_base() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let outcome = h
        .call(
            &mut state,
            &args(0, 0x1000, 0x3 /* RW */, ANON_PRIVATE, ANON_FD, 0),
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_MMAP_BASE),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE + 0x1000);
    assert_eq!(
        state.memory().page_permissions(DEFAULT_MMAP_BASE >> 12),
        Some(Permission::RW),
    );
}

#[test]
fn anon_unaligned_length_bumps_to_next_page() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    // 0x800 bytes — not page-aligned. Python's allocate_memory
    // bumps mmap_base by size, then aligns up to next page.
    h.call(&mut state, &args(0, 0x800, 0x3, ANON_PRIVATE, ANON_FD, 0))
        .expect("ok");
    // First call: allocated at DEFAULT_MMAP_BASE, returned that.
    // mmap_base should now be DEFAULT_MMAP_BASE + 0x1000 (aligned up).
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE + 0x1000);
    // The second call should land at the next page.
    let outcome = h
        .call(&mut state, &args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0))
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => {
            assert_eq!(ret, DEFAULT_MMAP_BASE + 0x1000);
        }
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE + 0x2000);
}

#[test]
fn anon_concrete_addr_succeeds_when_range_unmapped() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let target = 0x4000_0000;
    let outcome = h
        .call(
            &mut state,
            &args(target, 0x2000, 0x5 /* RX */, ANON_PRIVATE, ANON_FD, 0),
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, target),
        _ => panic!("expected Continue"),
    }
    // mmap_base does NOT advance for an explicit-addr request.
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
    assert_eq!(
        state.memory().page_permissions(target >> 12),
        Some(Permission::RX),
    );
    assert_eq!(
        state.memory().page_permissions((target >> 12) + 1),
        Some(Permission::RX),
    );
}

#[test]
fn map_shared_anonymous_also_works() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let outcome = h
        .call(&mut state, &args(0, 0x1000, 0x3, ANON_SHARED, ANON_FD, 0))
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_MMAP_BASE),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn bad_flags_returns_neg_one_no_shared_no_private() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    // flags has neither MAP_SHARED nor MAP_PRIVATE → Python returns -1.
    let outcome = h
        .call(&mut state, &args(0, 0x1000, 0x3, MAP_ANONYMOUS, ANON_FD, 0))
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX),
        _ => panic!("expected Continue"),
    }
    // No region mapped, mmap_base unchanged.
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
    assert!(
        state
            .memory()
            .page_permissions(DEFAULT_MMAP_BASE >> 12)
            .is_none()
    );
}

#[test]
fn bad_flags_returns_neg_one_both_shared_and_private() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let outcome = h
        .call(
            &mut state,
            &args(
                0,
                0x1000,
                0x3,
                MAP_ANONYMOUS | MAP_SHARED | MAP_PRIVATE,
                ANON_FD,
                0,
            ),
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn file_backed_falls_back() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    // fd = 3 (a real file descriptor) + no MAP_ANONYMOUS → Python.
    let err = h
        .call(&mut state, &args(0, 0x1000, 0x3, MAP_PRIVATE, 3, 0))
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
}

#[test]
fn anonymous_with_real_fd_falls_back() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    // MAP_ANONYMOUS but fd=3 (Linux ignores fd in this case but Python
    // checks `fd[31:0] != -1` for sim_fd lookup; mirror Python's path
    // by falling back so the (rare) sim_fd resolution runs there).
    let err = h
        .call(&mut state, &args(0, 0x1000, 0x3, ANON_PRIVATE, 3, 0))
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn collision_at_explicit_addr_falls_back() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let target = 0x4000_0000;
    // Pre-map one page in the path.
    state.map_memory(target, 0x1000, Permission::RW);

    let err = h
        .call(
            &mut state,
            &args(target, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0),
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
    // Pre-existing mapping intact (no partial mutation).
    assert_eq!(
        state.memory().page_permissions(target >> 12),
        Some(Permission::RW),
    );
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
}

#[test]
fn map_fixed_collision_unmaps_and_remaps_natively() {
    // POSIX MAP_FIXED: collision is not an error — the kernel
    // discards the colliding pages and maps the new range at the
    // requested address. angr-ttr7.
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let target = 0x4000_0000;
    // Pre-map one page R/W.
    state.map_memory(target, 0x1000, Permission::RW);
    assert_eq!(
        state.memory().page_permissions(target >> 12),
        Some(Permission::RW),
    );

    let outcome = h
        .call(
            &mut state,
            // New mapping: R/X (0x5). Verifies that the prior
            // perms (RW) are replaced, not merged.
            &args(target, 0x1000, 0x5, ANON_PRIVATE | MAP_FIXED, ANON_FD, 0),
        )
        .expect("MAP_FIXED collision must succeed natively");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, target),
        _ => panic!("expected Continue, got {outcome:?}"),
    }
    // The page is now RX (the new perms), not RW (the old perms).
    assert_eq!(
        state.memory().page_permissions(target >> 12),
        Some(Permission::RX),
    );
    // mmap_base must NOT advance for an explicit-addr request.
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
}

#[test]
fn map_fixed_multi_page_collision_discards_all_overlapped_pages() {
    // MAP_FIXED with a 3-page request that overlaps a single
    // pre-existing page in the middle. POSIX semantics: every
    // overlapped page is discarded and remapped with the new perms.
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let base = 0x4000_0000;
    // Pre-map only the middle page.
    state.map_memory(base + 0x1000, 0x1000, Permission::RW);

    let outcome = h
        .call(
            &mut state,
            &args(
                base,
                0x3000,
                0x7, /* RWX */
                ANON_PRIVATE | MAP_FIXED,
                ANON_FD,
                0,
            ),
        )
        .expect("MAP_FIXED multi-page collision must succeed natively");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, base),
        _ => panic!("expected Continue, got {outcome:?}"),
    }
    // All three pages now carry the new RWX perms.
    for i in 0..3 {
        assert_eq!(
            state.memory().page_permissions((base >> 12) + i),
            Some(Permission::RWX),
            "page {i} after MAP_FIXED remap",
        );
    }
}

#[test]
fn map_fixed_clean_addr_still_maps_without_unmap_noise() {
    // MAP_FIXED on a clean address: no collision, plain map.
    // Regression guard for the is_fixed shortcut path.
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let target = 0x4000_0000;

    let outcome = h
        .call(
            &mut state,
            &args(target, 0x1000, 0x3, ANON_PRIVATE | MAP_FIXED, ANON_FD, 0),
        )
        .expect("MAP_FIXED no-collision must succeed");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, target),
        _ => panic!("expected Continue"),
    }
    assert_eq!(
        state.memory().page_permissions(target >> 12),
        Some(Permission::RW),
    );
}

#[test]
fn map_fixed_misaligned_addr_returns_einval_and_maps_nothing() {
    // angr-sqfj8.110: MAP_FIXED must land at exactly `addr`, but
    // Memory::map is page-granular and would floor the request to its
    // containing page — mapping a range the caller never named. Linux
    // returns EINVAL; so do we, and nothing is mapped.
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let target = 0x4000_0123;

    let outcome = h
        .call(
            &mut state,
            &args(target, 0x1000, 0x3, ANON_PRIVATE | MAP_FIXED, ANON_FD, 0),
        )
        .expect("misaligned MAP_FIXED must complete natively, not fall back");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX, "expected -1 (EINVAL)"),
        _ => panic!("expected Continue"),
    }
    // Neither the containing page nor the next one may be mapped.
    assert_eq!(state.memory().page_permissions(target >> 12), None);
    assert_eq!(state.memory().page_permissions((target >> 12) + 1), None);
}

#[test]
fn map_fixed_misaligned_addr_leaves_existing_mapping_intact() {
    // The rejection must happen *before* the unmap+remap, otherwise a
    // bad request would still destroy a live mapping.
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let page = 0x4000_0000;
    state.memory_mut().map(page, 0x1000, Permission::RWX);

    let outcome = h
        .call(
            &mut state,
            &args(
                page + 0x800,
                0x1000,
                0x3,
                ANON_PRIVATE | MAP_FIXED,
                ANON_FD,
                0,
            ),
        )
        .expect("misaligned MAP_FIXED must complete natively, not fall back");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX, "expected -1 (EINVAL)"),
        _ => panic!("expected Continue"),
    }
    assert_eq!(
        state.memory().page_permissions(page >> 12),
        Some(Permission::RWX),
        "pre-existing mapping must survive a rejected MAP_FIXED",
    );
}

#[test]
fn misaligned_addr_without_map_fixed_is_a_hint_and_still_maps() {
    // Contrast with the two tests above: without MAP_FIXED the addr is
    // only a hint, so the page-granular round-down stays the behavior
    // (and matches procedures/posix/mmap.py).
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let target = 0x4000_0123;

    let outcome = h
        .call(
            &mut state,
            &args(target, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0),
        )
        .expect("misaligned hint addr must still map");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, target),
        _ => panic!("expected Continue"),
    }
    assert_eq!(
        state.memory().page_permissions(target >> 12),
        Some(Permission::RW),
    );
}

#[test]
fn map_fixed_with_addr_zero_does_not_engage_fixed_path() {
    // `is_fixed` requires addr != 0 — MAP_FIXED with addr=0 is a
    // nonsensical combination (Linux treats it as a portable
    // suggestion, mapping wherever convenient). Our native path
    // routes through the normal allocate_from_mmap_base flow:
    // if mmap_base is clean, it succeeds without unmapping.
    let h = NativeMmapSyscall;
    let mut state = fresh_state();

    let outcome = h
        .call(
            &mut state,
            &args(0, 0x1000, 0x3, ANON_PRIVATE | MAP_FIXED, ANON_FD, 0),
        )
        .expect("MAP_FIXED with addr=0 falls through to allocate");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_MMAP_BASE),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE + 0x1000);
}

#[test]
fn collision_at_mmap_base_falls_back() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    // Pre-map at the default mmap_base — pathological but possible
    // if a binary explicitly mapped at 0xC1000000.
    state.map_memory(DEFAULT_MMAP_BASE, 0x1000, Permission::RW);

    let err = h
        .call(&mut state, &args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0))
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
    // mmap_base must NOT advance on fallback.
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
}

#[test]
fn symbolic_addr_falls_back() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let mut a = args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0);
    a[0] = RustBV::symbolic(&ctx, "addr", 64);
    let err = h.call(&mut state, &a).expect_err("must fall back");
    match err {
        SyscallError::SymbolicArgument(msg) => {
            assert!(msg.contains("addr"), "got {msg:?}")
        }
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
}

#[test]
fn symbolic_length_falls_back() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let mut a = args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0);
    a[1] = RustBV::symbolic(&ctx, "length", 64);
    let err = h.call(&mut state, &a).expect_err("must fall back");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

#[test]
fn symbolic_flags_falls_back() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let mut a = args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0);
    a[3] = RustBV::symbolic(&ctx, "flags", 64);
    let err = h.call(&mut state, &a).expect_err("must fall back");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

#[test]
fn prot_zero_maps_no_access_pages() {
    // Python passes `prot[2:0]` to map_region, so prot=0 produces a
    // page with no read/write/execute. Mirror that.
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let outcome = h
        .call(&mut state, &args(0, 0x1000, 0, ANON_PRIVATE, ANON_FD, 0))
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { .. }));
    assert_eq!(
        state.memory().page_permissions(DEFAULT_MMAP_BASE >> 12),
        Some(Permission::NONE),
    );
}

#[test]
fn prot_high_bits_ignored() {
    // mmap.py extracts prot[2:0] (low 3 bits) before mapping. We do
    // `prot & 0x7` to match.
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    // prot = 0x107 → keeps low 3 bits = 0x7 (RWX).
    let outcome = h
        .call(
            &mut state,
            &args(0, 0x1000, 0x107, ANON_PRIVATE, ANON_FD, 0),
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { .. }));
    assert_eq!(
        state.memory().page_permissions(DEFAULT_MMAP_BASE >> 12),
        Some(Permission::RWX),
    );
}

#[test]
fn length_zero_returns_candidate_no_mapping() {
    let h = NativeMmapSyscall;
    let mut state = fresh_state();
    let outcome = h
        .call(&mut state, &args(0, 0, 0x3, ANON_PRIVATE, ANON_FD, 0))
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_MMAP_BASE),
        _ => panic!("expected Continue"),
    }
    // mmap_base unchanged for size=0 (matches python's allocate_memory
    // which would have left it where it is after aligning new_base=base).
    // Our impl skips set_mmap_base in the size==0 short-circuit.
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE);
    assert!(
        state
            .memory()
            .page_permissions(DEFAULT_MMAP_BASE >> 12)
            .is_none()
    );
}

#[test]
fn handler_metadata() {
    let h = NativeMmapSyscall;
    assert_eq!(h.name(), "mmap");
    assert_eq!(h.num_args(), 6);
}

#[test]
fn fork_preserves_mmap_base() {
    let mut state = fresh_state();
    state.set_mmap_base(0xC100_5000);
    let forked = state.fork();
    assert_eq!(forked.mmap_base(), 0xC100_5000);
    // Mutating parent does not affect fork.
    state.set_mmap_base(0xC100_9000);
    assert_eq!(forked.mmap_base(), 0xC100_5000);
}

// --- Legacy mmap (struct-arg) tests -----------------------------------

/// Write six little-endian 32-bit fields into memory at `ptr`,
/// matching the layout `mmap_arg_struct` reads
/// (`[addr, length, prot, flags, fd, offset]`).
fn write_struct_le(state: &mut RustSimState, ptr: u64, fields: [u32; 6]) {
    for (i, f) in fields.iter().enumerate() {
        state
            .memory_store(ptr + (i as u64) * 4, RustBV::concrete(*f as u128, 32))
            .expect("store");
    }
}

#[test]
fn old_mmap_dispatches_anonymous_struct_call() {
    let h = NativeOldMmapSyscall;
    let mut state = fresh_state();
    let ptr: u64 = 0x4000;
    state.map_memory(ptr, 0x1000, Permission::RW);
    // Anonymous private, RW perms, addr=0 (kernel chooses).
    write_struct_le(
        &mut state,
        ptr,
        [
            0,      // addr
            0x1000, // length
            0x3,    // prot RW
            ANON_PRIVATE as u32,
            ANON_FD as u32,
            0, // offset
        ],
    );

    let outcome = h
        .call(&mut state, &[RustBV::concrete(ptr as u128, 64)])
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_MMAP_BASE),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE + 0x1000);
    assert_eq!(
        state.memory().page_permissions(DEFAULT_MMAP_BASE >> 12),
        Some(Permission::RW),
    );
}

#[test]
fn old_mmap_bad_flags_returns_neg_one() {
    let h = NativeOldMmapSyscall;
    let mut state = fresh_state();
    let ptr: u64 = 0x4000;
    state.map_memory(ptr, 0x1000, Permission::RW);
    // MAP_ANONYMOUS only (no SHARED/PRIVATE) — Python returns -1.
    write_struct_le(
        &mut state,
        ptr,
        [0, 0x1000, 0x3, MAP_ANONYMOUS as u32, ANON_FD as u32, 0],
    );
    let outcome = h
        .call(&mut state, &[RustBV::concrete(ptr as u128, 64)])
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn old_mmap_file_backed_falls_back() {
    let h = NativeOldMmapSyscall;
    let mut state = fresh_state();
    let ptr: u64 = 0x4000;
    state.map_memory(ptr, 0x1000, Permission::RW);
    // fd=3 + no MAP_ANONYMOUS → Python.
    write_struct_le(&mut state, ptr, [0, 0x1000, 0x3, MAP_PRIVATE as u32, 3, 0]);
    let err = h
        .call(&mut state, &[RustBV::concrete(ptr as u128, 64)])
        .expect_err("fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn old_mmap_symbolic_field_falls_back() {
    let h = NativeOldMmapSyscall;
    let mut state = fresh_state();
    let ptr: u64 = 0x4000;
    state.map_memory(ptr, 0x1000, Permission::RW);
    // Initial concrete struct, then poison the `length` field with a
    // symbolic BV so the field read errors with SymbolicArgument.
    write_struct_le(
        &mut state,
        ptr,
        [0, 0x1000, 0x3, ANON_PRIVATE as u32, ANON_FD as u32, 0],
    );
    let ctx = SymContext::new();
    state
        .memory_store(ptr + 4, RustBV::symbolic(&ctx, "len", 32))
        .expect("store sym");
    let err = h
        .call(&mut state, &[RustBV::concrete(ptr as u128, 64)])
        .expect_err("fall back");
    match err {
        SyscallError::SymbolicArgument(msg) => {
            assert!(msg.contains("field 1"), "got {msg:?}")
        }
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
}

#[test]
fn old_mmap_handler_metadata() {
    let h = NativeOldMmapSyscall;
    assert_eq!(h.name(), "old_mmap");
    assert_eq!(h.num_args(), 1);
}

// --- mmap2 (page-offset 6-reg) tests ----------------------------------

#[test]
fn mmap2_dispatches_with_zero_offset() {
    let h = NativeMmap2Syscall;
    let mut state = fresh_state();
    let outcome = h
        .call(&mut state, &args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0))
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, DEFAULT_MMAP_BASE),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.mmap_base(), DEFAULT_MMAP_BASE + 0x1000);
}

#[test]
fn mmap2_offset_is_in_page_units() {
    // Anonymous mappings ignore offset, but the multiplication must
    // not overflow or otherwise crash for a non-zero page offset.
    // Page offset 0x20 → byte offset 0x20_000.
    let h = NativeMmap2Syscall;
    let mut state = fresh_state();
    let outcome = h
        .call(
            &mut state,
            &args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0x20),
        )
        .expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { .. }));
}

#[test]
fn mmap2_bad_flags_returns_neg_one() {
    let h = NativeMmap2Syscall;
    let mut state = fresh_state();
    let outcome = h
        .call(&mut state, &args(0, 0x1000, 0x3, MAP_ANONYMOUS, ANON_FD, 0))
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, u64::MAX),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn mmap2_file_backed_falls_back() {
    let h = NativeMmap2Syscall;
    let mut state = fresh_state();
    let err = h
        .call(&mut state, &args(0, 0x1000, 0x3, MAP_PRIVATE, 3, 0))
        .expect_err("fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn mmap2_symbolic_offset_falls_back() {
    let h = NativeMmap2Syscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let mut a = args(0, 0x1000, 0x3, ANON_PRIVATE, ANON_FD, 0);
    a[5] = RustBV::symbolic(&ctx, "offset", 64);
    let err = h.call(&mut state, &a).expect_err("fall back");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

#[test]
fn mmap2_handler_metadata() {
    let h = NativeMmap2Syscall;
    assert_eq!(h.name(), "mmap2");
    assert_eq!(h.num_args(), 6);
}
