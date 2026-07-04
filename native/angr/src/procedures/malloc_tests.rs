//! Tests for the malloc SimProcedure (extracted from malloc.rs).
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

#[test]
fn test_malloc_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeMalloc
        .call(&mut state, &[RustBV::concrete(100, 64)])
        .unwrap();
    let addr = result.unwrap().as_u64().unwrap();
    assert!(addr > 0);
    // Verify heap metadata tracking
    assert_eq!(state.heap_metadata().alloc_count(), 1);
    assert!(state.heap_metadata().is_allocated(addr));
    assert_eq!(state.heap_metadata().alloc_size(addr), Some(100));
}

#[test]
fn test_malloc_sequential_non_overlapping() {
    let mut state = RustSimState::new("amd64").unwrap();
    let r1 = NativeMalloc
        .call(&mut state, &[RustBV::concrete(32, 64)])
        .unwrap();
    let r2 = NativeMalloc
        .call(&mut state, &[RustBV::concrete(32, 64)])
        .unwrap();
    let a1 = r1.unwrap().as_u64().unwrap();
    let a2 = r2.unwrap().as_u64().unwrap();
    // Second allocation should be >= first + aligned size
    assert!(a2 >= a1 + 32);
    // Both should be tracked
    assert_eq!(state.heap_metadata().alloc_count(), 2);
}

#[test]
fn test_malloc_symbolic_size() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "size", 64);
    drop(ctx);
    let result = NativeMalloc.call(&mut state, &[sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_free_tracks_metadata() {
    let mut state = RustSimState::new("amd64").unwrap();
    // Allocate then free
    let r = NativeMalloc
        .call(&mut state, &[RustBV::concrete(64, 64)])
        .unwrap();
    let addr = r.unwrap().as_u64().unwrap();
    assert_eq!(state.heap_metadata().alloc_count(), 1);

    let result = NativeFree
        .call(&mut state, &[RustBV::concrete(addr as u128, 64)])
        .unwrap();
    assert!(result.is_none());
    // After free: removed from allocated, added to freed
    assert_eq!(state.heap_metadata().alloc_count(), 0);
    assert_eq!(state.heap_metadata().free_count(), 1);
    assert!(!state.heap_metadata().is_allocated(addr));
}

#[test]
fn test_free_null_no_crash() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeFree
        .call(&mut state, &[RustBV::concrete(0, 64)])
        .unwrap();
    assert!(result.is_none());
    // free(NULL) should not add to freed list
    assert_eq!(state.heap_metadata().free_count(), 0);
}

#[test]
fn test_calloc_zeroed() {
    let mut state = RustSimState::new("amd64").unwrap();
    // Map heap region (heap starts at 0xC0000000)
    state.map_memory(0xC000_0000, 0x10000, Permission::RWX);

    let result = NativeCalloc
        .call(
            &mut state,
            &[RustBV::concrete(4, 64), RustBV::concrete(8, 64)],
        )
        .unwrap();
    let addr = result.unwrap().as_u64().unwrap();
    // Verify zeroed memory
    let val = state.memory_load(addr, 8).unwrap();
    assert_eq!(val.as_u64(), Some(0));
}

#[test]
fn test_calloc_overflow() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeCalloc.call(
        &mut state,
        &[
            RustBV::concrete(u64::MAX as u128, 64),
            RustBV::concrete(2, 64),
        ],
    );
    assert!(result.is_err());
}

#[test]
fn test_calloc_too_large() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeCalloc.call(
        &mut state,
        &[RustBV::concrete(1, 64), RustBV::concrete(2_000_000, 64)],
    );
    assert!(result.is_err());
}

#[test]
fn test_realloc_null_ptr() {
    let mut state = RustSimState::new("amd64").unwrap();
    // realloc(NULL, size) should behave like malloc
    let result = NativeRealloc
        .call(
            &mut state,
            &[RustBV::concrete(0, 64), RustBV::concrete(64, 64)],
        )
        .unwrap();
    let addr = result.unwrap().as_u64().unwrap();
    assert!(addr > 0);
}

#[test]
fn test_realloc_tracks_metadata() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0xC000_0000, 0x10000, Permission::RWX);

    // Allocate initial block
    let r1 = NativeMalloc
        .call(&mut state, &[RustBV::concrete(16, 64)])
        .unwrap();
    let old_addr = r1.unwrap().as_u64().unwrap();
    assert_eq!(state.heap_metadata().alloc_count(), 1);

    // Realloc to larger size
    let r2 = NativeRealloc
        .call(
            &mut state,
            &[
                RustBV::concrete(old_addr as u128, 64),
                RustBV::concrete(32, 64),
            ],
        )
        .unwrap();
    let new_addr = r2.unwrap().as_u64().unwrap();

    // Old allocation should be freed, new one tracked
    assert!(!state.heap_metadata().is_allocated(old_addr));
    assert!(state.heap_metadata().is_allocated(new_addr));
    assert_eq!(state.heap_metadata().alloc_size(new_addr), Some(32));
    assert_eq!(state.heap_metadata().free_count(), 1);
}

#[test]
fn test_heap_metadata_cloned_on_fork() {
    let mut state = RustSimState::new("amd64").unwrap();
    NativeMalloc
        .call(&mut state, &[RustBV::concrete(100, 64)])
        .unwrap();
    assert_eq!(state.heap_metadata().alloc_count(), 1);

    let forked = state.fork();
    assert_eq!(forked.heap_metadata().alloc_count(), 1);
}

#[test]
fn test_realloc_copies_data() {
    let mut state = RustSimState::new("amd64").unwrap();
    // Map heap region
    state.map_memory(0xC000_0000, 0x10000, Permission::RWX);

    // Allocate and write data
    let r1 = NativeMalloc
        .call(&mut state, &[RustBV::concrete(16, 64)])
        .unwrap();
    let old_addr = r1.unwrap().as_u64().unwrap();
    state
        .memory_store(old_addr, RustBV::concrete(0xDEADBEEF, 32))
        .unwrap();

    // Realloc to larger size
    let r2 = NativeRealloc
        .call(
            &mut state,
            &[
                RustBV::concrete(old_addr as u128, 64),
                RustBV::concrete(32, 64),
            ],
        )
        .unwrap();
    let new_addr = r2.unwrap().as_u64().unwrap();

    // Data should be copied
    let val = state.memory_load(new_addr, 4).unwrap();
    assert_eq!(val.as_u64(), Some(0xDEADBEEF));
}

#[test]
fn test_memalign_returns_aligned_addr() {
    let mut state = RustSimState::new("amd64").unwrap();
    // Force the bump cursor to a non-aligned value so the alignment math
    // actually has to round up.
    let _ = NativeMalloc
        .call(&mut state, &[RustBV::concrete(1, 64)])
        .unwrap();
    for &alignment in &[32u64, 64, 128, 4096] {
        let r = NativeMemalign
            .call(
                &mut state,
                &[
                    RustBV::concrete(alignment as u128, 64),
                    RustBV::concrete(48, 64),
                ],
            )
            .unwrap();
        let addr = r.unwrap().as_u64().unwrap();
        assert!(addr > 0);
        assert_eq!(
            addr % alignment,
            0,
            "addr {addr:#x} not aligned to {alignment}"
        );
        assert!(state.heap_metadata().is_allocated(addr));
    }
}

#[test]
fn test_memalign_rejects_non_power_of_two() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeMemalign.call(
        &mut state,
        &[RustBV::concrete(24, 64), RustBV::concrete(16, 64)],
    );
    assert!(result.is_err());
}

#[test]
fn test_memalign_symbolic_size_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "size", 64);
    drop(ctx);
    let result = NativeMemalign.call(&mut state, &[RustBV::concrete(32, 64), sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_posix_memalign_writes_pointer_and_returns_zero() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0xC000_0000, 0x10000, Permission::RWX);
    // memptr lives in a writable region — reuse the heap region with a
    // 16-byte allocation as scratch space for the void** out-parameter.
    let scratch = NativeMalloc
        .call(&mut state, &[RustBV::concrete(16, 64)])
        .unwrap()
        .unwrap()
        .as_u64()
        .unwrap();

    let r = NativePosixMemalign
        .call(
            &mut state,
            &[
                RustBV::concrete(scratch as u128, 64),
                RustBV::concrete(64, 64),
                RustBV::concrete(100, 64),
            ],
        )
        .unwrap();
    // Return value is the errno int (0 on success).
    assert_eq!(r.unwrap().as_u64(), Some(0));

    // *memptr should hold an aligned heap pointer.
    let stored = state.memory_load(scratch, 8).unwrap();
    let stored_addr = stored.as_u64().unwrap();
    assert!(stored_addr > 0);
    assert_eq!(stored_addr % 64, 0);
    assert!(state.heap_metadata().is_allocated(stored_addr));
}

#[test]
fn test_posix_memalign_rejects_bad_alignment() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0xC000_0000, 0x10000, Permission::RWX);
    let scratch = NativeMalloc
        .call(&mut state, &[RustBV::concrete(16, 64)])
        .unwrap()
        .unwrap()
        .as_u64()
        .unwrap();

    // Alignment < sizeof(void*) (8) — must return EINVAL (22).
    let r = NativePosixMemalign
        .call(
            &mut state,
            &[
                RustBV::concrete(scratch as u128, 64),
                RustBV::concrete(4, 64),
                RustBV::concrete(100, 64),
            ],
        )
        .unwrap();
    assert_eq!(r.unwrap().as_u64(), Some(22));

    // Alignment not power of two — must return EINVAL.
    let r = NativePosixMemalign
        .call(
            &mut state,
            &[
                RustBV::concrete(scratch as u128, 64),
                RustBV::concrete(24, 64),
                RustBV::concrete(100, 64),
            ],
        )
        .unwrap();
    assert_eq!(r.unwrap().as_u64(), Some(22));
}

#[test]
fn test_posix_memalign_symbolic_size_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "size", 64);
    drop(ctx);
    let result = NativePosixMemalign.call(
        &mut state,
        &[
            RustBV::concrete(0xC000_0000, 64),
            RustBV::concrete(64, 64),
            sym,
        ],
    );
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}
