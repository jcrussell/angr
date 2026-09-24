// Unit tests for the code-invalidation arithmetic in icicle.rs.
//
// `invalidate_code_range` mixes two range conventions: lifted code groups cover
// an *inclusive* `[start, end]` byte range, while a write covers a *half-open*
// `[addr, addr + size)` range. An off-by-one either leaves stale JIT code live
// (under-invalidation) or drops the cached disassembly of an untouched
// instruction. Building an `icicle_vm::Vm` needs a sleigh processors_path, so
// the predicates are tested directly.

// The `Hitmap` / enum-conversion tests below are the angr-ph300.5 smoke layer:
// everything in icicle.rs that can be exercised without a sleigh install.

use super::{
    ExceptionCode, Hitmap, VmExit, checked_read_len, code_group_overlaps, disasm_addr_retained,
    written_range,
};

/// Convenience: does a group `[gs, ge]` get invalidated by a write of `size` at `addr`?
fn invalidated(gs: u64, ge: u64, addr: u64, size: u64) -> bool {
    match written_range(addr, size) {
        None => false,
        Some((a, end)) => code_group_overlaps(gs, ge, a, end),
    }
}

#[test]
fn empty_write_invalidates_nothing() {
    assert_eq!(written_range(0x1000, 0), None);
    // Even a write landing squarely inside the group is a no-op at size 0.
    assert!(!invalidated(0x1000, 0x1010, 0x1004, 0));
}

#[test]
fn written_range_is_half_open() {
    assert_eq!(written_range(0x1000, 1), Some((0x1000, 0x1001)));
    assert_eq!(written_range(0x1000, 0x10), Some((0x1000, 0x1010)));
}

#[test]
fn written_range_saturates_at_top_of_address_space() {
    // Would wrap to 0 with a plain add, making the range empty and the write
    // invisible to every group.
    assert_eq!(written_range(u64::MAX, 8), Some((u64::MAX, u64::MAX)));
    assert_eq!(
        written_range(u64::MAX - 3, 8),
        Some((u64::MAX - 3, u64::MAX))
    );
    // The group holding the final byte must still be invalidated.
    assert!(invalidated(u64::MAX - 8, u64::MAX, u64::MAX - 3, 8));
}

#[test]
fn write_strictly_before_group_is_ignored() {
    // Write ends exactly at the group start -> touches no group byte.
    assert!(!invalidated(0x1000, 0x1010, 0x0ff0, 0x10));
    assert!(!invalidated(0x1000, 0x1010, 0x0f00, 4));
}

#[test]
fn write_ending_one_past_group_start_hits_it() {
    // Last written byte is 0x1000, the group's first byte.
    assert!(invalidated(0x1000, 0x1010, 0x0ff0, 0x11));
}

#[test]
fn write_at_group_end_hits_it_because_group_end_is_inclusive() {
    // 0x1010 is the group's last byte, not one-past.
    assert!(invalidated(0x1000, 0x1010, 0x1010, 1));
    // One byte further is outside.
    assert!(!invalidated(0x1000, 0x1010, 0x1011, 1));
}

#[test]
fn write_starting_at_group_start_hits_it() {
    assert!(invalidated(0x1000, 0x1010, 0x1000, 1));
}

#[test]
fn fully_contained_and_fully_containing_writes_hit() {
    // Write strictly inside the group.
    assert!(invalidated(0x1000, 0x1010, 0x1004, 4));
    // Write swallowing the whole group.
    assert!(invalidated(0x1000, 0x1010, 0x0f00, 0x400));
    // Write exactly covering the group's inclusive extent.
    assert!(invalidated(0x1000, 0x1010, 0x1000, 0x11));
}

#[test]
fn straddling_writes_hit_from_either_side() {
    assert!(invalidated(0x1000, 0x1010, 0x0ffc, 0x8)); // overlaps the front
    assert!(invalidated(0x1000, 0x1010, 0x100c, 0x8)); // overlaps the back
}

#[test]
fn single_byte_group_boundaries() {
    // Degenerate group covering exactly one byte (start == end, inclusive).
    assert!(invalidated(0x2000, 0x2000, 0x2000, 1));
    assert!(!invalidated(0x2000, 0x2000, 0x1fff, 1));
    assert!(!invalidated(0x2000, 0x2000, 0x2001, 1));
}

#[test]
fn disasm_retain_window_is_half_open() {
    let (addr, end) = written_range(0x1000, 0x10).unwrap();
    // The first written byte's disassembly is stale.
    assert!(!disasm_addr_retained(0x1000, addr, end));
    assert!(!disasm_addr_retained(0x100f, addr, end));
    // One-past-the-end is untouched and must survive.
    assert!(disasm_addr_retained(0x1010, addr, end));
    assert!(disasm_addr_retained(0x0fff, addr, end));
}

#[test]
fn disasm_retain_drops_only_the_written_byte_for_a_one_byte_write() {
    let (addr, end) = written_range(0x1000, 1).unwrap();
    assert!(!disasm_addr_retained(0x1000, addr, end));
    assert!(disasm_addr_retained(0x1001, addr, end));
    assert!(disasm_addr_retained(0x0fff, addr, end));
}

#[test]
fn disasm_retain_at_top_of_address_space() {
    let (addr, end) = written_range(u64::MAX - 1, 4).unwrap();
    assert_eq!(end, u64::MAX);
    assert!(!disasm_addr_retained(u64::MAX - 1, addr, end));
    // Saturation makes `end == u64::MAX`, so the final byte's entry is kept
    // even though the write nominally reaches it. Documented, not desired:
    // an instruction cached at u64::MAX cannot be executed anyway.
    assert!(disasm_addr_retained(u64::MAX, addr, end));
}

// ---------------------------------------------------------------------------
// Hitmap — the edge-count buffer handed to the VM as a raw pointer
// ---------------------------------------------------------------------------

#[test]
fn hitmap_starts_zeroed_at_the_requested_length() {
    let hm = Hitmap::new(64);
    assert_eq!(hm.len(), 64);
    assert_eq!(hm.as_slice().len(), 64);
    assert!(hm.as_slice().iter().all(|&b| b == 0));
}

#[test]
fn hitmap_raw_pointer_aliases_the_slice() {
    // The VM writes edge counts through `as_mut_ptr()`; the Python-visible
    // reads go through `as_slice()`. They must be the same buffer, and the
    // pinned box must not move when the map is written through.
    let mut hm = Hitmap::new(8);
    let ptr = hm.as_mut_ptr();
    // SAFETY: `ptr` points at the pinned 8-byte buffer we just allocated, and
    // `hm` outlives the write.
    unsafe {
        *ptr.add(3) = 0xAB;
    }
    assert_eq!(hm.as_slice()[3], 0xAB);
    assert_eq!(hm.as_mut_ptr(), ptr, "pinned buffer must not relocate");

    hm.as_slice_mut()[7] = 0xCD;
    assert_eq!(hm.as_slice()[7], 0xCD);
}

#[test]
fn hitmap_of_zero_length_is_allowed() {
    let hm = Hitmap::new(0);
    assert_eq!(hm.len(), 0);
    assert!(hm.as_slice().is_empty());
}

// ---------------------------------------------------------------------------
// Enum bridging — icicle's types -> the Python-visible pyclass enums
// ---------------------------------------------------------------------------

#[test]
fn vm_exit_bridges_every_icicle_variant_distinctly() {
    use icicle_vm::VmExit as Src;
    let pairs = [
        (Src::Running, VmExit::Running),
        (Src::InstructionLimit, VmExit::InstructionLimit),
        (Src::Breakpoint, VmExit::Breakpoint),
        (Src::Interrupted, VmExit::Interrupted),
        (Src::Halt, VmExit::Halt),
        (Src::Killed, VmExit::Killed),
        (Src::Deadlock, VmExit::Deadlock),
        (Src::OutOfMemory, VmExit::OutOfMemory),
        (Src::Unimplemented, VmExit::Unimplemented),
    ];
    for (src, expected) in pairs {
        let got: VmExit = src.into();
        assert!(got.__eq__(&expected), "{src:?} bridged to {got:?}");
    }
    // `UnhandledException` carries a payload that the Python enum drops; the
    // code itself is read back via `get_exception_code()`.
    assert!(!VmExit::Halt.__eq__(&VmExit::Killed));
}

#[test]
fn exception_code_from_code_matches_the_from_impl() {
    // `from_code` is the u32 route Python takes; it must agree with the typed
    // `From<icicle_vm::cpu::ExceptionCode>` conversion for every code icicle
    // round-trips, and must not panic on an out-of-range one.
    use icicle_vm::cpu::ExceptionCode as Src;
    for code in 0u32..64 {
        let src = Src::from_u32(code);
        let expected: ExceptionCode = src.into();
        assert!(
            ExceptionCode::from_code(code).__eq__(&expected),
            "code {code} disagreed with the From impl"
        );
    }
    assert!(!ExceptionCode::Halt.__eq__(&ExceptionCode::None));
}

// `checked_read_len` — the pre-allocation guard on `Icicle::mem_read`'s
// Python-supplied `size`. Every rejection here is one the old code would have
// turned into an allocation attempt, i.e. a process abort under `panic =
// "abort"` rather than a catchable PyRuntimeError.

/// Stand-in for `Icicle::max_mem_read_bytes()` (50_000 pages * 4 KiB by
/// default); the exact value is the caller's business, the guard only compares.
const TEST_MAX: u64 = 50_000 * 4096;

#[test]
fn read_len_accepts_an_ordinary_request() {
    assert_eq!(checked_read_len(0x1000, 2, TEST_MAX), Ok(2));
    assert_eq!(checked_read_len(0x1000, 4096, TEST_MAX), Ok(4096));
    // Exactly at the ceiling is still served.
    assert_eq!(
        checked_read_len(0x1000, TEST_MAX, TEST_MAX),
        Ok(TEST_MAX as usize)
    );
}

#[test]
fn read_len_of_zero_is_an_empty_buffer_not_an_error() {
    // Unlike `written_range`, a zero-length read is a legal no-op that must
    // hand `read_bytes` an empty slice rather than be refused.
    assert_eq!(checked_read_len(0x1000, 0, TEST_MAX), Ok(0));
    // ...even at the very top of the address space, where `addr + size` is
    // still in range.
    assert_eq!(checked_read_len(u64::MAX, 0, TEST_MAX), Ok(0));
}

#[test]
fn read_len_rejects_a_range_that_wraps_the_address_space() {
    let err = checked_read_len(u64::MAX - 3, 8, TEST_MAX).unwrap_err();
    assert!(err.contains("wraps"), "{err}");
    // The wrap check runs before the size check, so a wrapping *small* read is
    // caught too -- `read_bytes` would otherwise walk `addr.wrapping_add(1)`
    // back around to 0 and read unrelated memory.
    let err = checked_read_len(u64::MAX, 2, TEST_MAX).unwrap_err();
    assert!(err.contains("wraps"), "{err}");
}

#[test]
fn read_len_rejects_a_size_beyond_the_memory_capacity() {
    let err = checked_read_len(0x1000, TEST_MAX + 1, TEST_MAX).unwrap_err();
    assert!(err.contains("exceeds"), "{err}");
    // The motivating case: a garbage size from Python. `u64::MAX` at addr 0
    // does not wrap, so the capacity check is the one that must catch it.
    let err = checked_read_len(0, u64::MAX, TEST_MAX).unwrap_err();
    assert!(err.contains("exceeds"), "{err}");
    // And a merely large one, well under `usize::MAX` but far past anything
    // the mmu could have mapped.
    let err = checked_read_len(0x1000, 1 << 40, TEST_MAX).unwrap_err();
    assert!(err.contains("exceeds"), "{err}");
}

#[test]
fn read_len_error_names_the_request() {
    // The message has to carry addr+size: it replaces the `ffi_result` context
    // the caller would otherwise have produced.
    let err = checked_read_len(0xdead_beef, u64::MAX, TEST_MAX).unwrap_err();
    assert!(err.contains("0xdeadbeef"), "{err}");
    assert!(err.contains(&u64::MAX.to_string()), "{err}");
}

#[test]
fn read_len_with_a_zero_capacity_refuses_every_nonempty_read() {
    // `max_mem_read_bytes` saturates, so a pathological mmu config degrades to
    // "nothing is readable" rather than to an unbounded allocation.
    assert_eq!(checked_read_len(0x1000, 0, 0), Ok(0));
    assert!(checked_read_len(0x1000, 1, 0).is_err());
}
