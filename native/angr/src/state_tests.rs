// angr-c2cv: RustSimState unit tests, extracted out of the former in-file
// `mod tests` (~590 lines) into a sibling file to shrink state.rs below the
// god-object threshold. Declared as a direct child of the crate-root state
// module so `use super::*` reaches the module's private items.

use super::*;

#[test]
fn test_state_creation() {
    let state = RustSimState::new("amd64").unwrap();
    assert_eq!(state.vex_arch(), VexArch::AMD64);
    assert_eq!(state.pc(), 0);
}

#[test]
fn test_state_fork() {
    let mut state1 = RustSimState::new("amd64").unwrap();
    state1.set_pc(0x1000);
    state1.set_register("rax", RustBV::concrete(42, 64));

    let state2 = state1.fork();

    // Both should have same values
    assert_eq!(state2.pc(), 0x1000);
    assert_eq!(state2.get_register("rax").unwrap().as_u64(), Some(42));

    // Different state IDs
    assert_ne!(state1.state_id(), state2.state_id());

    // state2's parent should be state1
    assert_eq!(state2.parent_id(), Some(state1.state_id()));
}

#[test]
fn test_set_detailed_history_honors_cap() {
    // set_detailed_history (called once per step from interpreter results)
    // must drain the oldest entries when the incoming buffer exceeds the
    // configured cap. Otherwise long blocks bypass max_history.
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(3);
    let entries: Vec<HistoryEntry> = (0..10)
        .map(|i| HistoryEntry {
            addr: 0x1000 + i,
            jumpkind: 0,
            jump_target: 0,
        })
        .collect();
    state.set_detailed_history(entries);
    let kept = state.detailed_history();
    assert_eq!(kept.len(), 3);
    // FIFO eviction: should retain the most-recent 3 entries.
    assert_eq!(kept[0].addr, 0x1007);
    assert_eq!(kept[2].addr, 0x1009);
}

#[test]
fn test_set_max_history_trims_retroactively() {
    // Lowering max_history on a state that already exceeds the new cap
    // must FIFO-evict the oldest entries down to the cap immediately.
    // Without this, add_to_history (which only removes one entry per
    // push when over cap) never converges and the buffer stays bloated.
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(0); // unlimited
    for i in 0..20u64 {
        state.add_to_history(0x3000 + i);
        state.add_history_entry(0x3000 + i, 0, 0);
    }
    assert_eq!(state.history().len(), 20);
    assert_eq!(state.detailed_history().len(), 20);

    // Retroactively cap to 4 — both buffers shrink to the most-recent 4.
    state.set_max_history(4);
    let kept = state.history();
    assert_eq!(kept.len(), 4);
    assert_eq!(kept, &[0x3010, 0x3011, 0x3012, 0x3013]);
    let kept_detailed = state.detailed_history();
    assert_eq!(kept_detailed.len(), 4);
    assert_eq!(kept_detailed[0].addr, 0x3010);
    assert_eq!(kept_detailed[3].addr, 0x3013);
}

#[test]
fn test_set_max_history_zero_no_trim() {
    // Switching to max=0 (unlimited) must NOT trim — existing entries stay.
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(5);
    for i in 0..5u64 {
        state.add_to_history(0x4000 + i);
    }
    state.set_max_history(0);
    assert_eq!(state.history().len(), 5);
}

#[test]
fn test_set_detailed_history_unlimited() {
    // max_history = 0 means no cap (legacy behavior).
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(0);
    let entries: Vec<HistoryEntry> = (0..50)
        .map(|i| HistoryEntry {
            addr: 0x2000 + i,
            jumpkind: 0,
            jump_target: 0,
        })
        .collect();
    state.set_detailed_history(entries);
    assert_eq!(state.detailed_history().len(), 50);
}

#[test]
fn test_state_memory() {
    let mut state = RustSimState::new("amd64").unwrap();

    // Map and write
    state.map_memory(0x1000, 0x1000, Permission::RWX);
    state
        .memory_store(0x1000, RustBV::concrete(0xDEADBEEF, 32))
        .unwrap();

    // Read back
    let val = state.memory_load(0x1000, 4).unwrap();
    assert_eq!(val.as_u64(), Some(0xDEADBEEF));
}

#[test]
fn test_state_fork_memory_cow() {
    let mut state1 = RustSimState::new("amd64").unwrap();
    state1.map_memory(0x1000, 0x1000, Permission::RWX);
    state1
        .memory_store(0x1000, RustBV::concrete(0xAAAA, 16))
        .unwrap();

    let mut state2 = state1.fork();

    // Modify state2
    state2
        .memory_store(0x1000, RustBV::concrete(0xBBBB, 16))
        .unwrap();

    // state1 should still have original value
    let val1 = state1.memory_load(0x1000, 2).unwrap();
    assert_eq!(val1.as_u64(), Some(0xAAAA));

    // state2 should have new value
    let val2 = state2.memory_load(0x1000, 2).unwrap();
    assert_eq!(val2.as_u64(), Some(0xBBBB));
}

#[test]
fn test_filesystem_default() {
    let fs = FileSystem::default();
    assert!(fs.is_open(0)); // stdin
    assert!(fs.is_open(1)); // stdout
    assert!(fs.is_open(2)); // stderr
    assert!(!fs.is_open(3));
    assert_eq!(fs.next_fd(), 3);
}

#[test]
fn test_filesystem_open_close() {
    let mut fs = FileSystem::default();
    let fd = fs.open("test.txt".to_string(), FdFlags::ReadOnly);
    assert_eq!(fd, 3);
    assert!(fs.is_open(3));

    let closed = fs.close(3);
    assert!(closed);
    assert!(!fs.is_open(3));

    // Double close returns false
    assert!(!fs.close(3));
}

#[test]
fn test_filesystem_write_read() {
    let mut fs = FileSystem::default();
    let fd = fs.open_with_content(
        "data.bin".to_string(),
        FdFlags::ReadOnly,
        b"hello world".to_vec(),
    );

    let data = fs.read(fd, 5);
    assert_eq!(data, b"hello");

    let data2 = fs.read(fd, 6);
    assert_eq!(data2, b" world");

    // Read past end
    let data3 = fs.read(fd, 10);
    assert!(data3.is_empty());
}

#[test]
fn test_filesystem_seek() {
    let mut fs = FileSystem::default();
    let fd = fs.open_with_content("data.bin".to_string(), FdFlags::ReadOnly, vec![0u8; 100]);

    // SEEK_SET
    assert_eq!(fs.seek(fd, 50, 0), Some(50));
    // SEEK_CUR
    assert_eq!(fs.seek(fd, 10, 1), Some(60));
    // SEEK_END
    assert_eq!(fs.seek(fd, -5, 2), Some(95));
    // Invalid whence
    assert_eq!(fs.seek(fd, 0, 99), None);
}

#[test]
fn test_filesystem_write_position_aware() {
    let mut fs = FileSystem::default();
    let fd = fs.open("out.bin".to_string(), FdFlags::WriteOnly);

    // Sequential writes (no seek): position-aware path is byte-identical to a
    // plain append, and advances the position to EOF each time.
    fs.write(fd, b"abc");
    fs.write(fd, b"def");
    assert_eq!(fs.fd_content(fd), b"abcdef");

    // Seek back and overwrite in place (the case append-only would corrupt).
    assert_eq!(fs.seek(fd, 0, 0), Some(0));
    fs.write(fd, b"XY");
    assert_eq!(fs.fd_content(fd), b"XYcdef");

    // A subsequent write continues from the advanced position (after "XY").
    fs.write(fd, b"Z");
    assert_eq!(fs.fd_content(fd), b"XYZdef");

    // Sparse seek past EOF zero-fills the gap, like write_at/pwrite.
    assert_eq!(fs.seek(fd, 8, 0), Some(8));
    fs.write(fd, b"!");
    assert_eq!(fs.fd_content(fd), b"XYZdef\x00\x00!");
}

#[test]
fn test_filesystem_fork_isolation() {
    let mut state = RustSimState::new("amd64").unwrap();
    state
        .file_system()
        .open("test.txt".to_string(), FdFlags::ReadOnly);
    assert!(state.file_system_ref().is_open(3));

    let mut forked = state.fork();
    // Close in forked state
    forked.file_system().close(3);
    assert!(!forked.file_system_ref().is_open(3));
    // Original should still be open
    assert!(state.file_system_ref().is_open(3));
}

#[test]
fn test_filesystem_backward_compat() {
    // fd_buffer/write_fd should still work through FileSystem
    let mut state = RustSimState::new("amd64").unwrap();
    state.write_stdout(b"hello");
    assert_eq!(state.stdout_buffer(), b"hello");
    assert!(state.has_stdout());

    state.write_fd(2, b"err");
    assert_eq!(state.fd_buffer(2), b"err");
}

#[test]
fn test_filesystem_open_fds() {
    let mut fs = FileSystem::default();
    let open = fs.open_fds();
    assert_eq!(open, vec![0, 1, 2]); // stdin, stdout, stderr

    fs.open("a.txt".to_string(), FdFlags::ReadOnly);
    fs.open("b.txt".to_string(), FdFlags::WriteOnly);
    let open = fs.open_fds();
    assert_eq!(open, vec![0, 1, 2, 3, 4]);

    fs.close(3);
    let open = fs.open_fds();
    assert_eq!(open, vec![0, 1, 2, 4]);
}

#[test]
fn test_filesystem_dup_dup2_pipe() {
    let mut fs = FileSystem::default();
    let fd = fs.open("a.txt".to_string(), FdFlags::ReadOnly);
    assert_eq!(fd, 3);

    // dup
    let dup_fd = fs.dup(fd).unwrap();
    assert_eq!(dup_fd, 4);
    assert_eq!(fs.fd_info(dup_fd).unwrap().0, "a.txt");

    // dup of closed fd returns None
    fs.close(fd);
    assert!(fs.dup(fd).is_none());

    // dup2 with fresh state
    let mut fs2 = FileSystem::default();
    let src = fs2.open("src.txt".to_string(), FdFlags::ReadOnly);
    let dst = fs2.dup2(src, 10).unwrap();
    assert_eq!(dst, 10);
    assert_eq!(fs2.fd_info(10).unwrap().0, "src.txt");
    // next_fd advanced past 10
    assert!(fs2.next_fd() > 10);

    // dup2(self, self) returns self when open
    assert_eq!(fs2.dup2(src, src), Some(src));

    // pipe allocates two consecutive fds
    let mut fs3 = FileSystem::default();
    let (r, w) = fs3.pipe();
    assert_eq!((r, w), (3, 4));
    assert!(fs3.is_open(r));
    assert!(fs3.is_open(w));
    assert_eq!(fs3.fd_info(r).unwrap().2, 0); // ReadOnly
    assert_eq!(fs3.fd_info(w).unwrap().2, 1); // WriteOnly
}

#[test]
fn test_inspection_default_inactive() {
    let mgr = InspectionManager::default();
    assert!(!mgr.is_active());
    assert!(!mgr.is_enabled(InspectEvent::MemRead));
}

#[test]
fn test_inspection_enable_disable() {
    let mut mgr = InspectionManager::default();
    mgr.enable(InspectEvent::MemRead);
    assert!(mgr.is_active());
    assert!(mgr.is_enabled(InspectEvent::MemRead));
    assert!(!mgr.is_enabled(InspectEvent::MemWrite));

    mgr.enable_all();
    assert!(mgr.is_enabled(InspectEvent::Fork));
    assert!(mgr.is_enabled(InspectEvent::Exit));

    mgr.disable(InspectEvent::MemRead);
    assert!(!mgr.is_enabled(InspectEvent::MemRead));
    assert!(mgr.is_enabled(InspectEvent::MemWrite));

    mgr.disable_all();
    assert!(!mgr.is_active());
}

#[test]
fn test_inspection_record_events() {
    let mut mgr = InspectionManager::default();
    mgr.enable(InspectEvent::MemWrite);

    mgr.record(InspectEvent::MemWrite, 0x1000, 4, 0x400000);
    mgr.record(InspectEvent::MemWrite, 0x1004, 8, 0x400010);

    assert_eq!(mgr.events().len(), 2);
    assert_eq!(mgr.event_counts()[InspectEvent::MemWrite as usize], 2);

    let e = &mgr.events()[0];
    assert_eq!(e.event, InspectEvent::MemWrite);
    assert_eq!(e.addr, 0x1000);
    assert_eq!(e.size, 4);
    assert_eq!(e.block_addr, 0x400000);
}

#[test]
fn test_inspection_ring_buffer() {
    let mut mgr = InspectionManager::default();
    mgr.set_max_events(3);
    mgr.enable(InspectEvent::MemRead);

    for i in 0..5 {
        mgr.record(InspectEvent::MemRead, i * 0x100, 4, 0);
    }

    // Only last 3 should remain
    assert_eq!(mgr.events().len(), 3);
    assert_eq!(mgr.events()[0].addr, 0x200);
    assert_eq!(mgr.events()[2].addr, 0x400);
    // But total count should be 5
    assert_eq!(mgr.event_counts()[InspectEvent::MemRead as usize], 5);
}

#[test]
fn test_inspection_on_state() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.inspection_mut().enable(InspectEvent::MemWrite);
    state.inspection_mut().enable(InspectEvent::MemRead);

    state.set_pc(0x400000);
    state.inspect_mem_write(0x1000, 8);
    state.inspect_mem_read(0x2000, 4);

    assert_eq!(
        state.inspection().event_counts()[InspectEvent::MemWrite as usize],
        1
    );
    assert_eq!(
        state.inspection().event_counts()[InspectEvent::MemRead as usize],
        1
    );
}

#[test]
fn test_inspection_fork_isolation() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.inspection_mut().enable(InspectEvent::MemWrite);
    state.inspect_mem_write(0x1000, 4);

    let mut forked = state.fork();
    forked.inspect_mem_write(0x2000, 4);

    // Parent should have 1 event
    assert_eq!(
        state.inspection().event_counts()[InspectEvent::MemWrite as usize],
        1
    );
    // Forked should have 2 (inherited 1 + new 1)
    assert_eq!(
        forked.inspection().event_counts()[InspectEvent::MemWrite as usize],
        2
    );
}

#[test]
fn test_inspection_disabled_no_record() {
    let mut state = RustSimState::new("amd64").unwrap();
    // Don't enable anything
    state.inspect_mem_write(0x1000, 4);
    state.inspect_mem_read(0x2000, 4);

    assert_eq!(state.inspection().events().len(), 0);
    assert_eq!(
        state.inspection().event_counts()[InspectEvent::MemWrite as usize],
        0
    );
}

// =========================================================================
// Snapshot / Serialization tests (angr-x04s.1.3)
// =========================================================================

/// Build a representative RustSimState that touches each bucket A/B/C
/// field (concrete + symbolic registers, mapped memory pages, solver
/// constraints, history, call stack, file system, hooks, environment,
/// flags). The Z3 ASTs are minted inside whichever Z3 context the test
/// is currently running under.
#[cfg(feature = "vex-engine-z3")]
fn build_populated_state() -> RustSimState {
    let mut s = RustSimState::new("amd64").unwrap();
    s.set_pc(0x4012a0);

    // Bucket B (registers): one concrete, one symbolic.
    s.set_register("rax", RustBV::concrete(0xdead_beef, 64));
    let rbx_sym = {
        let ctx = s.solver().borrow();
        RustBV::symbolic(&ctx, "rbx_sym", 64)
    };
    s.set_register("rbx", rbx_sym.clone());

    // Bucket B (memory): map a page with concrete bytes.
    s.memory_mut()
        .map_data(0x10_0000u64, &[1u8, 2, 3, 4, 5], Permission::RWX);

    // Bucket A: history + call stack.
    s.add_to_history(0x4011a0);
    s.add_to_history(0x4012a0);
    s.push_call(0x4012a0, 0x401400, 0x4012a5, 0x7fff_ffff_0000);
    // Heap metadata via the public heap_alloc/heap_free helpers.
    let a1 = s.heap_alloc(32);
    let _a2 = s.heap_alloc(64);
    let _ = s.heap_free(a1);

    // Bucket C: hook + environment + stdin symbols.
    s.add_hook(0x401200);
    s.setenv(b"PATH".to_vec(), b"/usr/bin".to_vec());
    s.setenv(b"HOME".to_vec(), b"/root".to_vec());
    s.record_stdin_symbol("stdin_chunk_0".to_string(), 16);

    // Flags.
    s.set_no_ip_concretization(true);
    s.set_keep_ip_symbolic(false);
    s.set_no_symbolic_jump_resolution(true);
    s.set_posix_brk(0x1B0_4000);
    s.set_mmap_base(0xC100_8000);

    // Solver constraints — `rbx > 10` must hold after restore.
    let cmp = {
        let ctx = s.solver().borrow();
        let ten = RustBV::concrete(10, 64);
        rbx_sym.ugt(&ten, &ctx)
    };
    s.solver().borrow().assume_true(&cmp);

    s
}

/// Round-trip assertions: bucket A scalars + bucket B field counts +
/// bucket C collection contents must match. Solver SAT + concretize
/// proves the assumed_constraints were faithfully replayed (the
/// SymContextSnapshot path is already covered separately, this just
/// confirms the wiring through RustSimStateSnapshot).
#[cfg(feature = "vex-engine-z3")]
fn assert_state_round_trip(orig: &RustSimState, restored: &RustSimState) {
    // Bucket A scalars.
    assert_eq!(restored.pc(), orig.pc());
    assert_eq!(restored.state_id(), orig.state_id());
    assert_eq!(restored.parent_id(), orig.parent_id());
    assert_eq!(restored.history().to_vec(), orig.history().to_vec());
    assert_eq!(
        restored.detailed_history().len(),
        orig.detailed_history().len()
    );
    assert_eq!(restored.heap_brk(), orig.heap_brk());
    assert_eq!(restored.posix_brk(), orig.posix_brk());
    assert_eq!(restored.mmap_base(), orig.mmap_base());
    assert_eq!(restored.no_ip_concretization(), orig.no_ip_concretization());
    assert_eq!(restored.keep_ip_symbolic(), orig.keep_ip_symbolic());
    assert_eq!(
        restored.no_symbolic_jump_resolution(),
        orig.no_symbolic_jump_resolution()
    );
    assert_eq!(restored.call_stack().len(), orig.call_stack().len());
    assert_eq!(restored.vex_arch(), orig.vex_arch());
    assert_eq!(restored.arch().name(), orig.arch().name());

    // Bucket A subset: heap metadata.
    assert_eq!(
        restored.heap_metadata().alloc_count(),
        orig.heap_metadata().alloc_count()
    );
    assert_eq!(
        restored.heap_metadata().free_count(),
        orig.heap_metadata().free_count()
    );

    // Bucket A: stdin symbols.
    assert_eq!(restored.stdin_symbols(), orig.stdin_symbols());

    // Bucket B: registers — concrete rax and symbolic rbx width.
    assert_eq!(
        restored.get_register("rax").and_then(|bv| bv.as_u64()),
        Some(0xdead_beef)
    );
    let restored_rbx = restored.get_register("rbx").expect("rbx present");
    assert_eq!(restored_rbx.width(), 64);

    // Bucket B: memory — first concrete bytes survived.
    let restored_bytes = restored
        .memory()
        .read_concrete_bytes_for_lift(crate::memory::Address::new(0x10_0000), 5)
        .expect("memory readable");
    assert_eq!(restored_bytes, vec![1, 2, 3, 4, 5]);

    // Bucket C: hooks + env.
    assert!(restored.is_hooked(0x401200));
    assert_eq!(
        restored.getenv(b"PATH").map(|v| v.to_vec()),
        Some(b"/usr/bin".to_vec())
    );
    assert_eq!(
        restored.getenv(b"HOME").map(|v| v.to_vec()),
        Some(b"/root".to_vec())
    );

    // Solver replayed — restored context must be SAT with the rbx > 10
    // constraint honored.
    assert!(restored.solver().borrow().is_sat());
    let restored_rbx_val = restored
        .solver()
        .borrow()
        .eval(&restored_rbx)
        .expect("rbx evaluable");
    assert!(
        restored_rbx_val > 10,
        "constraint rbx > 10 not honored after restore (got {restored_rbx_val})"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_state_snapshot_round_trip_buckets_a_b_c() {
    let orig = build_populated_state();
    let snap = orig.to_snapshot();
    let restored = RustSimState::from_snapshot(snap).expect("from_snapshot");
    assert_state_round_trip(&orig, &restored);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_state_to_from_serialized_round_trip() {
    let orig = build_populated_state();
    let bytes = orig.to_serialized();
    assert_eq!(
        bytes[0], SNAPSHOT_VERSION,
        "envelope must carry version byte"
    );
    let restored = RustSimState::from_serialized(&bytes).expect("from_serialized");
    assert_state_round_trip(&orig, &restored);
}

#[test]
fn test_state_from_serialized_empty_envelope() {
    match RustSimState::from_serialized(&[]) {
        Err(SnapshotError::EmptyEnvelope) => {}
        Err(other) => panic!("expected EmptyEnvelope, got {other:?}"),
        Ok(_) => panic!("empty envelope must fail"),
    }
}

#[test]
fn test_state_from_serialized_version_mismatch() {
    // Bump-byte trick: build a real envelope, replace version byte, expect
    // a fast VersionMismatch.
    let orig = RustSimState::new("amd64").unwrap();
    let mut bytes = orig.to_serialized();
    bytes[0] = SNAPSHOT_VERSION.wrapping_add(1);
    match RustSimState::from_serialized(&bytes) {
        Err(SnapshotError::VersionMismatch { found, expected }) => {
            assert_eq!(expected, SNAPSHOT_VERSION);
            assert_eq!(found, SNAPSHOT_VERSION.wrapping_add(1));
        }
        Err(other) => panic!("expected VersionMismatch, got {other:?}"),
        Ok(_) => panic!("bad version must fail"),
    }
}
