//! Snapshot round-trips: the full `to_snapshot`/`from_snapshot` field-by-field
//! contract (`assert_state_round_trip`), foreign state-id reservation on
//! restore, and the serialized-envelope form including its empty and
//! version-mismatch error paths; and the fields a deferred-fork sibling
//! takes from its `BranchSnapshot` (`fork_from_snapshot`).
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;
use super::helpers::build_populated_state;

/// Compare two serializable values through `serde_json::Value`: its objects
/// are `BTreeMap`s, so a `HashMap` field compares independent of iteration
/// order. Used for the field types that carry no `PartialEq`.
fn json<T: serde::Serialize>(v: &T) -> serde_json::Value {
    serde_json::to_value(v).expect("snapshot field serializes")
}

/// [`build_populated_state`] plus a non-default value in every snapshotted
/// field it leaves at its default, so a field that restores to a serde or
/// constructor default instead of its value fails
/// [`assert_state_round_trip`] (angr-vmi2l.202).
fn build_round_trip_state() -> RustSimState {
    let mut s = build_populated_state();
    s.detailed_history = [
        HistoryEntry {
            addr: 0x4011a0,
            jumpkind: HistoryEntry::JK_CALL,
            jump_target: 0x4012a0,
        },
        HistoryEntry {
            addr: 0x4012a0,
            jumpkind: HistoryEntry::JK_SYSCALL,
            jump_target: 0x4012a8,
        },
    ]
    .into_iter()
    .collect();
    s.max_history = 77;
    // PATH keeps no guest address, so both `EnvVar::addr` arms round-trip.
    s.set_env_addr(b"HOME", 0x7fff_0100);
    s.set_last_time(RustBV::symbol("sys_time_snapshot", 64));
    s.track_history = !s.track_history;
    s.block_count = 9;
    s.evicted_visits = Arc::new([(0x401000, 3)].into_iter().collect());
    s.removed_hooks = Arc::new([0x401300].into_iter().collect());
    s.concretizer.read_range_limit = 7;
    s.concretizer.enable_stride_detection = !s.concretizer.enable_stride_detection;
    s.concretizer.symbolic_write_addresses = !s.concretizer.symbolic_write_addresses;
    // fs: an fd with content at a non-zero offset, and a demoted path.
    let fd = s
        .fs
        .open_with_content("/tmp/f".to_string(), FdFlags::ReadWrite, b"hello".to_vec())
        .expect("fd space is not exhausted in tests");
    assert_eq!(s.fs.seek(fd, 2, 0), Some(2));
    s.fs
        .register_file_content("/tmp/d", super::helpers::sym_file_bytes(4, "rt"));
    let dfd = s
        .fs
        .open("/tmp/d".to_string(), FdFlags::ReadWrite)
        .expect("fd space is not exhausted in tests");
    assert!(s.fs.demote_symbolic_content(dfd));
    assert!(!s.fs.demoted_paths().is_empty());
    s.process_ids = ProcessIds { pid: 42, ppid: 41 };
    s.pending_syscall = Some(angr_vex::JumpKind::Sys_syscall);
    s.exit_code = Some(RustBV::concrete(3, 32));
    s.exit_pending = true;
    s.ctype_loc = CtypeLocPtrs {
        b: Some(0x603000),
        tolower: Some(0x603008),
        toupper: Some(0x603010),
    };
    s.stdin_read_lengths.push(StdinReadLength {
        first_leaf: "stdin_chunk_0".to_string(),
        leaves: 2,
        seeded: 1,
        length: Some(("stdin_len_0".to_string(), 64)),
    });
    s.stdin_host_packets.push(StdinHostPacket {
        after_leaves: 1,
        data: ("stdin_pkt_0".to_string(), 32),
        size: None,
    });
    s.removed_env_keys = Arc::new([b"OLDPWD".to_vec()].into_iter().collect());
    s.force_eager_forks = true;
    s.cgc_allocation_base = 0x4347_c000;
    s.cgc_sinkholes = vec![(0x1000, 0x2000)];
    s.sim_options = Arc::new(["SYMBOLIC_WRITE_ADDRESSES".to_string()].into_iter().collect());
    s.removed_sim_options = Arc::new(["LAZY_SOLVES".to_string()].into_iter().collect());
    s
}

/// The `to_snapshot`/`from_snapshot` contract, one field at a time. `orig`
/// is destructured with no `..`, so a new `RustSimState` field fails to
/// compile here until it is asserted (angr-vmi2l.202). Types with no
/// `PartialEq` compare through [`json`]. Memory and the solver are spot
/// checks: `SymbolicMemory` serializes only through a flush proof, and the
/// solver's replay is proved by SAT plus a constrained eval.
fn assert_state_round_trip(orig: &RustSimState, restored: &RustSimState) {
    let RustSimState {
        arch,
        vex_arch,
        registers,
        memory: _,
        solver: _,
        pc,
        state_id,
        parent_id,
        history,
        detailed_history,
        max_history,
        block_count,
        evicted_visits,
        hooks,
        removed_hooks,
        concretizer,
        track_history,
        fs,
        heap_brk,
        posix_brk,
        mmap_base,
        process_ids,
        pending_syscall,
        dirtied_code_pages,
        tracked_variables,
        exit_code,
        exit_pending,
        getopt_cursor,
        getopt_extern,
        native_resume_stack,
        ctype_loc,
        stdin_symbols,
        stdin_read_lengths,
        stdin_host_packets,
        call_stack,
        heap_metadata,
        environment,
        removed_env_keys,
        symbolic_pages: _,
        hook_symbolic_memory: _,
        addr_to_ast: _,
        last_time,
        no_ip_concretization,
        no_symbolic_jump_resolution,
        keep_ip_symbolic,
        force_eager_forks,
        cgc_allocation_base,
        cgc_sinkholes,
        sim_options,
        removed_sim_options,
    } = orig;

    // Scalars and `PartialEq` collections.
    assert_eq!(restored.arch.name(), arch.name());
    assert_eq!(restored.vex_arch, *vex_arch);
    assert_eq!(restored.pc, *pc);
    assert_eq!(restored.state_id, *state_id);
    assert_eq!(restored.parent_id, *parent_id);
    assert_eq!(restored.history, *history);
    assert_eq!(restored.max_history, *max_history);
    assert_eq!(restored.block_count, *block_count);
    assert_eq!(restored.evicted_visits, *evicted_visits);
    assert_eq!(restored.hooks, *hooks);
    assert!(restored.is_hooked(0x401200));
    assert_eq!(restored.removed_hooks, *removed_hooks);
    assert_eq!(restored.concretizer, *concretizer);
    assert_eq!(restored.track_history, *track_history);
    assert_eq!(restored.heap_brk, *heap_brk);
    assert_eq!(restored.posix_brk, *posix_brk);
    assert_eq!(restored.mmap_base, *mmap_base);
    assert_eq!(restored.process_ids, *process_ids);
    assert_eq!(restored.pending_syscall, *pending_syscall);
    assert!(!dirtied_code_pages.is_empty());
    assert_eq!(restored.dirtied_code_pages, *dirtied_code_pages);
    assert!(!tracked_variables.is_empty());
    assert_eq!(restored.tracked_variables, *tracked_variables);
    assert_eq!(restored.exit_pending, *exit_pending);
    assert_eq!(restored.getopt_cursor, *getopt_cursor);
    assert_eq!(restored.getopt_extern, *getopt_extern);
    assert_eq!(restored.ctype_loc, *ctype_loc);
    assert_eq!(restored.stdin_symbols, *stdin_symbols);
    assert_eq!(restored.stdin_read_lengths, *stdin_read_lengths);
    assert_eq!(restored.stdin_host_packets, *stdin_host_packets);
    assert_eq!(restored.environment, *environment);
    assert_eq!(
        restored.getenv(b"PATH").map(<[u8]>::to_vec),
        Some(b"/usr/bin".to_vec())
    );
    assert_eq!(restored.removed_env_keys, *removed_env_keys);
    assert_eq!(restored.no_ip_concretization, *no_ip_concretization);
    assert_eq!(
        restored.no_symbolic_jump_resolution,
        *no_symbolic_jump_resolution
    );
    assert_eq!(restored.keep_ip_symbolic, *keep_ip_symbolic);
    assert_eq!(restored.force_eager_forks, *force_eager_forks);
    assert_eq!(restored.cgc_allocation_base, *cgc_allocation_base);
    assert_eq!(restored.cgc_sinkholes, *cgc_sinkholes);
    assert_eq!(restored.sim_options, *sim_options);
    assert_eq!(restored.removed_sim_options, *removed_sim_options);

    // No `PartialEq`: compare the serialized form.
    assert_eq!(json(&restored.registers), json(registers));
    assert_eq!(
        restored.get_register("rax").and_then(|bv| bv.as_u64()),
        Some(0xdead_beef)
    );
    assert_eq!(json(&restored.detailed_history), json(detailed_history));
    assert_eq!(json(&restored.fs), json(fs));
    assert_eq!(restored.fs.demoted_paths(), fs.demoted_paths());
    assert_eq!(json(&restored.exit_code), json(exit_code));
    assert_eq!(json(&restored.last_time), json(last_time));
    assert_eq!(json(&restored.native_resume_stack), json(native_resume_stack));
    assert_eq!(json(&restored.call_stack), json(call_stack));
    assert_eq!(json(&restored.heap_metadata), json(heap_metadata));

    // Bucket D: not carried by the snapshot, so it restores empty (see
    // `RustSimStateSnapshot`).
    assert!(restored.symbolic_pages.is_empty());
    assert!(restored.hook_symbolic_memory.is_empty());
    assert!(restored.addr_to_ast.is_empty());

    // Memory: the first concrete bytes survived.
    let restored_bytes = restored
        .memory()
        .read_concrete_bytes_for_lift(crate::memory::Address::new(0x10_0000), 5)
        .expect("memory readable");
    assert_eq!(restored_bytes, vec![1, 2, 3, 4, 5]);

    // Solver replayed: the restored context is SAT with `rbx > 10` honored.
    let restored_rbx = restored.get_register("rbx").expect("rbx present");
    assert_eq!(restored_rbx.width(), 64);
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

#[test]
fn test_state_snapshot_round_trip_buckets_a_b_c() {
    let mut orig = build_round_trip_state();
    let snap = orig.to_snapshot();
    let restored = RustSimState::from_snapshot(snap).expect("from_snapshot");
    assert_state_round_trip(&orig, &restored);
}

/// angr-vmi2l.94: a solver residual Z3 rejects surfaces as
/// `SnapshotError::Residual` instead of restoring an under-constrained state.
#[test]
fn test_from_snapshot_rejects_malformed_residual() {
    let mut snap = build_populated_state().to_snapshot();
    snap.solver.residual_smtlib2 = "(assert (bvult undeclared_vmi2l94_state #x01))\n".into();
    assert!(matches!(
        RustSimState::from_snapshot(snap),
        Err(SnapshotError::Residual(_))
    ));
}

/// `state-id-never-reused` across a snapshot boundary: a restored state carries
/// an ID minted by a foreign counter, so `from_snapshot` must lift the local
/// counter above it. Otherwise a resumed manager re-mints IDs that are live in
/// its own stashes (angr-op0dn.13.14).
#[test]
fn test_from_snapshot_reserves_foreign_state_id() {
    let mut snap = build_populated_state().to_snapshot();
    // An ID far above anything this process's counter has issued.
    snap.state_id = u32::MAX as u64;
    let restored = RustSimState::from_snapshot(snap).expect("from_snapshot");
    assert_eq!(restored.state_id(), u32::MAX as u64);

    let fresh = build_populated_state();
    assert!(
        fresh.state_id() > restored.state_id(),
        "counter must be lifted past a restored ID; fresh={} restored={}",
        fresh.state_id(),
        restored.state_id(),
    );
    let forked = restored.fork_true(&RustBV::concrete(1, 1));
    assert!(
        forked.state_id() > restored.state_id(),
        "fork of a restored state must not re-mint a live ID; child={} parent={}",
        forked.state_id(),
        restored.state_id(),
    );
}

#[test]
fn test_state_to_from_serialized_round_trip() {
    let mut orig = build_round_trip_state();
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
    let mut orig = RustSimState::new("amd64").unwrap();
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

/// Which fields a deferred-fork sibling takes from its branch point. The
/// interpreter's `BranchSnapshot` carries the solver, registers, memory, call
/// stack and detailed history; every other field comes from the state
/// `fork_from_snapshot` is called on (the pre-syscall base, angr-vmi2l.195).
/// The base is mutated after the snapshot in each carried field, so a sibling
/// that read one from `self` would fail here (angr-vmi2l.202).
#[test]
fn test_fork_from_snapshot_takes_branch_point_fields() {
    let mut base = build_round_trip_state();
    let rbx = base.get_register("rbx").expect("rbx present");
    let branch_history: Vec<HistoryEntry> = base.detailed_history.iter().cloned().collect();
    let branch_calls = base.call_stack.len();
    let snapshot = |base: &RustSimState, with_memory: bool| BranchSnapshot {
        solver: base.solver.borrow().fork(),
        registers: base.registers.fork(),
        memory: with_memory.then(|| base.memory.fork()),
        call_stack: base.call_stack.clone(),
        detailed_history: base.detailed_history.iter().cloned().collect(),
    };
    let with_mem = snapshot(&base, true);
    let without_mem = snapshot(&base, false);

    // After the branch point: every carried field moves on the base.
    base.set_register("rax", RustBV::concrete(1, 64));
    base.push_call(0x401400, 0x401500, 0x401405, 0x7fff_fffe_0000);
    base.set_detailed_history(Vec::new());
    base.memory_mut()
        .store_concrete(0x10_0000u64, RustBV::concrete(9, 8))
        .expect("mapped page");
    base.solver().borrow().assume_true(&rbx.ult(&RustBV::concrete(5, 64)));
    assert!(!base.solver().borrow().is_sat(), "rbx > 10 && rbx < 5");
    // ...and so does a field that comes from `self`.
    let fd = base.fs.open_fds().into_iter().max().expect("an fd is open");
    assert_eq!(base.fs.seek(fd, 4, 0), Some(4));

    for (snap, mem_byte) in [(with_mem, 1u8), (without_mem, 9u8)] {
        let sib = base.fork_from_snapshot(snap);
        assert_eq!(sib.parent_id, Some(base.state_id));
        assert!(sib.state_id > base.state_id);
        assert_eq!(
            sib.get_register("rax").and_then(|bv| bv.as_u64()),
            Some(0xdead_beef)
        );
        assert_eq!(sib.call_stack.len(), branch_calls);
        assert_eq!(json(&sib.detailed_history), json(&branch_history));
        assert!(sib.solver().borrow().is_sat(), "branch-point solver");
        // Memory comes from the snapshot when it carries one, else from `self`.
        let byte = sib
            .memory()
            .read_concrete_bytes_for_lift(crate::memory::Address::new(0x10_0000), 1)
            .expect("memory readable");
        assert_eq!(byte, vec![mem_byte]);
        // Everything else is the base's current value.
        assert_eq!(json(&sib.fs), json(&base.fs));
        assert_eq!(sib.sim_options, base.sim_options);
        assert_eq!(sib.stdin_symbols, base.stdin_symbols);
        assert_eq!(sib.exit_pending, base.exit_pending);
    }
}
