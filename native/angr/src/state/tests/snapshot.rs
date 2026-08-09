//! Snapshot round-trips: the full `to_snapshot`/`from_snapshot` field-by-field
//! contract (`assert_state_round_trip`), foreign state-id reservation on
//! restore, and the serialized-envelope form including its empty and
//! version-mismatch error paths.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;
#[cfg(feature = "vex-engine-z3")]
use super::helpers::build_populated_state;

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
    assert_eq!(
        restored.history().iter().copied().collect::<Vec<_>>(),
        orig.history().iter().copied().collect::<Vec<_>>()
    );
    assert_eq!(
        restored.detailed_history().len(),
        orig.detailed_history().len()
    );
    assert_eq!(restored.heap_brk(), orig.heap_brk());
    assert_eq!(restored.posix_brk(), orig.posix_brk());
    assert_eq!(restored.mmap_base(), orig.mmap_base());
    assert_eq!(restored.tsc_counter(), orig.tsc_counter());
    assert_eq!(restored.getopt_cursor(), orig.getopt_cursor());
    {
        let (ro, oo) = (restored.getopt_extern(), orig.getopt_extern());
        assert_eq!(ro.optind, oo.optind);
        assert_eq!(ro.optarg, oo.optarg);
        assert_eq!(ro.optopt, oo.optopt);
    }
    {
        let (rs, os) = (restored.native_resume_stack(), orig.native_resume_stack());
        assert_eq!(rs.len(), os.len());
        for (rf, of) in rs.iter().zip(os.iter()) {
            assert_eq!(rf.proc_name, of.proc_name);
            assert_eq!(rf.resume_tag, of.resume_tag);
            assert_eq!(rf.saved_args.len(), of.saved_args.len());
            assert_eq!(rf.caller_return_addr, of.caller_return_addr);
        }
    }
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
        restored.getenv(b"PATH").map(<[u8]>::to_vec),
        Some(b"/usr/bin".to_vec())
    );
    assert_eq!(
        restored.getenv(b"HOME").map(<[u8]>::to_vec),
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
    let mut orig = build_populated_state();
    let snap = orig.to_snapshot();
    let restored = RustSimState::from_snapshot(snap).expect("from_snapshot");
    assert_state_round_trip(&orig, &restored);
}

/// `state-id-never-reused` across a snapshot boundary: a restored state carries
/// an ID minted by a foreign counter, so `from_snapshot` must lift the local
/// counter above it. Otherwise a resumed manager re-mints IDs that are live in
/// its own stashes (angr-op0dn.13.14).
#[cfg(feature = "vex-engine-z3")]
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

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_state_to_from_serialized_round_trip() {
    let mut orig = build_populated_state();
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
