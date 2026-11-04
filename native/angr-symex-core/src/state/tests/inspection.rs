//! The `state.inspect` event ring: enable/disable, event recording, the ring
//! buffer's eviction order, `set_max_events` shrink-from-the-front semantics,
//! and fork isolation of the recorded events.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

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

/// `set_max_events` shrinking drops the OLDEST events, keeping the newest
/// `max`, and a subsequent `record` still honours the new cap.
#[test]
fn test_inspection_set_max_events_shrinks_from_the_front() {
    let mut mgr = InspectionManager::default();
    mgr.enable(InspectEvent::MemRead);
    for i in 0..5 {
        mgr.record(InspectEvent::MemRead, i * 0x100, 4, 0);
    }
    assert_eq!(mgr.events().len(), 5);

    mgr.set_max_events(2);
    assert_eq!(mgr.events().len(), 2);
    assert_eq!(mgr.events()[0].addr, 0x300);
    assert_eq!(mgr.events()[1].addr, 0x400);

    mgr.record(InspectEvent::MemRead, 0x500, 4, 0);
    assert_eq!(mgr.events().len(), 2);
    assert_eq!(mgr.events()[0].addr, 0x400);
    assert_eq!(mgr.events()[1].addr, 0x500);
    assert_eq!(mgr.event_counts()[InspectEvent::MemRead as usize], 6);
}

/// `max_events == 0` means "count but retain nothing" — it must not panic
/// and must not leave a stray entry in the buffer.
#[test]
fn test_inspection_max_events_zero_retains_nothing() {
    let mut mgr = InspectionManager::default();
    mgr.enable(InspectEvent::MemWrite);
    mgr.set_max_events(0);

    mgr.record(InspectEvent::MemWrite, 0x1000, 4, 0);
    mgr.record(InspectEvent::MemWrite, 0x2000, 4, 0);

    assert!(mgr.events().is_empty());
    assert_eq!(mgr.event_counts()[InspectEvent::MemWrite as usize], 2);
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
