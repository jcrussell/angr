// Tests for procedures/stdin_common.rs (harness-seeded-stdin plumbing shared by
// every native fd-0 reader). The helpers were only exercised indirectly through
// scanf/fgets/read's own test files (angr-03vl4.48); these pin the seed-order,
// double-record and position-advance contracts directly, since those are what a
// new native stdin reader gets wrong (angr-ptf54).
use super::*;

/// Seed fd 0 with concrete bytes the way `_seed_stdin_to_rust` does.
#[cfg(feature = "vex-engine-z3")]
fn seed_stdin(state: &mut RustSimState, bytes: &[u8]) {
    let seed: Vec<RustBV> = bytes
        .iter()
        .map(|b| RustBV::concrete(u128::from(*b), 8))
        .collect();
    state.file_system().set_fd_content_sym(0, seed);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_mint_stdin_bytes_unseeded_records_every_leaf() {
    // No harness seed: every minted byte is a fresh unconstrained leaf and is
    // recorded so `posix.dumps(0)` can reconstruct the stream.
    let mut state = RustSimState::new("amd64").unwrap();
    let names: Vec<String> = (0..3).map(|i| format!("stdin_{i}")).collect();

    let bytes = mint_stdin_bytes(&mut state, &names);

    assert_eq!(bytes.len(), 3);
    for b in &bytes {
        assert_eq!(b.width(), 8);
        assert!(b.as_u64().is_none(), "minted byte is symbolic");
    }
    assert_eq!(state.stdin_symbols().len(), 3);
    let recorded: Vec<&str> = state
        .stdin_symbols()
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    assert_eq!(recorded, names);
    assert!(state.stdin_symbols().iter().all(|(_, bits)| *bits == 8));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_mint_stdin_bytes_binds_seed_and_skips_recording() {
    // Seeded bytes come back as plain leaves bound to the seed by constraint —
    // NOT stored directly (see `avoid-storing-extract-asts-in-guest-buffers`) —
    // and are deliberately not recorded, or `_inject_rust_stdin` would append a
    // duplicate of the harness's own chunk.
    let mut state = RustSimState::new("amd64").unwrap();
    seed_stdin(&mut state, b"AB");

    let names: Vec<String> = (0..2).map(|i| format!("stdin_{i}")).collect();
    let bytes = mint_stdin_bytes(&mut state, &names);

    for (b, want) in bytes.iter().zip([0x41u128, 0x42]) {
        assert!(b.as_u64().is_none(), "still a leaf symbol, not the seed BV");
        assert_eq!(state.eval(b), Some(want), "leaf binds to the seed byte");
    }
    assert!(state.stdin_symbols().is_empty(), "seeded bytes not recorded");
    // Position advanced by the bytes actually consumed, mirroring Python.
    assert_eq!(state.file_system_ref().fd_info(0).unwrap().1, 2);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_mint_stdin_bytes_partial_seed_records_only_the_tail() {
    // A read longer than the remaining seed binds the seeded prefix and mints
    // fresh recorded symbols for the rest.
    let mut state = RustSimState::new("amd64").unwrap();
    seed_stdin(&mut state, b"AB");

    let names: Vec<String> = (0..4).map(|i| format!("stdin_{i}")).collect();
    let bytes = mint_stdin_bytes(&mut state, &names);

    assert_eq!(bytes.len(), 4);
    assert_eq!(state.eval(&bytes[0]), Some(0x41));
    assert_eq!(state.eval(&bytes[1]), Some(0x42));
    // The unseeded tail is unconstrained: two distinct values are feasible.
    let ctx = state.solver().borrow();
    assert_ne!(ctx.min(&bytes[2], false), ctx.max(&bytes[2], false));
    drop(ctx);
    // Only the tail is recorded, in read order.
    let recorded: Vec<&str> = state
        .stdin_symbols()
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    assert_eq!(recorded, vec!["stdin_2", "stdin_3"]);
    // The position stops at the end of the seed, not at the read length.
    assert_eq!(state.file_system_ref().fd_info(0).unwrap().1, 2);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_mint_stdin_bytes_consumes_seed_in_read_order() {
    // Successive readers see successive seed bytes — the ordering contract
    // every native fd-0 reader shares.
    let mut state = RustSimState::new("amd64").unwrap();
    seed_stdin(&mut state, b"ABC");

    for (i, want) in [0x41u128, 0x42, 0x43].iter().enumerate() {
        let bytes = mint_stdin_bytes(&mut state, &[format!("stdin_{i}")]);
        assert_eq!(state.eval(&bytes[0]), Some(*want), "read {i}");
    }
    // Past the seed: fresh unconstrained, recorded.
    let bytes = mint_stdin_bytes(&mut state, &["stdin_3".to_string()]);
    assert!(bytes[0].as_u64().is_none());
    assert_eq!(state.stdin_symbols().len(), 1);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_mint_stdin_bytes_empty_names_is_a_noop() {
    let mut state = RustSimState::new("amd64").unwrap();
    seed_stdin(&mut state, b"AB");

    assert!(mint_stdin_bytes(&mut state, &[]).is_empty());
    assert!(state.stdin_symbols().is_empty());
    assert_eq!(state.file_system_ref().fd_info(0).unwrap().1, 0);
}

#[test]
fn test_stdin_seed_unconsumed_false_without_seed() {
    // No harness seed attached at all: nothing to leave unconsumed, so a
    // native reader is free to mint.
    let state = RustSimState::new("amd64").unwrap();
    assert!(!stdin_seed_unconsumed(&state));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_stdin_seed_unconsumed_tracks_position() {
    let mut state = RustSimState::new("amd64").unwrap();
    seed_stdin(&mut state, b"AB");
    assert!(stdin_seed_unconsumed(&state), "fresh seed is unconsumed");

    mint_stdin_bytes(&mut state, &["stdin_0".to_string()]);
    assert!(stdin_seed_unconsumed(&state), "one byte of two consumed");

    mint_stdin_bytes(&mut state, &["stdin_1".to_string()]);
    assert!(!stdin_seed_unconsumed(&state), "seed fully consumed");

    // Reading past the exhausted seed keeps it consumed (position clamps).
    mint_stdin_bytes(&mut state, &["stdin_2".to_string()]);
    assert!(!stdin_seed_unconsumed(&state));
}
