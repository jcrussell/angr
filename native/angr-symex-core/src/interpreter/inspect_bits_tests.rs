//! Tests for the [`InspectBit`] event table.
//!
//! The table ties each inspect bit number to its event name at compile time
//! (which dispatch call a gate precedes is a convention, see the
//! `inspect_bits` module doc); these tests cover the two ways the table
//! itself can go wrong — a duplicated/skipped bit, and drift away from the
//! Python `_INSPECT_EVENT_SPECS` table that builds the enabled-mask.

use super::InspectBit;

/// `ALL` must be dense and in bit order: `py_set_inspect_enabled` ships a
/// single `u32` mask, so a skipped or duplicated bit silently aliases two
/// events onto one flag.
#[test]
fn bits_are_dense_and_ordered() {
    for (idx, event) in InspectBit::ALL.iter().enumerate() {
        assert_eq!(
            event.bit() as usize,
            idx,
            "{event:?} is at index {idx} but claims bit {}",
            event.bit()
        );
    }
    assert!(
        InspectBit::ALL.len() <= 32,
        "enabled-mask is an AtomicU32; {} events do not fit",
        InspectBit::ALL.len()
    );
}

/// Event names are the Python `state.inspect` keys and the suffix of the
/// matching `call_inspect_*` method, so a duplicate would mean two events
/// dispatching into the same Python endpoint.
#[test]
fn event_names_are_unique_and_snake_case() {
    let mut seen = std::collections::HashSet::new();
    for event in InspectBit::ALL {
        let name = event.event_name();
        assert!(seen.insert(name), "duplicate event name {name}");
        assert!(
            name.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
            "{name} is not snake_case"
        );
    }
}

/// The Rust table is a mirror of Python's `_INSPECT_EVENT_SPECS`; the mask
/// Python sends is built from *its* numbering, so drift here mis-gates every
/// event above the divergence point. Parsed textually rather than imported so
/// the check does not need the angr package importable under `cargo test`.
#[test]
fn bits_match_python_inspect_event_specs() {
    let src = match std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../angr/exploration/rust_state_proxy.py"
    )) {
        Ok(s) => s,
        // SILENT(cat-a): running from a packaged sdist without the Python
        // tree next to the crate is expected; the Python-side test
        // `test_inspect_allowlist_complete_and_consistent` still covers it.
        Err(_) => return,
    };
    let table = src
        .split_once("_INSPECT_EVENT_SPECS: dict = {")
        .expect("specs table present")
        .1;
    let table = table
        .split_once("\n# Derived views")
        .expect("table terminator")
        .0;

    let mut python_bits = Vec::new();
    let mut pending_name: Option<String> = None;
    for line in table.lines() {
        if let Some(rest) = line.strip_prefix("    \"") {
            if let Some((name, _)) = rest.split_once("\": {") {
                pending_name = Some(name.to_string());
            }
        } else if let Some(rest) = line.strip_prefix("        \"bit\": ") {
            let bit: u8 = rest
                .trim_end_matches(',')
                .parse()
                .expect("bit is an integer");
            python_bits.push((bit, pending_name.take().expect("name precedes bit")));
        }
    }

    let rust_bits: Vec<(u8, String)> = InspectBit::ALL
        .iter()
        .map(|e| (e.bit(), e.event_name().to_string()))
        .collect();
    assert_eq!(
        rust_bits, python_bits,
        "InspectBit table drifted from _INSPECT_EVENT_SPECS"
    );
}
