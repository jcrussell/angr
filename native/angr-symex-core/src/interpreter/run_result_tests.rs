//! Tests for [`RunErrorKind::route`] — the error-kind → stash routing table.
//!
//! The routing decision used to live as unnamed guard clauses inside
//! `core_outcome.rs`'s match, so there was nothing to assert on; these tests
//! pin each named [`ErrorRoute`] to the (kind, pc) pair that produces it.

use super::{ErrorRoute, RunErrorKind};

// angr-91vj9.9: the pc-0 carve-out used to be an unnamed guard clause in
// `core_outcome.rs`'s routing match, so there was nothing to assert on.
#[test]
fn deadend_kind_routes_to_unliftable_at_any_addr() {
    assert_eq!(
        RunErrorKind::Deadend.route(0x40_1000),
        ErrorRoute::UnliftableDeadend
    );
    assert_eq!(
        RunErrorKind::Deadend.route(0),
        ErrorRoute::UnliftableDeadend
    );
}

#[test]
fn fatal_at_real_addr_routes_to_errored() {
    assert_eq!(RunErrorKind::Fatal.route(0x40_1000), ErrorRoute::Errored);
    assert_eq!(RunErrorKind::Fatal.route(1), ErrorRoute::Errored);
}

#[test]
fn fatal_at_null_addr_is_its_own_named_route() {
    let route = RunErrorKind::Fatal.route(0);
    assert_eq!(route, ErrorRoute::NullAddressDeadend);
    // The whole point of the variant: it is a deadend, but not the same
    // deadend as an unliftable block.
    assert_ne!(route, ErrorRoute::UnliftableDeadend);
}
