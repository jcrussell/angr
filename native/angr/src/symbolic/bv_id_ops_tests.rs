// Tests for symbolic/bv_id_ops.rs — the process-global symbol-id allocator
// (`NEXT_SYMBOL_ID` / `next_id` / `reserve_symbol_id` / `symbol_id_watermark`)
// and the thread-local `SymbolIdRebase` save/restore guard (angr-5mnx3.45:
// the guard stores `prev` rather than resetting to 0, and nothing exercised
// that LIFO contract).
//
// `NEXT_SYMBOL_ID` is process-global and `cargo test` runs these concurrently,
// so nothing here asserts an *absolute* id or watermark — only monotonicity,
// uniqueness and relative ordering, which hold under interleaving. The rebase
// offset is thread-local, so those tests are hermetic per test thread.
use super::*;

/// A fresh thread starts with the identity offset — worker-migration payloads
/// (same process, ids already unique) must deserialize unchanged.
#[test]
fn rebase_offset_defaults_to_zero() {
    assert_eq!(symbol_id_rebase_offset(), 0);
}

/// The single-guard case the doc describes: activate installs the offset, drop
/// puts the thread back to the identity.
#[test]
fn rebase_guard_installs_and_restores_offset() {
    {
        let _g = SymbolIdRebase::activate(0x4000);
        assert_eq!(symbol_id_rebase_offset(), 0x4000);
    }
    assert_eq!(symbol_id_rebase_offset(), 0);
}

/// The contract that motivates storing `prev` instead of resetting to 0: an
/// inner guard's drop must restore the *outer* guard's offset, not the
/// identity. A reset-to-0 Drop would silently un-rebase the remainder of the
/// outer restore, so its still-undeserialized leaves would land back in the
/// colliding id range `SymbolIdRebase` exists to escape.
#[test]
fn nested_rebase_guards_restore_in_lifo_order() {
    let _outer = SymbolIdRebase::activate(100);
    assert_eq!(symbol_id_rebase_offset(), 100);
    {
        let _inner = SymbolIdRebase::activate(200);
        assert_eq!(symbol_id_rebase_offset(), 200);
        {
            let _innermost = SymbolIdRebase::activate(300);
            assert_eq!(symbol_id_rebase_offset(), 300);
        }
        assert_eq!(symbol_id_rebase_offset(), 200);
    }
    assert_eq!(symbol_id_rebase_offset(), 100);
    drop(_outer);
    assert_eq!(symbol_id_rebase_offset(), 0);
}

/// Guards dropped out of nesting order are NOT a supported use — the stack is
/// a single cell, so the survivor's `prev` wins. Pinned so the limitation is
/// visible to anyone tempted to hold a guard in a collection: keep every
/// `SymbolIdRebase` in a plain lexical scope.
#[test]
fn out_of_order_drop_leaves_the_survivors_prev() {
    let outer = SymbolIdRebase::activate(100);
    let inner = SymbolIdRebase::activate(200);
    // Dropping the OUTER guard first restores *its* prev (0), clobbering the
    // inner guard's still-live 200.
    drop(outer);
    assert_eq!(symbol_id_rebase_offset(), 0);
    // ... and the inner guard then restores the outer's 100, which no longer
    // corresponds to any live rebase.
    drop(inner);
    assert_eq!(symbol_id_rebase_offset(), 100);
    // Leave the test thread clean for anything that runs after it.
    SYMBOL_ID_REBASE.with(|c| c.set(0));
}

/// The offset is thread-local: a guard live on this thread must not rebase ids
/// deserialized on any other. Parallel workers each own their own restore.
#[test]
fn rebase_offset_is_thread_local() {
    let _g = SymbolIdRebase::activate(0x9000);
    let seen = std::thread::spawn(symbol_id_rebase_offset)
        .join()
        .expect("probe thread");
    assert_eq!(seen, 0);
    assert_eq!(symbol_id_rebase_offset(), 0x9000);
}

/// `next_id` mints strictly-increasing, never-reused ids, and the watermark
/// stays an *exclusive* upper bound on everything minted so far.
#[test]
fn next_id_is_strictly_increasing_and_below_the_watermark() {
    let ctx = SymContext::new();
    let a = ctx.next_id();
    let b = ctx.next_id();
    assert!(b > a, "ids must be strictly increasing: {a} then {b}");
    assert!(
        symbol_id_watermark() > b,
        "watermark {} must exceed the last minted id {b}",
        symbol_id_watermark()
    );
}

/// Ids are allocated process-globally, not per context — two `SymContext`s in
/// one process must never hand out the same id (see `NEXT_SYMBOL_ID`'s doc for
/// the aliasing failure a per-context counter caused).
#[test]
fn ids_are_unique_across_contexts() {
    let a = SymContext::new();
    let b = SymContext::new();
    let ids = [a.next_id(), b.next_id(), a.next_id(), b.next_id()];
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "duplicate id across contexts: {ids:?}");
}

/// `reserve_symbol_id` is a `fetch_max`: it raises the watermark to the
/// requested exclusive bound and never lowers it, so a legacy snapshot's 0 is
/// a no-op rather than a counter reset.
#[test]
fn reserve_symbol_id_raises_but_never_lowers_the_watermark() {
    let target = symbol_id_watermark() + 0x1_0000;
    reserve_symbol_id(target);
    assert!(symbol_id_watermark() >= target);

    let high = symbol_id_watermark();
    reserve_symbol_id(0);
    assert!(
        symbol_id_watermark() >= high,
        "reserve_symbol_id(0) must not lower the watermark"
    );

    // The reserved bound is exclusive: the next id minted is at or above it,
    // so no id a restored snapshot's leaves already hold gets re-minted.
    let ctx = SymContext::new();
    assert!(ctx.next_id() >= target);
}

/// `new_bv` mints exactly one id and uses it for both halves of the value, so
/// the display name's suffix and the `Symbolic.id` identity key agree
/// (angr-sqfj8.90 — going via `unique_name` + `RustBV::symbolic` drew two).
#[test]
fn new_bv_name_suffix_matches_its_id() {
    let ctx = SymContext::new();
    let before = symbol_id_watermark();
    let bv = ctx.new_bv("stdin", 64);
    let RustBV::Symbolic { id, width, name, .. } = &bv else {
        panic!("new_bv must yield a Symbolic leaf, got {bv:?}");
    };
    assert_eq!(&**name, format!("stdin_{id}").as_str());
    assert_eq!(*width, 64);
    assert!(
        symbol_id_watermark() > before,
        "new_bv must mint at least one id"
    );
}

/// `unique_name` appends the freshly minted id and never repeats a name.
#[test]
fn unique_name_appends_a_fresh_id() {
    let ctx = SymContext::new();
    let first = ctx.unique_name("tmp");
    let second = ctx.unique_name("tmp");
    assert_ne!(first, second);
    for name in [&first, &second] {
        let suffix = name.strip_prefix("tmp_").expect("base prefix preserved");
        suffix.parse::<u64>().expect("suffix is the minted id");
    }
}
