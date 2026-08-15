//! `export_full`'s dump shape: the named-vs-symbolic register split, memory
//! pages and their load boundary, symbolic offsets, heap and open fds, and a
//! smoke test over history/callstack/arch.
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;
#[cfg(feature = "vex-engine-z3")]
use super::helpers::build_populated_state;

// =============================================================================
// export.rs — ExplorationStateSnapshot / RustSimState::export_full coverage
// (angr-n0irt.17). Before this, state/export.rs — the single choke point every
// found/errored/exported state crosses — had no #[cfg(test)] and no sibling
// *_tests.rs. These pin the zero-coverage branches named in the bead:
// the named-vs-symbolic register split (angr-4ju9e), memory_load page-boundary
// logic, get_page/page_addresses/get_symbolic_offsets, heap alloc/free, and
// open-fd export.
// =============================================================================

/// angr-4ju9e regression: export_full must (a) emit a *concrete* register in
/// `named_registers` (recoverable without an offset table on the Python side)
/// and (b) list a *symbolic* register in `symbolic_register_names` while
/// EXCLUDING it from `named_registers`, so Python attaches a lazy proxy instead
/// of silently dropping the Rust-computed symbolic value.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_export_full_named_vs_symbolic_register_split() {
    let mut s = RustSimState::new("amd64").unwrap();
    s.set_register("rax", RustBV::concrete(0xdead_beef, 64));
    let rbx_sym = {
        let ctx = s.solver().borrow();
        RustBV::symbolic(&ctx, "n0irt17_rbx", 64)
    };
    s.set_register("rbx", rbx_sym);

    let snap = s.export_full();

    let named = snap.get_registers_named();
    assert_eq!(
        named.get("rax"),
        Some(&(0xdead_beefu128, 64)),
        "concrete rax must be pre-computed into named_registers with its bit width"
    );
    assert!(
        !named.contains_key("rbx"),
        "symbolic rbx must be EXCLUDED from named_registers (else it is dropped)"
    );

    let sym_names = snap.get_symbolic_register_names();
    assert!(
        sym_names.iter().any(|n| n == "rbx"),
        "symbolic rbx must appear in symbolic_register_names for lazy-proxy recovery"
    );
    assert!(
        !sym_names.iter().any(|n| n == "rax"),
        "concrete rax must NOT be tagged symbolic"
    );
}

/// Memory export surface: page_count / page_addresses / get_page, and the
/// page-boundary offset+size arithmetic in `memory_load` (in-page hit, exact
/// end-of-page fit, one-byte overflow past the page → None, and unmapped →
/// None). Concrete-only pages report no symbolic offsets.
#[test]
fn test_export_full_memory_pages_and_load_boundary() {
    let mut s = RustSimState::new("amd64").unwrap();
    s.map_memory_data(0x10_0000, &[1u8, 2, 3, 4, 5, 6, 7, 8], Permission::RWX);
    s.map_memory_data(0x20_0000, &[0xAAu8; 4], Permission::RW);

    let snap = s.export_full();

    assert_eq!(snap.page_count(), 2, "two mapped pages must be exported");
    let addrs = snap.page_addresses();
    assert!(addrs.contains(&0x10_0000) && addrs.contains(&0x20_0000));

    // get_page returns a full PAGE_SIZE-wide, page-aligned tuple; OOB → None.
    let pg = snap.get_page(0).expect("page 0 exists");
    assert_eq!(pg.0 & 0xFFF, 0, "exported page addr must be page-aligned");
    assert_eq!(
        pg.1.len(),
        crate::memory::PAGE_SIZE as usize,
        "exported page must carry the whole page's concrete bytes"
    );
    assert!(snap.get_page(99).is_none(), "OOB page index → None");

    // In-page load at the mapped bytes.
    assert_eq!(
        snap.memory_load(0x10_0000, 4),
        Some(vec![1u8, 2, 3, 4]),
        "load of the seeded prefix must return those bytes"
    );
    // Exact end-of-page fit: offset 4092 + size 4 == PAGE_SIZE.
    assert_eq!(
        snap.memory_load(0x10_0000 + 4092, 4).map(|v| v.len()),
        Some(4),
        "a load that ends exactly at the page boundary must succeed"
    );
    // One byte past the page must NOT silently span into the next page.
    assert_eq!(
        snap.memory_load(0x10_0000 + 4093, 4),
        None,
        "a load overflowing the page boundary must return None"
    );
    // Unmapped address → None.
    assert_eq!(snap.memory_load(0x30_0000, 1), None);
    // angr-xloth.1: `size` crosses the `#[pymethods]` boundary untrusted, so a
    // value that makes `offset + size` wrap must be refused rather than
    // sneaking under the `<= page.len()` bound check and returning a short,
    // wrong slice. offset here is 0x100, so usize::MAX - 0xFF wraps to 0.
    assert_eq!(
        snap.memory_load(0x10_0100, usize::MAX - 0xFF),
        None,
        "a size that overflows offset+size must be refused, not wrapped"
    );

    // Concrete-only page → no symbolic offsets; unknown page → empty too.
    assert!(snap.get_symbolic_offsets(0x10_0000).is_empty());
    assert!(snap.get_symbolic_offsets(0xDEAD_0000).is_empty());
}

/// A symbolic store marks the page's symbolic bitmap, and export_full must
/// surface those byte offsets via get_symbolic_offsets (the non-empty branch).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_export_full_symbolic_offsets_nonempty() {
    let mut s = RustSimState::new("amd64").unwrap();
    s.map_memory(0x40_0000, 0x1000, Permission::RW);
    let sym = {
        let ctx = s.solver().borrow();
        RustBV::symbolic(&ctx, "n0irt17_byte", 8)
    };
    s.memory_store(0x40_0000, sym).expect("symbolic store");

    let snap = s.export_full();
    assert_eq!(
        snap.get_symbolic_offsets(0x40_0000),
        vec![0u16],
        "byte 0 of the page must be flagged symbolic after an 8-bit symbolic store"
    );
}

/// Heap and open-fd export: a freed allocation moves out of `heap_allocated`
/// into `heap_freed`, and an open file descriptor is exported with its name,
/// content length, and open flag.
#[test]
fn test_export_full_heap_and_open_fds() {
    let mut s = RustSimState::new("amd64").unwrap();
    let a1 = s.heap_alloc(32);
    let a2 = s.heap_alloc(64);
    assert_eq!(s.heap_free(a1), Some(32), "free returns the tracked size");

    let fd =
        s.fs.open_with_content("/flag.txt".to_string(), FdFlags::ReadOnly, vec![b'A'; 10]).expect("fd space is not exhausted in tests");

    let snap = s.export_full();

    // a2 stays allocated; a1 has moved to freed.
    assert_eq!(snap.get_heap_alloc_count(), 1);
    assert_eq!(snap.get_heap_allocated(), vec![(a2, 64)]);
    assert_eq!(snap.get_heap_free_count(), 1);
    assert_eq!(snap.get_heap_freed(), vec![a1]);

    // The opened fd is exported with name, content length, and open flag.
    let fds = snap.get_open_fds();
    assert_eq!(
        snap.get_fd_count(),
        fds.len(),
        "fd_count must match the exported fd list length"
    );
    let entry = fds
        .iter()
        .find(|e| e.0 == fd)
        .expect("the opened fd must be exported");
    assert_eq!(entry.1, "/flag.txt", "fd name must round-trip");
    assert_eq!(entry.4, 10, "content length must be exported");
    assert!(entry.5, "a freshly opened fd must be exported as open");
}

/// Broad smoke over the remaining getters on a fully-populated state: raw
/// register bytes, history, call stack (list + depth), and arch name are all
/// exported non-trivially.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_export_full_smoke_history_callstack_arch() {
    let snap = build_populated_state().export_full();

    assert!(
        !snap.get_registers_raw().is_empty(),
        "raw register bytes must be exported"
    );
    assert!(!snap.arch_name.is_empty(), "arch name must be exported");
    assert!(
        !snap.get_history().is_empty(),
        "populated state has basic-block history"
    );
    assert_eq!(
        snap.get_call_stack_depth(),
        snap.get_call_stack().len(),
        "call_stack_depth must match the exported call-stack length"
    );
    assert!(
        snap.get_call_stack_depth() >= 1,
        "populated state pushed a call frame"
    );
}
