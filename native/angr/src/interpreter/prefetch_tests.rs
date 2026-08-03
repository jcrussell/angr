//! Unit tests for [`super::prefetch`] page-prefetch helpers.
//!
//! Split out of `prefetch.rs` per the rust-mod-tests-sibling-extraction
//! convention to keep the implementation file focused. GIL-free: no PyO3.

use super::*;
use crate::vex::ir::Endness;

fn new_interp(ctx: &SymContext) -> VEXInterpreter<'_> {
    VEXInterpreter::new(VexArch::AMD64, ctx)
}

#[test]
fn set_page_prefetch_count_overrides_default() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // Default established in VEXInterpreter::new.
    let original = interp.page_prefetch_count;
    interp.set_page_prefetch_count(8);
    assert_eq!(interp.page_prefetch_count, 8);
    assert_ne!(interp.page_prefetch_count, original);
}

#[test]
fn get_stack_pointer_returns_rsp_value() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp
        .registers
        .put_reg("rsp", RustBV::concrete(0x1234_5678, 64));
    assert_eq!(interp.get_stack_pointer(), Some(0x1234_5678));
}

#[test]
fn is_stack_region_below_rsp_within_window() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp
        .registers
        .put_reg("rsp", RustBV::concrete(0x7fff_0000, 64));
    // Address 64KB below RSP — well within the 1MB window.
    assert!(interp.is_stack_region(0x7fff_0000 - 0x10000));
    // Address 2MB below RSP — outside the window.
    assert!(!interp.is_stack_region(0x7fff_0000 - 0x200000));
}

#[test]
fn is_stack_region_above_rsp_within_window() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp
        .registers
        .put_reg("rsp", RustBV::concrete(0x7fff_0000, 64));
    // Address 16KB above RSP — within 64KB upward margin.
    assert!(interp.is_stack_region(0x7fff_0000 + 0x4000));
    // Address 128KB above RSP — outside upward margin.
    assert!(!interp.is_stack_region(0x7fff_0000 + 0x20000));
}

#[test]
fn get_nearby_prefetch_list_main_page_only_without_rust_memory() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    // Without rust_memory the inner is_mapped check skips siblings,
    // leaving only the main page in the list.
    let pages = interp.get_nearby_prefetch_list(0x10000, 4);
    assert_eq!(pages, vec![0x10000]);
}

#[test]
fn get_eager_prefetch_list_caps_at_max_prefetch_batch() {
    // angr-1c88c gap 7/7: enable_eager_prefetch eager branch ->
    // get_region_prefetch_list capped at config.max_prefetch_batch.
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let mut mem = SymbolicMemory::new(Endness::Little);
    // 16-page lazy region (0x10000..0x20000), all unmapped.
    mem.add_lazy_region(0x10000u64, 0x10000u64);
    interp.set_rust_memory(mem);
    interp.config.enable_eager_prefetch = true;
    interp.config.max_prefetch_batch = 3;

    let pages = interp.get_eager_prefetch_list(0x10000);
    // The region has 16 unmapped pages but the batch cap stops enumeration at 3.
    assert_eq!(pages.len(), 3, "must cap at max_prefetch_batch");
    assert_eq!(pages, vec![0x10000, 0x11000, 0x12000]);
}

#[test]
fn get_eager_prefetch_list_falls_back_to_main_page_without_region() {
    // No lazy region containing the trigger -> get_region_prefetch_list None ->
    // fallback to just the main page.
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let mem = SymbolicMemory::new(Endness::Little);
    interp.set_rust_memory(mem);
    interp.config.enable_eager_prefetch = true;
    interp.config.max_prefetch_batch = 8;

    let pages = interp.get_eager_prefetch_list(0x55000);
    assert_eq!(pages, vec![0x55000]);
}

#[test]
fn get_eager_prefetch_list_falls_back_to_main_page_without_rust_memory() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    // No rust_memory attached -> fallback to the main page.
    let pages = interp.get_eager_prefetch_list(0x22000);
    assert_eq!(pages, vec![0x22000]);
}

/// angr-9ke6b.30: both fetch paths must decline identically when no
/// `fetch_page` callback is registered. `fetch_page` has always returned
/// `Ok(false)`; `fetch_pages_batch` used to fall through to
/// `call_batch_fetch_pages`, whose unset-batch fallback hard-errors with
/// "fetch_page callback not set". That asymmetry is the one unguarded path to
/// a mandatory-callback error outside the three slots
/// `PythonCallbacks::is_ready` checks — keep them in step, or `is_ready`'s
/// doc comment stops being true.
#[test]
fn fetch_paths_decline_without_a_fetch_page_callback() {
    pyo3::Python::initialize();
    let callbacks = PythonCallbacks::new();
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.set_rust_memory(SymbolicMemory::new(Endness::Little));

    assert!(
        matches!(interp.fetch_page(&callbacks, 0x1000), Ok(false)),
        "single-page fetch must decline, not error",
    );
    assert!(
        matches!(
            interp.fetch_pages_batch(&callbacks, &[0x1000, 0x2000]),
            Ok(0)
        ),
        "batch fetch must decline, not error",
    );
}
