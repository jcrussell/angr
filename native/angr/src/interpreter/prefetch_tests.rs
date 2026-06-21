//! Unit tests for [`super::prefetch`] page-prefetch helpers.
//!
//! Split out of `prefetch.rs` per the rust-mod-tests-sibling-extraction
//! convention to keep the implementation file focused. GIL-free: no PyO3.

use super::*;
use crate::vex::ir::{Endness, IRType};

fn new_interp(ctx: &SymContext) -> VEXInterpreter<'_> {
    VEXInterpreter::new(VexArch::AMD64, ctx)
}

fn make_irsb_load(addr: u64, load_addr: u64, size: IRType) -> IRSB {
    let mut irsb = IRSB::new(addr, VexArch::AMD64);
    irsb.statements.push(IRStmt::IMark {
        addr,
        len: 4,
        delta: 0,
    });
    // Allocate a temp for the load destination.
    let tmp = irsb.tyenv.new_temp(size);
    irsb.statements.push(IRStmt::WrTmp {
        tmp,
        data: IRExpr::Load {
            addr: Box::new(IRExpr::Const(IRConst::U64(load_addr))),
            ty: size,
            endness: Endness::Little,
        },
    });
    irsb
}

#[test]
fn set_load_prefetch_toggles_flag() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    assert!(!interp.use_load_prefetch);
    interp.set_load_prefetch(true);
    assert!(interp.use_load_prefetch);
    interp.set_load_prefetch(false);
    assert!(!interp.use_load_prefetch);
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
fn clear_prefetch_cache_empties_map() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.load_prefetch_cache.insert(
        (0x4000, 4),
        PrefetchedLoad {
            value: RustBV::concrete(0xaa, 32),
            is_symbolic: false,
        },
    );
    assert!(interp.get_prefetched_load(0x4000, 4).is_some());
    interp.clear_prefetch_cache();
    assert!(interp.get_prefetched_load(0x4000, 4).is_none());
}

#[test]
fn get_prefetched_load_returns_value_on_hit() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.load_prefetch_cache.insert(
        (0x8000, 8),
        PrefetchedLoad {
            value: RustBV::concrete(0xbb, 64),
            is_symbolic: false,
        },
    );
    let hit = interp.get_prefetched_load(0x8000, 8).expect("hit");
    assert!(!hit.is_symbolic);
    assert_eq!(hit.value.as_u64(), Some(0xbb));
    // Miss on different size.
    assert!(interp.get_prefetched_load(0x8000, 4).is_none());
}

#[test]
fn scan_loads_in_irsb_finds_concrete_load() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let irsb = make_irsb_load(0x1000, 0x4000, IRType::I32);
    let mut loads = Vec::new();
    interp.scan_loads_in_irsb(&irsb, &mut loads);
    assert!(loads.contains(&(0x4000, 4)));
}

#[test]
fn scan_loads_skips_addresses_in_concrete_memory() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // Cache 0x4000 in concrete memory so the scanner skips it.
    interp.add_concrete_memory(0x4000, vec![0u8; 0x100]);
    let irsb = make_irsb_load(0x1000, 0x4000, IRType::I32);
    let mut loads = Vec::new();
    interp.scan_loads_in_irsb(&irsb, &mut loads);
    assert!(!loads.iter().any(|&(a, _)| a == 0x4000));
}

#[test]
fn try_eval_expr_concrete_handles_const_u64() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let env = TypeEnv::new();
    let expr = IRExpr::Const(IRConst::U64(0xdead));
    assert_eq!(interp.try_eval_expr_concrete(&expr, &env), Some(0xdead));
}

#[test]
fn try_eval_expr_concrete_handles_concrete_register() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp
        .registers
        .put_reg("rsp", RustBV::concrete(0x7fff_0000, 64));
    let env = TypeEnv::new();
    // AMD64 RSP = offset 48
    let expr = IRExpr::Get {
        offset: 48,
        ty: IRType::I64,
    };
    assert_eq!(
        interp.try_eval_expr_concrete(&expr, &env),
        Some(0x7fff_0000)
    );
}

#[test]
fn try_eval_expr_concrete_returns_none_for_symbolic_register() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let sym = RustBV::symbolic(&ctx, "rsp_sym", 64);
    interp.registers.put_reg("rsp", sym);
    let env = TypeEnv::new();
    let expr = IRExpr::Get {
        offset: 48,
        ty: IRType::I64,
    };
    assert!(interp.try_eval_expr_concrete(&expr, &env).is_none());
}

#[test]
fn try_eval_expr_concrete_returns_none_for_unop() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let env = TypeEnv::new();
    let expr = IRExpr::RdTmp(0);
    // RdTmp isn't handled by simplified concrete eval.
    assert_eq!(interp.try_eval_expr_concrete(&expr, &env), None);
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
