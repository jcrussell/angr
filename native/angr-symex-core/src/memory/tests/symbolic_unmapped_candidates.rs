//! `load_symbolic_unified`'s `Multiple` arm over candidates whose pages Rust
//! does not hold (angr-vmi2l.177): no candidate is dropped, so the address
//! domain keeps every concretizer solution; a fetchable page is sent back to
//! the caller; and each unmapped read gets its own filler (or 0 under
//! ZERO_FILL_UNCONSTRAINED_MEMORY) instead of a name-interned shared symbol.

use super::super::*;
use crate::symbolic::concretize::AddressConcretizer;

/// A symbolic address constrained to `base + offs[i]`. The offsets must be
/// irregular: an arithmetic progression concretizes as `Strided`, not the
/// `Multiple` arm under test.
fn addr_over(ctx: &SymContext, name: &str, base: u64, offs: &[u64]) -> RustBV {
    let addr = RustBV::symbol(name, 64);
    let mut clause: Option<RustBV> = None;
    for &off in offs {
        let eq = addr.eq(&RustBV::concrete(u128::from(base + off), 64));
        clause = Some(clause.map_or(eq.clone(), |c| c.or(&eq)));
    }
    ctx.assume_true(&clause.expect("non-empty offsets"));
    assert!(ctx.is_sat(), "candidate constraint must be SAT");
    addr
}

/// Two all-unmapped loads read independent fillers: their difference is
/// satisfiable. The old arm returned one `mem_all_unmapped_<size>` symbol
/// for both, so `a == b` was valid.
#[test]
fn all_unmapped_loads_do_not_alias() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let offs = [0u64, 1, 3, 0x40];
    let a_addr = addr_over(&ctx, "a_idx", 0x5000_0000, &offs);
    let b_addr = addr_over(&ctx, "b_idx", 0x6000_0000, &offs);
    let a = mem
        .load_symbolic_unified(a_addr, 1, &ctx, &concretizer)
        .expect("an all-unmapped load reads fillers");
    let b = mem
        .load_symbolic_unified(b_addr, 1, &ctx, &concretizer)
        .expect("an all-unmapped load reads fillers");

    ctx.push();
    ctx.assume_true(&a.ne(&b));
    assert!(ctx.is_sat(), "independent unmapped reads must not be forced equal");
    ctx.pop();
}

/// Under ZERO_FILL_UNCONSTRAINED_MEMORY an all-unmapped load is concrete 0,
/// as Python returns.
#[test]
fn all_unmapped_load_zero_fill_is_zero() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.set_zero_fill_unconstrained(true);

    let addr = addr_over(&ctx, "zf_idx", 0x5000_0000, &[0, 1, 3, 0x40]);
    let v = mem
        .load_symbolic_unified(addr, 4, &ctx, &concretizer)
        .expect("an all-unmapped load reads zero fill");
    assert_eq!(ctx.eval_upto(&v, 2), vec![0], "zero fill must read 0");
}

/// Candidates split across a mapped page and a page neither side holds keep
/// every address solution, and the mapped candidates still read their bytes.
/// The old arm dropped the unmapped candidates and asserted the address
/// into the mapped survivors.
#[test]
fn straddle_keeps_every_candidate() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.store_concrete(0x1FFE, RustBV::concrete(0x4241, 16))
        .expect("store 0x1FFE");

    let addr = addr_over(&ctx, "st_idx", 0x1FFE, &[0, 1, 2, 5]);
    let v = mem
        .load_symbolic_unified(addr.clone(), 1, &ctx, &concretizer)
        .expect("a straddling load reads its mapped bytes and fillers");

    let mut seen = ctx.eval_upto(&addr, 8);
    seen.sort_unstable();
    assert_eq!(seen, vec![0x1FFE, 0x1FFF, 0x2000, 0x2003]);

    ctx.push();
    ctx.assume_true(&addr.eq(&RustBV::concrete(0x1FFF, 64)));
    assert_eq!(ctx.eval_upto(&v, 2), vec![0x42], "mapped candidate keeps its byte");
    ctx.pop();
}

/// A missing candidate page inside a lazy region, or one the host claims,
/// surfaces as `UnmappedPageInRegion` so the interpreter fetches it.
#[test]
fn fetchable_candidate_page_is_requested() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    let addr = addr_over(&ctx, "lz_idx", 0x1FFE, &[0, 1, 2, 5]);

    let hosted = mem.load_symbolic_unified_hosted(addr.clone(), 1, &ctx, &concretizer, &|page| {
        page == 0x2000
    });
    assert!(
        matches!(hosted, Err(MemoryError::UnmappedPageInRegion { page_addr: 0x2000 })),
        "a host page must be fetched, got {hosted:?}"
    );

    mem.add_lazy_region(0x2000, 0x1000);
    let lazy = mem.load_symbolic_unified(addr, 1, &ctx, &concretizer);
    assert!(
        matches!(lazy, Err(MemoryError::UnmappedPageInRegion { page_addr: 0x2000 })),
        "a lazy-region page must be fetched, got {lazy:?}"
    );
}

/// A leaf whose tail runs onto a page neither side holds keeps the bytes the
/// mapped page holds, in either byte order; only the tail is a filler.
#[test]
fn cross_page_leaf_keeps_mapped_bytes() {
    for endness in [Endness::Little, Endness::Big] {
        let ctx = SymContext::new_mock();
        let mut mem = SymbolicMemory::new(endness);
        mem.map(0x1000, 0x1000, Permission::RWX);
        mem.store_concrete(0x1FFE, RustBV::concrete(0x4142, 16))
            .expect("store 0x1FFE");

        let v = mem.load_concrete_or_unconstrained(0x1FFEu64, 4, &ctx);
        assert_eq!(v.width(), 32);
        let mapped = match endness {
            Endness::Little => v.extract(15, 0),
            Endness::Big => v.extract(31, 16),
        };
        assert_eq!(
            ctx.eval_upto(&mapped, 2),
            vec![0x4142],
            "{endness:?}: mapped bytes must survive the unmapped tail"
        );
    }
}

/// The multiwrite store over a candidate page Rust has not fetched from its
/// lazy region reports `UnmappedPageInRegion` and installs nothing, so the
/// shell fetches the host's bytes and retries. Its old bare install mapped
/// the page as zeros, and every byte the store did not cover then read 0
/// instead of the host's data (angr-vmi2l.350).
#[test]
fn multiwrite_store_over_unfetched_lazy_page_maps_nothing() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.add_lazy_region(0x7000, 0x1000);

    let addr = addr_over(&ctx, "mw_idx", 0x7010, &[0, 1, 3]);
    let result = mem.store_symbolic_unified_multi(addr, RustBV::concrete(0xAA, 8), &ctx, &concretizer);
    assert!(
        matches!(result, Err(MemoryError::UnmappedPageInRegion { page_addr: 0x7000 })),
        "expected a lazy-page fault for 0x7000, got {result:?}"
    );
    assert!(!mem.is_mapped(0x7000), "the faulting store must not map the page");
}

/// A candidate page in no lazy region comes back `Unmapped` with nothing
/// mapped either, and the shell hands the store back to the proxy.
#[test]
fn multiwrite_store_over_unknown_page_maps_nothing() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);

    let addr = addr_over(&ctx, "mw_unk_idx", 0x9010, &[0, 1, 3]);
    let result = mem.store_symbolic_unified_multi(addr, RustBV::concrete(0xAA, 8), &ctx, &concretizer);
    assert!(
        matches!(result, Err(MemoryError::Unmapped { .. })),
        "expected Unmapped, got {result:?}"
    );
    assert!(!mem.is_mapped(0x9000), "the faulting store must not map the page");
}
