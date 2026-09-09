//! Unit tests for `SymbolicMemory::load_concrete_or_unconstrained`'s
//! unmapped-address fallback — the Err path that only fires for unmapped
//! addresses, since the ITE build path always hits Ok (angr-szg45.4).

use super::super::*;

/// Extract the variable name from a `Symbolic` BV, panicking otherwise.
fn sym_name(bv: &RustBV) -> String {
    match bv {
        RustBV::Symbolic { name, .. } => name.to_string(),
        other => panic!("expected Symbolic BV, got {other:?}"),
    }
}

/// With `zero_fill_unconstrained=true`, an unmapped load returns concrete 0 of
/// width `size*8`.
#[test]
fn test_load_concrete_or_unconstrained_zero_fill() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.set_zero_fill_unconstrained(true);

    let v = mem.load_concrete_or_unconstrained(0x5000u64, 4, &ctx);
    assert_eq!(v.width(), 32, "width must be size*8");
    assert_eq!(v.as_u64(), Some(0), "zero-fill must produce concrete 0");
}

/// With `zero_fill_unconstrained=false` (the default), an unmapped load returns
/// a fresh `unc_mem_`-prefixed symbolic of width `size*8`, and successive calls
/// produce distinct names — guaranteeing distinct constraint identity across
/// unmapped leaves.
#[test]
fn test_load_concrete_or_unconstrained_symbolic_distinct_names() {
    let ctx = SymContext::new_mock();
    let mem = SymbolicMemory::new(Endness::Little);
    assert!(!mem.zero_fill_unconstrained(), "default must be false");

    let v1 = mem.load_concrete_or_unconstrained(0x5000u64, 4, &ctx);
    let v2 = mem.load_concrete_or_unconstrained(0x5000u64, 4, &ctx);

    assert_eq!(v1.width(), 32, "non-byte size=4 must give a 32-bit value");
    let n1 = sym_name(&v1);
    let n2 = sym_name(&v2);
    assert!(
        n1.starts_with("unc_mem_"),
        "name must be unc_mem_-prefixed: {n1}"
    );
    assert!(
        n2.starts_with("unc_mem_"),
        "name must be unc_mem_-prefixed: {n2}"
    );
    assert_ne!(
        n1, n2,
        "successive unmapped loads must produce distinct names"
    );
}

/// angr-03vl4.41: uniqueness must hold *across* independent ITE-tree builds,
/// not just within one. The builder used to seed a fresh `counter = 0` per
/// call, so the first unmapped leaf of every build stringified identically
/// (`unc_mem_<addr>_1`) — and since `RustBV::from_parts` interns the Z3
/// constant **by name**, two logically-distinct unconstrained reads aliased in
/// the solver. Two separate builds over the same unmapped address set must
/// yield disjoint symbol names.
#[test]
fn test_ite_builds_do_not_reuse_unconstrained_names() {
    let ctx = SymContext::new_mock();
    let mem = SymbolicMemory::new(Endness::Little);
    let addrs = [0x5000u64, 0x6000u64];
    let addr_expr = RustBV::symbolic(&ctx, "ite_addr", 64);

    let names = |bv: &RustBV| -> Vec<String> {
        let mut out = Vec::new();
        collect_symbol_names(bv, &mut out);
        out
    };

    let t1 = mem
        .build_balanced_ite_load_after_prep(&addr_expr, &addrs, 4, &ctx)
        .expect("unmapped leaves fall back, they do not error");
    let t2 = mem
        .build_balanced_ite_load_after_prep(&addr_expr, &addrs, 4, &ctx)
        .expect("unmapped leaves fall back, they do not error");

    let n1 = names(&t1);
    let n2 = names(&t2);
    assert!(
        n1.iter().any(|n| n.starts_with("unc_mem_")),
        "both leaves are unmapped, so the tree must contain unc_mem_ symbols: {n1:?}"
    );
    for name in n1.iter().filter(|n| n.starts_with("unc_mem_")) {
        assert!(
            !n2.contains(name),
            "unconstrained name {name} reused across independent ITE builds"
        );
    }
}

/// Collect every `Symbolic` leaf name reachable from `bv`, walking the
/// `Expression` children an ITE tree is built from.
fn collect_symbol_names(bv: &RustBV, out: &mut Vec<String>) {
    match bv {
        RustBV::Symbolic { name, .. } => out.push(name.to_string()),
        RustBV::Expression { operands, .. } => {
            for operand in operands.iter() {
                collect_symbol_names(operand, out);
            }
        }
        _ => {}
    }
}
