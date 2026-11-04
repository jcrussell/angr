//! Tests for [`super`] that cross-check the core `vex::ccall` helpers against
//! the shell's [`super::SUPPORTED`] list. They live here, not in
//! `vex/ccall_tests.rs`, because `vex/` names nothing outside `vex` and
//! `angr_bv` (angr-mx00p.51.6).

use angr_bv::SymBV;

use super::SUPPORTED;
use crate::vex::ccall::ccall_symbolic;

/// Every served helper, over every selector, builds a `ret_bits`-wide result
/// with no `unbuildable_op` stand-in: under `ccall_symbolic`'s width guard no
/// op can mismatch widths (the width `debug_assert!`s are on in test builds).
#[test]
fn ccall_symbolic_selector_sweep_builds_every_result() {
    for &name in SUPPORTED {
        let width = if name.starts_with("amd64g") || name.starts_with("arm64g") {
            64
        } else {
            32
        };
        let selectors: Vec<u64> = if name.ends_with("calculate_condition") {
            if name.starts_with("arm") {
                (0..256).collect()
            } else {
                (0..18).flat_map(|c| (0..67).map(move |op| c << 8 | op)).collect()
            }
        } else {
            (0..67).collect()
        };
        let mut served = 0;
        for sel in selectors {
            let deps = ["d1", "d2", "d3"].map(|d| SymBV::symbol(format!("sweep_{d}"), width));
            let mut args = if name.ends_with("calculate_condition") && !name.starts_with("arm") {
                vec![
                    SymBV::concrete(u128::from(sel >> 8), width),
                    SymBV::concrete(u128::from(sel & 0xff), width),
                ]
            } else {
                vec![SymBV::concrete(u128::from(sel), width)]
            };
            args.extend(deps);
            let Some(result) = ccall_symbolic(name, &args, width) else {
                continue;
            };
            served += 1;
            assert_eq!(result.width(), width, "{name} selector {sel:#x}");
            if let Ok(term) = result.to_clarirs() {
                assert!(
                    !term
                        .variables()
                        .iter()
                        .any(|v| v.as_str().starts_with("unbuildable_op_")),
                    "{name} selector {sel:#x} built an unbuildable op"
                );
            }
        }
        assert!(served > 0, "{name} served no selector");
    }
}
