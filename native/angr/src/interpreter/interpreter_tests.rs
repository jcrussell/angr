//! Unit tests for the [`super::VEXInterpreter`] struct defined in
//! `interpreter/mod.rs` itself — construction and the hook set. Op-level and
//! statement-level coverage lives in the per-submodule `*_tests.rs` siblings.

use super::*;

#[test]
fn test_interpreter_creation() {
    let ctx = SymContext::new_mock();
    let interp = VEXInterpreter::new(VexArch::AMD64, &ctx);
    assert_eq!(interp.get_pc(), 0);
}

#[test]
fn test_hook_management() {
    let ctx = SymContext::new_mock();
    let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);

    interp.add_hook(0x1000);
    assert!(interp.is_hooked(0x1000));
    assert!(!interp.is_hooked(0x2000));

    interp.remove_hook(0x1000);
    assert!(!interp.is_hooked(0x1000));
}
