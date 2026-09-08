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

/// `mov.d $f0, $f2` as VEX lifts it under FR=0: an I32 copy of the `fN_lo`
/// sub-field sitting at each register's base offset. On a big-endian MIPS32
/// target that field is the register's HIGH word, so the standalone
/// interpreter must land the value in `f0`'s high half — the same answer
/// `arch::register_file_tests::be_register_file_mirrors_sub_register_access` pins for
/// the register file, now driven through a whole block (angr-21cz6).
fn run_mips_movd_lo_copy(is_le: bool) -> u64 {
    let ctx = SymContext::new_mock();
    let mut interp = VEXInterpreter::with_config_endian(
        VexArch::MIPS32,
        &ctx,
        crate::callbacks::ExecutionConfig::default(),
        is_le,
    );
    let f0 = interp.registers.arch().register_offset("f0").unwrap();
    let f2 = interp.registers.arch().register_offset("f2").unwrap();
    interp
        .registers
        .put_reg("f2", RustBV::concrete(0xAABB_CCDD_1122_3344, 64));

    let mut irsb = IRSB::new(0x400_000, VexArch::MIPS32);
    irsb.tyenv.new_temp(IRType::I32);
    irsb.statements.push(IRStmt::IMark {
        addr: 0x400_000,
        len: 4,
        delta: 0,
    });
    irsb.statements.push(IRStmt::WrTmp {
        tmp: 0,
        data: IRExpr::Get {
            offset: f2,
            ty: IRType::I32,
        },
    });
    irsb.statements.push(IRStmt::Put {
        offset: f0,
        data: IRExpr::RdTmp(0),
    });

    Python::initialize();
    let callbacks = PythonCallbacks::new();
    Python::attach(|_py| {
        interp
            .execute_block(&callbacks, &irsb)
            .expect("mov.d block executes");
    });
    interp.registers.get_reg("f0", &ctx).unwrap().as_u64().unwrap()
}

#[test]
fn be_standalone_interpreter_mirrors_sub_register_put() {
    assert_eq!(run_mips_movd_lo_copy(false), 0xAABB_CCDD_0000_0000);
}

/// The little-endian default is untouched: `f2_lo` is the low word there.
#[test]
fn le_standalone_interpreter_does_not_mirror_sub_register_put() {
    assert_eq!(run_mips_movd_lo_copy(true), 0x1122_3344);
}
