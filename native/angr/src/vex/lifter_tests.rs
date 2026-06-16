use super::*;

#[test]
fn test_irsb_builder() {
    let mut builder = IRSBBuilder::new(0x1000, VexArch::AMD64);
    builder.offs_ip(184); // AMD64 RIP offset

    // mov rax, 42
    builder.imark(0x1000, 5);
    let t0 = builder.new_tmp(IRType::I64);
    builder.wrtmp(t0, IRExpr::const_u64(42));
    builder.put(16, IRExpr::tmp(t0)); // RAX offset

    // Set next instruction
    builder.next(IRExpr::const_u64(0x1005), JumpKind::Boring);

    let irsb = builder.build();
    assert_eq!(irsb.addr, 0x1000);
    assert_eq!(irsb.num_instructions(), 1);
    assert_eq!(irsb.statements.len(), 3); // IMark + WrTmp + Put
}

#[test]
fn test_ir_expr_helpers() {
    let expr = IRExpr::binop(
        IROp::Add(IRType::I64),
        IRExpr::get(16, IRType::I64),
        IRExpr::const_u64(1),
    );

    match expr {
        IRExpr::Binop {
            op,
            left: _,
            right: _,
        } => {
            assert!(matches!(op, IROp::Add(IRType::I64)));
        }
        _ => panic!("expected Binop"),
    }
}
