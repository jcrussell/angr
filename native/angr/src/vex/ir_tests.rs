use super::*;

#[test]
fn test_ir_type_sizes() {
    assert_eq!(IRType::I1.bits(), 1);
    assert_eq!(IRType::I8.bits(), 8);
    assert_eq!(IRType::I32.bits(), 32);
    assert_eq!(IRType::I64.bits(), 64);
    assert_eq!(IRType::V128.bits(), 128);
}

#[test]
fn test_ir_const_types() {
    assert_eq!(IRConst::U8(42).get_type(), IRType::I8);
    assert_eq!(IRConst::U32(42).get_type(), IRType::I32);
    assert_eq!(IRConst::U64(42).get_type(), IRType::I64);
}

#[test]
fn test_irsb_creation() {
    let irsb = IRSB::new(0x1000, VexArch::AMD64);
    assert_eq!(irsb.addr, 0x1000);
    assert_eq!(irsb.num_instructions(), 0);
}
