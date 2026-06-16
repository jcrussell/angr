use super::*;

#[test]
fn test_bytes_to_bv() {
    let bytes = vec![0x78, 0x56, 0x34, 0x12];
    let bv = bytes_to_bv(&bytes, 32);
    assert_eq!(bv.as_u64(), Some(0x12345678));
}

#[test]
fn test_bv_to_bytes() {
    let bv = RustBV::concrete(0x12345678, 32);
    let bytes = bv_to_bytes(&bv);
    assert_eq!(bytes, vec![0x78, 0x56, 0x34, 0x12]);
}

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
