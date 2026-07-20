use super::*;

#[test]
fn test_arm_basics() {
    let arch = ARM;

    assert_eq!(arch.bits(), 32);
    assert_eq!(arch.name(), "ARM");
    assert!(arch.is_little_endian());
}

#[test]
fn test_register_lookup() {
    let arch = ARM;

    assert_eq!(arch.register_offset("r0"), Some(8));
    assert_eq!(arch.register_size("r0"), Some(4));

    assert_eq!(arch.register_offset("sp"), Some(60));
    assert_eq!(arch.register_offset("r13"), Some(60));

    assert_eq!(arch.register_offset("lr"), Some(64));
    assert_eq!(arch.register_offset("r14"), Some(64));

    assert_eq!(arch.register_offset("pc"), Some(68));
}

#[test]
fn test_special_registers() {
    let arch = ARM;

    assert_eq!(arch.ip_offset(), 68); // PC
    assert_eq!(arch.sp_offset(), 60); // SP
    assert_eq!(arch.bp_offset(), Some(52)); // R11/FP
}

#[test]
fn test_neon_q_and_d_registers() {
    // Smoke test for the NEON D/Q scaffolding (angr-bkcs.1).
    // ARM has D0..D31 (64-bit) and Q0..Q15 (128-bit) overlapping in
    // pairs of D registers: Q0=D0..D1, Q1=D2..D3, ..., Q15=D30..D31.
    let arch = ARM;

    // D registers are 8 bytes wide. D0 is at VEX offset 128 (after the
    // 108-127 emnote/cmstart/cmlen/nraddr/ip_at_syscall block); the pre-fix
    // table had D0=112 and was -16 off the real layout (angr-ihfe5).
    assert_eq!(arch.register_offset("d0"), Some(128));
    assert_eq!(arch.register_size("d0"), Some(8));
    assert_eq!(arch.register_offset("d31"), Some(128 + 31 * 8));

    // Q registers are 16 bytes wide and start at the matching even
    // D register: Qn lives at offset of D(2n).
    for n in 0..16usize {
        let q = format!("q{n}");
        let d_even = format!("d{}", 2 * n);
        assert_eq!(arch.register_offset(&q), arch.register_offset(&d_even));
        assert_eq!(arch.register_size(&q), Some(16));
    }
}
