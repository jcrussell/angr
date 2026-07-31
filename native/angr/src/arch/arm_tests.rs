use super::*;

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
