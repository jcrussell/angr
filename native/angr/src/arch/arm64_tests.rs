// Unit tests for arch/arm64.rs, extracted to a sibling file.
// See rust-mod-tests-sibling-extraction (bd memory) for the pattern.

use super::*;

#[test]
fn test_arm64_basics() {
    let arch = ARM64;

    assert_eq!(arch.bits(), 64);
    assert_eq!(arch.name(), "ARM64");
    assert!(arch.is_little_endian());
}

#[test]
fn test_register_lookup() {
    let arch = ARM64;

    assert_eq!(arch.register_offset("x0"), Some(16));
    assert_eq!(arch.register_size("x0"), Some(8));

    assert_eq!(arch.register_offset("w0"), Some(16));
    assert_eq!(arch.register_size("w0"), Some(4));

    assert_eq!(arch.register_offset("sp"), Some(264));
    assert_eq!(arch.register_offset("xsp"), Some(264));

    assert_eq!(arch.register_offset("lr"), Some(256));
    assert_eq!(arch.register_offset("x30"), Some(256));

    assert_eq!(arch.register_offset("fp"), Some(248));
    assert_eq!(arch.register_offset("x29"), Some(248));
}

#[test]
fn test_special_registers() {
    let arch = ARM64;

    assert_eq!(arch.ip_offset(), 272); // PC
    assert_eq!(arch.sp_offset(), 264); // SP
    assert_eq!(arch.bp_offset(), Some(248)); // FP (X29)
}

#[test]
fn test_neon_q_registers() {
    // Smoke test for the NEON Q/V register scaffolding (angr-bkcs.1):
    // Q0..Q31 (and the V0..V31 aliases) must resolve to the 128-bit
    // SIMD register file slots, and D0..D31 must reference the lower
    // 64 bits of the matching Q register so D/Q overlap correctly.
    let arch = ARM64;

    // Q registers are 16 bytes wide and live at Q0=320..Q31=816,
    // 16 bytes apart.
    assert_eq!(arch.register_offset("q0"), Some(320));
    assert_eq!(arch.register_size("q0"), Some(16));
    assert_eq!(arch.register_offset("q31"), Some(320 + 31 * 16));
    assert_eq!(arch.register_size("q31"), Some(16));

    // V registers alias Q registers (same offset, same width).
    for i in 0..32 {
        let q = format!("q{i}");
        let v = format!("v{i}");
        assert_eq!(arch.register_offset(&q), arch.register_offset(&v));
        assert_eq!(arch.register_size(&q), arch.register_size(&v));
    }

    // D registers are the lower 64 bits of the matching Q register.
    for i in 0..32 {
        let q = format!("q{i}");
        let d = format!("d{i}");
        assert_eq!(arch.register_offset(&q), arch.register_offset(&d));
        assert_eq!(arch.register_size(&d), Some(8));
    }

    // The saturation flag sits just past the V register file; it is a U128,
    // so FPCR lands 56 bytes further on, past emnote/cmstart/cmlen/nraddr/
    // ip_at_syscall (angr-a4xix, angr-zxzi3).
    assert_eq!(arch.register_offset("qcflag"), Some(832));
    assert_eq!(arch.register_size("qcflag"), Some(16));
    assert_eq!(arch.register_offset("fpcr"), Some(888));
    assert_eq!(arch.register_size("fpcr"), Some(4));
    assert_eq!(arch.register_offset("ip_at_syscall"), Some(880));
}
