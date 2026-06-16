use super::*;

#[test]
fn test_mips32_basics() {
    let arch = MIPS32;

    assert_eq!(arch.bits(), 32);
    assert_eq!(arch.name(), "MIPS32");
}

#[test]
fn test_mips64_basics() {
    let arch = MIPS64;

    assert_eq!(arch.bits(), 64);
    assert_eq!(arch.name(), "MIPS64");
}

#[test]
fn test_mips32_register_lookup() {
    let arch = MIPS32;

    assert_eq!(arch.register_offset("v0"), Some(16));
    assert_eq!(arch.register_offset("$2"), Some(16));
    assert_eq!(arch.register_offset("r2"), Some(16));

    assert_eq!(arch.register_offset("sp"), Some(124));
    assert_eq!(arch.register_offset("$29"), Some(124));

    assert_eq!(arch.register_size("v0"), Some(4));
}

#[test]
fn test_mips64_register_lookup() {
    let arch = MIPS64;

    assert_eq!(arch.register_offset("v0"), Some(32));
    assert_eq!(arch.register_size("v0"), Some(8));
}
