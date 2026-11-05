//! Tests for the `errno` negative-errno constants (extracted from errno.rs).

use super::*;

#[test]
fn neg_errno_values_match_kernel_abi() {
    assert_eq!(NEG_ONE, 0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(NEG_EBADF, 0xFFFF_FFFF_FFFF_FFF7);
    assert_eq!(NEG_EFAULT, 0xFFFF_FFFF_FFFF_FFF2);
    assert_eq!(NEG_EINVAL, 0xFFFF_FFFF_FFFF_FFEA);
    assert_eq!(NEG_ENOTTY, 0xFFFF_FFFF_FFFF_FFE7);
    assert_eq!(NEG_ERANGE, 0xFFFF_FFFF_FFFF_FFDE);
}

#[test]
fn neg_is_the_two_s_complement_of_the_errno() {
    for errno in [1_u32, 9, 14, 22, 25, 34, 4095] {
        assert_eq!(neg(errno).wrapping_add(u64::from(errno)), 0);
    }
}
