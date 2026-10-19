//! Tests for `pyvex_reader.rs`'s pure string and constant helpers.
//!
//! Reading statements and expressions needs live pyvex objects, which `cargo
//! test` does not have; those tests are in
//! `tests/engines/rust/test_pyvex_reader.py`.

use super::*;

#[test]
fn test_parse_loadg_op_carries_source_width() {
    // Identity variants — no widening.
    assert_eq!(parse_loadg_op("ILGop_Ident32"), IRLoadGOp::Identity);
    assert_eq!(parse_loadg_op("ILGop_Ident64"), IRLoadGOp::Identity);
    assert_eq!(parse_loadg_op("ILGop_IdentV128"), IRLoadGOp::Identity);
    // The four canonical VEX widening ops (all widen to 32 bits).
    assert_eq!(
        parse_loadg_op("ILGop_8Uto32"),
        IRLoadGOp::WidenZ { src_bits: 8 }
    );
    assert_eq!(
        parse_loadg_op("ILGop_8Sto32"),
        IRLoadGOp::WidenS { src_bits: 8 }
    );
    assert_eq!(
        parse_loadg_op("ILGop_16Uto32"),
        IRLoadGOp::WidenZ { src_bits: 16 }
    );
    assert_eq!(
        parse_loadg_op("ILGop_16Sto32"),
        IRLoadGOp::WidenS { src_bits: 16 }
    );
    // Defensive *to64 forms.
    assert_eq!(
        parse_loadg_op("ILGop_16Uto64"),
        IRLoadGOp::WidenZ { src_bits: 16 }
    );
    assert_eq!(
        parse_loadg_op("ILGop_16Sto64"),
        IRLoadGOp::WidenS { src_bits: 16 }
    );
    assert_eq!(
        parse_loadg_op("ILGop_32Uto64"),
        IRLoadGOp::WidenZ { src_bits: 32 }
    );
    assert_eq!(
        parse_loadg_op("ILGop_32Sto64"),
        IRLoadGOp::WidenS { src_bits: 32 }
    );
    // Unknown strings are flagged, not silently treated as Identity.
    assert_eq!(parse_loadg_op("ILGop_INVALID"), IRLoadGOp::Unknown);
    assert_eq!(parse_loadg_op("nonsense"), IRLoadGOp::Unknown);
}

#[test]
fn test_arch_parsing() {
    assert_eq!(parse_arch("AMD64"), Some(VexArch::AMD64));
    assert_eq!(parse_arch("amd64"), Some(VexArch::AMD64));
    assert_eq!(parse_arch("X86"), Some(VexArch::X86));
    assert_eq!(parse_arch("ARMEL"), Some(VexArch::ARM));
    assert_eq!(parse_arch("ARMHF"), Some(VexArch::ARM));
    assert_eq!(parse_arch("ARMCortexM"), Some(VexArch::ARM));
    assert_eq!(parse_arch("AARCH64"), Some(VexArch::ARM64));
    assert_eq!(parse_arch("MIPS32"), Some(VexArch::MIPS32));
    assert_eq!(parse_arch("MIPS64"), Some(VexArch::MIPS64));
    assert_eq!(parse_arch("invalid"), None);
}

/// The two inert/logged string fallbacks: an unknown MBE event is a full
/// fence, an unknown Dirty effect is `Ifx_None` (logged at debug).
#[test]
fn test_unknown_mbe_and_dirty_fx_strings_fall_back() {
    assert_eq!(parse_mbe_event("Imbe_NotAThing"), MBusEvent::Fence);
    assert_eq!(parse_mbe_event("Imbe_CancelReservation"), MBusEvent::Fence);
    assert_eq!(parse_dirty_fx("Ifx_NotAThing"), DirtyFx::None);
    assert_eq!(parse_dirty_fx("Ifx_Modify"), DirtyFx::Modify);
}

#[test]
fn test_exit_destination_takes_the_low_64_bits() {
    assert_eq!(const_to_u64(&IRConst::U32(0x2000)), 0x2000);
    assert_eq!(const_to_u64(&IRConst::U64(u64::MAX)), u64::MAX);
    assert_eq!(const_to_u64(&IRConst::U128(1 << 64 | 7)), 7);
    assert_eq!(const_to_u64(&IRConst::U1(true)), 1);
}
