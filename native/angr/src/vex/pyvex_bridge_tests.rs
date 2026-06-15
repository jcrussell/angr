// Tests for pyvex_bridge.rs — extracted from the inline `mod tests` block.
// See parent module for the code under test.

use super::super::ir::{IROp, JumpKind};
use super::*;

#[test]
fn test_simple_irsb_deserialization() {
    let json = r#"{
        "addr": 4096,
        "arch": "AMD64",
        "statements": [
            {"tag": "Ist_IMark", "addr": 4096, "len": 3, "delta": 0},
            {"tag": "Ist_Put", "offset": 16, "data": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 42}}}
        ],
        "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 4099}},
        "jumpkind": "Ijk_Boring",
        "offsIP": 184,
        "tyenv": {"types": ["Ity_I64", "Ity_I32"]}
    }"#;

    let irsb = deserialize_irsb(json).unwrap();
    assert_eq!(irsb.addr, 4096);
    assert_eq!(irsb.statements.len(), 2);
    assert!(matches!(irsb.arch, VexArch::AMD64));
    assert!(matches!(irsb.jumpkind, JumpKind::Boring));
}

#[test]
fn test_parse_loadg_op_carries_source_width() {
    use super::super::ir::IRLoadGOp;
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
fn test_binop_expression() {
    let json = r#"{
        "addr": 4096,
        "arch": "AMD64",
        "statements": [
            {"tag": "Ist_IMark", "addr": 4096, "len": 4, "delta": 0},
            {"tag": "Ist_WrTmp", "tmp": 0, "data": {
                "tag": "Iex_Binop",
                "op": "Iop_Add64",
                "args": [
                    {"tag": "Iex_Get", "offset": 16, "ty": "Ity_I64"},
                    {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 8}}
                ]
            }}
        ],
        "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 4100}},
        "jumpkind": "Ijk_Boring",
        "offsIP": 184,
        "tyenv": {"types": ["Ity_I64"]}
    }"#;

    let irsb = deserialize_irsb(json).unwrap();
    assert_eq!(irsb.statements.len(), 2);

    if let IRStmt::WrTmp { tmp, data } = &irsb.statements[1] {
        assert_eq!(*tmp, 0);
        if let IRExpr::Binop { op, .. } = data {
            assert!(matches!(op, IROp::Add(IRType::I64)));
        } else {
            panic!("Expected Binop expression");
        }
    } else {
        panic!("Expected WrTmp statement");
    }
}

#[test]
fn test_exit_statement() {
    let json = r#"{
        "addr": 4096,
        "arch": "AMD64",
        "statements": [
            {"tag": "Ist_IMark", "addr": 4096, "len": 2, "delta": 0},
            {"tag": "Ist_Exit", "guard": {"tag": "Iex_RdTmp", "tmp": 0}, "dst": {"tag": "Ico_U64", "value": 8192}, "jk": "Ijk_Boring", "offsIP": 184}
        ],
        "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 4098}},
        "jumpkind": "Ijk_Boring",
        "offsIP": 184,
        "tyenv": {"types": ["Ity_I1"]}
    }"#;

    let irsb = deserialize_irsb(json).unwrap();

    if let IRStmt::Exit { guard, dst, jk, .. } = &irsb.statements[1] {
        assert!(matches!(guard, IRExpr::RdTmp(0)));
        assert_eq!(*dst, 8192);
        assert!(matches!(jk, JumpKind::Boring));
    } else {
        panic!("Expected Exit statement");
    }
}

#[test]
fn test_arch_parsing() {
    assert!(matches!(parse_arch("AMD64"), Ok(VexArch::AMD64)));
    assert!(matches!(parse_arch("amd64"), Ok(VexArch::AMD64)));
    assert!(matches!(parse_arch("x86"), Ok(VexArch::X86)));
    assert!(matches!(parse_arch("ARM"), Ok(VexArch::ARM)));
    assert!(matches!(parse_arch("arm64"), Ok(VexArch::ARM64)));
    assert!(matches!(parse_arch("AARCH64"), Ok(VexArch::ARM64)));
    assert!(parse_arch("invalid").is_err());
}

#[test]
fn test_addss_irsb_parsing() {
    // This is the exact JSON from ADDSS xmm0, xmm1 (f3 0f 58 c1)
    let json = r#"{
        "addr": 4096,
        "arch": "X86",
        "statements": [
            {"tag": "Ist_IMark", "addr": 4096, "len": 4, "delta": 0},
            {"tag": "Ist_WrTmp", "tmp": 1, "data": {"tag": "Iex_Get", "offset": 176, "ty": "Ity_V128"}},
            {"tag": "Ist_WrTmp", "tmp": 2, "data": {"tag": "Iex_Get", "offset": 160, "ty": "Ity_V128"}},
            {"tag": "Ist_WrTmp", "tmp": 0, "data": {
                "tag": "Iex_Binop",
                "op": "Iop_Add32F0x4",
                "args": [
                    {"tag": "Iex_RdTmp", "tmp": 2},
                    {"tag": "Iex_RdTmp", "tmp": 1}
                ]
            }},
            {"tag": "Ist_Put", "offset": 160, "data": {"tag": "Iex_RdTmp", "tmp": 0}}
        ],
        "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U32", "value": 4100}},
        "jumpkind": "Ijk_Boring",
        "offsIP": 68,
        "tyenv": {"types": ["Ity_V128", "Ity_V128", "Ity_V128", "Ity_I32"]}
    }"#;

    let irsb = deserialize_irsb(json).unwrap();
    assert_eq!(irsb.addr, 4096);
    assert_eq!(irsb.statements.len(), 5);
    assert!(matches!(irsb.arch, VexArch::X86));

    // Check that the binop is VFAddS
    if let IRStmt::WrTmp { tmp, data } = &irsb.statements[3] {
        assert_eq!(*tmp, 0);
        if let IRExpr::Binop {
            op,
            left: _,
            right: _,
        } = data
        {
            println!("Parsed opcode: {:?}", op);
            assert!(
                matches!(op, IROp::VFAddS { elem: IRType::F32 }),
                "Expected VFAddS{{F32}}, got {:?}",
                op
            );
        } else {
            panic!("Expected Binop expression");
        }
    } else {
        panic!("Expected WrTmp statement");
    }
}
