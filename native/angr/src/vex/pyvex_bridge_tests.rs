// Tests for pyvex_bridge.rs — extracted from the inline `mod tests` block.
// See parent module for the code under test.

use super::super::ir::{Endness, IROp, JumpKind};
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
            assert!(
                matches!(op, IROp::VFAddS { elem: IRType::F32 }),
                "Expected VFAddS{{F32}}, got {op:?}"
            );
        } else {
            panic!("Expected Binop expression");
        }
    } else {
        panic!("Expected WrTmp statement");
    }
}

// ---------------------------------------------------------------------------
// convert_stmt: per-IRStmt-variant deserialization coverage (angr-03vl4.73)
//
// The tests above only reach IMark / Put / WrTmp / Exit by name. The block
// below covers the remaining variants `convert_stmt` handles. Every case goes
// through `deserialize_irsb` rather than calling `convert_stmt` directly, so
// each one also pins the serde tag/field renames on `PyVexStmt` (`oldHi`,
// `expdLo`, `mFx`, `nFxState`, ...) that a rename would otherwise break
// silently at the JSON boundary.
// ---------------------------------------------------------------------------

/// Wrap one statement's JSON in a minimal AMD64 IRSB and return the single
/// converted `IRStmt`.
fn stmt_from_json(stmt_json: &str) -> IRStmt {
    let json = r#"{"addr": 4096, "arch": "AMD64", "statements": ["#.to_string()
        + stmt_json
        + r#"],
        "next": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 4100}},
        "jumpkind": "Ijk_Boring",
        "offsIP": 184,
        "tyenv": {"types": ["Ity_I64"]}}"#;

    let irsb = deserialize_irsb(&json).expect("IRSB deserialization failed");
    assert_eq!(irsb.statements.len(), 1);
    irsb.statements.into_iter().next().unwrap()
}

#[test]
fn test_stmt_noop_and_mbe() {
    assert!(matches!(
        stmt_from_json(r#"{"tag": "Ist_NoOp"}"#),
        IRStmt::NoOp
    ));

    for (event, expected) in [
        ("Imbe_Fence", MBusEvent::Fence),
        ("Imbe_SFence", MBusEvent::SFence),
        ("Imbe_LFence", MBusEvent::LFence),
        ("Imbe_MFence", MBusEvent::MFence),
    ] {
        let json = format!(r#"{{"tag": "Ist_MBE", "event": "{event}"}}"#);
        match stmt_from_json(&json) {
            IRStmt::MBE(ev) => assert_eq!(ev, expected, "MBE event {event}"),
            other => panic!("Expected MBE statement, got {other:?}"),
        }
    }
}

#[test]
fn test_stmt_abihint() {
    let stmt = stmt_from_json(
        r#"{"tag": "Ist_AbiHint",
            "base": {"tag": "Iex_Get", "offset": 48, "ty": "Ity_I64"},
            "len": 128,
            "nia": {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 4200}}}"#,
    );

    match stmt {
        IRStmt::AbiHint { base, len, nia } => {
            assert!(matches!(
                *base,
                IRExpr::Get {
                    offset: 48,
                    ty: IRType::I64
                }
            ));
            assert_eq!(len, 128);
            assert!(matches!(*nia, IRExpr::Const(IRConst::U64(4200))));
        }
        other => panic!("Expected AbiHint statement, got {other:?}"),
    }
}

#[test]
fn test_stmt_puti() {
    let stmt = stmt_from_json(
        r#"{"tag": "Ist_PutI",
            "descr": {"base": 776, "elemTy": "Ity_F64", "nElems": 8},
            "ix": {"tag": "Iex_RdTmp", "tmp": 3},
            "bias": 2,
            "data": {"tag": "Iex_RdTmp", "tmp": 4}}"#,
    );

    match stmt {
        IRStmt::PutI {
            descr,
            ix,
            bias,
            data,
        } => {
            assert_eq!(descr.base, 776);
            assert_eq!(descr.elemTy, IRType::F64);
            assert_eq!(descr.nElems, 8);
            assert!(matches!(*ix, IRExpr::RdTmp(3)));
            assert_eq!(bias, 2);
            assert!(matches!(*data, IRExpr::RdTmp(4)));
        }
        other => panic!("Expected PutI statement, got {other:?}"),
    }
}

#[test]
fn test_stmt_store_threads_endness() {
    for (end, expected) in [("Iend_LE", Endness::Little), ("Iend_BE", Endness::Big)] {
        let json = format!(
            r#"{{"tag": "Ist_Store",
                 "addr": {{"tag": "Iex_RdTmp", "tmp": 1}},
                 "data": {{"tag": "Iex_Const", "con": {{"tag": "Ico_U32", "value": 7}}}},
                 "end": "{end}"}}"#
        );
        match stmt_from_json(&json) {
            IRStmt::Store {
                addr,
                data,
                endness,
            } => {
                assert!(matches!(addr, IRExpr::RdTmp(1)));
                assert!(matches!(data, IRExpr::Const(IRConst::U32(7))));
                assert_eq!(endness, expected, "endness for {end}");
            }
            other => panic!("Expected Store statement, got {other:?}"),
        }
    }
}

#[test]
fn test_stmt_storeg() {
    let stmt = stmt_from_json(
        r#"{"tag": "Ist_StoreG",
            "addr": {"tag": "Iex_RdTmp", "tmp": 2},
            "data": {"tag": "Iex_RdTmp", "tmp": 3},
            "guard": {"tag": "Iex_RdTmp", "tmp": 4},
            "end": "Iend_BE"}"#,
    );

    match stmt {
        IRStmt::StoreG {
            addr,
            data,
            guard,
            endness,
        } => {
            assert!(matches!(*addr, IRExpr::RdTmp(2)));
            assert!(matches!(*data, IRExpr::RdTmp(3)));
            assert!(matches!(*guard, IRExpr::RdTmp(4)));
            assert_eq!(endness, Endness::Big);
        }
        other => panic!("Expected StoreG statement, got {other:?}"),
    }
}

#[test]
fn test_stmt_loadg() {
    let stmt = stmt_from_json(
        r#"{"tag": "Ist_LoadG",
            "dst": 5,
            "addr": {"tag": "Iex_RdTmp", "tmp": 6},
            "alt": {"tag": "Iex_Const", "con": {"tag": "Ico_U32", "value": 0}},
            "guard": {"tag": "Iex_RdTmp", "tmp": 7},
            "cvt": "ILGop_16Sto32",
            "end": "Iend_LE"}"#,
    );

    match stmt {
        IRStmt::LoadG {
            dst,
            addr,
            alt,
            guard,
            cvt,
            endness,
        } => {
            assert_eq!(dst, 5);
            assert!(matches!(*addr, IRExpr::RdTmp(6)));
            assert!(matches!(*alt, IRExpr::Const(IRConst::U32(0))));
            assert!(matches!(*guard, IRExpr::RdTmp(7)));
            assert_eq!(cvt, IRLoadGOp::WidenS { src_bits: 16 });
            assert_eq!(endness, Endness::Little);
        }
        other => panic!("Expected LoadG statement, got {other:?}"),
    }
}

#[test]
fn test_stmt_cas_single_width() {
    // Single-width CAS: every *Hi half is absent, and must stay `None` rather
    // than being filled in with a placeholder.
    let stmt = stmt_from_json(
        r#"{"tag": "Ist_CAS",
            "oldHi": null,
            "oldLo": 9,
            "addr": {"tag": "Iex_RdTmp", "tmp": 1},
            "expdHi": null,
            "expdLo": {"tag": "Iex_RdTmp", "tmp": 2},
            "dataHi": null,
            "dataLo": {"tag": "Iex_RdTmp", "tmp": 3},
            "end": "Iend_LE"}"#,
    );

    match stmt {
        IRStmt::CAS {
            old_hi,
            old_lo,
            addr,
            expdHi,
            expdLo,
            dataHi,
            dataLo,
            endness,
        } => {
            assert_eq!(old_hi, None);
            assert_eq!(old_lo, 9);
            assert!(matches!(*addr, IRExpr::RdTmp(1)));
            assert!(expdHi.is_none());
            assert!(matches!(*expdLo, IRExpr::RdTmp(2)));
            assert!(dataHi.is_none());
            assert!(matches!(*dataLo, IRExpr::RdTmp(3)));
            assert_eq!(endness, Endness::Little);
        }
        other => panic!("Expected CAS statement, got {other:?}"),
    }
}

#[test]
fn test_stmt_cas_double_width() {
    let stmt = stmt_from_json(
        r#"{"tag": "Ist_CAS",
            "oldHi": 11,
            "oldLo": 10,
            "addr": {"tag": "Iex_RdTmp", "tmp": 1},
            "expdHi": {"tag": "Iex_RdTmp", "tmp": 4},
            "expdLo": {"tag": "Iex_RdTmp", "tmp": 2},
            "dataHi": {"tag": "Iex_RdTmp", "tmp": 5},
            "dataLo": {"tag": "Iex_RdTmp", "tmp": 3},
            "end": "Iend_BE"}"#,
    );

    match stmt {
        IRStmt::CAS {
            old_hi,
            old_lo,
            expdHi,
            dataHi,
            endness,
            ..
        } => {
            assert_eq!(old_hi, Some(11));
            assert_eq!(old_lo, 10);
            assert!(matches!(expdHi.as_deref(), Some(IRExpr::RdTmp(4))));
            assert!(matches!(dataHi.as_deref(), Some(IRExpr::RdTmp(5))));
            assert_eq!(endness, Endness::Big);
        }
        other => panic!("Expected CAS statement, got {other:?}"),
    }
}

#[test]
fn test_stmt_llsc_load_linked_and_store_conditional() {
    // Load-linked: no store data.
    let ll = stmt_from_json(
        r#"{"tag": "Ist_LLSC",
            "storedata": null,
            "result": 12,
            "addr": {"tag": "Iex_RdTmp", "tmp": 1},
            "end": "Iend_BE"}"#,
    );
    match ll {
        IRStmt::LLSC {
            storedata,
            result,
            addr,
            endness,
        } => {
            assert!(storedata.is_none());
            assert_eq!(result, 12);
            assert!(matches!(*addr, IRExpr::RdTmp(1)));
            assert_eq!(endness, Endness::Big);
        }
        other => panic!("Expected LLSC statement, got {other:?}"),
    }

    // Store-conditional: store data present.
    let sc = stmt_from_json(
        r#"{"tag": "Ist_LLSC",
            "storedata": {"tag": "Iex_RdTmp", "tmp": 2},
            "result": 13,
            "addr": {"tag": "Iex_RdTmp", "tmp": 1},
            "end": "Iend_LE"}"#,
    );
    match sc {
        IRStmt::LLSC {
            storedata,
            result,
            endness,
            ..
        } => {
            assert!(matches!(storedata.as_deref(), Some(IRExpr::RdTmp(2))));
            assert_eq!(result, 13);
            assert_eq!(endness, Endness::Little);
        }
        other => panic!("Expected LLSC statement, got {other:?}"),
    }
}

#[test]
fn test_stmt_dirty_with_result_tmp() {
    let stmt = stmt_from_json(
        r#"{"tag": "Ist_Dirty",
            "cee": {"name": "amd64g_dirtyhelper_RDTSC", "addr": 140737488, "mcx_mask": 0},
            "guard": null,
            "tmp": 14,
            "mFx": "Ifx_None",
            "mAddr": null,
            "mSize": 0,
            "nFxState": 0,
            "args": [{"tag": "Iex_GSPTR"}]}"#,
    );

    match stmt {
        IRStmt::Dirty(d) => {
            assert_eq!(d.cee.name, "amd64g_dirtyhelper_RDTSC");
            assert_eq!(d.cee.addr, 140_737_488);
            assert_eq!(d.cee.mcx_mask, 0);
            assert!(d.guard.is_none());
            assert_eq!(d.tmp, Some(14));
            assert_eq!(d.mFx, DirtyFx::None);
            assert!(d.mAddr.is_none());
            assert_eq!(d.mSize, 0);
            assert_eq!(d.nFxState, 0);
            assert_eq!(d.args.len(), 1);
            assert!(matches!(d.args[0], IRExpr::GSPTR));
        }
        other => panic!("Expected Dirty statement, got {other:?}"),
    }
}

#[test]
fn test_stmt_dirty_guarded_memory_effect() {
    // A guarded dirty call with no result tmp and a declared memory effect —
    // the shape that exercises every optional field in the other direction.
    let stmt = stmt_from_json(
        r#"{"tag": "Ist_Dirty",
            "cee": {"name": "amd64g_dirtyhelper_FXSAVE", "addr": 4096, "mcx_mask": 7},
            "guard": {"tag": "Iex_RdTmp", "tmp": 15},
            "tmp": null,
            "mFx": "Ifx_Modify",
            "mAddr": {"tag": "Iex_RdTmp", "tmp": 16},
            "mSize": 512,
            "nFxState": 2,
            "args": [{"tag": "Iex_GSPTR"}, {"tag": "Iex_RdTmp", "tmp": 16}]}"#,
    );

    match stmt {
        IRStmt::Dirty(d) => {
            assert_eq!(d.cee.name, "amd64g_dirtyhelper_FXSAVE");
            assert_eq!(d.cee.mcx_mask, 7);
            assert!(matches!(d.guard.as_deref(), Some(IRExpr::RdTmp(15))));
            assert_eq!(d.tmp, None);
            assert_eq!(d.mFx, DirtyFx::Modify);
            assert!(matches!(d.mAddr.as_deref(), Some(IRExpr::RdTmp(16))));
            assert_eq!(d.mSize, 512);
            assert_eq!(d.nFxState, 2);
            assert_eq!(d.args.len(), 2);
        }
        other => panic!("Expected Dirty statement, got {other:?}"),
    }
}

/// The two untagged string->enum fallbacks on this boundary (angr-03vl4.69):
/// an unrecognized `Imbe_*` collapses to `Fence` (inert — the interpreter
/// discards the payload) and an unrecognized `Ifx_*` to the logged
/// `DirtyFx::None`. Driven through the JSON layer so the `#[serde(rename)]`s
/// stay covered too.
#[test]
fn test_unknown_mbe_and_dirty_fx_strings_fall_back() {
    match stmt_from_json(r#"{"tag": "Ist_MBE", "event": "Imbe_NotAThing"}"#) {
        IRStmt::MBE(ev) => assert_eq!(ev, MBusEvent::Fence),
        other => panic!("Expected MBE statement, got {other:?}"),
    }

    let stmt = stmt_from_json(
        r#"{"tag": "Ist_Dirty",
            "cee": {"name": "helper", "addr": 4096, "mcx_mask": 0},
            "guard": null,
            "tmp": null,
            "mFx": "Ifx_NotAThing",
            "mAddr": null,
            "mSize": 0,
            "nFxState": 0,
            "args": []}"#,
    );
    match stmt {
        IRStmt::Dirty(d) => assert_eq!(d.mFx, DirtyFx::None),
        other => panic!("Expected Dirty statement, got {other:?}"),
    }
}

/// `parse_endness`'s logged little-endian fallback, reached the way the live
/// path reaches it: an `Ist_Store` carrying an endness string neither marshal
/// path knows. The unit-level coverage lives in `opcode_map_tests.rs`.
#[test]
fn test_store_with_unknown_endness_falls_back_to_little() {
    let stmt = stmt_from_json(
        r#"{"tag": "Ist_Store", "end": "Iend_ME",
            "addr": {"tag": "Iex_RdTmp", "tmp": 3},
            "data": {"tag": "Iex_RdTmp", "tmp": 4}}"#,
    );
    match stmt {
        IRStmt::Store { endness, .. } => assert_eq!(endness, Endness::Little),
        other => panic!("Expected Store statement, got {other:?}"),
    }
}
