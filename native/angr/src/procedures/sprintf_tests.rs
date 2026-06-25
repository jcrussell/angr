//! Tests for the sprintf/snprintf SimProcedure (extracted from sprintf.rs).
use super::*;
use crate::memory::Permission;

fn setup_state() -> RustSimState {
    // Map destination buffer.
    crate::procedures::test_util::amd64_state_with_regions(&[(0x2000, 0x1000)])
}

#[test]
fn test_sprintf_simple_string() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);

    let result = NativeSprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dest
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(0, 64),      // unused vararg
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(11));
    // Verify written data
    for (i, &expected) in b"hello world\x00".iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert_eq!(byte.as_u64().unwrap() as u8, expected);
    }
}

#[test]
fn test_sprintf_percent_d() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"val=%d\x00", Permission::RWX);

    let result = NativeSprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(42, 64), // %d arg
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(6)); // "val=42"
    let mut out = Vec::new();
    for i in 0..6u64 {
        out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    assert_eq!(&out, b"val=42");
}

#[test]
fn test_sprintf_percent_x() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%x\x00", Permission::RWX);

    let result = NativeSprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(255, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(2)); // "ff"
    let mut out = Vec::new();
    for i in 0..2u64 {
        out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    assert_eq!(&out, b"ff");
}

#[test]
fn test_sprintf_percent_s() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"hi %s!\x00", Permission::RWX);
    state.map_memory_data(0x3000, b"world\x00", Permission::RWX);

    let result = NativeSprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x3000, 64), // pointer to "world"
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(9)); // "hi world!"
    let mut out = Vec::new();
    for i in 0..9u64 {
        out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    assert_eq!(&out, b"hi world!");
}

#[test]
fn test_sprintf_percent_c() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%c%c%c\x00", Permission::RWX);

    let result = NativeSprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(b'A' as u128, 64),
                RustBV::concrete(b'B' as u128, 64),
                RustBV::concrete(b'C' as u128, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(3));
    let mut out = Vec::new();
    for i in 0..3u64 {
        out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    assert_eq!(&out, b"ABC");
}

#[test]
fn test_sprintf_escaped_percent() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"100%%\x00", Permission::RWX);

    let result = NativeSprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(4)); // "100%"
}

#[test]
fn test_sprintf_width_padding() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%08x\x00", Permission::RWX);

    let result = NativeSprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0xAB, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(8)); // "000000ab"
    let mut out = Vec::new();
    for i in 0..8u64 {
        out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    assert_eq!(&out, b"000000ab");
}

#[test]
fn test_sprintf_negative_int() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%d\x00", Permission::RWX);

    // -1 as u64
    let neg_one = (-1i32) as u32 as u64;
    let result = NativeSprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(neg_one as u128, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(2)); // "-1"
    let mut out = Vec::new();
    for i in 0..2u64 {
        out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    assert_eq!(&out, b"-1");
}

#[test]
fn test_snprintf_truncation() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);

    let result = NativeSnprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(6, 64), // size = 6
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    // Returns full would-have-been length
    assert_eq!(result.unwrap().as_u64(), Some(11));
    // But only wrote 5 chars + null
    let mut out = Vec::new();
    for i in 0..6u64 {
        out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    assert_eq!(&out, b"hello\x00");
}

#[test]
fn test_snprintf_zero_size() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

    let result = NativeSnprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64), // size = 0
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    // Returns full length, writes nothing
    assert_eq!(result.unwrap().as_u64(), Some(5));
}

#[test]
fn test_sprintf_symbolic_dest() {
    let mut state = setup_state();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "dest", 64);
    drop(ctx);

    let result = NativeSprintf.call(
        &mut state,
        &[
            sym,
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
        ],
    );
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_sprintf_pointer() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%p\x00", Permission::RWX);

    let result = NativeSprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0xdeadbeef, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(10)); // "0xdeadbeef"
    let mut out = Vec::new();
    for i in 0..10u64 {
        out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    assert_eq!(&out, b"0xdeadbeef");
}

#[test]
fn test_sprintf_multiple_args() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%d+%d=%d\x00", Permission::RWX);

    let result = NativeSprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(2, 64),
                RustBV::concrete(3, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(5)); // "1+2=3"
    let mut out = Vec::new();
    for i in 0..5u64 {
        out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    assert_eq!(&out, b"1+2=3");
}

// --- vsnprintf: matches Python's no-op stub (no %-substitution) ---

#[test]
fn test_vsnprintf_stub_writes_nul_returns_one() {
    let mut state = setup_state();
    // Pre-fill the destination so we can confirm only a single NUL lands.
    state.map_memory_data(0x2000, b"XXXX\x00", Permission::RWX);

    let result = NativeVsnprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // str
                RustBV::concrete(16, 64),     // size != 0
                RustBV::concrete(0x1000, 64), // format (ignored)
                RustBV::concrete(0, 64),      // va_list (ignored)
            ],
        )
        .unwrap();

    // Stub returns 1 and stores exactly one NUL byte at str[0].
    assert_eq!(result.unwrap().as_u64(), Some(1));
    assert_eq!(state.memory_load(0x2000, 1).unwrap().as_u64().unwrap(), 0);
    // Byte after the terminator is untouched (no formatting occurred).
    assert_eq!(
        state.memory_load(0x2001, 1).unwrap().as_u64().unwrap(),
        b'X' as u64
    );
}

#[test]
fn test_vsnprintf_zero_size_returns_zero() {
    let mut state = setup_state();
    state.map_memory_data(0x2000, b"XXXX\x00", Permission::RWX);

    let result = NativeVsnprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64), // size == 0
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    // size == 0: returns 0 and writes nothing.
    assert_eq!(result.unwrap().as_u64(), Some(0));
    assert_eq!(
        state.memory_load(0x2000, 1).unwrap().as_u64().unwrap(),
        b'X' as u64
    );
}

#[test]
fn test_sprintf_percent_n_falls_back() {
    // %n must defer to Python (Err), NOT write the char count natively.
    // Python's format_parser.py::FormatString.replace raises SimProcedureError
    // on %n; a native count-write would diverge. Pins the faithful fallback so a
    // future iter doesn't naively "implement" it. See `format-n-no-native-parity`.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"abc%n\x00", Permission::RWX);

    let result = NativeSprintf.call(
        &mut state,
        &[
            RustBV::concrete(0x2000, 64), // dest
            RustBV::concrete(0x1000, 64), // format
            RustBV::concrete(0x2800, 64), // %n int* arg
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
        ],
    );
    assert!(result.is_err(), "%n in sprintf should fall back to Python");
}

#[test]
fn test_sprintf_float_specifiers_fall_back() {
    // %f/%e/%g must defer to Python (Err), NOT format a float natively. Python's
    // format_parser.py::FormatString.replace has no float arm and hits
    // `raise SimProcedureError("Unimplemented format specifier ...")` for any
    // spec outside {s,d,i,u,c,x,o,p}. A native float formatter would succeed
    // where Python errors, diverging from the engine we mirror — same wall as
    // %n. Pins the faithful fallback so a future iter doesn't naively
    // "implement" it. See bd memory `format-float-no-native-parity`.
    for spec in [b"%f\x00", b"%e\x00", b"%g\x00"] {
        let mut state = setup_state();
        state.map_memory_data(0x1000, spec, Permission::RWX);

        let result = NativeSprintf.call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dest
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        );
        assert!(
            result.is_err(),
            "float specifier in sprintf should fall back to Python"
        );
    }
}

#[test]
fn test_asprintf_simple_string() {
    // asprintf mallocs the buffer, writes the pointer to *strp, returns length.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);

    let result = NativeAsprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // strp (char **)
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(11));
    // *strp now points at a heap buffer holding the formatted string + NUL.
    let dst = state.memory_load(0x2000, 8).unwrap().as_u64().unwrap();
    assert!(dst != 0);
    assert!(state.heap_metadata().is_allocated(dst));
    for (i, &expected) in b"hello world\x00".iter().enumerate() {
        let byte = state.memory_load(dst + i as u64, 1).unwrap();
        assert_eq!(byte.as_u64().unwrap() as u8, expected);
    }
}

#[test]
fn test_asprintf_percent_d() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"val=%d\x00", Permission::RWX);

    let result = NativeAsprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // strp
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(42, 64),     // %d arg
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(6)); // "val=42"
    let dst = state.memory_load(0x2000, 8).unwrap().as_u64().unwrap();
    let mut out = Vec::new();
    for i in 0..6u64 {
        out.push(state.memory_load(dst + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    assert_eq!(&out, b"val=42");
}

#[test]
fn test_asprintf_symbolic_format_falls_back() {
    // A symbolic format string must defer to Python (read_string errors on a
    // symbolic byte), matching sprintf's fallback semantics.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"\x00\x00\x00\x00", Permission::RWX);
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "fmt_byte", 8)
    };
    state.memory_store(0x1000, sym).unwrap();

    let result = NativeAsprintf.call(
        &mut state,
        &[
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
        ],
    );
    assert!(
        result.is_err(),
        "symbolic format in asprintf should fall back to Python"
    );
}
