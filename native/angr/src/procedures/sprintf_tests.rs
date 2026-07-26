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
fn test_sprintf_pointer_falls_back() {
    // %p must defer to Python (Err): native emitted "0xdeadbeef" (with a 0x
    // prefix) while format_parser.py emits bare hex "deadbeef", and Python
    // sign-folds bit-63-set pointers. A native "success" diverges from the
    // engine we mirror. See `format-p-no-native-parity` (angr-3i88a).
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%p\x00", Permission::RWX);

    let result = NativeSprintf.call(
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
    );
    assert!(result.is_err(), "%p in sprintf should fall back to Python");
}

#[test]
fn test_sprintf_unsigned_high_bit_falls_back() {
    // %u/%x/%o with the high bit set diverge: Python's buggy signed sign-folds
    // the value; native renders it C-correct unsigned. Defer for parity.
    // Low-bit values (test_sprintf_percent_x with 255) still format natively.
    // See `format-unsigned-highbit-no-native-parity` (angr-3i88a).
    for spec in [b"%x\x00", b"%u\x00", b"%o\x00"] {
        let mut state = setup_state();
        state.map_memory_data(0x1000, spec, Permission::RWX);

        let result = NativeSprintf.call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0xdeadbeef, 64), // bit 31 set
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        );
        assert!(
            result.is_err(),
            "high-bit unsigned spec should fall back to Python"
        );
    }
}

#[test]
fn test_sprintf_digit_precision_falls_back() {
    // "%.3d" / "%.3s": Python's _match_spec mis-slices the '.', drops the
    // conversion letter, and FormatString.replace raises SimProcedureError
    // (state errored). Native previously continued — defer for parity.
    // See `format-digit-precision-no-native-parity` (angr-3i88a).
    for spec in [b"%.3d\x00".as_slice(), b"%.3s\x00".as_slice()] {
        let mut state = setup_state();
        state.map_memory_data(0x1000, spec, Permission::RWX);
        state.map_memory_data(0x3000, b"hello\x00", Permission::RWX);

        let result = NativeSprintf.call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x3000, 64), // usable as int 7 or str ptr
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        );
        assert!(
            result.is_err(),
            "digit precision should fall back to Python"
        );
    }
}

#[test]
fn test_sprintf_star_width_falls_back() {
    // "%*d": Python's extract_components swallows '%*' without consuming a
    // width arg, shifting later variadic args. Native can't reproduce that
    // shift; defer for parity. See `format-star-width-no-native-parity`.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%*d\x00", Permission::RWX);

    let result = NativeSprintf.call(
        &mut state,
        &[
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(5, 64),  // width
            RustBV::concrete(42, 64), // value
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
        ],
    );
    assert!(
        result.is_err(),
        "'*' dynamic width should fall back to Python"
    );
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

// --- vsprintf: raw format write, no %-substitution (va_list unmodeled) ---

#[test]
fn test_vsprintf_writes_raw_string() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);

    let result = NativeVsprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // str
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(0, 64),      // va_list (ignored)
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(11));
    for (i, &expected) in b"hello world\x00".iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert_eq!(byte.as_u64().unwrap() as u8, expected);
    }
}

#[test]
fn test_vsprintf_no_substitution() {
    // A %d format must be copied VERBATIM (no va_list read), unlike NativeSprintf.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"val=%d\x00", Permission::RWX);

    let result = NativeVsprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0, 64), // va_list (ignored — NOT a %d arg)
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(6)); // "val=%d"
    for (i, &expected) in b"val=%d\x00".iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert_eq!(byte.as_u64().unwrap() as u8, expected);
    }
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

// --- Integer length-modifier width narrowing (angr-n0irt.4) ---
//
// format_parser.py masks each concretized int arg to `size*8` bits (line 96)
// before formatting: hh->8, h->16, none->32, l/ll->64. Native must narrow to
// the same width or %hd/%hhd/%hu/%hx/%ho diverge from Python. These mirror
// scanf_tests.rs's assert_scanf_store_width suite on the formatting side.

/// Run a single-vararg sprintf and return the bytes written to the dest buffer
/// (excluding the trailing NUL), or `Err` if the proc deferred to Python.
fn sprintf_one(fmt: &[u8], arg: u64) -> Result<Vec<u8>, ()> {
    let mut state = setup_state();
    // Build "<fmt>\0" in memory at 0x1000.
    let mut fmtbuf = fmt.to_vec();
    fmtbuf.push(0);
    state.map_memory_data(0x1000, &fmtbuf, Permission::RWX);

    let result = NativeSprintf.call(
        &mut state,
        &[
            RustBV::concrete(0x2000, 64), // dest
            RustBV::concrete(0x1000, 64), // format
            RustBV::concrete(arg as u128, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
        ],
    );
    let n = result.map_err(|_| ())?.ok_or(())?.as_u64().ok_or(())?;
    let mut out = Vec::new();
    for i in 0..n {
        out.push(state.memory_load(0x2000 + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    Ok(out)
}

#[test]
fn test_sprintf_short_narrows_to_16_bits() {
    // %hd of 0x10001 masks to 16 bits -> 1 (Python prints "1"); an unpatched
    // native path printed the full 65537. High bit (bit 15) of the masked value
    // is clear, so no fallback.
    assert_eq!(sprintf_one(b"%hd", 0x10001).unwrap(), b"1");
    // %hu likewise masks to 16 bits.
    assert_eq!(sprintf_one(b"%hu", 0x1_0000 + 42).unwrap(), b"42");
    // %hx masks to 16 bits: 0xAB_1234 -> 0x1234 -> "1234".
    assert_eq!(sprintf_one(b"%hx", 0x00AB_1234).unwrap(), b"1234");
    // %ho masks to 16 bits: 0o777777 fits in 16 bits + a set upper bit dropped.
    assert_eq!(sprintf_one(b"%ho", 0xFFFF_0007).unwrap(), b"7");
}

#[test]
fn test_sprintf_char_narrows_to_8_bits() {
    // %hhd of 0x101 masks to 8 bits -> 1.
    assert_eq!(sprintf_one(b"%hhd", 0x101).unwrap(), b"1");
    // %hhu masks to 8 bits: 0x17F -> 0x7F -> 127 (bit 7 clear, no fallback).
    assert_eq!(sprintf_one(b"%hhu", 0x17F).unwrap(), b"127");
    // %hhx masks to 8 bits: 0xAB12 -> 0x12 -> "12".
    assert_eq!(sprintf_one(b"%hhx", 0xAB12).unwrap(), b"12");
}

#[test]
fn test_sprintf_short_signed_fold() {
    // %hd sign-folds at bit 15: 0xFFFF (16-bit -1) prints "-1", mirroring
    // Python's `c_val -= 1<<16` when the width high bit is set.
    assert_eq!(sprintf_one(b"%hd", 0xFFFF).unwrap(), b"-1");
    // %hhd sign-folds at bit 7: 0xFF -> -1.
    assert_eq!(sprintf_one(b"%hhd", 0xFF).unwrap(), b"-1");
    // Upper bits beyond the width are ignored before the fold.
    assert_eq!(sprintf_one(b"%hd", 0xDEAD_8000).unwrap(), b"-32768");
}

#[test]
fn test_sprintf_narrow_unsigned_high_bit_falls_back() {
    // The high-bit guard now checks the modifier-derived width, not just bit
    // 31/63: %hx of a value whose bit 15 is set must defer to Python (whose
    // buggy signed fold renders it negative), just like the 32-bit case.
    assert!(sprintf_one(b"%hx", 0x8000).is_err());
    assert!(sprintf_one(b"%hu", 0x8000).is_err());
    assert!(sprintf_one(b"%ho", 0x8000).is_err());
    // %hhx of a value with bit 7 set likewise defers.
    assert!(sprintf_one(b"%hhx", 0x80).is_err());
    // ...but with the width high bit clear it still formats natively.
    assert_eq!(sprintf_one(b"%hx", 0x7FFF).unwrap(), b"7fff");
}

/// Regression test for angr-mi56k: an explicit field width whose digit run
/// encodes a huge value must be clamped to `MAX_OUTPUT_LEN` before
/// `pad_and_push` uses it as its padding-loop bound. The
/// `output.len() > MAX_OUTPUT_LEN` check in `format_string` only runs AFTER
/// `pad_and_push` returns, so left unclamped this is an unbounded
/// allocation/loop (OOM territory), not merely a large-but-finite one — the
/// wall-clock bound below is what actually catches a regression back to the
/// unclamped behavior (a correctness-only assertion on output length would
/// still pass after hanging for a very long time first).
#[test]
fn test_sprintf_huge_explicit_width_is_clamped_and_fast() {
    let mut state = crate::procedures::test_util::amd64_state_with_regions(&[(0x2000, 0x3000)]);
    state.map_memory_data(0x1000, b"%9999999999d\x00", Permission::RWX);

    let start = std::time::Instant::now();
    let result = NativeSprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dest
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(7u128, 64),  // %d arg
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "sprintf with a huge explicit width must not run an unbounded \
         padding loop; took {elapsed:?}"
    );

    let n = result.unwrap().as_u64().expect("length is concrete");
    assert!(
        n <= MAX_OUTPUT_LEN as u64,
        "clamped output length {n} exceeds MAX_OUTPUT_LEN"
    );

    // Bytes actually written to dest must match the reported (clamped)
    // length — no truncated/garbage tail from an aborted unbounded loop.
    for i in 0..n {
        let byte = state.memory_load(0x2000 + i, 1).unwrap();
        assert!(byte.as_u64().is_some(), "byte {i} of padded output missing");
    }
}
