//! Tests for scanf/sscanf native SimProcedures (extracted from scanf.rs).

use super::*;
use crate::memory::Permission;

fn setup_state() -> RustSimState {
    crate::procedures::test_util::amd64_state_with_regions(&[(0x2000, 0x1000)])
}

#[test]
fn test_scanf_percent_d() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%d\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(0x2000, 64), // &int_var
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    // Should return 1 (one conversion)
    assert_eq!(result.unwrap().as_u64(), Some(1));

    // Value at 0x2000 should be symbolic (32-bit)
    let val = state.memory_load(0x2000, 4).unwrap();
    assert!(val.as_u64().is_none(), "scanf %d result should be symbolic");
}

#[test]
fn test_scanf_two_ints() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%d %d\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64), // &a
                RustBV::concrete(0x2010, 64), // &b
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(2));

    // Both should be symbolic
    let a = state.memory_load(0x2000, 4).unwrap();
    let b = state.memory_load(0x2010, 4).unwrap();
    assert!(a.as_u64().is_none());
    assert!(b.as_u64().is_none());
}

#[test]
fn test_scanf_percent_s() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%s\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64), // char buf[]
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1));

    // First byte should be symbolic
    let first = state.memory_load(0x2000, 1).unwrap();
    assert!(
        first.as_u64().is_none(),
        "scanf %s first byte should be symbolic"
    );

    // NUL terminator at max_str_len offset
    let nul = state.memory_load(0x2000 + MAX_SCANF_STR_LEN, 1).unwrap();
    assert_eq!(nul.as_u64(), Some(0));
}

#[test]
fn test_scanf_percent_s_with_width() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%10s\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1));

    // NUL at offset 10
    let nul = state.memory_load(0x200A, 1).unwrap();
    assert_eq!(nul.as_u64(), Some(0));
}

#[test]
fn test_scanf_percent_c() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%c\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64), // &char_var
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1));

    // Single symbolic byte
    let val = state.memory_load(0x2000, 1).unwrap();
    assert!(val.as_u64().is_none());
}

#[test]
fn test_scanf_percent_x() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%x\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1));

    let val = state.memory_load(0x2000, 4).unwrap();
    assert!(val.as_u64().is_none());
}

#[test]
fn test_scanf_long() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%ld\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64), // &long_var
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1));

    // 64-bit symbolic value
    let val = state.memory_load(0x2000, 8).unwrap();
    assert!(val.as_u64().is_none());
}

#[test]
fn test_scanf_suppressed() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%*d %d\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64), // only one pointer (first %d is suppressed)
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    // One successful conversion (suppressed doesn't count)
    assert_eq!(result.unwrap().as_u64(), Some(1));
}

#[test]
fn test_scanf_symbolic_format() {
    let mut state = setup_state();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "fmt", 64);
    drop(ctx);

    let result = NativeScanf.call(
        &mut state,
        &[
            sym,
            RustBV::concrete(0x2000, 64),
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
fn test_scanf_symbolic_ptr() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%d\x00", Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "ptr", 64);
    drop(ctx);

    let result = NativeScanf.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64),
            sym,
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
fn test_scanf_mixed_specifiers() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%d %c %s\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64), // &int_var
                RustBV::concrete(0x2010, 64), // &char_var
                RustBV::concrete(0x2020, 64), // char buf[]
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(3));
}

#[test]
fn test_isoc99_scanf() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%d\x00", Permission::RWX);

    let result = NativeIsoc99Scanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1));
}

#[test]
fn test_sscanf_defers_to_python() {
    // sscanf must PARSE its concrete source to constrain the output; native
    // free-BVS minting would make impossible paths feasible. NativeSscanf::call
    // therefore always returns an error so dispatch bounces to Python, which
    // parses the source faithfully (angr-8onrp).
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"42\x00", Permission::RWX); // source string
    state.map_memory_data(0x1100, b"%d\x00", Permission::RWX); // format

    let result = NativeSscanf.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64), // str
            RustBV::concrete(0x1100, 64), // format
            RustBV::concrete(0x2000, 64), // &int_var
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
        ],
    );

    assert!(result.is_err(), "sscanf must defer to Python");
    // Nothing was written — the output slot stays untouched.
    let val = state.memory_load(0x2000, 4).unwrap();
    assert_eq!(val.as_u64(), Some(0));
}

#[test]
fn test_scanf_escaped_percent() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%%d\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
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

    // "%%" is literal %, "d" is literal — no conversions
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_scanf_no_conversions() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
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

    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_scanf_stdin_tracking() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%d %s\x00", Permission::RWX);

    NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x2100, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    // Should have recorded stdin symbols
    let symbols = state.stdin_symbols();
    assert!(!symbols.is_empty(), "scanf should record stdin symbols");
    // One 32-bit symbol for %d + MAX_SCANF_STR_LEN 8-bit symbols for %s
    assert_eq!(symbols.len(), 1 + MAX_SCANF_STR_LEN as usize);
}

// ---- fscanf / __isoc99_fscanf -------------------------------------------

/// AMD64 `_IO_FILE._fileno` byte offset (mirrors io_file_data_for_arch).
const AMD64_FD_OFF: u64 = 112;

/// Map a FILE struct at `file_ptr` and store `fd` at its `_fileno` field.
fn setup_file_struct(state: &mut RustSimState, file_ptr: u64, fd: i32) {
    state.map_memory_data(file_ptr & !0xfff, &vec![0u8; 0x4000], Permission::RWX);
    state
        .memory_store(
            file_ptr + AMD64_FD_OFF,
            RustBV::concrete(fd as u32 as u128, 32),
        )
        .unwrap();
}

#[test]
fn test_fscanf_basic() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%d\x00", Permission::RWX);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 3);

    let result = NativeFscanf
        .call(
            &mut state,
            &[
                RustBV::concrete(file_ptr as u128, 64), // FILE*
                RustBV::concrete(0x1000, 64),           // format
                RustBV::concrete(0x2000, 64),           // &int_var
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1));
    let val = state.memory_load(0x2000, 4).unwrap();
    assert!(
        val.as_u64().is_none(),
        "fscanf %d result should be symbolic"
    );
    // A non-stdin fd must NOT pollute the stdin reconstruction.
    assert!(
        state.stdin_symbols().is_empty(),
        "fscanf on fd 3 must not record stdin symbols"
    );
}

#[test]
fn test_fscanf_negative_fd_returns_minus_one() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%d\x00", Permission::RWX);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, -1);

    let result = NativeFscanf
        .call(
            &mut state,
            &[
                RustBV::concrete(file_ptr as u128, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    // Closed/negative fd → -1 (matches Python fscanf when simfd is None).
    assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
}

#[test]
fn test_fscanf_stdin_fd_records_symbols() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%d\x00", Permission::RWX);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 0); // fscanf(stdin, ...)

    NativeFscanf
        .call(
            &mut state,
            &[
                RustBV::concrete(file_ptr as u128, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    // fd 0 IS stdin → symbol must surface for posix.dumps(0).
    assert_eq!(state.stdin_symbols().len(), 1);
}

#[test]
fn test_isoc99_fscanf_basic() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%d %d\x00", Permission::RWX);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 4);

    let result = NativeIsoc99Fscanf
        .call(
            &mut state,
            &[
                RustBV::concrete(file_ptr as u128, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x2010, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(2));
    assert!(state.memory_load(0x2000, 4).unwrap().as_u64().is_none());
    assert!(state.memory_load(0x2010, 4).unwrap().as_u64().is_none());
}

#[test]
fn test_scanf_scanset_basic() {
    // %[abc] is a string-like conversion; mints symbolic bytes + NUL.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%[abc]\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64), // char buf[]
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1));
    let first = state.memory_load(0x2000, 1).unwrap();
    assert!(
        first.as_u64().is_none(),
        "scanset first byte should be symbolic"
    );
    let nul = state.memory_load(0x2000 + MAX_SCANF_STR_LEN, 1).unwrap();
    assert_eq!(nul.as_u64(), Some(0));
}

#[test]
fn test_scanf_scanset_negated_newline() {
    // %[^\n] is the common "read a whole line" idiom.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%[^\n]\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1));
    assert!(state.memory_load(0x2000, 1).unwrap().as_u64().is_none());
}

#[test]
fn test_scanf_scanset_with_width_and_literal_bracket() {
    // Field width caps the NUL offset; a leading ']' is a literal set member,
    // so the set body is "]a-z" and the conversion terminates at the 2nd ']'.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%5[]a-z]\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1));
    // NUL at field-width offset 5.
    let nul = state.memory_load(0x2005, 1).unwrap();
    assert_eq!(nul.as_u64(), Some(0));
}

#[test]
fn test_scanf_scanset_suppressed_then_int() {
    // %*[^\n] consumes no pointer arg; the following %d takes the first ptr.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%*[^\n]%d\x00", Permission::RWX);

    let result = NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64), // &int_var (for %d)
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    // Only the %d performs assignment.
    assert_eq!(result.unwrap().as_u64(), Some(1));
    let val = state.memory_load(0x2000, 4).unwrap();
    assert!(
        val.as_u64().is_none(),
        "%d after suppressed scanset should be symbolic"
    );
}

#[test]
fn test_scanf_scanset_unterminated_falls_back() {
    // An unterminated set has no closing ']' — must error so the caller falls
    // back to Python rather than silently mis-parsing.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%[abc\x00", Permission::RWX);

    let result = NativeScanf.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
        ],
    );
    assert!(result.is_err(), "unterminated scanset should fall back");
}

#[test]
fn test_scanf_percent_n_falls_back() {
    // %n must defer to Python (Err), NOT mint/store a count natively. Python's
    // format_parser.py::FormatString.interpret raises SimProcedureError on %n in
    // the addr-based (sscanf) path and does a numeric read in the SimPackets path;
    // a single native behavior would diverge from one of them. Pins the faithful
    // fallback. See bd memory `format-n-no-native-parity`.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%n\x00", Permission::RWX);

    let result = NativeScanf.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(0, 64),
        ],
    );
    assert!(result.is_err(), "%n in scanf should fall back to Python");
}

#[test]
fn test_scanf_float_specifiers_fall_back() {
    // %f/%e/%g must defer to Python (Err), NOT mint a symbolic float natively.
    // Python's format_parser.py::FormatString.interpret only handles
    // {d,i,u,o,x,p,s,c} and raises SimProcedureError on anything else; a native
    // float read would diverge from the engine we mirror — same wall as %n.
    // Pins the faithful fallback. See bd memory `format-float-no-native-parity`.
    for spec in [b"%f\x00", b"%e\x00", b"%g\x00"] {
        let mut state = setup_state();
        state.map_memory_data(0x1000, spec, Permission::RWX);

        let result = NativeScanf.call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        );
        assert!(
            result.is_err(),
            "float specifier in scanf should fall back to Python"
        );
    }
}

/// angr-ptf54: `%s` off stdin consumes the harness-seeded fd-0 bytes (bound by
/// constraint), while sscanf — which reads no stdin — is untouched.
#[test]
fn test_scanf_percent_s_consumes_seeded_stdin() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%s\x00", Permission::RWX);
    let seed: Vec<RustBV> = vec![RustBV::concrete(0x41, 8), RustBV::concrete(0x42, 8)];
    state.file_system().set_fd_content_sym(0, seed);

    NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(0x2000, 64), // char buf[]
            ],
        )
        .expect("scanf served natively");

    for (i, want) in [0x41u128, 0x42].iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert!(byte.as_u64().is_none(), "byte {i} is still a leaf symbol");
        assert_eq!(state.eval(&byte), Some(*want), "byte {i} binds to the seed");
    }
    assert_eq!(state.file_system_ref().fd_info(0).unwrap().1, 2);
}

/// angr-ggb66: `%c` reads one byte off the stream, so it consumes one seeded
/// fd-0 byte per conversion, in order — same byte-for-byte mapping as `%s`.
#[test]
fn test_scanf_percent_c_consumes_seeded_stdin() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%c%c\x00", Permission::RWX);
    let seed: Vec<RustBV> = vec![RustBV::concrete(0x41, 8), RustBV::concrete(0x42, 8)];
    state.file_system().set_fd_content_sym(0, seed);

    NativeScanf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(0x2000, 64), // char c1
                RustBV::concrete(0x2001, 64), // char c2
            ],
        )
        .expect("scanf served natively");

    for (i, want) in [0x41u128, 0x42].iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert!(byte.as_u64().is_none(), "byte {i} is still a leaf symbol");
        assert_eq!(state.eval(&byte), Some(*want), "byte {i} binds to the seed");
    }
    // Both seeded bytes consumed, and neither is re-recorded as a stdin symbol.
    assert_eq!(state.file_system_ref().fd_info(0).unwrap().1, 2);
    assert_eq!(state.stdin_symbols().len(), 0);
}

/// angr-ggb66: a numeric conversion cannot map the seed byte-for-byte (it models
/// a digit parse), so while fd 0 has unread seed the whole call defers to Python
/// — and it defers *before* any store, so the earlier `%s` in the format has not
/// consumed the seed either.
#[test]
fn test_scanf_numeric_over_seeded_stdin_falls_back() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%s %d\x00", Permission::RWX);
    let seed: Vec<RustBV> = vec![RustBV::concrete(0x41, 8), RustBV::concrete(0x42, 8)];
    state.file_system().set_fd_content_sym(0, seed);

    let result = NativeScanf.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64), // format
            RustBV::concrete(0x2000, 64), // char buf[]
            RustBV::concrete(0x2100, 64), // &int_var
        ],
    );

    assert!(
        result.is_err(),
        "numeric conversion over seeded stdin should fall back to Python"
    );
    // Pristine state for the Python SimProcedure: seed unconsumed, no stores.
    assert_eq!(state.file_system_ref().fd_info(0).unwrap().1, 0);
    assert_eq!(state.stdin_symbols().len(), 0);
    assert_eq!(state.memory_load(0x2000, 1).unwrap().as_u64(), Some(0));
}

/// The seeded-stdin fallback is scoped to *unread* seed on stdin: an unseeded
/// `scanf("%d")` has nothing to mismatch, so it keeps the native `%d` fast path.
/// (`sscanf` reads a memory buffer, never fd 0, and now defers to Python
/// unconditionally — see `test_sscanf_defers_to_python` — so it does not touch
/// the seeded-stdin logic at all.)
#[test]
fn test_scanf_numeric_native_without_unread_seed() {
    // scanf("%d") once an earlier reader drained the seed: native again.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"%c\x00", Permission::RWX);
    state.map_memory_data(0x1100, b"%d\x00", Permission::RWX);
    state
        .file_system()
        .set_fd_content_sym(0, vec![RustBV::concrete(0x41, 8)]);

    NativeScanf
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .expect("%c drains the one seeded byte");
    NativeScanf
        .call(
            &mut state,
            &[RustBV::concrete(0x1100, 64), RustBV::concrete(0x2100, 64)],
        )
        .expect("nothing left to mismatch, so %d stays native");
    let val = state.memory_load(0x2100, 4).unwrap();
    assert!(val.as_u64().is_none(), "%d still mints a symbolic value");
}

/// Run `scanf(fmt, &dst)` with concrete 0xAA guard bytes filling the 8 bytes at
/// `dst`, and assert exactly `expect_bytes` low bytes were overwritten with a
/// symbolic value while every higher guard byte survived untouched.
fn assert_scanf_store_width(fmt: &[u8], expect_bytes: u64) {
    let mut state = setup_state();
    state.map_memory_data(0x1000, fmt, Permission::RWX);
    for off in 0..8u64 {
        state
            .memory_store(0x2000 + off, RustBV::concrete(0xAA, 8))
            .unwrap();
    }

    let result = NativeScanf
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(1));

    for off in 0..8u64 {
        let byte = state.memory_load(0x2000 + off, 1).unwrap();
        if off < expect_bytes {
            assert!(
                byte.as_u64().is_none(),
                "byte {off} of {} should be symbolic",
                String::from_utf8_lossy(fmt)
            );
        } else {
            assert_eq!(
                byte.as_u64(),
                Some(0xAA),
                "byte {off} of {} must not be clobbered",
                String::from_utf8_lossy(fmt)
            );
        }
    }
}

#[test]
fn test_scanf_short_modifier_stores_two_bytes() {
    // %hd targets a `short*` — writing 4 bytes clobbers the neighbour.
    assert_scanf_store_width(b"%hd\x00", 2);
}

#[test]
fn test_scanf_char_modifier_stores_one_byte() {
    // %hhd targets a `signed char*`.
    assert_scanf_store_width(b"%hhd\x00", 1);
}

#[test]
fn test_scanf_int_and_long_store_widths_unchanged() {
    assert_scanf_store_width(b"%d\x00", 4);
    assert_scanf_store_width(b"%ld\x00", 8);
    assert_scanf_store_width(b"%lld\x00", 8);
    assert_scanf_store_width(b"%zu\x00", 8);
    assert_scanf_store_width(b"%hx\x00", 2);
}
