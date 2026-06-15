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
fn test_sscanf_basic() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"42\x00", Permission::RWX); // source string
    state.map_memory_data(0x1100, b"%d\x00", Permission::RWX); // format

    let result = NativeSscanf
        .call(
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
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1));
    // Value should be symbolic (we don't actually parse the source)
    let val = state.memory_load(0x2000, 4).unwrap();
    assert!(val.as_u64().is_none());
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
