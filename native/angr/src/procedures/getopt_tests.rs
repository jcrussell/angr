//! Tests for the native `getopt(3)` short-option engine (bhk0a.2).
//!
//! These exercise the concrete-argv fast path and assert byte-for-byte parity
//! with the Python `_GetOptBase._getopt` semantics: the `int` return, the
//! `optind`/`optchar` cursor advance, and the `optarg`/`optopt` extern-global
//! writes. Python-side parity (the cross-engine check) lives in the pytest
//! suite.
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::GetoptExternAddrs;

// Memory layout for the fixtures:
//   0x1000 — optstring + argv element strings
//   0x2000 — argv pointer array (8-byte slots, amd64)
//   0x3000 — extern globals: optind @0x3000, optarg @0x3008, optopt @0x3010
const STR_BASE: u64 = 0x1000;
const ARGV_BASE: u64 = 0x2000;
const OPTIND_ADDR: u64 = 0x3000;
const OPTARG_ADDR: u64 = 0x3008;
const OPTOPT_ADDR: u64 = 0x3010;

fn setup() -> RustSimState {
    let mut state = crate::procedures::test_util::amd64_state_with_regions(&[
        (STR_BASE, 0x1000),
        (ARGV_BASE, 0x1000),
        (0x3000, 0x1000),
    ]);
    state.set_getopt_extern(GetoptExternAddrs {
        optind: Some(OPTIND_ADDR),
        optarg: Some(OPTARG_ADDR),
        optopt: Some(OPTOPT_ADDR),
    });
    state
}

/// Lay out the optstring and the argv vector. `args` are the argv elements
/// (argv[0] is the program name). Returns `argc`. Each string is NUL-terminated
/// and placed sequentially from `STR_BASE`; their addresses are written as
/// 8-byte little-endian pointers into the `ARGV_BASE` array.
fn load_argv(state: &mut RustSimState, optstring: &[u8], args: &[&[u8]]) -> u32 {
    let mut cursor = STR_BASE;
    // optstring first (NUL-terminated)
    let mut bytes = optstring.to_vec();
    bytes.push(0);
    state.map_memory_data(cursor, &bytes, Permission::RWX);
    cursor += bytes.len() as u64;

    for (i, a) in args.iter().enumerate() {
        let mut b = a.to_vec();
        b.push(0);
        state.map_memory_data(cursor, &b, Permission::RWX);
        // write argv[i] = cursor
        state
            .memory_store(
                ARGV_BASE + (i as u64) * 8,
                RustBV::concrete(cursor as u128, 64),
            )
            .unwrap();
        cursor += b.len() as u64;
    }
    args.len() as u32
}

fn call(state: &mut RustSimState, argc: u32) -> i64 {
    let r = NativeGetopt
        .call(
            state,
            &[
                RustBV::concrete(argc as u128, 64),
                RustBV::concrete(ARGV_BASE as u128, 64),
                RustBV::concrete(STR_BASE as u128, 64),
            ],
        )
        .unwrap()
        .unwrap();
    // getopt returns a 32-bit int; sign-extend for -1 comparisons.
    r.as_u64().unwrap() as u32 as i32 as i64
}

fn read_optind(state: &RustSimState) -> u32 {
    state.memory_load(OPTIND_ADDR, 4).unwrap().as_u64().unwrap() as u32
}
fn read_optarg(state: &RustSimState) -> u64 {
    state.memory_load(OPTARG_ADDR, 8).unwrap().as_u64().unwrap()
}
fn read_optopt(state: &RustSimState) -> u32 {
    state.memory_load(OPTOPT_ADDR, 4).unwrap().as_u64().unwrap() as u32
}

#[test]
fn test_flag_then_required_arg() {
    let mut state = setup();
    // optstring "ab:" : -a is a flag, -b takes a required argument
    let argc = load_argv(
        &mut state,
        b"ab:",
        &[b"prog", b"-a", b"-b", b"val", b"file"],
    );

    // -a (flag)
    assert_eq!(call(&mut state, argc), b'a' as i64);
    assert_eq!(read_optind(&state), 2);

    // -b val (required arg from next element)
    assert_eq!(call(&mut state, argc), b'b' as i64);
    assert_eq!(read_optind(&state), 4);
    let optarg = read_optarg(&state);
    // optarg should point at "val"
    let mut s = Vec::new();
    for i in 0..3u64 {
        s.push(state.memory_load(optarg + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    assert_eq!(&s, b"val");

    // "file" is a non-option operand -> stop (-1), optind unchanged
    assert_eq!(call(&mut state, argc), -1);
    assert_eq!(read_optind(&state), 4);
}

#[test]
fn test_unknown_option_sets_optopt() {
    let mut state = setup();
    let argc = load_argv(&mut state, b"a", &[b"prog", b"-x"]);
    assert_eq!(call(&mut state, argc), b'?' as i64);
    assert_eq!(read_optopt(&state), b'x' as u32);
    assert_eq!(read_optind(&state), 2);
}

#[test]
fn test_combined_short_flags() {
    let mut state = setup();
    // -ab => two flags packed into one element
    let argc = load_argv(&mut state, b"ab", &[b"prog", b"-ab"]);
    assert_eq!(call(&mut state, argc), b'a' as i64);
    assert_eq!(read_optind(&state), 1); // still inside the same element
    assert_eq!(call(&mut state, argc), b'b' as i64);
    assert_eq!(read_optind(&state), 2);
    assert_eq!(call(&mut state, argc), -1);
}

#[test]
fn test_attached_required_arg() {
    let mut state = setup();
    // -aval : required arg attached in the same element
    let argc = load_argv(&mut state, b"a:", &[b"prog", b"-aval"]);
    assert_eq!(call(&mut state, argc), b'a' as i64);
    assert_eq!(read_optind(&state), 2);
    let optarg = read_optarg(&state);
    let mut s = Vec::new();
    for i in 0..3u64 {
        s.push(state.memory_load(optarg + i, 1).unwrap().as_u64().unwrap() as u8);
    }
    assert_eq!(&s, b"val");
}

#[test]
fn test_missing_required_arg_leading_colon() {
    let mut state = setup();
    // leading ':' selects the ':' return on a missing required argument
    let argc = load_argv(&mut state, b":a:", &[b"prog", b"-a"]);
    assert_eq!(call(&mut state, argc), b':' as i64);
    assert_eq!(read_optopt(&state), b'a' as u32);
    assert_eq!(read_optind(&state), 2);
}

#[test]
fn test_missing_required_arg_no_colon() {
    let mut state = setup();
    let argc = load_argv(&mut state, b"a:", &[b"prog", b"-a"]);
    assert_eq!(call(&mut state, argc), b'?' as i64);
    assert_eq!(read_optopt(&state), b'a' as u32);
}

#[test]
fn test_double_dash_terminator() {
    let mut state = setup();
    let argc = load_argv(&mut state, b"a", &[b"prog", b"--", b"-a"]);
    // "--" stops option scanning, advancing optind past it
    assert_eq!(call(&mut state, argc), -1);
    assert_eq!(read_optind(&state), 2);
}

#[test]
fn test_guest_optind_reset_full() {
    let mut state = setup();
    let argc = load_argv(&mut state, b"a", &[b"prog", b"-a"]);
    // consume -a
    assert_eq!(call(&mut state, argc), b'a' as i64);
    assert_eq!(read_optind(&state), 2);
    // guest resets optind to 0 -> glibc full rescan from index 1
    state
        .memory_store(OPTIND_ADDR, RustBV::concrete(0, 32))
        .unwrap();
    assert_eq!(call(&mut state, argc), b'a' as i64);
    assert_eq!(read_optind(&state), 2);
}

#[test]
fn test_symbolic_argc_defers() {
    let mut state = setup();
    load_argv(&mut state, b"a", &[b"prog", b"-a"]);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "argc", 64);
    drop(ctx);
    let res = NativeGetopt.call(
        &mut state,
        &[
            sym,
            RustBV::concrete(ARGV_BASE as u128, 64),
            RustBV::concrete(STR_BASE as u128, 64),
        ],
    );
    assert!(matches!(res, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_registered_in_registry() {
    let registry = crate::procedures::NativeProcedureRegistry::new();
    assert!(registry.has_native("getopt"));
}
