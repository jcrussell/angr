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
fn test_shrunk_argv_element_defers_not_panics() {
    // angr-qwyti.19: the guest shrinks its own argv element between getopt
    // calls, leaving the engine's restored `optchar` cursor past the new end.
    // The native proc must defer (Err), not panic-index the buffer.
    let mut state = setup();
    // optstring "ab": -a and -b are both flags. "-ab" packs two flags, so the
    // first call parks the cursor mid-element (optchar=2, optind stays 1).
    let argc = load_argv(&mut state, b"ab", &[b"prog", b"-ab"]);
    assert_eq!(call(&mut state, argc), b'a' as i64);
    assert_eq!(read_optind(&state), 1); // cursor parked inside the element

    // Strings are laid out sequentially from STR_BASE: optstring "ab\0" (3B) at
    // 0x1000, "prog\0" (5B) at 0x1003, "-ab\0" at 0x1008. Shrink argv[1] to "-a"
    // by NUL-ing offset 2 — the restored optchar (2) now exceeds arg.len() (2).
    let argv1 = STR_BASE + 3 + 5;
    state
        .memory_store(argv1 + 2, RustBV::concrete(0, 8))
        .unwrap();

    // Second call: optchar==2 indexes a length-2 element. Must Err (defer to
    // Python), not panic on the raw `arg[optchar]`.
    let res = NativeGetopt.call(
        &mut state,
        &[
            RustBV::concrete(argc as u128, 64),
            RustBV::concrete(ARGV_BASE as u128, 64),
            RustBV::concrete(STR_BASE as u128, 64),
        ],
    );
    assert!(
        matches!(res, Err(ProcedureError::Other(_))),
        "expected defer-to-Python Err on shrunk argv, got {res:?}"
    );
}

#[test]
fn test_registered_in_registry() {
    let registry = crate::procedures::NativeProcedureRegistry::new();
    assert!(registry.has_native("getopt"));
}

// --- guest-controlled address wrapping (angr-03vl4.44) ---------------------
//
// `argv_ptr` and the argv element pointers are guest values, so every address
// computation in the proc uses `wrapping_add`. These three tests pin one site
// each: a plain `+` panics under the release-checked profile that CI runs, and
// silently produces a wrong address in the shipped release `.so`. Each test is
// non-vacuous only because both the pre-wrap page and page 0 are mapped, so the
// wrapped access is servable and the assertion reaches past the arithmetic.

/// Map the two pages a `u64::MAX`-adjacent access straddles.
fn map_wrap_pages(state: &mut RustSimState) {
    state.map_memory(0xFFFF_FFFF_FFFF_F000, 0x1000, Permission::RWX);
    state.map_memory(0, 0x1000, Permission::RWX);
}

/// Extern-global layout with `optind` unset, so `load_cursor` uses the plugin
/// cursor these tests seed directly instead of a guest word.
fn wrap_setup() -> RustSimState {
    let mut state = setup();
    state.set_getopt_extern(GetoptExternAddrs {
        optind: None,
        optarg: Some(OPTARG_ADDR),
        optopt: Some(OPTOPT_ADDR),
    });
    map_wrap_pages(&mut state);
    state
}

fn store_ptr_at(state: &mut RustSimState, addr: u64, value: u64) {
    state
        .memory_store(addr, RustBV::concrete(value as u128, 64))
        .unwrap();
}

fn call_at(state: &mut RustSimState, argc: u32, argv_ptr: u64) -> i64 {
    let r = NativeGetopt
        .call(
            state,
            &[
                RustBV::concrete(argc as u128, 64),
                RustBV::concrete(argv_ptr as u128, 64),
                RustBV::concrete(STR_BASE as u128, 64),
            ],
        )
        .expect("must not panic on a wrapping argv address")
        .unwrap();
    r.as_u64().unwrap() as u32 as i32 as i64
}

#[test]
fn getopt_argv_ptr_near_u64_max_wraps_element_slot() {
    let mut state = wrap_setup();
    state.map_memory_data(STR_BASE, b"a\0", Permission::RWX);
    let arg1 = STR_BASE + 2;
    state.map_memory_data(arg1, b"-a\0", Permission::RWX);

    // argv[1]'s slot is `argv_ptr + 1 * 8`, which wraps to address 0.
    let argv_ptr = u64::MAX - 7;
    store_ptr_at(&mut state, 0, arg1);
    state.set_getopt_cursor(1, 0);

    assert_eq!(call_at(&mut state, 2, argv_ptr), b'a' as i64);
}

#[test]
fn getopt_argv_ptr_near_u64_max_wraps_next_element_slot() {
    let mut state = wrap_setup();
    state.map_memory_data(STR_BASE, b"a:\0", Permission::RWX);
    let arg0 = STR_BASE + 3;
    let arg1 = STR_BASE + 6;
    state.map_memory_data(arg0, b"-a\0", Permission::RWX);
    state.map_memory_data(arg1, b"val\0", Permission::RWX);

    // argv[0]'s slot sits in the top page; argv[1]'s — the required argument
    // that `-a` consumes — is `argv_ptr + 1 * 8`, which wraps to address 0.
    let argv_ptr = u64::MAX - 7;
    store_ptr_at(&mut state, argv_ptr, arg0);
    store_ptr_at(&mut state, 0, arg1);
    state.set_getopt_cursor(0, 0);

    assert_eq!(call_at(&mut state, 2, argv_ptr), b'a' as i64);
    assert_eq!(read_optarg(&state), arg1);
}

#[test]
fn getopt_element_near_u64_max_wraps_inline_optarg() {
    let mut state = wrap_setup();
    state.map_memory_data(STR_BASE, b"a:\0", Permission::RWX);

    // The argv element itself straddles the wrap: "-av\0" starts at u64::MAX,
    // so the inline optarg (`elem_ptr + optchar + 1`) lands at address 1.
    let elem_ptr = u64::MAX;
    for (i, b) in b"-av\0".iter().enumerate() {
        state
            .memory_store(elem_ptr.wrapping_add(i as u64), RustBV::concrete(*b as u128, 8))
            .unwrap();
    }
    store_ptr_at(&mut state, ARGV_BASE, elem_ptr);
    state.set_getopt_cursor(0, 0);

    assert_eq!(call_at(&mut state, 1, ARGV_BASE), b'a' as i64);
    assert_eq!(read_optarg(&state), 1);
}

/// Harness 6 boundary sweep for angr-03vl4.44: every value in the shared
/// `test_boundary_values` table, not just the single `u64::MAX - 7` pivot
/// `getopt_argv_ptr_near_u64_max_wraps_element_slot` pins, must resolve
/// argv[1]'s slot (`argv_ptr.wrapping_add(1 * ps)`) at the exact wrapped
/// address.
#[test]
fn getopt_argv_ptr_boundary_sweep_wraps_element_slot() {
    for &argv_ptr in &crate::test_boundary_values::boundary_addresses() {
        let mut state = setup();
        state.set_getopt_extern(GetoptExternAddrs {
            optind: None,
            optarg: Some(OPTARG_ADDR),
            optopt: Some(OPTOPT_ADDR),
        });
        state.map_memory_data(STR_BASE, b"a\0", Permission::RWX);
        let arg1 = STR_BASE + 2;
        state.map_memory_data(arg1, b"-a\0", Permission::RWX);

        // argv[1]'s slot is `argv_ptr + 1 * ps` — the `wrapping_add` this
        // sweep exercises across the shared boundary table.
        let slot1 = argv_ptr.wrapping_add(8);
        let page = slot1 & !0xFFFu64;
        if state.memory().page_permissions(page >> 12).is_none() {
            state.map_memory(page, 0x1000, Permission::RWX);
        }
        store_ptr_at(&mut state, slot1, arg1);
        state.set_getopt_cursor(1, 0);

        assert_eq!(
            call_at(&mut state, 2, argv_ptr),
            b'a' as i64,
            "argv_ptr={argv_ptr:#x} must read argv[1] through the wrapped slot {slot1:#x}"
        );
    }
}
