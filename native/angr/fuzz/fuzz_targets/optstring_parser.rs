//! Fuzz `getopt::parse_optstring`, the short-option engine's optstring
//! grammar walk (guest-supplied `argv[1]` of the SimProcedure, so hostile
//! input). It is a pure function over arbitrary bytes with no ASCII/UTF-8 or
//! NUL-termination assumption and no debug-assert precondition, so it fuzzes
//! hermetically without needing to reconstruct a caller contract. Must never
//! panic, and the reported per-option arity must always be one of the three
//! `getopt` arities (no-argument / required / optional).
#![no_main]

use libfuzzer_sys::fuzz_target;

use rustylib::fuzz_api::parse_optstring;

fuzz_target!(|data: &[u8]| {
    let (opts, _leading_colon) = parse_optstring(data);
    for (&opt_char, &nargs) in &opts {
        assert!(
            nargs <= 2,
            "parse_optstring produced an out-of-range arity {nargs} for option {opt_char:?}"
        );
    }
});
