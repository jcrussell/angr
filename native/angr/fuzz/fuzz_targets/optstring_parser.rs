//! Fuzz `getopt::parse_optstring`, the short-option engine's optstring
//! grammar walk (guest-supplied `argv[1]` of the SimProcedure, so hostile
//! input). It is a pure function over arbitrary bytes with no ASCII/UTF-8 or
//! NUL-termination assumption and no debug-assert precondition, so it fuzzes
//! hermetically without needing to reconstruct a caller contract. Must never
//! panic; the option map can't outgrow the input (resource-cap bound, each
//! entry consumes at least one input byte); and `leading_colon` must agree
//! with an independent re-derivation of its documented contract ("a leading
//! ':' after any leading '+'/'-' mode chars selects the missing-arg return")
//! -- unlike a bound on the fixed {0,1,2} arity set, this actually varies
//! with the input and so gives the fuzzer something to falsify.
#![no_main]

use libfuzzer_sys::fuzz_target;

use rustylib::fuzz_api::parse_optstring;

fuzz_target!(|data: &[u8]| {
    let (opts, leading_colon) = parse_optstring(data);

    assert!(
        opts.len() <= data.len(),
        "parse_optstring reported {} options from only {} input bytes",
        opts.len(),
        data.len()
    );
    for (&opt_char, &nargs) in &opts {
        assert!(
            nargs <= 2,
            "parse_optstring produced an out-of-range arity {nargs} for option {opt_char:?}"
        );
    }

    let mut i = 0;
    while i < data.len() && (data[i] == b'+' || data[i] == b'-') {
        i += 1;
    }
    let expected_leading_colon = i < data.len() && data[i] == b':';
    assert_eq!(
        leading_colon, expected_leading_colon,
        "leading_colon={leading_colon} disagrees with the documented ':' contract for {data:?}"
    );
});
