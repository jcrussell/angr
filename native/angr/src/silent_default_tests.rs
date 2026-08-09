//! Behavioural tests for the `silent_default!` macro (`lib.rs`).
//!
//! The macro is the one-line spelling of CLAUDE.md's `SILENT(cat-x)` +
//! `log::warn!` convention (angr-91vj9.6). These lock in the two things a
//! caller relies on: the success payload passes through untouched, and the
//! failure arm yields exactly `$default` rather than panicking or diverging.
//! Log *level* selection (cat_b -> debug, cat_c -> warn) is enforced at
//! compile time — an unrecognized category has no matching `@log` arm — so it
//! needs no runtime assertion here.

#[test]
fn option_some_passes_through() {
    let got: u64 = silent_default!(cat_c, Some(0x4000_u64), 0, "unused");
    assert_eq!(got, 0x4000);
}

#[test]
fn option_none_yields_the_default() {
    let context = "option_none_yields_the_default";
    let got: u64 = silent_default!(cat_c, None::<u64>, 0, "no value ({context})");
    assert_eq!(got, 0);
}

#[test]
fn result_ok_passes_through() {
    let got: u64 = silent_default!(
        cat_b,
        Ok::<u64, String>(0x1234),
        0,
        |err| "unreachable: {err}"
    );
    assert_eq!(got, 0x1234);
}

#[test]
fn result_err_yields_the_default_and_binds_the_error() {
    // `err` is in scope for the message; the macro would not compile if the
    // message dropped it (unused variable under the crate's `-D warnings`).
    let got: u64 = silent_default!(
        cat_b,
        Err::<u64, String>("boom".to_string()),
        7,
        |err| "conversion failed: {err}"
    );
    assert_eq!(got, 7);
}

#[test]
fn default_expression_may_diverge() {
    // `$default` sits in tail position of a match arm, so a `return`/`continue`
    // is a legal default — the shape `push_assumed_constraint_or_log`
    // (`exploration/state_api.rs`) uses to skip a failed conversion.
    fn first_ok(items: &[Result<u64, String>]) -> Vec<u64> {
        let mut out = Vec::new();
        for item in items {
            let value = silent_default!(
                cat_b,
                item.as_ref(),
                continue,
                |err| "skipping item: {err}"
            );
            out.push(*value);
        }
        out
    }
    let got = first_ok(&[Ok(1), Err("bad".to_string()), Ok(3)]);
    assert_eq!(got, vec![1, 3]);
}

#[test]
fn fallible_expression_is_evaluated_exactly_once() {
    let calls = std::cell::Cell::new(0_u32);
    let probe = || {
        calls.set(calls.get() + 1);
        None::<u64>
    };
    let got: u64 = silent_default!(cat_c, probe(), 0, "probe returned None");
    assert_eq!(got, 0);
    assert_eq!(calls.get(), 1);
}
