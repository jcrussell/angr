//! End-to-end proof that the macros' diagnostics actually *fail the build*.
//!
//! The unit tests in `src/lib.rs` assert that a `compile_error!` token reaches
//! the expansion; only running rustc over a real invocation proves the message
//! a contributor sees, and that nothing else drowns it out. Each `tests/ui/*.rs`
//! case is paired with a `.stderr` snapshot — regenerate them after an
//! intentional wording change with `TRYBUILD=overwrite cargo test -p
//! angr-macros --test compile_fail`.
//!
//! Coverage is meant to be *total*: every diagnostic any of the three macro
//! bodies can emit — `steady_guarded_impl`, `steady_guard_checked_impl`,
//! `derive_merge_policy_impl` — has a case here, including the ones whose
//! `compile_error!` is forwarded from a `syn::parse2` failure. A new
//! diagnostic without a `tests/ui/` case is proven only by an in-process
//! `to_string().contains(...)` unit test, which cannot see what rustc renders
//! or whether a second error drowns it out (angr-0jh0j.76).
//!
//! Where two of a macro's validations can fire on the *same* invocation, the
//! combination gets its own case too: the bodies accumulate diagnostics
//! instead of returning at the first one, and only rustc can confirm it
//! renders both rather than truncating (angr-0jh0j.79). Those cases are
//! `steady_guarded_both_misuses.rs`, `merge_manual_duplicate_blank_reason.rs`
//! (two faults within one attribute), and
//! `merge_policy_and_manual_both_misused.rs` — the last crossing the two
//! independently-validated attributes `derive_merge_policy_impl` reads, so an
//! early return added to either branch drops the other's diagnostic
//! (angr-0jh0j.86).
//!
//! The two function-like macros -- `callback_setters_impl` and
//! `inspect_test_entries_impl` -- are deliberately *not* covered here: their
//! expansion emits a `#[::pyo3::pymethods]` block, so a `tests/ui/` case would
//! need `pyo3` as a dev-dependency of this crate, built with features that do
//! not unify with `rustylib`'s. Their diagnostics -- including
//! `inspect_test_entries!`'s accumulate-don't-return pair -- are held by the
//! `count_compile_errors` assertions in `inspect_test_entries_tests`
//! (angr-0jh0j.85).
//!
//! The snapshots carry rustc's own rendering of the span, so they are
//! toolchain-sensitive; that is safe here because `rust-toolchain.toml` pins
//! the compiler. A snapshot diff immediately after a toolchain bump means
//! regenerate, not "the macro broke".
#[test]
fn macro_misuse_fails_to_compile_with_the_documented_message() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/*.rs");
}
