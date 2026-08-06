//! Proc macros that turn "call this at every site" conventions into
//! compile-time obligations, instead of relying on every author remembering.
//!
//! Written in response to the angr-9ke6b / angr-sqfj8 audit pattern: the same
//! invariant (steady-config guarding, Multi-cell flushing, per-field merge
//! reconciliation) kept getting upheld at some call sites and silently
//! skipped at others. See the plan at
//! `/home/ubuntu/.claude/plans/review-the-last-two-effervescent-starlight.md`.

use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

/// Injects `self.steady_config_guard();` as the first statement of the
/// annotated `&mut self` method body.
///
/// `steady_config_guard()` is cheap and idempotent when no wave-parallel
/// session is live, so applying it uniformly (including to methods that
/// already call it by hand) is safe — the point is to make "a new mutator
/// forgot to call it" impossible rather than "hopefully remembered."
#[proc_macro_attribute]
pub fn steady_guarded(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut func = parse_macro_input!(item as ItemFn);

    let guard_call: syn::Stmt = syn::parse_quote! {
        self.steady_config_guard();
    };
    func.block.stmts.insert(0, guard_call);

    quote! { #func }.into()
}
