//! Proc macros that turn "call this at every site" conventions into
//! compile-time obligations, instead of relying on every author remembering.
//!
//! Written in response to the angr-9ke6b / angr-sqfj8 audit pattern: the same
//! invariant (steady-config guarding, Multi-cell flushing, per-field merge
//! reconciliation) kept getting upheld at some call sites and silently
//! skipped at others. See the plan at
//! `/home/ubuntu/.claude/plans/review-the-last-two-effervescent-starlight.md`.

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, ItemFn, Lit, Meta, parse_macro_input, spanned::Spanned};

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

/// Policy values [`derive_merge_policy`] accepts. Each names a treatment
/// already implemented by hand in `state/fork.rs::RustSimState::merge` — this
/// derive does not generate merge code (several fields need whole-slice or
/// cross-field logic a per-field derive can't express safely, e.g. `fs`'s
/// longest-stdout scan across every branch at once, or `hooks` needing to
/// read its `removed_hooks` sibling) — it only proves every field made a
/// documented, conscious choice.
const VALID_MERGE_POLICIES: &[&str] = &[
    // Carried from `self` unchanged; the field is not expected to diverge
    // across merge branches (e.g. `arch`, `pc`).
    "self_wins",
    // Freshly computed for the merged state, not derived from any branch's
    // prior value (e.g. `state_id`, `parent_id`).
    "computed",
    // Combined via the field type's own `.merge(...)`/`.union_from(...)`
    // method (e.g. `registers`, `memory`, `solver`, `heap_metadata`).
    "delegate",
    // Set-like union across every branch (e.g. `stdin_symbols`, the boolean
    // SimOption mirrors via logical OR).
    "union",
    "max",
    "min",
    // Kept from `self` like `self_wins`, but `merge` calls
    // `warn_config_divergence` first — divergence across branches is loud,
    // not silent (e.g. `getopt_optind`, `ctype_loc`).
    "warn_on_diverge",
    // Computed jointly with one or more sibling fields, or by scanning every
    // branch at once — no single-field strategy applies (e.g. `fs`,
    // `hooks`+`removed_hooks`, `cgc_sinkholes`, the symbolic-overlay maps).
    "joint",
];

/// Proves every field of the annotated struct carries a
/// `#[merge_policy = "..."]` attribute naming one of [`VALID_MERGE_POLICIES`].
/// Emits no code beyond validation — see that constant's doc for why the
/// actual merge logic stays hand-written in `state/fork.rs`.
///
/// Adding a field to `RustSimState` without a `#[merge_policy]` (or with a
/// mistyped one) is a compile error, so "silently reuses the struct-literal's
/// bare `self.x.clone()` shape without anyone deciding that's actually
/// correct" (the angr-9ke6b.121 / angr-sqfj8.85/.86/.88 bug family) can no
/// longer hide unlabeled among the other 40-odd fields.
#[proc_macro_derive(MergePolicy, attributes(merge_policy))]
pub fn derive_merge_policy(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let Data::Struct(data) = &input.data else {
        return quote_spanned! { input.span() =>
            compile_error!("MergePolicy can only be derived for structs");
        }
        .into();
    };
    let Fields::Named(fields) = &data.fields else {
        return quote_spanned! { input.span() =>
            compile_error!("MergePolicy requires named fields");
        }
        .into();
    };

    let mut errors = proc_macro2::TokenStream::new();
    for field in &fields.named {
        // `Fields::Named` (matched above) guarantees every field has an
        // ident — only tuple structs/enum variants lack one.
        #[allow(clippy::expect_used, reason = "Fields::Named guarantees an ident")]
        let name = field
            .ident
            .as_ref()
            .expect("Fields::Named guarantees an ident");
        let policy_attrs: Vec<_> = field
            .attrs
            .iter()
            .filter(|a| a.path().is_ident("merge_policy"))
            .collect();

        match policy_attrs.as_slice() {
            [] => {
                let msg = format!(
                    "field `{name}` has no #[merge_policy = \"...\"] — document how it is \
                     combined across merge branches in state/fork.rs::merge (see the \
                     `angr_macros::MergePolicy` derive doc for the allowed values)"
                );
                errors.extend(quote_spanned! { field.span() => compile_error!(#msg); });
            }
            [attr] => {
                let value = match &attr.meta {
                    Meta::NameValue(nv) => match &nv.value {
                        syn::Expr::Lit(syn::ExprLit {
                            lit: Lit::Str(s), ..
                        }) => Some(s.value()),
                        _ => None,
                    },
                    _ => None,
                };
                match value {
                    Some(v) if VALID_MERGE_POLICIES.contains(&v.as_str()) => {}
                    Some(v) => {
                        let msg = format!(
                            "field `{name}`: unknown merge_policy \"{v}\" — expected one of \
                             {VALID_MERGE_POLICIES:?}"
                        );
                        errors.extend(quote_spanned! { attr.span() => compile_error!(#msg); });
                    }
                    None => {
                        let msg = "expected #[merge_policy = \"...\"] (string literal)";
                        errors.extend(quote_spanned! { attr.span() => compile_error!(#msg); });
                    }
                }
            }
            _ => {
                let msg = format!("field `{name}` has more than one #[merge_policy] attribute");
                errors.extend(quote_spanned! { field.span() => compile_error!(#msg); });
            }
        }
    }

    errors.into()
}
