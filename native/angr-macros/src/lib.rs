//! Proc macros that turn "call this at every site" conventions into
//! compile-time obligations, instead of relying on every author remembering.
//!
//! Written in response to the angr-9ke6b / angr-sqfj8 audit pattern: the same
//! invariant (steady-config guarding, Multi-cell flushing, per-field merge
//! reconciliation) kept getting upheld at some call sites and silently
//! skipped at others. See the plan at
//! `/home/ubuntu/.claude/plans/review-the-last-two-effervescent-starlight.md`.

use proc_macro::TokenStream;
use quote::{format_ident, quote, quote_spanned};
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
/// implemented in `state/fork.rs::RustSimState::merge`. The five policies in
/// [`MECHANICAL_MERGE_POLICIES`] are *generated* by this derive as
/// `merge_field_<name>()` methods that `merge` must call, so the label and the
/// merge line can no longer disagree; the rest (`delegate`, `computed`,
/// `joint`) name treatments a per-field derive cannot express safely — e.g.
/// `fs`'s longest-stdout scan across every branch at once, or `hooks` needing
/// to read its `removed_hooks` sibling — and stay hand-written, with the label
/// only proving a conscious choice was made.
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
    // Kept from `self` like `self_wins`, but `warn_config_divergence` runs
    // first — divergence across branches is loud, not silent (e.g.
    // `getopt_optind`, `ctype_loc`).
    "warn_on_diverge",
    // Computed jointly with one or more sibling fields, or by scanning every
    // branch at once — no single-field strategy applies (e.g. `fs`,
    // `hooks`+`removed_hooks`, `cgc_sinkholes`, the symbolic-overlay maps).
    "joint",
];

/// The subset of [`VALID_MERGE_POLICIES`] this derive can express as a
/// per-field expression, and therefore generates a `merge_field_<name>()`
/// method for. A field whose type does not fit the generated shape (a `max`
/// over `Option<RustBV>`, a `union` over a `Vec`, a `warn_on_diverge` whose
/// divergence test is not `!=`) opts out with `#[merge_manual = "<why>"]`.
const MECHANICAL_MERGE_POLICIES: &[&str] = &["self_wins", "union", "max", "min", "warn_on_diverge"];

/// Proves every field of the annotated struct carries a
/// `#[merge_policy = "..."]` attribute naming one of [`VALID_MERGE_POLICIES`],
/// and *generates* the merge expression for the mechanical ones (see
/// [`MECHANICAL_MERGE_POLICIES`]).
///
/// Adding a field to `RustSimState` without a `#[merge_policy]` (or with a
/// mistyped one) is a compile error, so "silently reuses the struct-literal's
/// bare `self.x.clone()` shape without anyone deciding that's actually
/// correct" (the angr-9ke6b.121 / angr-sqfj8.85/.86/.88 bug family) can no
/// longer hide unlabeled among the other 40-odd fields.
///
/// # What the generated methods buy
///
/// The label alone only recorded an intent; nothing tied it to the line
/// `state/fork.rs::RustSimState::merge` actually wrote for that field, so a
/// field labelled `max` could be merged self-wins (or vice versa) and no
/// tooling would notice. Each mechanical field now gets a private
///
/// ```ignore
/// fn merge_field_<name>(&self, others: &[&Self]) -> <FieldTy>
/// ```
///
/// whose body *is* the policy. `merge` must call it — a private method nobody
/// calls is `dead_code`, which CI's `-D warnings` turns into a build failure —
/// so a field's declared policy and its merge line cannot drift apart.
///
/// # Requirements at the derive site
///
/// - `warn_config_divergence(&str, bool)` must be in scope (used by the
///   generated `warn_on_diverge` bodies).
/// - `union` fields must be `bool`, `max`/`min` fields `Ord`, and
///   `warn_on_diverge` fields `PartialEq` — otherwise the generated body does
///   not compile and the field needs `#[merge_manual = "<why>"]`.
#[proc_macro_derive(MergePolicy, attributes(merge_policy, merge_manual))]
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
    let mut methods = proc_macro2::TokenStream::new();
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

        let policy = match policy_attrs.as_slice() {
            [] => {
                let msg = format!(
                    "field `{name}` has no #[merge_policy = \"...\"] — document how it is \
                     combined across merge branches in state/fork.rs::merge (see the \
                     `angr_macros::MergePolicy` derive doc for the allowed values)"
                );
                errors.extend(quote_spanned! { field.span() => compile_error!(#msg); });
                None
            }
            [attr] => match str_attr_value(attr) {
                Some(v) if VALID_MERGE_POLICIES.contains(&v.as_str()) => Some(v),
                Some(v) => {
                    let msg = format!(
                        "field `{name}`: unknown merge_policy \"{v}\" — expected one of \
                         {VALID_MERGE_POLICIES:?}"
                    );
                    errors.extend(quote_spanned! { attr.span() => compile_error!(#msg); });
                    None
                }
                None => {
                    let msg = "expected #[merge_policy = \"...\"] (string literal)";
                    errors.extend(quote_spanned! { attr.span() => compile_error!(#msg); });
                    None
                }
            },
            _ => {
                let msg = format!("field `{name}` has more than one #[merge_policy] attribute");
                errors.extend(quote_spanned! { field.span() => compile_error!(#msg); });
                None
            }
        };

        let manual_attrs: Vec<_> = field
            .attrs
            .iter()
            .filter(|a| a.path().is_ident("merge_manual"))
            .collect();
        let manual = match manual_attrs.as_slice() {
            [] => false,
            [attr] => {
                if str_attr_value(attr).is_none_or(|v| v.trim().is_empty()) {
                    let msg = "expected #[merge_manual = \"<why the generated body cannot \
                               express this policy>\"] (non-empty string literal)";
                    errors.extend(quote_spanned! { attr.span() => compile_error!(#msg); });
                }
                true
            }
            _ => {
                let msg = format!("field `{name}` has more than one #[merge_manual] attribute");
                errors.extend(quote_spanned! { field.span() => compile_error!(#msg); });
                true
            }
        };

        let Some(policy) = policy else { continue };
        let mechanical = MECHANICAL_MERGE_POLICIES.contains(&policy.as_str());
        if manual {
            if !mechanical {
                let msg = format!(
                    "field `{name}`: #[merge_manual] is redundant on merge_policy \
                     \"{policy}\" — only {MECHANICAL_MERGE_POLICIES:?} are generated, every \
                     other policy is hand-written by definition"
                );
                errors.extend(quote_spanned! { field.span() => compile_error!(#msg); });
            }
            continue;
        }
        if !mechanical {
            continue;
        }

        let ty = &field.ty;
        let method = format_ident!("merge_field_{}", name);
        let name_str = name.to_string();
        let doc = format!("Merges `{name_str}` per its `#[merge_policy = \"{policy}\"]` label.");
        // Fields of a Copy type reach `.clone()` through the same generated
        // body as the rest; the lint would fire on those and only those.
        let copy_ok = quote! {
            #[allow(clippy::clone_on_copy, reason = "one generated body serves Copy and non-Copy fields alike")]
        };
        methods.extend(match policy.as_str() {
            "self_wins" => quote_spanned! { field.span() =>
                #[doc = #doc]
                #copy_ok
                fn #method(&self, _others: &[&Self]) -> #ty {
                    self.#name.clone()
                }
            },
            "union" => quote_spanned! { field.span() =>
                #[doc = #doc]
                fn #method(&self, others: &[&Self]) -> #ty {
                    self.#name || others.iter().any(|o| o.#name)
                }
            },
            "max" => quote_spanned! { field.span() =>
                #[doc = #doc]
                fn #method(&self, others: &[&Self]) -> #ty {
                    others.iter().fold(self.#name, |acc, o| acc.max(o.#name))
                }
            },
            "min" => quote_spanned! { field.span() =>
                #[doc = #doc]
                fn #method(&self, others: &[&Self]) -> #ty {
                    others.iter().fold(self.#name, |acc, o| acc.min(o.#name))
                }
            },
            // `warn_on_diverge` keeps `self`'s value like `self_wins`, but
            // says so out loud first — see VALID_MERGE_POLICIES.
            _ => quote_spanned! { field.span() =>
                #[doc = #doc]
                #copy_ok
                fn #method(&self, others: &[&Self]) -> #ty {
                    warn_config_divergence(#name_str, others.iter().any(|o| o.#name != self.#name));
                    self.#name.clone()
                }
            },
        });
    }

    let struct_ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    quote! {
        #errors

        impl #impl_generics #struct_ident #ty_generics #where_clause {
            #methods
        }
    }
    .into()
}

/// Extracts `"value"` from a `#[name = "value"]` attribute.
fn str_attr_value(attr: &syn::Attribute) -> Option<String> {
    match &attr.meta {
        Meta::NameValue(nv) => match &nv.value {
            syn::Expr::Lit(syn::ExprLit {
                lit: Lit::Str(s), ..
            }) => Some(s.value()),
            _ => None,
        },
        _ => None,
    }
}
