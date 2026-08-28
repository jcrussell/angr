//! Proc macros that turn "call this at every site" conventions into
//! compile-time obligations, instead of relying on every author remembering.
//!
//! Written in response to the angr-9ke6b / angr-sqfj8 audit pattern: the same
//! invariant (steady-config guarding, Multi-cell flushing, per-field merge
//! reconciliation) kept getting upheld at some call sites and silently
//! skipped at others. See the plan at
//! `/home/ubuntu/.claude/plans/review-the-last-two-effervescent-starlight.md`.

use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::{
    Data, DeriveInput, ExprLit, Fields, FnArg, ImplItem, ItemFn, ItemImpl, Lit, Meta,
    MetaNameValue, Receiver, spanned::Spanned,
};

/// Injects `self.steady_config_guard();` as the first statement of the
/// annotated `&mut self` method body.
///
/// `steady_config_guard()` is cheap and idempotent when no wave-parallel
/// session is live, so applying it uniformly (including to methods that
/// already call it by hand) is safe — the point is to make "a new mutator
/// forgot to call it" impossible rather than "hopefully remembered."
///
/// Takes no attribute arguments, and rejects anything but a `&mut self`
/// receiver: `steady_config_guard` itself needs `&mut self`, so a misapplied
/// attribute would otherwise surface as a confusing error *inside* the
/// injected statement ("cannot find value `self`" / "cannot borrow as
/// mutable") rather than pointing at the attribute.
#[proc_macro_attribute]
pub fn steady_guarded(attr: TokenStream, item: TokenStream) -> TokenStream {
    steady_guarded_impl(attr.into(), item.into()).into()
}

/// `proc_macro2` body of [`steady_guarded`], so the expansion is reachable
/// from unit tests (the `proc_macro` types only exist inside a real macro
/// invocation).
///
/// Diagnostics are emitted as `compile_error!` *next to* the function rather
/// than in place of it, so a misuse reports one clear message instead of also
/// burying the caller in "no method named ..." follow-ons.
fn steady_guarded_impl(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let mut func = match syn::parse2::<ItemFn>(item) {
        Ok(func) => func,
        Err(err) => return err.to_compile_error(),
    };

    let mut errors = proc_macro2::TokenStream::new();
    if !attr.is_empty() {
        let msg = "#[steady_guarded] takes no arguments — the guard call it injects is not \
                   configurable; write a bare `#[angr_macros::steady_guarded]`";
        errors.extend(quote_spanned! { attr.span() => compile_error!(#msg); });
    }

    let receiver_ok =
        matches!(func.sig.inputs.first(), Some(FnArg::Receiver(recv)) if takes_mut_self(recv));
    if receiver_ok {
        let guard_call: syn::Stmt = syn::parse_quote! {
            self.steady_config_guard();
        };
        func.block.stmts.insert(0, guard_call);
    } else {
        // Skip the injection too: without a `&mut self` receiver it cannot
        // compile, and its errors would drown out the one above.
        let msg = "#[steady_guarded] requires a `&mut self` receiver — it injects \
                   `self.steady_config_guard();`, which takes `&mut self`";
        errors.extend(quote_spanned! { func.sig.span() => compile_error!(#msg); });
    }

    quote! { #errors #func }
}

/// Whether a method receiver binds `self` mutably by reference.
fn takes_mut_self(recv: &Receiver) -> bool {
    if recv.colon_token.is_some() {
        // Typed receiver (`self: &mut Self`): syn leaves `reference` /
        // `mutability` unset for these, so the type is the only signal.
        matches!(&*recv.ty, syn::Type::Reference(r) if r.mutability.is_some())
    } else {
        recv.reference.is_some() && recv.mutability.is_some()
    }
}

/// Requires every method with a `&mut self` receiver in the annotated `impl`
/// block — `pub` or not — to carry either `#[angr_macros::steady_guarded]`
/// or `#[angr_macros::steady_guard_exempt(reason = "...")]` — no unlabeled
/// third path. No visibility filter: PyO3 exposes every method in a
/// `#[pymethods]` block to Python regardless of Rust-level visibility, so
/// gating on `pub` would silently exempt a private mutator too.
///
/// [`steady_guarded`] alone is opt-in per function, so a newly added mutator
/// can silently skip it exactly like `set_max_history` did (angr-c7xno.21).
/// `tools/audit_steady_guard_coverage.py` closed that gap with a baseline-gated
/// script (since removed), but only for `pub fn` names matching a fixed
/// mutator-prefix heuristic (`set_`/`clear_`/`register_`/...) — a
/// differently-named or non-`pub` real mutator (e.g. `clear_native_techniques`,
/// which mutates the same `self.native_techniques` its sibling `register_*`
/// methods guard, or `rebuild_stop_addrs`, private but still Python-callable)
/// was invisible to it. Applying this attribute at the `impl` block level
/// instead makes the choice mandatory for every eligible method regardless of
/// its name or visibility, the same "opt-out derive" shape
/// [`derive_merge_policy`] uses for struct fields.
///
/// `#[steady_guard_exempt]` is inert: recognized and stripped here, never
/// itself resolved as an attribute macro. Using it outside a
/// `#[steady_guard_checked]` impl block is a hard "cannot find attribute
/// macro" compile error rather than a silent no-op.
#[proc_macro_attribute]
pub fn steady_guard_checked(attr: TokenStream, item: TokenStream) -> TokenStream {
    steady_guard_checked_impl(attr.into(), item.into()).into()
}

/// `proc_macro2` body of [`steady_guard_checked`]; see [`steady_guarded_impl`]
/// for why this split exists.
fn steady_guard_checked_impl(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let mut item_impl = match syn::parse2::<ItemImpl>(item) {
        Ok(item_impl) => item_impl,
        Err(err) => return err.to_compile_error(),
    };

    let mut errors = proc_macro2::TokenStream::new();
    if !attr.is_empty() {
        let msg = "#[steady_guard_checked] takes no arguments";
        errors.extend(quote_spanned! { attr.span() => compile_error!(#msg); });
    }

    for item in &mut item_impl.items {
        let ImplItem::Fn(m) = item else { continue };
        // No visibility filter: PyO3 exposes every method in a `#[pymethods]`
        // impl block to Python regardless of its Rust-level visibility (a
        // private `fn` is just as callable from Python as a `pub` one), so
        // gating on `pub` would silently exempt e.g. `rebuild_stop_addrs` —
        // which mutates `self.stop_addrs`, a `StepContext`-snapshotted field
        // — from ever needing a choice.
        let receiver_ok =
            matches!(m.sig.inputs.first(), Some(FnArg::Receiver(recv)) if takes_mut_self(recv));
        if !receiver_ok {
            continue;
        }

        let has_guarded = m
            .attrs
            .iter()
            .any(|a| path_ends_with(a.path(), "steady_guarded"));
        let exempt_idx = m
            .attrs
            .iter()
            .position(|a| path_ends_with(a.path(), "steady_guard_exempt"));

        match (has_guarded, exempt_idx) {
            (true, Some(idx)) => {
                let msg = format!(
                    "fn `{}`: carries both #[steady_guarded] and #[steady_guard_exempt] — pick one",
                    m.sig.ident
                );
                errors.extend(quote_spanned! { m.attrs[idx].span() => compile_error!(#msg); });
                // Inert: consumed here regardless of which arm fired, so it is
                // never independently resolved as an attribute macro and this
                // case reports exactly one error rather than also tripping
                // "cannot find attribute macro `steady_guard_exempt`".
                m.attrs.remove(idx);
            }
            (false, None) => {
                let msg = format!(
                    "fn `{}`: a `&mut self` method in a `#[steady_guard_checked]` impl block must \
                     carry either `#[angr_macros::steady_guarded]` (if it mutates exploration \
                     config or the active stash) or `#[angr_macros::steady_guard_exempt(reason = \
                     \"...\")]` (documenting why not)",
                    m.sig.ident
                );
                errors.extend(quote_spanned! { m.sig.span() => compile_error!(#msg); });
            }
            (true, None) => {}
            (false, Some(idx)) => {
                let reason_ok = match m.attrs[idx].parse_args::<MetaNameValue>() {
                    Ok(nv) if nv.path.is_ident("reason") => match &nv.value {
                        syn::Expr::Lit(ExprLit {
                            lit: Lit::Str(s), ..
                        }) => !s.value().trim().is_empty(),
                        _ => false,
                    },
                    _ => false,
                };
                if !reason_ok {
                    let msg = "expected #[angr_macros::steady_guard_exempt(reason = \"<why this \
                               mutator doesn't need steady_config_guard>\")] (non-empty string \
                               literal)";
                    errors.extend(quote_spanned! { m.attrs[idx].span() => compile_error!(#msg); });
                }
                m.attrs.remove(idx);
            }
        }
    }

    quote! { #errors #item_impl }
}

/// Whether `path`'s last segment is `name`, regardless of how it was
/// qualified (`angr_macros::steady_guarded` vs. an imported bare
/// `steady_guarded`).
fn path_ends_with(path: &syn::Path, name: &str) -> bool {
    path.segments.last().is_some_and(|s| s.ident == name)
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
    // `self_wins` for an *in-place* merge — one that mutates `self` rather
    // than building a fresh struct (e.g. `SymbolicMemory::merge`). Such a
    // merge keeps `self`'s value by simply never assigning the field, so
    // there is no line for a generated `merge_field_<name>()` to replace and
    // the policy is non-mechanical by construction. Use `self_wins` (+
    // `#[merge_manual]` if needed) for struct-literal merges instead.
    "in_place_self",
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
/// `#[merge_policy = "..."]` attribute naming one of the crate-private
/// `VALID_MERGE_POLICIES` list, and *generates* the merge expression for the
/// mechanical ones (the `MECHANICAL_MERGE_POLICIES` subset). Both consts are
/// private, so they are named here rather than linked — a rustdoc intra-doc
/// link from this public item would resolve only under
/// `--document-private-items` and is denied by the workspace rustdoc gate.
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
/// ```text
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
/// - the field type must satisfy what its policy's generated body actually
///   asks for — otherwise the body does not compile and the field needs
///   `#[merge_manual = "<why>"]`:
///   - `self_wins`: `Clone` (the body is `self.<f>.clone()`).
///   - `union`: `bool`.
///   - `max`/`min`: `Copy + Ord`. `Ord` alone is not enough: the body folds
///     with `self.<f>` as the seed and calls `Ord::max`/`Ord::min`, both of
///     which take their operands *by value*, so a non-`Copy` `Ord` type
///     (`String`, `Vec<T>`, a `BigNum` newtype) fails to move out from behind
///     `&self` rather than failing an `Ord` bound.
///   - `warn_on_diverge`: `Clone + PartialEq` — `PartialEq` for the `!=`
///     divergence test, `Clone` because the value it then keeps is returned
///     the same way `self_wins` returns it.
///
/// # Example
///
/// ```
/// #[derive(angr_macros::MergePolicy)]
/// struct Config {
///     #[merge_policy = "self_wins"]
///     arch: &'static str,
///     #[merge_policy = "union"]
///     tainted: bool,
///     #[merge_policy = "max"]
///     depth: u64,
///     #[merge_policy = "delegate"]
///     memory: Vec<u8>,
/// }
///
/// let a = Config { arch: "x86", tainted: false, depth: 3, memory: vec![] };
/// let b = Config { arch: "amd64", tainted: true, depth: 7, memory: vec![] };
///
/// assert_eq!(a.merge_field_arch(&[&b]), "x86");
/// assert!(a.merge_field_tainted(&[&b]));
/// assert_eq!(a.merge_field_depth(&[&b]), 7);
/// // No `merge_field_memory` — `delegate` is hand-written by definition.
/// ```
///
/// The `Copy` half of the `max`/`min` requirement, pinned: `String` is `Ord`,
/// but the generated fold cannot move it out from behind `&self`.
///
/// ```compile_fail
/// #[derive(angr_macros::MergePolicy)]
/// struct Config {
///     #[merge_policy = "max"]
///     name: String,
/// }
/// ```
#[proc_macro_derive(MergePolicy, attributes(merge_policy, merge_manual))]
pub fn derive_merge_policy(input: TokenStream) -> TokenStream {
    derive_merge_policy_impl(input.into()).into()
}

/// `proc_macro2` body of [`derive_merge_policy`], so the expansion is
/// reachable from unit tests (the `proc_macro` types only exist inside a real
/// macro invocation).
fn derive_merge_policy_impl(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<DeriveInput>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };

    let Data::Struct(data) = &input.data else {
        return quote_spanned! { input.span() =>
            compile_error!("MergePolicy can only be derived for structs");
        };
    };
    let Fields::Named(fields) = &data.fields else {
        return quote_spanned! { input.span() =>
            compile_error!("MergePolicy requires named fields");
        };
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
        let manual = if manual_attrs.is_empty() {
            false
        } else {
            if manual_attrs.len() > 1 {
                let msg = format!("field `{name}` has more than one #[merge_manual] attribute");
                errors.extend(quote_spanned! { field.span() => compile_error!(#msg); });
            }
            // Every attribute's reason is checked, not just the sole one in the
            // non-duplicate case: a field carrying both a duplicate-attribute
            // typo and a blank reason has two problems, and reporting them one
            // compile at a time costs the contributor a needless round-trip.
            for attr in &manual_attrs {
                if str_attr_value(attr).is_none_or(|v| v.trim().is_empty()) {
                    let msg = "expected #[merge_manual = \"<why the generated body cannot \
                               express this policy>\"] (non-empty string literal)";
                    errors.extend(quote_spanned! { attr.span() => compile_error!(#msg); });
                }
            }
            true
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

/// Generates the one-line PyO3 setter (`set_<slot>`) for every callback slot
/// named in the invocation.
///
/// Invoked as `callback_setters! { Type => slot_a, slot_b, ... }`, and in
/// practice driven by `rustylib`'s `with_callback_fields!` list so the slot
/// set has exactly one spelling:
///
/// ```ignore
/// with_callback_fields!(callback_setters PythonCallbacks =>);
/// ```
///
/// Each slot expands to a `pub fn set_<slot>(&mut self, cb: Py<PyAny>)` whose
/// body is `self.<slot> = Some(cb);`, wrapped in its own `#[pymethods] impl`
/// block (PyO3's `multiple-pymethods` feature) — an attribute proc macro
/// cannot see through an unexpanded `macro_rules!` invocation *inside* an
/// impl block, so the generated block has to carry `#[pymethods]` itself.
///
/// The point is the name/slot pairing: hand-written, `set_inspect_reg_read`
/// assigning to `inspect_reg_write` compiles and silently misroutes the
/// callback (angr-0jh0j.6). Deriving the method name from the field ident
/// makes that state unrepresentable, the same way [`steady_guarded`] derives
/// its guard call and `require_callback!` derives its error text.
///
/// Per-slot prose (the expected Python signature, degraded paths, sibling
/// slots it is confusable with) belongs on the **struct field's** doc comment,
/// which the generated doc links to — that keeps one description per slot
/// rather than a field/setter pair that can drift.
#[proc_macro]
pub fn callback_setters(input: TokenStream) -> TokenStream {
    callback_setters_impl(input.into()).into()
}

/// `proc_macro2` body of [`callback_setters`], so the expansion is reachable
/// from unit tests (the `proc_macro` types only exist inside a real macro
/// invocation).
fn callback_setters_impl(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let spec = match syn::parse2::<CallbackSetters>(input) {
        Ok(spec) => spec,
        Err(err) => return err.to_compile_error(),
    };

    let ty = &spec.ty;
    if spec.slots.is_empty() {
        return quote_spanned! { ty.span() =>
            compile_error!("callback_setters! needs at least one slot ident after `=>`");
        };
    }

    let setters = spec.slots.iter().map(|slot| {
        let setter = format_ident!("set_{}", slot);
        let doc = format!("Set the [`{ty}::{slot}`] callback slot.");
        quote! {
            #[doc = #doc]
            #[doc = ""]
            #[doc = " Generated by `angr_macros::callback_setters!`, so the method name and"]
            #[doc = " the slot it writes cannot disagree. The expected Python signature is"]
            #[doc = " documented on the field itself."]
            pub fn #setter(&mut self, cb: ::pyo3::Py<::pyo3::types::PyAny>) {
                self.#slot = Some(cb);
            }
        }
    });

    quote! {
        #[allow(
            unreachable_pub,
            reason = "pyo3 `#[pymethods]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
        )]
        #[::pyo3::pymethods]
        impl #ty {
            #(#setters)*
        }
    }
}

/// Generates the `call_inspect_<event>` Python test entry points from bodies
/// written against a `self.forward(..)` placeholder.
///
/// Invoked as `inspect_test_entries! { Type => fn <event>(&self, ..) -> .. { .. } ... }`.
/// Each entry expands to
///
/// ```ignore
/// #[pyo3(name = "call_inspect_<event>")]
/// pub fn py_call_inspect_<event>(&self, ..) -> .. { /* body, with
///     `self.forward(..)` rewritten to `self.call_inspect_<event>(..)` */ }
/// ```
///
/// so the three names that must agree — the Rust method, the Python-visible
/// name, and the dispatch method being exercised — all come from the one
/// `<event>` ident. Hand-written, the same-signature pairs
/// (`reg_read`/`reg_write`, `tmp_read`/`tmp_write`, `mem_read`/`mem_write`,
/// `call`/`return`) let a copy-paste compile while testing the *sibling*
/// dispatch, which is exactly the class angr-0jh0j.6 raised for this file.
///
/// A body must call `self.forward(..)` exactly once, and may not name a
/// `call_inspect_*` method directly — the placeholder is the only route to the
/// dispatch method, so it cannot be bypassed back into the drift it removes.
/// Everything else in the body is passed through untouched, which is what lets
/// the `constraints` entry import its guard AST before forwarding. Caller
/// attributes (notably `#[pyo3(signature = ..)]`, which PyO3 requires for
/// `Option<T>` parameters) are preserved.
#[proc_macro]
pub fn inspect_test_entries(input: TokenStream) -> TokenStream {
    inspect_test_entries_impl(input.into()).into()
}

/// `proc_macro2` body of [`inspect_test_entries`], so the expansion is
/// reachable from unit tests (the `proc_macro` types only exist inside a real
/// macro invocation).
fn inspect_test_entries_impl(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let spec = match syn::parse2::<InspectTestEntries>(input) {
        Ok(spec) => spec,
        Err(err) => return err.to_compile_error(),
    };

    let ty = &spec.ty;
    let mut errors = proc_macro2::TokenStream::new();
    let mut methods = proc_macro2::TokenStream::new();

    for func in &spec.entries {
        let event = &func.sig.ident;
        // A raw ident (`r#return`, the only event whose name is a keyword)
        // stringifies with its `r#` prefix; strip it so the derived names read
        // `call_inspect_return` rather than `call_inspect_r#return`.
        let raw = event.to_string();
        let event = raw.strip_prefix("r#").unwrap_or(&raw);
        let target = format_ident!("call_inspect_{}", event);
        let py_name = format!("call_inspect_{event}");
        let mut sig = func.sig.clone();
        sig.ident = format_ident!("py_call_inspect_{}", event);

        let mut hits = 0usize;
        let mut leaked = false;
        let body = rewrite_forward(
            func.block.to_token_stream(),
            &target,
            &mut hits,
            &mut leaked,
        );
        if hits != 1 {
            let msg = format!(
                "inspect_test_entries: `{event}` must call `self.forward(..)` exactly once \
                 (found {hits}) — that placeholder is what binds the entry to \
                 `{target}`"
            );
            errors.extend(quote_spanned! { func.block.span() => compile_error!(#msg); });
        }
        if leaked {
            let msg = format!(
                "inspect_test_entries: `{event}` names a `call_inspect_*` method directly — \
                 call `self.forward(..)` instead so the target is derived from the entry name"
            );
            errors.extend(quote_spanned! { func.block.span() => compile_error!(#msg); });
        }

        let attrs = &func.attrs;
        let vis = &func.vis;
        methods.extend(quote! {
            #(#attrs)*
            #[pyo3(name = #py_name)]
            #vis #sig #body
        });
    }

    if methods.is_empty() {
        return quote_spanned! { ty.span() =>
            compile_error!("inspect_test_entries! needs at least one `fn <event>(..)` entry after `=>`");
        };
    }

    quote! {
        #errors

        #[allow(
            unreachable_pub,
            reason = "pyo3 `#[pymethods]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
        )]
        #[::pyo3::pymethods]
        impl #ty {
            #methods
        }
    }
}

/// Rewrites the `forward` placeholder ident to `target` throughout a body,
/// counting the rewrites and flagging any direct mention of a
/// `call_inspect_*` method (which would bypass the derivation).
fn rewrite_forward(
    tokens: proc_macro2::TokenStream,
    target: &syn::Ident,
    hits: &mut usize,
    leaked: &mut bool,
) -> proc_macro2::TokenStream {
    tokens
        .into_iter()
        .map(|tt| match tt {
            proc_macro2::TokenTree::Group(g) => {
                let inner = rewrite_forward(g.stream(), target, hits, leaked);
                let mut new = proc_macro2::Group::new(g.delimiter(), inner);
                new.set_span(g.span());
                proc_macro2::TokenTree::Group(new)
            }
            proc_macro2::TokenTree::Ident(id) => {
                if id == "forward" {
                    *hits += 1;
                    let mut replacement = target.clone();
                    replacement.set_span(id.span());
                    proc_macro2::TokenTree::Ident(replacement)
                } else {
                    if id.to_string().starts_with("call_inspect_") {
                        *leaked = true;
                    }
                    proc_macro2::TokenTree::Ident(id)
                }
            }
            other => other,
        })
        .collect()
}

/// Parsed form of an `inspect_test_entries! { Type => fn .. }` invocation.
struct InspectTestEntries {
    ty: syn::Ident,
    entries: Vec<ItemFn>,
}

impl syn::parse::Parse for InspectTestEntries {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let ty: syn::Ident = input.parse()?;
        input.parse::<syn::Token![=>]>()?;
        let mut entries = Vec::new();
        while !input.is_empty() {
            entries.push(input.parse()?);
        }
        Ok(Self { ty, entries })
    }
}

/// Parsed form of a `callback_setters! { Type => slot, slot, ... }` invocation.
struct CallbackSetters {
    ty: syn::Ident,
    slots: syn::punctuated::Punctuated<syn::Ident, syn::Token![,]>,
}

impl syn::parse::Parse for CallbackSetters {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let ty: syn::Ident = input.parse()?;
        input.parse::<syn::Token![=>]>()?;
        let slots = syn::punctuated::Punctuated::parse_terminated(input)?;
        Ok(Self { ty, slots })
    }
}

#[cfg(test)]
mod steady_guarded_tests {
    use super::steady_guarded_impl;
    use quote::quote;

    /// Token-stream text of the expansion. `to_string()` spaces every token,
    /// so the *injected statement* reads `self . steady_config_guard`, which
    /// distinguishes it from the unspaced `self.steady_config_guard();` quoted
    /// inside the diagnostic message.
    fn expand(attr: proc_macro2::TokenStream, item: proc_macro2::TokenStream) -> String {
        steady_guarded_impl(attr, item).to_string()
    }

    const INJECTED: &str = "self . steady_config_guard";

    #[test]
    fn mut_self_method_gets_the_guard_as_its_first_statement() {
        let out = expand(quote! {}, quote! { fn set_x(&mut self) { self.x = 1; } });
        assert!(!out.contains("compile_error"), "{out}");
        let guard = out.find(INJECTED);
        let body = out.find("self . x = 1");
        assert!(guard.is_some(), "guard call injected: {out}");
        assert!(body.is_some(), "original body kept: {out}");
        assert!(guard < body, "guard must come first: {out}");
    }

    #[test]
    fn typed_mut_self_receiver_is_accepted() {
        let out = expand(quote! {}, quote! { fn set_x(self: &mut Self) {} });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains(INJECTED), "{out}");
    }

    #[test]
    fn shared_self_receiver_is_rejected_without_injecting() {
        let out = expand(quote! {}, quote! { fn set_x(&self) {} });
        assert!(out.contains("requires a `&mut self` receiver"), "{out}");
        // The bad injection is skipped so its own errors cannot drown out the
        // diagnostic above.
        assert!(!out.contains(INJECTED), "{out}");
        // The function itself still reaches the compiler, so its callers do
        // not additionally fail with "no method named set_x".
        assert!(out.contains("fn set_x"), "{out}");
    }

    #[test]
    fn free_function_is_rejected() {
        let out = expand(quote! {}, quote! { fn set_x(a: u8) {} });
        assert!(out.contains("requires a `&mut self` receiver"), "{out}");
        assert!(!out.contains(INJECTED), "{out}");
    }

    #[test]
    fn by_value_self_receiver_is_rejected() {
        let out = expand(quote! {}, quote! { fn set_x(mut self) {} });
        assert!(out.contains("requires a `&mut self` receiver"), "{out}");
    }

    #[test]
    fn attribute_arguments_are_rejected_but_the_guard_still_lands() {
        let out = expand(quote! { every = "step" }, quote! { fn set_x(&mut self) {} });
        assert!(out.contains("takes no arguments"), "{out}");
        assert!(out.contains(INJECTED), "{out}");
    }

    #[test]
    fn both_validations_fire_together_on_a_doubly_misused_attribute() {
        // The two checks are independent and accumulate into `errors` rather
        // than returning early, so a contributor who made both mistakes sees
        // both messages from one compile instead of one per rebuild.
        let out = expand(quote! { every = "step" }, quote! { fn set_x(&self) {} });
        assert!(out.contains("takes no arguments"), "{out}");
        assert!(out.contains("requires a `&mut self` receiver"), "{out}");
        assert_eq!(out.matches("compile_error").count(), 2, "{out}");
        // Receiver rejection still wins over injection, and the function still
        // reaches the compiler so its callers do not fail a second way.
        assert!(!out.contains(INJECTED), "{out}");
        assert!(out.contains("fn set_x"), "{out}");
        // Argument diagnostic first: it is spanned at the attribute, which
        // sorts above the signature a reader scans downward from.
        assert!(
            out.find("takes no arguments") < out.find("requires a `&mut self` receiver"),
            "{out}"
        );
    }

    #[test]
    fn non_function_item_reports_a_parse_error() {
        let out = expand(quote! {}, quote! { struct Nope; });
        assert!(out.contains("compile_error"), "{out}");
    }
}

#[cfg(test)]
mod steady_guard_checked_tests {
    use super::steady_guard_checked_impl;
    use quote::quote;

    fn expand(item: proc_macro2::TokenStream) -> String {
        steady_guard_checked_impl(quote! {}, item).to_string()
    }

    #[test]
    fn guarded_and_exempt_methods_pass_without_error() {
        let out = expand(quote! {
            impl S {
                #[angr_macros::steady_guarded]
                pub fn set_x(&mut self) { self.steady_config_guard(); }

                #[angr_macros::steady_guard_exempt(reason = "reads only")]
                pub fn get_x(&mut self) -> u64 { 0 }
            }
        });
        assert!(!out.contains("compile_error"), "{out}");
        // The exempt attribute is stripped so it is never independently
        // resolved as an attribute macro.
        assert!(!out.contains("steady_guard_exempt"), "{out}");
        assert!(out.contains("steady_guarded"), "{out}");
    }

    #[test]
    fn unlabeled_mut_self_pub_fn_is_an_error() {
        let out = expand(quote! {
            impl S {
                pub fn set_x(&mut self) {}
            }
        });
        assert!(out.contains("must carry either"), "{out}");
    }

    #[test]
    fn shared_self_methods_are_ignored_but_private_mut_self_is_not() {
        let out = expand(quote! {
            impl S {
                pub fn get_x(&self) -> u64 { 0 }
                fn helper(&mut self) {}
            }
        });
        // A private `&mut self` method is still PyO3-callable inside a
        // `#[pymethods]` block, so it must make the same choice a `pub` one
        // does — exactly one error, naming `helper`, not `get_x`.
        assert!(out.contains("fn `helper`: a"), "{out}");
        assert!(!out.contains("fn `get_x`:"), "{out}");
        assert_eq!(out.matches("must carry either").count(), 1, "{out}");
    }

    #[test]
    fn private_mut_self_method_can_be_labeled() {
        let out = expand(quote! {
            impl S {
                #[angr_macros::steady_guard_exempt(reason = "internal bookkeeping only")]
                fn helper(&mut self) {}
            }
        });
        assert!(!out.contains("compile_error"), "{out}");
    }

    #[test]
    fn exempt_without_a_reason_is_an_error() {
        for attr in [
            quote! { #[angr_macros::steady_guard_exempt] },
            quote! { #[angr_macros::steady_guard_exempt(reason = "  ")] },
            quote! { #[angr_macros::steady_guard_exempt(wrong = "x")] },
        ] {
            let out = expand(quote! {
                impl S {
                    #attr
                    pub fn set_x(&mut self) {}
                }
            });
            assert!(out.contains("non-empty string"), "{attr}: {out}");
        }
    }

    #[test]
    fn both_guarded_and_exempt_is_an_error() {
        let out = expand(quote! {
            impl S {
                #[angr_macros::steady_guarded]
                #[angr_macros::steady_guard_exempt(reason = "x")]
                pub fn set_x(&mut self) {}
            }
        });
        assert!(out.contains("pick one"), "{out}");
        // The exempt attribute must be stripped here too, or this case trips
        // a second, unrelated "cannot find attribute macro" error on top of
        // the diagnostic above. The diagnostic message legitimately mentions
        // "steady_guard_exempt" by name, so check for the re-emitted
        // *attribute* token specifically, not a bare substring match.
        assert!(
            !out.contains("# [angr_macros :: steady_guard_exempt"),
            "{out}"
        );
    }

    #[test]
    fn non_impl_item_reports_a_parse_error() {
        let out = expand(quote! { fn nope() {} });
        assert!(out.contains("compile_error"), "{out}");
    }

    #[test]
    fn block_level_attributes_like_pymethods_are_preserved() {
        let out = expand(quote! {
            #[pymethods]
            impl S {
                #[angr_macros::steady_guarded]
                pub fn set_x(&mut self) {}
            }
        });
        assert!(out.contains("pymethods"), "{out}");
    }
}

#[cfg(test)]
mod merge_policy_tests {
    use super::{MECHANICAL_MERGE_POLICIES, VALID_MERGE_POLICIES, derive_merge_policy_impl};
    use quote::quote;

    /// Token-stream text of the expansion. `to_string()` spaces every token,
    /// so a generated signature reads `fn merge_field_depth (& self , ...)`;
    /// matching on the bare `merge_field_<name>` ident is spacing-independent.
    fn expand(input: proc_macro2::TokenStream) -> String {
        derive_merge_policy_impl(input).to_string()
    }

    /// Expansion with every space removed, for assertions about a generated
    /// *body* — `to_string()`'s inter-token spacing is an implementation
    /// detail of `proc_macro2` and not worth pinning.
    fn squished(input: proc_macro2::TokenStream) -> String {
        expand(input).replace(' ', "")
    }

    /// One field per mechanical policy, so a single expansion covers all five
    /// generated bodies.
    fn all_mechanical() -> proc_macro2::TokenStream {
        quote! {
            struct S {
                #[merge_policy = "self_wins"]
                arch: String,
                #[merge_policy = "union"]
                tainted: bool,
                #[merge_policy = "max"]
                depth: u64,
                #[merge_policy = "min"]
                floor: u64,
                #[merge_policy = "warn_on_diverge"]
                optind: i32,
            }
        }
    }

    #[test]
    fn every_mechanical_policy_generates_a_merge_field_method() {
        let out = expand(all_mechanical());
        assert!(!out.contains("compile_error"), "{out}");
        for name in ["arch", "tainted", "depth", "floor", "optind"] {
            assert!(
                out.contains(&format!("merge_field_{name}")),
                "{name}: {out}"
            );
        }
    }

    #[test]
    fn generated_bodies_match_their_policy() {
        let out = squished(all_mechanical());
        // `self_wins` keeps `self`'s value and never reads `others`.
        assert!(out.contains("fnmerge_field_arch(&self,_others:"), "{out}");
        assert!(out.contains("self.arch.clone()"), "{out}");
        // `union` is a logical OR across every branch.
        assert!(out.contains("self.tainted||others.iter().any"), "{out}");
        assert!(out.contains("acc.max(o.depth)"), "{out}");
        assert!(out.contains("acc.min(o.floor)"), "{out}");
        // `warn_on_diverge` is `self_wins` plus the loud divergence check.
        assert!(out.contains("warn_config_divergence(\"optind\""), "{out}");
        assert!(out.contains("self.optind.clone()"), "{out}");
    }

    #[test]
    fn non_mechanical_policies_generate_nothing() {
        // Every valid policy that is not mechanical: labelled, accepted, but
        // hand-written in state/fork.rs::merge.
        for policy in VALID_MERGE_POLICIES
            .iter()
            .filter(|p| !MECHANICAL_MERGE_POLICIES.contains(p))
        {
            let out = expand(quote! {
                struct S {
                    #[merge_policy = #policy]
                    f: u64,
                }
            });
            assert!(!out.contains("compile_error"), "{policy}: {out}");
            assert!(!out.contains("merge_field_f"), "{policy}: {out}");
        }
    }

    #[test]
    fn missing_merge_policy_is_an_error() {
        let out = expand(quote! { struct S { f: u64, } });
        assert!(out.contains("has no #[merge_policy"), "{out}");
    }

    #[test]
    fn unknown_merge_policy_is_an_error() {
        let out = expand(quote! {
            struct S {
                #[merge_policy = "whatever"]
                f: u64,
            }
        });
        assert!(out.contains("unknown merge_policy"), "{out}");
        assert!(out.contains("whatever"), "{out}");
        assert!(!out.contains("merge_field_f"), "{out}");
    }

    #[test]
    fn non_string_merge_policy_is_an_error() {
        let out = expand(quote! {
            struct S {
                #[merge_policy(self_wins)]
                f: u64,
            }
        });
        assert!(out.contains("string literal"), "{out}");
        assert!(!out.contains("merge_field_f"), "{out}");
    }

    #[test]
    fn duplicate_merge_policy_is_an_error() {
        let out = expand(quote! {
            struct S {
                #[merge_policy = "max"]
                #[merge_policy = "min"]
                f: u64,
            }
        });
        assert!(out.contains("more than one #[merge_policy]"), "{out}");
        // Neither of the two policies may win by accident.
        assert!(!out.contains("merge_field_f"), "{out}");
    }

    #[test]
    fn merge_manual_opts_a_mechanical_field_out_of_generation() {
        let out = expand(quote! {
            struct S {
                #[merge_policy = "max"]
                #[merge_manual = "Option<RustBV> has no Ord"]
                f: Option<u64>,
            }
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(!out.contains("merge_field_f"), "{out}");
    }

    #[test]
    fn merge_manual_without_a_reason_is_an_error() {
        for attr in [
            quote! { #[merge_manual] },
            quote! { #[merge_manual = "  "] },
        ] {
            let out = expand(quote! {
                struct S {
                    #[merge_policy = "max"]
                    #attr
                    f: u64,
                }
            });
            assert!(out.contains("non-empty string literal"), "{out}");
        }
    }

    #[test]
    fn duplicate_merge_manual_is_an_error() {
        let out = expand(quote! {
            struct S {
                #[merge_policy = "max"]
                #[merge_manual = "a"]
                #[merge_manual = "b"]
                f: u64,
            }
        });
        assert!(out.contains("more than one #[merge_manual]"), "{out}");
        // Well-formed reasons on both copies: the duplicate is the only problem.
        assert_eq!(out.matches("compile_error").count(), 1, "{out}");
    }

    #[test]
    fn duplicate_merge_manual_also_reports_each_blank_reason() {
        // Two independent typos on one field must surface in one compile, not
        // one per rebuild — so the reason check runs over *every* copy of the
        // attribute, not just the sole one in the non-duplicate case.
        let out = expand(quote! {
            struct S {
                #[merge_policy = "max"]
                #[merge_manual = "  "]
                #[merge_manual = "ok"]
                f: u64,
            }
        });
        assert!(out.contains("more than one #[merge_manual]"), "{out}");
        assert!(out.contains("non-empty string literal"), "{out}");
        assert_eq!(out.matches("compile_error").count(), 2, "{out}");
    }

    #[test]
    fn merge_manual_on_a_non_mechanical_policy_is_redundant() {
        let out = expand(quote! {
            struct S {
                #[merge_policy = "delegate"]
                #[merge_manual = "already hand-written"]
                f: u64,
            }
        });
        assert!(out.contains("redundant"), "{out}");
    }

    #[test]
    fn enums_and_tuple_structs_are_rejected() {
        let out = expand(quote! { enum E { A } });
        assert!(out.contains("only be derived for structs"), "{out}");

        let out = expand(quote! { struct S(u64); });
        assert!(out.contains("requires named fields"), "{out}");
    }

    #[test]
    fn unparseable_input_reports_a_parse_error() {
        let out = expand(quote! { fn nope() {} });
        assert!(out.contains("compile_error"), "{out}");
    }

    #[test]
    fn generics_and_where_clauses_are_carried_onto_the_impl() {
        let out = squished(quote! {
            struct S<T> where T: Ord + Clone {
                #[merge_policy = "max"]
                f: T,
            }
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("impl<T>S<T>"), "{out}");
        assert!(out.contains("whereT:Ord+Clone"), "{out}");
    }

    /// A struct with no fields still gets an (empty) impl block rather than a
    /// diagnostic — the derive has nothing to prove about zero fields.
    #[test]
    fn empty_struct_expands_to_an_empty_impl() {
        let out = expand(quote! { struct S {} });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("impl S"), "{out}");
    }
}

#[cfg(test)]
mod callback_setters_tests {
    use super::callback_setters_impl;
    use quote::quote;

    fn expand(item: proc_macro2::TokenStream) -> String {
        callback_setters_impl(item).to_string()
    }

    #[test]
    fn every_slot_gets_a_setter_named_after_it() {
        let out = expand(quote! { Holder => alpha, beta, });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("impl Holder"), "{out}");
        assert!(out.contains("pub fn set_alpha"), "{out}");
        assert!(out.contains("pub fn set_beta"), "{out}");
    }

    #[test]
    fn setter_body_writes_the_slot_its_name_came_from() {
        // The whole point of the macro: `set_alpha` cannot assign to `beta`.
        let out = expand(quote! { Holder => alpha, beta });
        let alpha = out.find("pub fn set_alpha").unwrap_or(usize::MAX);
        let beta = out.find("pub fn set_beta").unwrap_or(usize::MAX);
        assert!(alpha < beta, "{out}");
        let alpha_body = &out[alpha..beta];
        assert!(alpha_body.contains("self . alpha = Some (cb)"), "{out}");
        assert!(!alpha_body.contains("self . beta"), "{out}");
    }

    #[test]
    fn the_block_carries_pymethods_itself() {
        // `#[pymethods]` on the *caller's* impl block cannot see through an
        // unexpanded macro invocation, so the expansion must attribute its own.
        let out = expand(quote! { Holder => alpha });
        assert!(out.contains(":: pyo3 :: pymethods"), "{out}");
        assert!(out.contains("unreachable_pub"), "{out}");
    }

    #[test]
    fn empty_slot_list_is_rejected() {
        let out = expand(quote! { Holder => });
        assert!(out.contains("at least one slot ident"), "{out}");
    }

    #[test]
    fn missing_fat_arrow_reports_a_parse_error() {
        let out = expand(quote! { Holder alpha });
        assert!(out.contains("compile_error"), "{out}");
    }
}

#[cfg(test)]
mod inspect_test_entries_tests {
    use super::inspect_test_entries_impl;
    use quote::quote;

    fn expand(item: proc_macro2::TokenStream) -> String {
        inspect_test_entries_impl(item).to_string()
    }

    /// How many `compile_error!` invocations the expansion carries — one per
    /// diagnostic, so this distinguishes "both validations fired" from "one
    /// fired and the other was swallowed".
    fn count_compile_errors(out: &str) -> usize {
        out.matches("compile_error").count()
    }

    #[test]
    fn entry_name_drives_all_three_derived_names() {
        let out = expand(quote! {
            Holder =>
            pub fn reg_read(&self, when: &str) -> PyResult<()> { self.forward(when) }
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("pub fn py_call_inspect_reg_read"), "{out}");
        assert!(out.contains("name = \"call_inspect_reg_read\""), "{out}");
        assert!(out.contains("self . call_inspect_reg_read (when)"), "{out}");
    }

    #[test]
    fn surrounding_body_statements_are_preserved() {
        // The `constraints` entry imports its guard AST before forwarding.
        let out = expand(quote! {
            Holder =>
            fn constraints(&self, g: u8) -> PyResult<()> {
                let guard = import(g)?;
                self.forward(&guard)
            }
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("let guard = import (g) ?"), "{out}");
        assert!(
            out.contains("self . call_inspect_constraints (& guard)"),
            "{out}"
        );
    }

    #[test]
    fn caller_attributes_survive() {
        let out = expand(quote! {
            Holder =>
            /// doc line
            #[pyo3(signature = (v))]
            fn expr(&self, v: Option<u8>) -> PyResult<()> { self.forward(v.as_ref()) }
        });
        assert!(out.contains("signature = (v)"), "{out}");
        assert!(out.contains("doc line"), "{out}");
    }

    #[test]
    fn a_body_without_the_placeholder_is_rejected() {
        let out = expand(quote! {
            Holder =>
            fn fork(&self) -> PyResult<()> { Ok(()) }
        });
        assert!(out.contains("exactly once (found 0)"), "{out}");
        assert_eq!(count_compile_errors(&out), 1, "{out}");
    }

    #[test]
    fn naming_a_dispatch_method_directly_is_rejected() {
        // The misroute the macro exists to prevent: `reg_read`'s entry calling
        // the identically-typed `call_inspect_reg_write`. The body still
        // forwards exactly once, so `leaked` is the *only* validation that
        // fires — the combined case is
        // `both_diagnostics_fire_on_one_entry` below.
        let out = expand(quote! {
            Holder =>
            fn reg_read(&self, w: &str) -> PyResult<()> {
                self.call_inspect_reg_write(w)?;
                self.forward(w)
            }
        });
        assert!(
            out.contains("names a `call_inspect_*` method directly"),
            "{out}"
        );
        assert_eq!(count_compile_errors(&out), 1, "{out}");
    }

    #[test]
    fn both_diagnostics_fire_on_one_entry() {
        // The two validations are independent and accumulated, not
        // early-returned: a body that names a dispatch method directly *and*
        // never forwards trips both. Collapsing them into one `return` would
        // leave a contributor fixing one mistake, recompiling, and only then
        // learning about the second — the regression
        // `steady_guarded_both_misuses.rs` guards against for the sibling
        // macro. `inspect_test_entries!` has no `tests/ui/` case of its own
        // (its expansion needs `pyo3` in scope, which the macros crate does
        // not depend on), so the count is asserted here instead.
        let out = expand(quote! {
            Holder =>
            fn reg_read(&self, w: &str) -> PyResult<()> { self.call_inspect_reg_write(w) }
        });
        assert!(out.contains("exactly once (found 0)"), "{out}");
        assert!(
            out.contains("names a `call_inspect_*` method directly"),
            "{out}"
        );
        assert_eq!(count_compile_errors(&out), 2, "{out}");
    }

    #[test]
    fn multiple_entries_each_get_their_own_target() {
        let out = expand(quote! {
            Holder =>
            fn fork(&self) -> PyResult<()> { self.forward() }
            fn irsb(&self) -> PyResult<()> { self.forward() }
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("self . call_inspect_fork ()"), "{out}");
        assert!(out.contains("self . call_inspect_irsb ()"), "{out}");
    }

    #[test]
    fn empty_entry_list_is_rejected() {
        let out = expand(quote! { Holder => });
        assert!(out.contains("at least one"), "{out}");
    }
}
