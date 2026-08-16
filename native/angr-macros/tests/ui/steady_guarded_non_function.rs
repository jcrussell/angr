//! `#[steady_guarded]` on something that is not a function: `syn::parse2` fails
//! and its error is forwarded verbatim, which must still be a build failure
//! rather than the item passing through unannotated.
#[angr_macros::steady_guarded]
struct NotAFunction;

fn main() {}
