//! `#[steady_guard_checked]` on something that is not an `impl` block: the
//! `syn::parse2` error is forwarded verbatim, which must still fail the build
//! rather than let the item through unchecked.
#[angr_macros::steady_guard_checked]
struct NotAnImpl;

fn main() {}
