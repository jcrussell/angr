//! An `inspect_dispatch!` invocation with no entries after `=>` is a mistake,
//! not an empty-but-valid impl block: it would silently delete the whole
//! dispatch family.
struct Holder;

angr_macros::inspect_dispatch! {
    Holder =>
}

fn main() {}
