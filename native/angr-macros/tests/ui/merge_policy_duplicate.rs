//! Two `#[merge_policy]` attributes on one field: whichever the derive picked
//! would be arbitrary, and the losing label would read as documentation of a
//! treatment the field does not get.
#[derive(angr_macros::MergePolicy)]
struct State {
    #[merge_policy = "self_wins"]
    #[merge_policy = "union"]
    arch: u8,
}

fn main() {}
