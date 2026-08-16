//! A duplicated `#[merge_manual]` *and* a blank reason on the same field: two
//! independent typos, both of which must surface in one compile. The reason
//! check runs over every copy of the attribute, not just the sole one in the
//! non-duplicate case (angr-0jh0j.78).
#[derive(angr_macros::MergePolicy)]
struct State {
    #[merge_policy = "self_wins"]
    #[merge_manual = "  "]
    #[merge_manual = "hand-written in fork.rs"]
    arch: u8,
}

fn main() {}
