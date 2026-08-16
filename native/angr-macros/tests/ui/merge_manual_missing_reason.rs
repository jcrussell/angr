//! `#[merge_manual]` opts a field out of the generated body, so its reason is
//! the only record of *why* — both an absent value and a blank one must fail.
#[derive(angr_macros::MergePolicy)]
struct State {
    #[merge_policy = "self_wins"]
    #[merge_manual]
    arch: u8,

    #[merge_policy = "union"]
    #[merge_manual = "   "]
    seen: u64,
}

fn main() {}
