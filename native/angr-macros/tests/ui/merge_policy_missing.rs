//! A field with no `#[merge_policy]` — the angr-9ke6b.121 bug family this
//! derive exists to make impossible.
#[derive(angr_macros::MergePolicy)]
struct State {
    #[merge_policy = "self_wins"]
    arch: u8,
    unlabelled: u64,
}

fn main() {}
