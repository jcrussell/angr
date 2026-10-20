//! A field with no `#[merge_policy]` — the bug family this derive exists to
//! make impossible (angr-9ke6b.121).
#[derive(angr_macros::MergePolicy)]
struct State {
    #[merge_policy = "self_wins"]
    arch: u8,
    unlabelled: u64,
}

fn main() {}
