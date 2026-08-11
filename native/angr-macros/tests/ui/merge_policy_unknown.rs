//! A misspelled policy name must not silently mean "no treatment".
#[derive(angr_macros::MergePolicy)]
struct State {
    #[merge_policy = "self-wins"]
    arch: u8,
}

fn main() {}
