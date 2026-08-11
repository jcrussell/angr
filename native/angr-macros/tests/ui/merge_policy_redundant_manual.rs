//! `#[merge_manual]` opts a field out of a *generated* body; on a policy that
//! is hand-written by definition it means the author misread the derive.
#[derive(angr_macros::MergePolicy)]
struct State {
    #[merge_policy = "delegate"]
    #[merge_manual = "hand-written in fork.rs"]
    memory: u64,
}

fn main() {}
