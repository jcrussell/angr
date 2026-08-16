//! Two `#[merge_manual]` attributes on one field: the second reason would be
//! silently dropped, leaving a rationale in the source that documents nothing.
#[derive(angr_macros::MergePolicy)]
struct State {
    #[merge_policy = "self_wins"]
    #[merge_manual = "hand-written in fork.rs"]
    #[merge_manual = "and also in snapshot.rs"]
    arch: u8,
}

fn main() {}
