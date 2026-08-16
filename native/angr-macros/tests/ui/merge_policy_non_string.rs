//! `#[merge_policy]` with a non-string value: the policy name is matched
//! against `VALID_MERGE_POLICIES` as text, so anything that is not a string
//! literal has to be rejected outright rather than read as "unlabelled".
#[derive(angr_macros::MergePolicy)]
struct State {
    #[merge_policy = 7]
    arch: u8,
}

fn main() {}
