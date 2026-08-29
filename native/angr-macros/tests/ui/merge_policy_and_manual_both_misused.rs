//! One field misusing `#[merge_policy]` *and* `#[merge_manual]` at once. The
//! two attributes are validated by independent branches of
//! `derive_merge_policy_impl`, which accumulate into a shared `errors` stream
//! rather than returning at the first failure; a refactor that made the policy
//! branch return early would silently swallow the `#[merge_manual]` diagnostic,
//! and the single-attribute cases could not see it (angr-0jh0j.86).
#[derive(angr_macros::MergePolicy)]
struct State {
    #[merge_policy = "nonsense"]
    #[merge_manual = "   "]
    arch: u8,
}

fn main() {}
