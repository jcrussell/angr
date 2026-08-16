//! A tuple struct has no field names to attach a policy to (and none to name in
//! the diagnostic), so it is rejected before the per-field walk.
#[derive(angr_macros::MergePolicy)]
struct State(u8, u64);

fn main() {}
