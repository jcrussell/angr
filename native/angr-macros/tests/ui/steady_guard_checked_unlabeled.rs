//! A `&mut self pub fn` inside a `#[steady_guard_checked]` impl block must
//! carry either `#[steady_guarded]` or `#[steady_guard_exempt(reason = "...")]`
//! — an unlabeled mutator is rejected rather than silently passing through.
struct Manager;

impl Manager {
    fn steady_config_guard(&mut self) {}
}

#[angr_macros::steady_guard_checked]
impl Manager {
    pub fn set_x(&mut self) {}
}

fn main() {}
