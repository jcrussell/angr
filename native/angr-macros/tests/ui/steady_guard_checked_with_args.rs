//! `#[steady_guard_checked]` takes no arguments — an attribute that looks
//! configurable (e.g. an apparent opt-out list) must be rejected rather than
//! silently ignored while the block is still checked in full.
struct Manager;

impl Manager {
    fn steady_config_guard(&mut self) {}
}

#[angr_macros::steady_guard_checked(skip = "set_x")]
impl Manager {
    #[angr_macros::steady_guarded]
    pub fn set_x(&mut self) {}
}

fn main() {}
