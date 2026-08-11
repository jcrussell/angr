//! `#[steady_guarded]` takes no arguments — an attribute that looks
//! configurable must be rejected rather than silently ignored.
struct Manager;

impl Manager {
    fn steady_config_guard(&mut self) {}

    #[angr_macros::steady_guarded(every = "step")]
    fn set_x(&mut self) {}
}

fn main() {}
