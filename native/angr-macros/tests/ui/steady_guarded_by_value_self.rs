//! `#[steady_guarded]` on a by-value `self` method: the guard takes
//! `&mut self`, and a consuming receiver is not it.
struct Manager;

impl Manager {
    fn steady_config_guard(&mut self) {}

    #[angr_macros::steady_guarded]
    fn set_x(self) {}
}

fn main() {}
