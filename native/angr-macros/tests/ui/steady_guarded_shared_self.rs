//! `#[steady_guarded]` on a `&self` method: the injected guard needs
//! `&mut self`, so the attribute must say so instead of letting the injection
//! fail with "cannot borrow as mutable".
struct Manager;

impl Manager {
    fn steady_config_guard(&mut self) {}

    #[angr_macros::steady_guarded]
    fn set_x(&self) {}
}

fn main() {}
