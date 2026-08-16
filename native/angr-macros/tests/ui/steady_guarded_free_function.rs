//! `#[steady_guarded]` on a free function: there is no `self` to guard, so the
//! injected `self.steady_config_guard();` could not resolve at all.
#[angr_macros::steady_guarded]
fn set_x() {}

fn main() {}
