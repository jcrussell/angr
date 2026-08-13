//! Carrying both `#[steady_guarded]` and `#[steady_guard_exempt]` on the same
//! method is rejected as exactly one error — the exempt attribute must be
//! stripped in this arm too, or the case also trips "cannot find attribute
//! macro `steady_guard_exempt`" on top of the diagnostic below.
struct Manager;

impl Manager {
    fn steady_config_guard(&mut self) {}
}

#[angr_macros::steady_guard_checked]
impl Manager {
    #[angr_macros::steady_guarded]
    #[angr_macros::steady_guard_exempt(reason = "x")]
    pub fn set_x(&mut self) {}
}

fn main() {}
