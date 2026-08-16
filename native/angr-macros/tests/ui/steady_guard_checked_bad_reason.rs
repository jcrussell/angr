//! `#[steady_guard_exempt]`'s whole value is the greppable rationale, so a
//! blank one — or one keyed by anything other than `reason` — is no better
//! than the unlabeled third path the checked block exists to forbid.
struct Manager;

impl Manager {
    fn steady_config_guard(&mut self) {}
}

#[angr_macros::steady_guard_checked]
impl Manager {
    #[angr_macros::steady_guard_exempt(reason = "  ")]
    pub fn set_x(&mut self) {}

    #[angr_macros::steady_guard_exempt(because = "guard runs in the caller")]
    pub fn set_y(&mut self) {}
}

fn main() {}
