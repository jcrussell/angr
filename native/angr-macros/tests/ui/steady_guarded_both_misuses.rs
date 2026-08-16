//! Both `#[steady_guarded]` validations firing on one invocation: attribute
//! arguments *and* a non-`&mut self` receiver. The two `compile_error!`s are
//! accumulated rather than early-returned, so rustc must report *both* — a
//! refactor that turns either `errors.extend` back into a `return` would leave
//! a contributor fixing one typo, recompiling, and only then learning about the
//! second.
struct Manager;

impl Manager {
    fn steady_config_guard(&mut self) {}

    #[angr_macros::steady_guarded(every = "step")]
    fn set_x(&self) {}
}

fn main() {}
