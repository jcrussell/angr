use super::*;
use std::mem::size_of;

/// Documentation-grade size probe. Captured to validate the iter-61
/// decision to suppress `clippy::result_large_err` / `large_enum_variant`
/// instead of boxing the inline `RustSimState` / `PendingCallback`. Run
/// with `cargo test --release -p angr -- --nocapture sizes::print`.
#[test]
fn print_step_error_sizes() {
    let step_err = size_of::<StepError>();
    let result_unit = size_of::<Result<(), StepError>>();
    let pending = size_of::<PendingCallback>();
    let sim_state = size_of::<RustSimState>();
    let boxed_result = size_of::<Result<(), Box<StepError>>>();
    println!("StepError                       = {step_err} bytes");
    println!("Result<(), StepError>           = {result_unit} bytes");
    println!("Result<(), Box<StepError>>      = {boxed_result} bytes");
    println!("PendingCallback                 = {pending} bytes");
    println!("RustSimState                    = {sim_state} bytes");
    println!(
        "boxing-would-save               = {} bytes per Err return",
        result_unit.saturating_sub(boxed_result)
    );
}
