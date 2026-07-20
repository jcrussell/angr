use std::time::Duration;

use backtrace::Backtrace;
use libafl::executors::ExitKind;

use super::apply_wall_clock_timeout;

// Pins the wall-clock timeout semantics enforced in
// PyExecutorInner::run_target (angr-ph300.69): before this fix the Fuzzer
// accepted a `timeout=` kwarg, coerced None to Some(0ms), and never read it.

#[test]
fn no_timeout_leaves_clean_exit_untouched() {
    // None budget == "run as long as it takes".
    let out = apply_wall_clock_timeout(Ok(ExitKind::Ok), None, Duration::from_secs(3600));
    assert!(matches!(out, Ok(ExitKind::Ok)));
}

#[test]
fn zero_budget_is_treated_as_disabled() {
    // A zero-length budget means "no timeout", not "always timeout".
    let out = apply_wall_clock_timeout(
        Ok(ExitKind::Ok),
        Some(Duration::ZERO),
        Duration::from_secs(1),
    );
    assert!(matches!(out, Ok(ExitKind::Ok)));
}

#[test]
fn under_budget_run_is_not_a_timeout() {
    let out = apply_wall_clock_timeout(
        Ok(ExitKind::Ok),
        Some(Duration::from_millis(100)),
        Duration::from_millis(40),
    );
    assert!(matches!(out, Ok(ExitKind::Ok)));
}

#[test]
fn overrun_clean_exit_becomes_timeout() {
    let out = apply_wall_clock_timeout(
        Ok(ExitKind::Ok),
        Some(Duration::from_millis(50)),
        Duration::from_millis(200),
    );
    assert!(matches!(out, Ok(ExitKind::Timeout)));
}

#[test]
fn budget_boundary_is_inclusive() {
    // elapsed == budget counts as a timeout (>= comparison).
    let out = apply_wall_clock_timeout(
        Ok(ExitKind::Ok),
        Some(Duration::from_millis(50)),
        Duration::from_millis(50),
    );
    assert!(matches!(out, Ok(ExitKind::Timeout)));
}

#[test]
fn overrun_crash_is_still_reported_as_crash() {
    // A crash found during a slow run is worth keeping — do not mask it.
    let out = apply_wall_clock_timeout(
        Ok(ExitKind::Crash),
        Some(Duration::from_millis(50)),
        Duration::from_millis(200),
    );
    assert!(matches!(out, Ok(ExitKind::Crash)));
}

#[test]
fn overrun_error_propagates_unchanged() {
    let err = Err(libafl::Error::Unknown("boom".to_string(), Backtrace::new()));
    let out = apply_wall_clock_timeout(
        err,
        Some(Duration::from_millis(50)),
        Duration::from_millis(200),
    );
    assert!(out.is_err());
}
