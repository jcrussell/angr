//! `NativeTechnique` — exploration techniques that run entirely in Rust.
//!
//! Split out of `exploration::mod` (angr-zel8z.3). These variants avoid Python
//! callback overhead for common technique patterns; they are applied in the
//! Rust exploration loop (see `exploration::helpers`).

/// Native exploration technique variants.
///
/// These techniques run entirely in Rust during the exploration loop,
/// avoiding Python callback overhead for common technique patterns.
#[derive(Debug, Clone)]
pub(crate) enum NativeTechnique {
    /// Limits path length by block count. States exceeding `max_length` blocks
    /// are moved to "cut" (or "_DROP" if `drop` is true).
    LengthLimiter { max_length: usize, drop: bool },
    /// Wall-clock timeout. Exploration stops after `timeout_secs` seconds.
    Timeout {
        timeout_secs: f64,
        start_time: Option<std::time::Instant>,
    },
    /// Basic loop bounding: limits how many times a single address can appear
    /// in a state's history. States exceeding the bound are moved to `discard_stash`.
    LoopBound { bound: usize, discard_stash: String },
    /// Native `ManualMergepoint` parity (angr-op0dn.11.5). States whose pc
    /// reaches `address` are parked in `wait_stash`; once the active stash
    /// drains (or `wait_counter_limit` post-step rounds elapse since the last
    /// arrival), the waiters are grouped by callstack and every group with ≥2
    /// members is merged in-Rust via `_merge_states`. `counter` is the live
    /// post-step round counter — reset to 0 whenever a fresh waiter arrives.
    MergePoint {
        address: u64,
        wait_counter_limit: usize,
        counter: usize,
        wait_stash: String,
    },
}
