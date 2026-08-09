//! Run-loop wire types: results and memory-load payloads.
//!
//! Split out of `callbacks.rs` (god-module decomposition, angr-zel8z.2).

/// How a [`RunResult::Error`] should be routed by the exploration stepping
/// loop. Carried as a typed signal so routing does not depend on matching
/// substrings of the human-readable error message (angr-zzju9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunErrorKind {
    /// Graceful deadend: the block could not be lifted (the Python lift
    /// callback returned the empty-IRSB sentinel, e.g. on
    /// `SimEngineError: No bytes in memory`). Routed to the deadended stash,
    /// matching the vanilla Python engine, which deadends states that reach
    /// invalid/unliftable code addresses.
    Deadend,
    /// Genuine execution error (malformed IRSB, memory/op/temp/callback
    /// failure). Routed to the errored stash — *unless* it was reported at
    /// the null address, see [`RunErrorKind::route`].
    Fatal,
}

/// Which stash a [`RunResult::Error`] actually lands in, once the
/// address-dependent carve-out is taken into account.
///
/// `RunErrorKind` alone does not decide this: a `Fatal` reported at pc 0 is
/// deadended, not errored. That carve-out used to live as an unnamed
/// `RunErrorKind::Fatal if addr == 0` guard inline in the routing `match`,
/// which read as a special case with no stated meaning and no way to assert
/// on it (angr-c7xno.30 / angr-91vj9.9). Naming it as its own variant makes
/// each real case explicit and forces the routing site to choose per case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorRoute {
    /// The block was unliftable ([`RunErrorKind::Deadend`]) — a graceful stop
    /// at invalid/unmapped code. Deadended stash.
    UnliftableDeadend,
    /// A [`RunErrorKind::Fatal`] reported at pc 0. The state jumped to the
    /// null address (classically a `ret` into a zeroed return-address slot
    /// after a clean exit), so whatever the interpreter then failed on is a
    /// consequence of the null jump rather than an engine fault. The vanilla
    /// Python engine deadends such states, so we do too.
    NullAddressDeadend,
    /// A genuine execution error at a real address. Errored stash.
    Errored,
}

impl RunErrorKind {
    /// Classify a [`RunResult::Error`] for stash routing. `addr` is the pc the
    /// error was reported at (the value the state's pc is set to).
    pub(crate) fn route(self, addr: u64) -> ErrorRoute {
        match self {
            RunErrorKind::Deadend => ErrorRoute::UnliftableDeadend,
            RunErrorKind::Fatal if addr == 0 => ErrorRoute::NullAddressDeadend,
            RunErrorKind::Fatal => ErrorRoute::Errored,
        }
    }
}

/// Result of running the execution loop.
#[derive(Debug, Clone)]
pub(crate) enum RunResult {
    /// Reached max blocks limit - continue later.
    MaxBlocks { pc: u64 },
    /// Hit a hook address - need Python to handle.
    /// This is the legacy variant without pre-extracted arguments.
    Hook { addr: u64 },
    /// SimProcedure hook hit with pre-extracted arguments.
    /// This allows Python to directly use the arguments without re-extracting.
    SimProcedure {
        /// Address where the SimProcedure is hooked.
        addr: u64,
        /// Name of the SimProcedure (e.g., "strlen", "malloc").
        name: String,
        /// Number of arguments extracted (for Python to know how many to use).
        num_args: usize,
        /// Return address (from stack for calls, or 0 if unknown).
        return_addr: u64,
    },
    /// Syscall encountered - need Python to handle.
    ///
    /// `num` is `None` when the syscall-number register is symbolic
    /// (angr-gffd) — the dispatch loop must force a Python callback in that
    /// case rather than picking a native handler.
    Syscall { num: Option<u64>, pc: u64 },
    /// Symbolic branch - need Python to fork states.
    /// This is returned when use_deferred_forks is false or max_deferred_forks is reached.
    SymbolicBranch {
        condition_id: u64,
        true_target: u64,
        false_target: u64,
    },
    /// Normal block end.
    ///
    /// Never constructed in production: the interpreter's own
    /// `BlockResult::BlockEnd` is consumed inside `run_until_event`, which
    /// loops to the next block instead of surfacing a `RunResult`. Only
    /// `callbacks_tests` / `events_tests` build it (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    BlockEnd {
        next_addr: u64,
        /// `core_outcome` reads only `next_addr` (`BlockEnd { next_addr: pc, .. }`);
        /// the jumpkind is carried for symmetry with the other jump variants
        /// (angr-9ke6b.218 item 5).
        #[allow(dead_code)]
        jumpkind: String,
    },
    /// Error during execution. `kind` tells the stepping loop whether to
    /// deadend the state (unliftable block) or move it to the errored stash
    /// (genuine error), without inspecting the message text.
    Error {
        message: String,
        addr: u64,
        kind: RunErrorKind,
    },
    /// Rust VEX interpreter hit an unsupported operation - need Python VEX engine fallback.
    NeedPythonVEX { addr: u64, reason: String },
    /// Reached max deferred forks limit - return to Python with accumulated forks.
    MaxDeferredForks { pc: u64 },
    /// Symbolic jump target - multiple concrete targets after concretization.
    /// This is returned when a jump target (e.g., ret instruction) is symbolic
    /// but can be concretized to a bounded set of concrete addresses.
    SymbolicJumpTarget {
        /// Concrete target addresses after concretization.
        targets: Vec<u64>,
        /// ID for the stored symbolic expression (for constraint addition).
        condition_id: u64,
        /// Jump kind (Ijk_Ret, Ijk_Call, etc.). Populated from the
        /// `BlockResult::SymbolicJumpTarget` produced by
        /// `interpreter::exits::concretize_jump_target`, but no consumer reads
        /// it: `core_outcome` destructures it as `jumpkind: _`. Retained as
        /// the wire contract (angr-9ke6b.218 item 5).
        #[allow(dead_code)]
        jumpkind: String,
    },
    /// Unconstrained jump - too many targets, exceeds limit.
    /// The state should be moved to the "unconstrained" stash.
    ///
    /// Every field is populated from the `BlockResult::UnconstrainedJump`
    /// built by `interpreter::exits`, but nothing reads them: `core_outcome`
    /// matches this variant as `UnconstrainedJump { .. }`. Retained as the
    /// wire contract (angr-9ke6b.214, angr-9ke6b.218 item 5).
    UnconstrainedJump {
        /// Minimum possible target address.
        #[allow(dead_code)]
        min_target: u64,
        /// Maximum possible target address.
        #[allow(dead_code)]
        max_target: u64,
        /// The configured limit that was exceeded.
        #[allow(dead_code)]
        limit: usize,
        /// Jump kind (Ijk_Ret, Ijk_Call, etc.).
        #[allow(dead_code)]
        jumpkind: String,
    },
    /// Unmodeled function call - need Python to resolve.
    /// This is returned when execution reaches a CALL to an address that isn't hooked.
    /// Python should check if a SimProcedure exists for this address.
    UnmodeledCall {
        /// Address of the unmodeled function.
        addr: u64,
        /// Return address (where to continue after the call).
        return_addr: u64,
        /// Symbol name if available from binary.
        symbol_name: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::{ErrorRoute, RunErrorKind};

    // angr-91vj9.9: the pc-0 carve-out used to be an unnamed guard clause in
    // `core_outcome.rs`'s routing match, so there was nothing to assert on.
    #[test]
    fn deadend_kind_routes_to_unliftable_at_any_addr() {
        assert_eq!(
            RunErrorKind::Deadend.route(0x40_1000),
            ErrorRoute::UnliftableDeadend
        );
        assert_eq!(
            RunErrorKind::Deadend.route(0),
            ErrorRoute::UnliftableDeadend
        );
    }

    #[test]
    fn fatal_at_real_addr_routes_to_errored() {
        assert_eq!(RunErrorKind::Fatal.route(0x40_1000), ErrorRoute::Errored);
        assert_eq!(RunErrorKind::Fatal.route(1), ErrorRoute::Errored);
    }

    #[test]
    fn fatal_at_null_addr_is_its_own_named_route() {
        let route = RunErrorKind::Fatal.route(0);
        assert_eq!(route, ErrorRoute::NullAddressDeadend);
        // The whole point of the variant: it is a deadend, but not the same
        // deadend as an unliftable block.
        assert_ne!(route, ErrorRoute::UnliftableDeadend);
    }
}
