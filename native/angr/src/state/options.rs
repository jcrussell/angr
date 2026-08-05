//! Options/flags cluster for `RustSimState`.
//!
//! The symex-relevant SimOption flags and per-state behavior toggles:
//! memory-permission enforcement (STRICT_PAGE_ACCESS / ENABLE_NX), the
//! symbolic-jump handling gates (NO_IP_CONCRETIZATION /
//! NO_SYMBOLIC_JUMP_RESOLUTION / KEEP_IP_SYMBOLIC), the generic
//! `set_option`/`has_option` SimOption set, eager-fork forcing, and the
//! `SharedLineageSolver` materialization opt-in. Split out of `mod.rs` per the
//! god-object decomposition (angr-0mqkc.5); mirrors the `construction.rs` /
//! `fork.rs` / `registers.rs` / `memory.rs` extension-impl pattern.

use super::*;

impl RustSimState {
    /// Enable or disable strict memory permission enforcement.
    /// Mirrors angr's STRICT_PAGE_ACCESS option.
    pub fn set_enforce_permissions(&mut self, enabled: bool) {
        self.memory.set_enforce_permissions(enabled);
    }

    /// Whether strict memory permission enforcement is enabled.
    pub fn enforce_permissions(&self) -> bool {
        self.memory.enforce_permissions()
    }

    /// Enable or disable non-executable page enforcement on instruction fetch.
    /// Mirrors angr's ENABLE_NX option. The X check fires only when this AND
    /// `enforce_permissions` (STRICT_PAGE_ACCESS) are both on, matching
    /// Python's heavy VEX engine.
    pub fn set_enforce_nx(&mut self, enabled: bool) {
        self.memory.set_enforce_nx(enabled);
    }

    /// Whether non-executable page enforcement is enabled.
    pub fn enforce_nx(&self) -> bool {
        self.memory.enforce_nx()
    }

    /// Enable or disable IP concretization at block boundaries.
    /// Mirrors angr's NO_IP_CONCRETIZATION option. When true, a symbolic jump
    /// target routes the state to the unconstrained stash without warning
    /// (matches engines/successors.py:292-296). Default off.
    pub fn set_no_ip_concretization(&mut self, enabled: bool) {
        self.no_ip_concretization = enabled;
    }

    /// Whether IP concretization is suppressed for symbolic jump targets.
    pub fn no_ip_concretization(&self) -> bool {
        self.no_ip_concretization
    }

    /// Enable or disable resolution of symbolic jump targets.
    /// Mirrors angr's NO_SYMBOLIC_JUMP_RESOLUTION option. When true, any
    /// symbolic jump target routes the state to the unconstrained stash
    /// instead of enumerating concretizations (matches
    /// engines/successors.py:234-239). Default off.
    pub fn set_no_symbolic_jump_resolution(&mut self, enabled: bool) {
        self.no_symbolic_jump_resolution = enabled;
    }

    /// Whether symbolic jump targets are routed to unconstrained without
    /// enumeration.
    pub fn no_symbolic_jump_resolution(&self) -> bool {
        self.no_symbolic_jump_resolution
    }

    /// Enable or disable preservation of the symbolic IP after concretization.
    /// Mirrors angr's KEEP_IP_SYMBOLIC option. When true, the engine still
    /// concretizes the next pc, but the IP register on each successor is left
    /// holding the original symbolic expression and no narrowing constraint is
    /// added (matches engines/successors.py:297-307,326-331). Default off.
    pub fn set_keep_ip_symbolic(&mut self, enabled: bool) {
        self.keep_ip_symbolic = enabled;
    }

    /// Whether the IP register should be kept symbolic across block boundaries.
    pub fn keep_ip_symbolic(&self) -> bool {
        self.keep_ip_symbolic
    }

    /// Add or remove a symex-relevant SimOption flag (angr-kzjv6). CoW via
    /// `Arc::make_mut` so unforked siblings keep sharing the original set.
    /// `name` is the angr option string (e.g. `"SHORT_READS"`); only the
    /// symex-relevant subset that native SimProcedures consult is threaded
    /// across the FFI in `_add_rust_state`.
    pub fn set_option(&mut self, name: &str, enabled: bool) {
        let opts = Arc::make_mut(&mut self.sim_options);
        if enabled {
            opts.insert(name.to_string());
            // No longer "removed since fork" if it was — see `removed_sim_options`.
            if self.removed_sim_options.contains(name) {
                Arc::make_mut(&mut self.removed_sim_options).remove(name);
            }
        } else if opts.remove(name) {
            Arc::make_mut(&mut self.removed_sim_options).insert(name.to_string());
        }
    }

    /// Whether the named SimOption is active on this state. Lets native
    /// SimProcedures branch on options like `SHORT_READS` (angr-kzjv6) without
    /// the Python round-trip through `rust_state_proxy.options`.
    pub fn has_option(&self, name: &str) -> bool {
        self.sim_options.contains(name)
    }

    /// angr-027h: force eager (immediate) forking for this state regardless of
    /// the manager's `use_deferred_forks` setting. Set on loop-exit forks
    /// resumed at an UnconstrainedJump so they BFS to the find target instead
    /// of recursively re-deferring. Cloned on fork.
    pub fn set_force_eager_forks(&mut self, enabled: bool) {
        self.force_eager_forks = enabled;
    }

    /// Whether this state forces eager forking. See [`Self::set_force_eager_forks`].
    pub fn force_eager_forks(&self) -> bool {
        self.force_eager_forks
    }

    /// Set the fork-time `SharedLineageSolver` materialization opt-in on
    /// this state's solver context (angr-3ms1 step 1b).
    ///
    /// Forwards to [`SymContext::set_use_shared_lineage_solver`]. Setting
    /// on a seed state propagates to every descendant via `fork()` (the
    /// child SymContext inherits the parent's flag), so a single call at
    /// state-creation time is sufficient.
    ///
    /// Inert in this slice — the materialization gate (step 1c) is the
    /// first consumer.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_use_shared_lineage_solver(&self, enabled: bool) {
        self.solver.borrow().set_use_shared_lineage_solver(enabled);
    }

    /// Whether fork-time `SharedLineageSolver` materialization is opted
    /// in on this state's solver context (angr-3ms1 step 1b).
    #[cfg(feature = "vex-engine-z3")]
    pub fn use_shared_lineage_solver(&self) -> bool {
        self.solver.borrow().use_shared_lineage_solver()
    }
}
