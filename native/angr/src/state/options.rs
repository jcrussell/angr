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

/// Read a flag out of whichever slot `$storage` names. See `state_flags!`.
macro_rules! state_flag_read {
    (field, $this:ident, $getter:ident) => {
        $this.$getter
    };
    (memory, $this:ident, $getter:ident) => {
        $this.memory.$getter()
    };
}

/// Write a flag into whichever slot `$storage` names. See `state_flags!`.
macro_rules! state_flag_write {
    (field, $this:ident, $setter:ident, $getter:ident, $value:ident) => {
        $this.$getter = $value
    };
    (memory, $this:ident, $setter:ident, $getter:ident, $value:ident) => {
        $this.memory.$setter($value)
    };
}

/// Emit the Python-facing half of a flag, or nothing for a `rust_only` one.
/// See `state_flags!`.
macro_rules! state_flag_export {
    (rust_only, $setter:ident, $getter:ident) => {};
    (python, $setter:ident, $getter:ident) => {
        #[pymethods]
        impl PyRustSimState {
            #[doc = concat!(
                                "Python binding for [`RustSimState::", stringify!($setter), "`]."
                            )]
            pub fn $setter(&mut self, enabled: bool) {
                self.inner.$setter(enabled);
            }

            #[doc = concat!(
                                "Python binding for [`RustSimState::", stringify!($getter), "`]."
                            )]
            pub fn $getter(&self) -> bool {
                self.inner.$getter()
            }
        }
    };
}

/// Declare the boolean state flags — both layers of each — in one place.
///
/// Every entry expands to the inherent `set_x` / `x` pair on [`RustSimState`]
/// *and*, unless it is marked `rust_only`, the matching `#[pymethods]` block
/// on `PyRustSimState` exposing them to Python under the same names. Before
/// angr-12jjk.13 those layers were hand-written in two files, so a flag added
/// here and forgotten in `state/pymethods.rs` was silently unreachable from
/// Python with no compile error. Adding a flag is now a single entry below.
///
/// `$storage` says where the bit lives: `field` for a plain `RustSimState`
/// field named after the getter, `memory` for one owned by `self.memory`
/// (whose accessors share the flag's names).
///
/// `use_shared_lineage_solver` is deliberately *not* in this list: it takes
/// `&self` rather than `&mut self` (the bit lives behind `self.solver`) and is
/// `#[cfg(feature = "vex-engine-z3")]`-gated, so it keeps its hand-written
/// pair — and its hand-written `#[pyo3]` wrappers in `state/pymethods.rs`.
macro_rules! state_flags {
    ($(
        $storage:ident, $export:ident {
            $(#[$set_doc:meta])*
            fn $setter:ident(bool);
            $(#[$get_doc:meta])*
            fn $getter:ident() -> bool;
        }
    )*) => {
        impl RustSimState {
            $(
                $(#[$set_doc])*
                pub fn $setter(&mut self, enabled: bool) {
                    state_flag_write!($storage, self, $setter, $getter, enabled);
                }

                $(#[$get_doc])*
                pub fn $getter(&self) -> bool {
                    state_flag_read!($storage, self, $getter)
                }
            )*
        }

        $( state_flag_export!($export, $setter, $getter); )*
    };
}

state_flags! {
    memory, python {
        /// Enable or disable strict memory permission enforcement.
        /// Mirrors angr's STRICT_PAGE_ACCESS option. Default off.
        fn set_enforce_permissions(bool);
        /// Whether strict memory permission enforcement is enabled.
        fn enforce_permissions() -> bool;
    }

    memory, python {
        /// Enable or disable non-executable page enforcement on instruction
        /// fetch. Mirrors angr's ENABLE_NX option. The X check fires only when
        /// this AND `enforce_permissions` (STRICT_PAGE_ACCESS) are both on,
        /// matching Python's heavy VEX engine. Default off.
        fn set_enforce_nx(bool);
        /// Whether non-executable page enforcement is enabled.
        fn enforce_nx() -> bool;
    }

    field, python {
        /// Enable or disable IP concretization at block boundaries.
        /// Mirrors angr's NO_IP_CONCRETIZATION option. When true, a symbolic
        /// jump target routes the state to the unconstrained stash without
        /// warning (matches `engines/successors.py`, the
        /// `NO_IP_CONCRETIZATION` branch of `_categorize_successor`).
        /// Default off.
        fn set_no_ip_concretization(bool);
        /// Whether IP concretization is suppressed for symbolic jump targets.
        fn no_ip_concretization() -> bool;
    }

    field, python {
        /// Enable or disable resolution of symbolic jump targets.
        /// Mirrors angr's NO_SYMBOLIC_JUMP_RESOLUTION option. When true, any
        /// symbolic jump target routes the state to the unconstrained stash
        /// instead of enumerating concretizations (matches
        /// `engines/successors.py`, the `NO_SYMBOLIC_JUMP_RESOLUTION` branch
        /// of `_eval_target_jumptable`'s caller). Default off.
        fn set_no_symbolic_jump_resolution(bool);
        /// Whether symbolic jump targets are routed to unconstrained without
        /// enumeration.
        fn no_symbolic_jump_resolution() -> bool;
    }

    field, python {
        /// Enable or disable preservation of the symbolic IP after
        /// concretization. Mirrors angr's KEEP_IP_SYMBOLIC option. When true,
        /// the engine still concretizes the next pc, but the IP register on
        /// each successor is left holding the original symbolic expression and
        /// no narrowing constraint is added (matches `engines/successors.py`,
        /// the `KEEP_IP_SYMBOLIC` branches of `_categorize_successor`).
        /// Default off.
        fn set_keep_ip_symbolic(bool);
        /// Whether the IP register should be kept symbolic across block
        /// boundaries.
        fn keep_ip_symbolic() -> bool;
    }

    field, rust_only {
        /// angr-027h: force eager (immediate) forking for this state
        /// regardless of the manager's `use_deferred_forks` setting. Set on
        /// loop-exit forks resumed at an UnconstrainedJump so they BFS to the
        /// find target instead of recursively re-deferring. Cloned on fork.
        ///
        /// `rust_only`: the sole caller is
        /// `exploration::core_outcome_handlers`, and no Python-side knob wants
        /// to override that heuristic per state.
        fn set_force_eager_forks(bool);
        /// Whether this state forces eager forking.
        /// See [`RustSimState::set_force_eager_forks`].
        fn force_eager_forks() -> bool;
    }
}

/// angr's `SYMBOLIC_INITIAL_VALUES` SimOption string (angr-c7xno.61).
///
/// Gates whether an "unconstrained" value is a fresh symbol or a concrete
/// zero — `SimSolver::Unconstrained` in `angr/state_plugins/solver.py` returns
/// `BVV(0, bits)` unless this is in `state.options`. The native consumer is
/// `procedures/stub.rs::NativeReturnUnconstrained`; the Python mirror is
/// `_NATIVE_SIMOPTIONS` in `angr/exploration/rust_manager.py`.
///
/// Unlike the other threaded options this one is seeded ON by
/// `RustSimState::with_solver_endian` — see the comment there for why.
pub const SYMBOLIC_INITIAL_VALUES: &str = "SYMBOLIC_INITIAL_VALUES";

impl RustSimState {
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
