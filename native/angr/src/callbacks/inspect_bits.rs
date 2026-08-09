//! Single source of truth for the `state.inspect` event-bit assignment
//! (angr-12jjk.21).
//!
//! Every inspect dispatch site used to hard-code its bit as a bare `u8`
//! literal with a prose comment naming the event ("// TmpRead bit assigned
//! in `_INSPECT_EVENT_SPECS`"). Nothing tied the number to the callback it
//! gated, so pairing `inspect_event_enabled(13)` with
//! `call_inspect_tmp_write` compiled just as happily as the correct pairing
//! and silently mis-gated the breakpoint.
//!
//! The `inspect_events!` table below declares (variant, bit, event name)
//! once; [`InspectBit`] is generated from it and is the only type
//! [`PythonCallbacks::inspect_event_enabled`] accepts, so a dispatch site
//! now names its event instead of its bit.
//!
//! The bit numbers mirror the Python-side `_INSPECT_EVENT_SPECS` table in
//! `angr/exploration/rust_state_proxy.py`, which is what actually builds the
//! enabled-mask Python hands to `py_set_inspect_enabled`. That table stays
//! authoritative for the wire format; adding an event means adding a row in
//! both places (the 5-touchpoint recipe documented above
//! `_INSPECT_EVENT_SPECS`). The first six bits additionally coincide with
//! `crate::state::InspectEvent`, which `inspect_bits_tests` asserts.
//!
//! [`PythonCallbacks::inspect_event_enabled`]: super::PythonCallbacks::inspect_event_enabled

/// Generate [`InspectBit`] from a `(variant, bit, event name)` table.
///
/// Kept as a macro rather than a hand-written enum so `ALL`, `bit()` and
/// `event_name()` cannot drift out of sync with the variant list — the three
/// of them are what the tests and the Python cross-check read.
macro_rules! inspect_events {
    ($( $(#[$attr:meta])* $variant:ident = $bit:literal => $name:literal ),+ $(,)?) => {
        /// A `state.inspect` event, identified by its bit in the
        /// `PythonCallbacks` enabled-mask (an `AtomicU32`, so bits 0..=31 are
        /// representable).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(u8)]
        #[allow(
            dead_code,
            reason = "the table is deliberately complete: the Python-dispatched events (SimProcedure/Syscall/Dirty) have no Rust gate reading their bit, but omitting them would let a future event silently reuse a taken bit"
        )]
        pub(crate) enum InspectBit {
            $( $(#[$attr])* $variant = $bit, )+
        }

        #[allow(
            dead_code,
            reason = "ALL / event_name exist for the table-consistency tests (bit density, and the cross-checks against crate::state::InspectEvent and Python's _INSPECT_EVENT_SPECS); the engine only ever calls bit()"
        )]
        impl InspectBit {
            /// Every event, in bit order. Used by the tests to check bit
            /// density/uniqueness and the `InspectEvent` overlap.
            pub(crate) const ALL: &'static [InspectBit] = &[ $( InspectBit::$variant, )+ ];

            /// The event's bit position in the enabled-mask.
            #[inline(always)]
            pub(crate) const fn bit(self) -> u8 {
                self as u8
            }

            /// The Python `state.inspect` event name (the key in
            /// `_INSPECT_EVENT_SPECS` and the suffix of the matching
            /// `call_inspect_*` / `_cb_inspect_*` method).
            pub(crate) const fn event_name(self) -> &'static str {
                match self {
                    $( InspectBit::$variant => $name, )+
                }
            }
        }
    };
}

inspect_events! {
    MemRead = 0 => "mem_read",
    MemWrite = 1 => "mem_write",
    RegRead = 2 => "reg_read",
    RegWrite = 3 => "reg_write",
    Fork = 4 => "fork",
    Exit = 5 => "exit",
    Instruction = 6 => "instruction",
    Irsb = 7 => "irsb",
    Call = 8 => "call",
    Return = 9 => "return",
    /// Dispatched from Python (`dispatch_origin: 'python'`), so no Rust
    /// gate reads this bit — it exists to keep the table complete.
    SimProcedure = 10 => "simprocedure",
    /// Python-dispatched; see [`InspectBit::SimProcedure`].
    Syscall = 11 => "syscall",
    /// Python-dispatched; see [`InspectBit::SimProcedure`].
    Dirty = 12 => "dirty",
    TmpRead = 13 => "tmp_read",
    TmpWrite = 14 => "tmp_write",
    Statement = 15 => "statement",
    Expr = 16 => "expr",
    AddressConcretization = 17 => "address_concretization",
    SymbolicVariable = 18 => "symbolic_variable",
    Constraints = 19 => "constraints",
    VexLift = 20 => "vex_lift",
}

test_submod!("inspect_bits_tests.rs" => inspect_bits_tests);
