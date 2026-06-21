//! Macros for declaring native SimProcedures in lockstep.
//!
//! `declare_proc!` emits a unit struct + `NativeSimProcedure` impl from a
//! single declarative form. The procedure's `name()`, `num_args()`, and the
//! per-argument extraction code are all derived from the same `args = [...]`
//! list, so it is impossible to register a procedure whose declared arity
//! disagrees with the actual extraction count.
//!
//! # Example
//!
//! ```ignore
//! declare_proc! {
//!     name = "strlen",
//!     struct = NativeStrlen,
//!     args = [addr: concrete],
//!     call |state| {
//!         scan_for_null(state, addr, MAX_STRLEN as u64)
//!     }
//! }
//! ```
//!
//! The argument kinds are:
//!   * `concrete` — call `extract_concrete_arg` and bind a `u64`. Symbolic
//!     callers fall back to Python.
//!   * `bv` — clone the raw `RustBV` and bind it; the body decides how to
//!     handle symbolic / concrete cases.
//!
//! An optional `no_return = true` flag overrides the default `no_return()`
//! from the trait (terminal procedures like `exit`, `abort`).
//!
//! An optional `aliases = ["name", ...]` list registers the same native impl
//! under additional names (e.g. glibc large-file variants `fseeko`/`ftello`
//! that Python angr aliases `fseeko = fseek`). Mirrors the trait's
//! `aliases()` default; see `NativeProcedureRegistry::register`.
//!
//! Registration with `NativeProcedureRegistry::new` remains hand-written;
//! the macro only enforces the declaration-side lockstep.

#[macro_export]
macro_rules! declare_proc {
    (
        $(#[$attr:meta])*
        name = $name:literal,
        struct = $struct:ident,
        args = [ $($arg:ident : $kind:ident),* $(,)? ],
        $(no_return = $no_ret:literal,)?
        $(aliases = [ $($alias:literal),* $(,)? ],)?
        call |$state:ident| $body:block
    ) => {
        $(#[$attr])*
        pub struct $struct;

        impl $crate::procedures::NativeSimProcedure for $struct {
            #[inline]
            fn name(&self) -> &'static str { $name }

            #[inline]
            fn num_args(&self) -> usize {
                $crate::declare_proc!(@count $( $arg )*)
            }

            $(
                #[inline]
                fn no_return(&self) -> bool { $no_ret }
            )?

            $(
                #[inline]
                fn aliases(&self) -> &'static [&'static str] {
                    &[ $($alias),* ]
                }
            )?

            fn call(
                &self,
                $state: &mut $crate::state::RustSimState,
                args: &[$crate::symbolic::RustBV],
            ) -> ::std::result::Result<
                ::std::option::Option<$crate::symbolic::RustBV>,
                $crate::procedures::ProcedureError,
            > {
                let mut _idx: usize = 0;
                $(
                    let $arg = $crate::declare_proc!(
                        @extract $kind, args, _idx, stringify!($arg)
                    )?;
                    _idx += 1;
                )*
                let _ = _idx;
                // Reference `args` unconditionally so a zero-argument
                // procedure (e.g. `clearenv`, `rand`) does not trip the
                // unused-variable lint on the generated parameter.
                let _ = &args;
                $body
            }
        }
    };

    // ---- count helpers ----
    (@count) => { 0usize };
    (@count $head:ident $($tail:ident)*) => {
        1usize + $crate::declare_proc!(@count $($tail)*)
    };

    // ---- extraction helpers ----
    (@extract concrete, $args:ident, $idx:ident, $name:expr) => {
        $crate::procedures::extract_concrete_arg(&$args[$idx], $name)
    };
    (@extract bv, $args:ident, $idx:ident, $name:expr) => {
        ::std::result::Result::<
            $crate::symbolic::RustBV,
            $crate::procedures::ProcedureError,
        >::Ok($args[$idx].clone())
    };
}
