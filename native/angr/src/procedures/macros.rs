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
//! Registration happens in `NativeProcedureRegistry::new` via
//! `register_procs!`, which takes the flat list of unit-struct paths and
//! emits the `register(Arc::new(..))` call for each.
//!
//! Declaring a procedure and forgetting to add it to that list is a **build
//! failure**, not a silent no-op: the unit struct `declare_proc!` emits is
//! `pub(crate)`, and the list is the only place in the crate that constructs
//! it, so an unregistered procedure trips rustc's `dead_code` lint
//! ("struct `NativeFoo` is never constructed") — an error under CI's clippy
//! `-D warnings` gate. Verified empirically in angr-12jjk.6; a trait `impl`
//! does not count as a construction, so the lint sees through the macro.

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
        pub(crate) struct $struct;

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

/// Declare a zero-argument procedure that always returns the same constant,
/// sized to the architecture word width.
///
/// Thin wrapper over [`declare_proc!`] for the common "return a fixed word"
/// shape — the POSIX identity getters (`getuid`/`geteuid`/`getgid`/`getegid`)
/// each just `return 1000`. The `$value` expression is evaluated per call and
/// widened to `state.arch().bits()` via `RustBV::concrete`, exactly as the
/// expanded `declare_proc!` body would.
///
/// # Example
///
/// ```ignore
/// declare_const_proc! {
///     /// ```c
///     /// uid_t getuid(void);
///     /// ```
///     name = "getuid",
///     struct = NativeGetuid,
///     value = DEFAULT_UID_GID,
/// }
/// ```
#[macro_export]
macro_rules! declare_const_proc {
    (
        $(#[$attr:meta])*
        name = $name:literal,
        struct = $struct:ident,
        value = $value:expr $(,)?
    ) => {
        $crate::declare_proc! {
            $(#[$attr])*
            name = $name,
            struct = $struct,
            args = [],
            call |state| {
                ::std::result::Result::Ok(::std::option::Option::Some(
                    $crate::symbolic::RustBV::concrete($value, state.arch().bits())
                ))
            }
        }
    };
}

/// Register a batch of native procedures into a
/// [`NativeProcedureRegistry`](crate::procedures::NativeProcedureRegistry).
///
/// Each row is the path to a unit struct declared by [`declare_proc!`] (or
/// hand-written); the macro wraps it in `Arc::new(..)` and calls `register`,
/// which also installs every `aliases()` name. Mirrors `register_syscalls!`
/// in `syscalls/mod.rs` — the syscall table was already list-driven, the
/// procedure table was 147 hand-written `registry.register(Arc::new(..));`
/// lines (angr-12jjk.6).
///
/// The list is also what makes a declared-but-unregistered procedure a
/// `dead_code` build failure rather than a silently unreachable one — see
/// the module doc above.
///
/// # Example
///
/// ```ignore
/// register_procs!(
///     registry,
///     [
///         strlen::NativeStrlen,
///         // aliases() covers memmove_unlocked etc.
///         memcpy::NativeMemcpy,
///     ]
/// );
/// ```
macro_rules! register_procs {
    ($registry:expr, [ $( $proc:path ),* $(,)? ]) => {
        $(
            $registry.register(::std::sync::Arc::new($proc));
        )*
    };
}
