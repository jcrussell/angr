//! Method delegation for the two-variant `Dyn*` enums in this module.
//!
//! `DynCorpus` (`corpus.rs`) and `DynMutator` (`mutator.rs`) exist only to pick
//! a LibAFL implementation at runtime, so every trait method they implement is
//! the same `match self { A(inner) => inner.m(..), B(inner) => inner.m(..) }`
//! body. Written by hand that is ~140 lines in which a pasted arm can silently
//! call the wrong method — a real hazard for the adjacent
//! `next`/`prev`/`first`/`last` family on `Corpus`. `delegate_two_variant!`
//! writes the bodies from the signatures instead (angr-12jjk.25).

/// Generate forwarding bodies for a two-variant enum's trait methods.
///
/// Invoke inside the `impl` block, passing the enum, its two variants, and the
/// signatures to forward — each terminated by `;`:
///
/// ```ignore
/// impl<I: Input> Corpus<I> for DynCorpus<I> {
///     delegate_two_variant! {
///         DynCorpus { InMem, OnDisk }
///         fn count(&self) -> usize;
///         fn add(&mut self, testcase: Testcase<I>) -> Result<CorpusId, Error>;
///     }
/// }
/// ```
///
/// `&self` and `&mut self` get separate munch arms because an optional-`mut`
/// fragment before `self` is a local ambiguity the macro matcher rejects.
/// A return type is mandatory; write `-> ()` for a unit method.
macro_rules! delegate_two_variant {
    ($enum:ident { $a:ident, $b:ident } $($rest:tt)*) => {
        $crate::fuzzer::delegate::delegate_two_variant!(@munch $enum, $a, $b, $($rest)*);
    };

    (@munch $enum:ident, $a:ident, $b:ident,
     fn $name:ident(&self $(, $arg:ident : $ty:ty)* $(,)?) -> $ret:ty; $($rest:tt)*) => {
        fn $name(&self $(, $arg: $ty)*) -> $ret {
            match self {
                $enum::$a(inner) => inner.$name($($arg),*),
                $enum::$b(inner) => inner.$name($($arg),*),
            }
        }
        $crate::fuzzer::delegate::delegate_two_variant!(@munch $enum, $a, $b, $($rest)*);
    };

    (@munch $enum:ident, $a:ident, $b:ident,
     fn $name:ident(&mut self $(, $arg:ident : $ty:ty)* $(,)?) -> $ret:ty; $($rest:tt)*) => {
        fn $name(&mut self $(, $arg: $ty)*) -> $ret {
            match self {
                $enum::$a(inner) => inner.$name($($arg),*),
                $enum::$b(inner) => inner.$name($($arg),*),
            }
        }
        $crate::fuzzer::delegate::delegate_two_variant!(@munch $enum, $a, $b, $($rest)*);
    };

    (@munch $enum:ident, $a:ident, $b:ident,) => {};
}

pub(crate) use delegate_two_variant;
