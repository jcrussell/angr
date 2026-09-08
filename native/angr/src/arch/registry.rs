//! The registry of architectures the Rust engine supports.
//!
//! Sibling of `arch/mod.rs`. [`ALL_ARCHES`] is the single table naming every
//! supported architecture, the spellings [`arch_desc_from_name`] accepts for
//! it, and the constructors for its [`Arch`] and
//! [`CallingConvention`] singletons; `arch/mod.rs` re-exports the lookups so
//! `crate::arch::arch_from_name` keeps working. Extracted in angr-5mnx3.1.

use super::*;

impl Clone for Box<dyn Arch> {
    fn clone(&self) -> Self {
        // This is a bit of a hack - we rely on architectures being singletons.
        // The dispatch is exhaustive: every `dyn Arch` implementor is one of
        // the six supported singletons, so PPC32/PPC64/S390X can never be the
        // dynamic type here. We delegate to `arch_from_vex` so the singleton
        // lookup table lives in exactly one place; it panics loudly rather
        // than silently mis-cloning should a new VexArch implementor appear.
        arch_from_vex(self.vex_arch())
    }
}

/// Panic/error message for a VexArch the Rust engine does not implement.
///
/// The Rust engine supports six arches (X86, AMD64, ARM, ARM64, MIPS32,
/// MIPS64). For PPC32/PPC64/S390X — which Python angr supports — callers
/// should fall back to the Python engine. `RustExplorationManager`
/// construction rejects these loudly via `arch_from_name`; this message
/// covers the downstream snapshot-restore / interpreter-fork paths that
/// take a `VexArch` directly.
fn unsupported_arch_msg(arch: VexArch) -> String {
    format!(
        "Rust engine does not support {arch:?}; supported arches are X86, AMD64, \
         ARM, ARM64, MIPS32, MIPS64. Use the Python engine for this architecture."
    )
}

/// One row of [`ALL_ARCHES`]: everything the engine knows about a supported
/// architecture, keyed by the names a caller may spell it with.
pub(crate) struct ArchDesc {
    /// Canonical name; equals `Arch::name()` of the arch `make_arch` builds.
    pub name: &'static str,
    /// Accepted spellings other than `name`. Matched case-insensitively,
    /// same as `name` itself.
    pub aliases: &'static [&'static str],
    pub vex: VexArch,
    /// Constructor for the arch singleton. A `&'static dyn Arch` field would
    /// read more directly, but `arch_from_name` / `arch_from_vex` hand out
    /// owned `Box<dyn Arch>` and `dyn Arch` is not `Clone` on its own.
    pub make_arch: fn() -> Box<dyn Arch>,
    pub make_cc: fn() -> Box<dyn CallingConvention>,
}

/// The single registry of supported architectures.
///
/// Before this table the alias lists were restated in four places
/// (`arch_from_name`'s match arms, a per-CC `ARCH_ALIASES` const,
/// `cc_for_arch`'s if/else chain, and a hand-written list in
/// `calling_conventions_tests`), which let them drift: `mips64be` was
/// registered on the MIPS N64 calling convention but never accepted by
/// `arch_from_name`. It is **not** listed here — MIPS64 big-endian states are
/// built as `mips64` plus `Iend_BE`, so a separate arch name would be a second
/// spelling for a distinction we do not model. Widening `arch_from_name` to
/// accept it would be a behavior change, not a refactor.
///
/// Every `name`/`aliases` entry must resolve through both `make_arch` and
/// `make_cc`; `calling_conventions_tests::test_arch_from_name_names_all_have_a_cc`
/// enforces that, and `test_arch_aliases_disjoint` enforces that no spelling
/// belongs to two rows.
pub(crate) const ALL_ARCHES: &[ArchDesc] = &[
    ArchDesc {
        name: "X86",
        aliases: &["x86", "i386", "i486", "i586", "i686"],
        vex: VexArch::X86,
        make_arch: || Box::new(X86),
        make_cc: || Box::new(Cdecl),
    },
    ArchDesc {
        name: "AMD64",
        aliases: &["amd64", "x86_64", "x64"],
        vex: VexArch::AMD64,
        make_arch: || Box::new(AMD64),
        make_cc: || Box::new(SystemVAMD64),
    },
    ArchDesc {
        name: "ARM",
        aliases: &["arm", "armel", "armhf", "armv7", "armv7l"],
        vex: VexArch::ARM,
        make_arch: || Box::new(ARM),
        make_cc: || Box::new(ARMEABI),
    },
    ArchDesc {
        name: "ARM64",
        aliases: &["arm64", "aarch64", "armv8"],
        vex: VexArch::ARM64,
        make_arch: || Box::new(ARM64),
        make_cc: || Box::new(AArch64),
    },
    ArchDesc {
        name: "MIPS32",
        aliases: &["mips", "mips32", "mipsel", "mipsle"],
        vex: VexArch::MIPS32,
        make_arch: || Box::new(MIPS32),
        make_cc: || Box::new(MipsO32),
    },
    ArchDesc {
        name: "MIPS64",
        aliases: &["mips64", "mips64el", "mips64le"],
        vex: VexArch::MIPS64,
        make_arch: || Box::new(MIPS64),
        make_cc: || Box::new(MipsN64),
    },
];

/// Look up the [`ArchDesc`] whose canonical name or alias list contains
/// `name`. Matching is case-insensitive.
pub(crate) fn arch_desc_from_name(name: &str) -> Option<&'static ArchDesc> {
    ALL_ARCHES.iter().find(|d| {
        d.name.eq_ignore_ascii_case(name) || d.aliases.iter().any(|a| a.eq_ignore_ascii_case(name))
    })
}

/// Create an architecture by name.
pub(crate) fn arch_from_name(name: &str) -> Option<Box<dyn Arch>> {
    arch_desc_from_name(name).map(|d| (d.make_arch)())
}

/// Create an architecture from VexArch.
pub(crate) fn arch_from_vex(arch: VexArch) -> Box<dyn Arch> {
    match ALL_ARCHES.iter().find(|d| d.vex == arch) {
        Some(d) => (d.make_arch)(),
        // PPC32/PPC64/S390X are the only VexArch values with no row.
        None => panic!("{}", unsupported_arch_msg(arch)),
    }
}

test_submod!("registry_tests.rs" => tests);
