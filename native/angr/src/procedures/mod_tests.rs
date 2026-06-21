use super::*;

#[test]
fn test_registry_creation() {
    let registry = NativeProcedureRegistry::new();
    assert!(registry.is_enabled());
    assert!(registry.has_native("strlen"));
    assert!(registry.has_native("memcpy"));
    assert!(registry.has_native("strcmp"));
}

#[test]
fn test_registry_disable_all() {
    let mut registry = NativeProcedureRegistry::new();
    registry.disable_all();
    assert!(!registry.is_enabled());
    assert!(registry.get("strlen").is_none());
}

#[test]
fn test_registry_disable_specific() {
    let mut registry = NativeProcedureRegistry::new();
    registry.disable("strlen");
    assert!(registry.get("strlen").is_none());
    assert!(registry.get("memcpy").is_some());
}

#[test]
fn test_python_override() {
    let mut registry = NativeProcedureRegistry::new();
    registry.set_python_override("memcpy");
    assert!(registry.has_python_override("memcpy"));
    assert!(registry.get("memcpy").is_none());
    assert!(registry.get("strlen").is_some());
}

#[test]
fn test_stdio_unlocked_aliases_dispatch() {
    // Python angr aliases `x_unlocked = x` for stdio procs. The alias mechanism
    // (NativeSimProcedure::aliases + register) must surface the same native impl
    // under the `_unlocked` dispatch name so glibc-heavy binaries don't round-trip
    // to Python for the unlocked variants.
    let registry = NativeProcedureRegistry::new();
    for (base, alias) in [
        ("fwrite", "fwrite_unlocked"),
        ("fputs", "fputs_unlocked"),
        ("feof", "feof_unlocked"),
        ("fflush", "fflush_unlocked"),
    ] {
        assert!(registry.has_native(base), "{base} should be native");
        assert!(
            registry.has_native(alias),
            "{alias} should resolve via alias"
        );
        // The alias must resolve to the same procedure name as the base.
        assert_eq!(
            registry.get(alias).map(|p| p.name()),
            registry.get(base).map(|p| p.name()),
            "{alias} should dispatch to the {base} impl",
        );
    }
}

#[test]
fn test_fileops_largefile_aliases_dispatch() {
    // Python angr aliases the glibc large-file seek variants `fseeko = fseek`
    // and `ftello = ftell` (same off_t-vs-long signatures on LP64). The
    // declare_proc! `aliases = [...]` mechanism must surface the same native
    // impl under the `o`-suffixed name so `_FILE_OFFSET_BITS=64` binaries that
    // emit `fseeko`/`ftello` don't round-trip to Python.
    let registry = NativeProcedureRegistry::new();
    for (base, alias) in [("fseek", "fseeko"), ("ftell", "ftello")] {
        assert!(registry.has_native(base), "{base} should be native");
        assert!(
            registry.has_native(alias),
            "{alias} should resolve via alias"
        );
        assert_eq!(
            registry.get(alias).map(|p| p.name()),
            registry.get(base).map(|p| p.name()),
            "{alias} should dispatch to the {base} impl",
        );
    }
}

#[test]
fn test_char_io_unlocked_aliases_dispatch() {
    // Python angr aliases the single-char stdio variants `x_unlocked = x`
    // (fputc.py: `fputc_unlocked = putc_unlocked = fputc`; fgetc.py:
    // `getc = fgetc_unlocked = getc_unlocked = fgetc`; getchar.py:
    // `getchar_unlocked = getchar`). All five `_unlocked` names resolve in
    // SIM_PROCEDURES["libc"], so without the alias a glibc-heavy binary that
    // emits the unlocked symbol round-trips to Python. The native impls must
    // surface under the `_unlocked` dispatch name too.
    let registry = NativeProcedureRegistry::new();
    for (base, alias) in [
        ("fputc", "fputc_unlocked"),
        ("putc", "putc_unlocked"),
        ("fgetc", "fgetc_unlocked"),
        ("getc", "getc_unlocked"),
        ("getchar", "getchar_unlocked"),
    ] {
        assert!(registry.has_native(base), "{base} should be native");
        assert!(
            registry.has_native(alias),
            "{alias} should resolve via alias"
        );
        assert_eq!(
            registry.get(alias).map(|p| p.name()),
            registry.get(base).map(|p| p.name()),
            "{alias} should dispatch to the {base} impl",
        );
    }
}
