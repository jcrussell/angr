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
