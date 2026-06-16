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
