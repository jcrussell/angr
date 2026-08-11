// Tests for exploration/manager_methods_procedures.rs — the native-procedure
// registry corner of the `#[pymethods]` surface (angr-03vl4.13).
//
// Two contracts live at this boundary rather than inside
// `NativeProcedureRegistry`:
//
//   * `register_unconstrained_stubs` is the only mutator here that *filters*
//     its input. A `SimLibrary` hands out `ReturnUnconstrained` for every
//     symbol it has no model for, so Python passes a batch keyed on the
//     binary's own symbol names; a zero return width, or a name that already
//     has a real native implementation, must not overwrite anything. Every
//     other method in the file forwards verbatim.
//   * `has_native_procedure` answers "is there an implementation", NOT "will
//     it be dispatched". `disable_native_procedure{,s}` and
//     `set_python_override` both make `NativeProcedureRegistry::get` return
//     `None` while leaving `has_native` true, so a caller that treats the two
//     as synonyms reports a suppressed procedure as live.
use super::*;

/// Address of the registry entry behind `name`, or `None` when the name is
/// unregistered. Raw address rather than the `Arc` so the borrow of `mgr` ends
/// before the caller mutates it; `Arc<dyn NativeSimProcedure>` is a fat
/// pointer, hence `ptr::addr_eq` at the comparison sites.
fn entry_addr(mgr: &RustExplorationManager, name: &str) -> Option<*const ()> {
    mgr.native_procedures
        .get(name)
        .map(|proc| Arc::as_ptr(proc).cast::<()>())
}

/// `register_unconstrained_stubs` registers only the names that are both
/// non-zero-width and not already backed by a real native implementation.
#[test]
fn register_unconstrained_stubs_filters_zero_width_and_real_natives() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    assert!(mgr.has_native_procedure("strlen"), "baseline: strlen is native");
    let strlen_before = entry_addr(&mgr, "strlen").expect("strlen entry");

    mgr.register_unconstrained_stubs(vec![
        // Real native implementation already present — the stub loses.
        ("strlen".to_string(), 64),
        // Zero return width: nothing to hand back, so no stub is made.
        ("binary_void_helper".to_string(), 0),
        // Neither exclusion applies: this one lands.
        ("binary_unmodelled_helper".to_string(), 64),
    ]);

    let strlen_after = entry_addr(&mgr, "strlen").expect("strlen entry survived");
    assert!(
        std::ptr::addr_eq(strlen_before, strlen_after),
        "a ReturnUnconstrained stub must never displace a real native procedure",
    );
    assert!(
        !mgr.has_native_procedure("binary_void_helper"),
        "a zero return width is skipped, not registered as a 0-bit stub",
    );
    assert!(
        mgr.has_native_procedure("binary_unmodelled_helper"),
        "the one name with neither exclusion is registered",
    );
}

/// Re-registering the same unmodelled name is idempotent in effect: the second
/// call is filtered by the `has_native` check the first call's own
/// registration turned true, so the entry is left alone rather than replaced.
#[test]
fn register_unconstrained_stubs_is_idempotent_for_a_name_it_already_added() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    mgr.register_unconstrained_stubs(vec![("binary_helper".to_string(), 32)]);
    let first = entry_addr(&mgr, "binary_helper").expect("stub registered");

    // A different width, to catch a silent overwrite rather than only a
    // pointer-equal re-insert.
    mgr.register_unconstrained_stubs(vec![("binary_helper".to_string(), 64)]);
    let second = entry_addr(&mgr, "binary_helper").expect("stub still registered");

    assert!(
        std::ptr::addr_eq(first, second),
        "the second batch is filtered by has_native; the 32-bit stub stays",
    );
}

/// `has_native_procedure` reports implementation presence, not dispatchability:
/// both suppression mechanisms leave it true while stopping dispatch.
#[test]
fn has_native_procedure_ignores_dispatch_suppression() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    mgr.disable_native_procedure("strlen");
    mgr.set_python_override("memcpy");

    assert!(
        mgr.has_native_procedure("strlen") && mgr.has_native_procedure("memcpy"),
        "suppression does not remove the implementation",
    );
    assert!(
        entry_addr(&mgr, "strlen").is_none() && entry_addr(&mgr, "memcpy").is_none(),
        "both suppression forms make the dispatcher fall through to Python",
    );

    mgr.enable_native_procedure("strlen");
    mgr.remove_python_override("memcpy");
    assert!(
        entry_addr(&mgr, "strlen").is_some() && entry_addr(&mgr, "memcpy").is_some(),
        "each suppression is undone by its own inverse",
    );
}

/// The global switch and the per-name sets are independent layers: flipping
/// `enabled` back on does not resurrect a name that was individually disabled.
#[test]
fn global_disable_and_per_name_disable_are_independent() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    mgr.disable_native_procedure("strlen");
    mgr.disable_native_procedures();
    assert!(!mgr.native_procedures_enabled());
    assert!(
        entry_addr(&mgr, "memcpy").is_none(),
        "the global switch suppresses names that were never individually disabled",
    );

    mgr.enable_native_procedures();
    assert!(mgr.native_procedures_enabled());
    assert!(
        entry_addr(&mgr, "memcpy").is_some(),
        "memcpy comes back with the global switch",
    );
    assert!(
        entry_addr(&mgr, "strlen").is_none(),
        "strlen's per-name disable outlives the global re-enable — \
         enable_native_procedures is not a reset",
    );
}

/// `list_native_procedures` mirrors the registry keys, so it grows with a
/// registered stub and — like `has_native_procedure` — is blind to suppression.
#[test]
fn list_native_procedures_tracks_registration_not_suppression() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let baseline = mgr.list_native_procedures().len();

    mgr.register_unconstrained_stubs(vec![("binary_helper".to_string(), 64)]);
    let listed = mgr.list_native_procedures();
    assert_eq!(listed.len(), baseline + 1);
    assert!(listed.iter().any(|name| name == "binary_helper"));

    mgr.disable_native_procedures();
    assert_eq!(
        mgr.list_native_procedures().len(),
        baseline + 1,
        "the listing is a registry inventory, not a dispatch preview",
    );
}
