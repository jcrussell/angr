use super::*;

#[test]
fn test_cpuid_not_handled_natively() {
    // CPUID/RDTSCP receive a GSPTR arg and are deliberately routed to the
    // Python engine; they must NOT be registered for native dispatch
    // (angr-t1ok).
    let dispatch = DirtyHelperDispatch::new();
    for name in [
        "amd64g_dirtyhelper_CPUID_baseline",
        "amd64g_dirtyhelper_CPUID_avx2",
        "x86g_dirtyhelper_CPUID_sse3",
        "amd64g_dirtyhelper_RDTSCP",
    ] {
        assert!(dispatch.try_call(name, &[]).is_none(), "{name}");
    }
}

/// angr-vmi2l.69: RDTSC and IN return fresh symbols in the Python engine, so
/// a native concrete stand-in would drop one side of any branch over them.
#[test]
fn test_rdtsc_and_in_not_handled_natively() {
    let dispatch = DirtyHelperDispatch::new();
    for name in [
        "amd64g_dirtyhelper_RDTSC",
        "x86g_dirtyhelper_RDTSC",
        "amd64g_dirtyhelper_IN",
        "x86g_dirtyhelper_IN",
    ] {
        assert!(!dispatch.handles(name), "{name}");
        assert!(dispatch.try_call(name, &[0x80, 1]).is_none(), "{name}");
    }
}

#[test]
fn test_unknown_helper() {
    let dispatch = DirtyHelperDispatch::new();
    assert!(dispatch.try_call("unknown_helper", &[]).is_none());
}

#[test]
fn test_out_port() {
    let dispatch = DirtyHelperDispatch::new();
    let out_result = dispatch
        .try_call("amd64g_dirtyhelper_OUT", &[0x80, 0x00])
        .unwrap();
    assert!(out_result.return_value.is_none());
}
