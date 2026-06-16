use super::*;

#[test]
fn test_cpuid_not_handled_natively() {
    // CPUID/RDTSCP receive a GSPTR arg and are deliberately routed to the
    // Python engine; they must NOT be registered for native dispatch
    // (angr-t1ok: dead-code removal).
    let dispatch = DirtyHelperDispatch::new();
    assert!(!dispatch.can_handle("amd64g_dirtyhelper_CPUID_baseline"));
    assert!(!dispatch.can_handle("amd64g_dirtyhelper_CPUID_avx2"));
    assert!(!dispatch.can_handle("x86g_dirtyhelper_CPUID_sse3"));
    assert!(!dispatch.can_handle("amd64g_dirtyhelper_RDTSCP"));
}

#[test]
fn test_rdtsc() {
    let dispatch = DirtyHelperDispatch::new();

    assert!(dispatch.can_handle("amd64g_dirtyhelper_RDTSC"));

    let result1 = dispatch.try_call("amd64g_dirtyhelper_RDTSC", &[]).unwrap();
    let result2 = dispatch.try_call("amd64g_dirtyhelper_RDTSC", &[]).unwrap();

    // TSC should increment
    assert!(result2.return_value.unwrap() > result1.return_value.unwrap());
}

#[test]
fn test_unknown_helper() {
    let dispatch = DirtyHelperDispatch::new();

    assert!(!dispatch.can_handle("unknown_helper"));
    assert!(dispatch.try_call("unknown_helper", &[]).is_none());
}

#[test]
fn test_in_out_ports() {
    let dispatch = DirtyHelperDispatch::new();

    // IN should return 0xFF
    let in_result = dispatch.try_call("amd64g_dirtyhelper_IN", &[0x80]).unwrap();
    assert_eq!(in_result.return_value, Some(0xFF));

    // OUT should have no return value
    let out_result = dispatch
        .try_call("amd64g_dirtyhelper_OUT", &[0x80, 0x00])
        .unwrap();
    assert!(out_result.return_value.is_none());
}
