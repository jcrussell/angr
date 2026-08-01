use super::*;

#[test]
fn test_cpuid_not_handled_natively() {
    // CPUID/RDTSCP receive a GSPTR arg and are deliberately routed to the
    // Python engine; they must NOT be registered for native dispatch
    // (angr-t1ok: dead-code removal).
    let dispatch = DirtyHelperDispatch::new();
    let mut hs = DirtyHelperState::default();
    assert!(
        dispatch
            .try_call(&mut hs, "amd64g_dirtyhelper_CPUID_baseline", &[])
            .is_none()
    );
    assert!(
        dispatch
            .try_call(&mut hs, "amd64g_dirtyhelper_CPUID_avx2", &[])
            .is_none()
    );
    assert!(
        dispatch
            .try_call(&mut hs, "x86g_dirtyhelper_CPUID_sse3", &[])
            .is_none()
    );
    assert!(
        dispatch
            .try_call(&mut hs, "amd64g_dirtyhelper_RDTSCP", &[])
            .is_none()
    );
}

#[test]
fn test_rdtsc() {
    let dispatch = DirtyHelperDispatch::new();
    let mut hs = DirtyHelperState::default();

    let result1 = dispatch
        .try_call(&mut hs, "amd64g_dirtyhelper_RDTSC", &[])
        .unwrap();
    let result2 = dispatch
        .try_call(&mut hs, "amd64g_dirtyhelper_RDTSC", &[])
        .unwrap();

    // TSC should increment
    assert!(result2.return_value.unwrap() > result1.return_value.unwrap());
}

/// angr-9ke6b.173: the TSC used to be a single process-wide `AtomicU64`, so
/// the value a state observed depended on how many RDTSCs *unrelated* states
/// had already executed. With the counter scoped to `DirtyHelperState`, two
/// independent states see byte-identical sequences, and interleaving their
/// calls does not perturb either one.
#[test]
fn test_rdtsc_is_per_state_not_process_wide() {
    let dispatch = DirtyHelperDispatch::new();

    let rdtsc = |hs: &mut DirtyHelperState| {
        dispatch
            .try_call(hs, "amd64g_dirtyhelper_RDTSC", &[])
            .unwrap()
            .return_value
            .unwrap()
    };

    // Two states, run one after the other.
    let mut a = DirtyHelperState::default();
    let mut b = DirtyHelperState::default();
    let seq_a: Vec<u64> = (0..4).map(|_| rdtsc(&mut a)).collect();
    let seq_b: Vec<u64> = (0..4).map(|_| rdtsc(&mut b)).collect();
    assert_eq!(
        seq_a, seq_b,
        "independent states must observe identical RDTSC sequences"
    );

    // Same two states, but interleaved (what parallel exploration does).
    let mut c = DirtyHelperState::default();
    let mut d = DirtyHelperState::default();
    let mut seq_c = Vec::new();
    let mut seq_d = Vec::new();
    for _ in 0..4 {
        seq_c.push(rdtsc(&mut c));
        seq_d.push(rdtsc(&mut d));
    }
    assert_eq!(seq_c, seq_a, "interleaving must not perturb a state's TSC");
    assert_eq!(seq_d, seq_a, "interleaving must not perturb a state's TSC");

    // And the absolute values are fixed, not just mutually consistent — the
    // same binary replayed in a fresh process gets the same numbers.
    assert_eq!(
        seq_a,
        vec![
            TSC_INITIAL,
            TSC_INITIAL + TSC_STEP,
            TSC_INITIAL + 2 * TSC_STEP,
            TSC_INITIAL + 3 * TSC_STEP,
        ]
    );
}

/// A forked path continues the parent's timeline (it does not restart at
/// `TSC_INITIAL`) and then advances independently of its sibling.
#[test]
fn test_rdtsc_fork_continues_parent_timeline() {
    let dispatch = DirtyHelperDispatch::new();
    let mut parent = DirtyHelperState::default();
    dispatch
        .try_call(&mut parent, "amd64g_dirtyhelper_RDTSC", &[])
        .unwrap();

    let mut child = parent.clone();
    let parent_next = dispatch
        .try_call(&mut parent, "amd64g_dirtyhelper_RDTSC", &[])
        .unwrap()
        .return_value
        .unwrap();
    let child_next = dispatch
        .try_call(&mut child, "amd64g_dirtyhelper_RDTSC", &[])
        .unwrap()
        .return_value
        .unwrap();

    assert_eq!(parent_next, TSC_INITIAL + TSC_STEP);
    assert_eq!(child_next, parent_next, "fork inherits the parent's clock");
}

/// The counter saturates instead of wrapping: a pathological RDTSC loop must
/// never make simulated time appear to run backwards.
#[test]
fn test_rdtsc_saturates_at_u64_max() {
    let dispatch = DirtyHelperDispatch::new();
    let mut hs = DirtyHelperState { tsc: u64::MAX - 1 };
    let first = dispatch
        .try_call(&mut hs, "amd64g_dirtyhelper_RDTSC", &[])
        .unwrap()
        .return_value
        .unwrap();
    let second = dispatch
        .try_call(&mut hs, "amd64g_dirtyhelper_RDTSC", &[])
        .unwrap()
        .return_value
        .unwrap();
    assert_eq!(first, u64::MAX - 1);
    assert_eq!(second, u64::MAX);
    assert!(second >= first);
}

#[test]
fn test_unknown_helper() {
    let dispatch = DirtyHelperDispatch::new();
    let mut hs = DirtyHelperState::default();

    assert!(dispatch.try_call(&mut hs, "unknown_helper", &[]).is_none());
}

#[test]
fn test_in_out_ports() {
    let dispatch = DirtyHelperDispatch::new();
    let mut hs = DirtyHelperState::default();

    // IN should return 0xFF
    let in_result = dispatch
        .try_call(&mut hs, "amd64g_dirtyhelper_IN", &[0x80])
        .unwrap();
    assert_eq!(in_result.return_value, Some(0xFF));

    // OUT should have no return value
    let out_result = dispatch
        .try_call(&mut hs, "amd64g_dirtyhelper_OUT", &[0x80, 0x00])
        .unwrap();
    assert!(out_result.return_value.is_none());

    // Stateless helpers must not touch the TSC.
    assert_eq!(hs.tsc, TSC_INITIAL);
}
