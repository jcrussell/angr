//! Behavioural coverage for the `AVOID_MULTIVALUED_READS` /
//! `AVOID_MULTIVALUED_WRITES` short-circuits (angr-6cp06.67).
//!
//! Both options are user-reachable SimOptions that silently change results:
//! a symbolic-address load stops reading memory and mints an unconstrained
//! value, and a symbolic-address store is dropped on the floor. Before this
//! module the five memory-level gates —
//! [`SymbolicMemory::load_symbolic`], [`SymbolicMemory::load_symbolic_unified`],
//! [`SymbolicMemory::store_symbolic`], [`SymbolicMemory::store_symbolic_unified`]
//! and [`SymbolicMemory::store_symbolic_unified_multi`] — had no Rust-side test
//! at all, so deleting any one branch stayed green.
//!
//! Every test pins the symbolic address to a *single* concrete solution, so the
//! ON/OFF pair differs only by the short-circuit: with the option off the
//! address concretizes to exactly that location and the access goes through.

use super::super::*;

/// The one location every test in this module reads and writes.
const ADDR: u64 = 0x1000;
/// Backer value planted at [`ADDR`], distinguishable from 0 and from
/// [`STORED`].
const BACKER: u128 = 0xDEAD_BEEF_CAFE_BABE;
/// Value the store tests try to write over [`BACKER`].
const STORED: u128 = 0x0102_0304_0506_0708;

/// Mapped memory with [`BACKER`] planted at [`ADDR`].
fn backed_memory() -> SymbolicMemory {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(ADDR, 0x1000, Permission::RWX);
    mem.store_concrete(ADDR, RustBV::concrete(BACKER, 64))
        .expect("planting the backer must succeed");
    mem
}

/// A 64-bit BVS pinned by constraint to [`ADDR`]. `as_u64()` stays `None`, so
/// `should_avoid_multivalued_{read,write}` fires, but the concretizer has
/// exactly one solution when it does not.
#[cfg(feature = "vex-engine-z3")]
fn pinned_symbolic_addr(ctx: &SymContext, name: &str) -> RustBV {
    let addr = RustBV::symbolic(ctx, name, 64);
    ctx.assume_true(&addr.eq(&RustBV::concrete(ADDR as u128, 64), ctx));
    assert!(ctx.is_sat(), "pinning constraint must be SAT");
    assert!(
        addr.as_u64().is_none(),
        "test fixture: the address must stay structurally symbolic"
    );
    addr
}

/// Concretizer with only the avoid-multivalued flags flipped.
fn concretizer_with(reads: bool, writes: bool) -> AddressConcretizer {
    let mut c = AddressConcretizer::new();
    c.configure_strategies(false, None, None, false, reads, writes);
    c
}

/// `(min, max)` solver bounds of `bv`.
#[cfg(feature = "vex-engine-z3")]
fn bounds(ctx: &SymContext, bv: &RustBV) -> (u128, u128) {
    (
        ctx.min(bv, false).expect("min must be SAT"),
        ctx.max(bv, false).expect("max must be SAT"),
    )
}

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------

/// `load_symbolic` with AVOID_MULTIVALUED_READS off reads the backer; with it
/// on it returns a fully unconstrained value that never touched the page.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn load_symbolic_short_circuits_to_unconstrained_under_avoid_multivalued_reads() {
    let ctx = SymContext::new_mock();
    let mem = backed_memory();

    let off = mem
        .load_symbolic(
            pinned_symbolic_addr(&ctx, "mv_off_addr"),
            8,
            &ctx,
            &concretizer_with(false, false),
        )
        .expect("load with the option off must succeed");
    assert_eq!(
        bounds(&ctx, &off),
        (BACKER, BACKER),
        "option off: the pinned address must concretize and read the backer"
    );

    let on = mem
        .load_symbolic(
            pinned_symbolic_addr(&ctx, "mv_on_addr"),
            8,
            &ctx,
            &concretizer_with(true, false),
        )
        .expect("load with the option on must succeed");
    assert_eq!(
        bounds(&ctx, &on),
        (0, u64::MAX as u128),
        "option on: the load must be fully unconstrained, not the backer"
    );
}

/// Same contract for the unified entry point, which has its own copy of the
/// short-circuit.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn load_symbolic_unified_short_circuits_to_unconstrained_under_avoid_multivalued_reads() {
    let ctx = SymContext::new_mock();
    let mut mem = backed_memory();

    let off = mem
        .load_symbolic_unified(
            pinned_symbolic_addr(&ctx, "mvu_off_addr"),
            8,
            &ctx,
            &concretizer_with(false, false),
        )
        .expect("unified load with the option off must succeed");
    assert_eq!(
        bounds(&ctx, &off),
        (BACKER, BACKER),
        "option off: unified load must read the backer"
    );

    let on = mem
        .load_symbolic_unified(
            pinned_symbolic_addr(&ctx, "mvu_on_addr"),
            8,
            &ctx,
            &concretizer_with(true, false),
        )
        .expect("unified load with the option on must succeed");
    assert_eq!(
        bounds(&ctx, &on),
        (0, u64::MAX as u128),
        "option on: unified load must be fully unconstrained"
    );
}

/// The short-circuit routes through `unconstrained_read_value`, so
/// `zero_fill_unconstrained` still decides the shape of the fabricated value.
/// A test that only asserted "not the backer" would pass on either.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn avoid_multivalued_read_short_circuit_honors_zero_fill_unconstrained() {
    let ctx = SymContext::new_mock();
    let mut mem = backed_memory();
    mem.set_zero_fill_unconstrained(true);

    let concretizer = concretizer_with(true, false);
    for (name, loaded) in [
        (
            "load_symbolic",
            mem.load_symbolic(pinned_symbolic_addr(&ctx, "zf_a"), 8, &ctx, &concretizer),
        ),
        (
            "load_symbolic_unified",
            mem.load_symbolic_unified(pinned_symbolic_addr(&ctx, "zf_b"), 8, &ctx, &concretizer),
        ),
    ] {
        let value = loaded.unwrap_or_else(|e| panic!("{name} must succeed: {e:?}"));
        assert_eq!(
            value.as_u64(),
            Some(0),
            "{name}: zero_fill_unconstrained must make the short-circuit yield concrete 0"
        );
    }
}

/// The gate is `option && addr.as_u64().is_none()` — a concrete address must
/// read through even with the option on, matching Python (the short-circuit
/// lives past the concrete fast path).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn avoid_multivalued_reads_does_not_fire_for_concrete_addresses() {
    let ctx = SymContext::new_mock();
    let mut mem = backed_memory();
    let concretizer = concretizer_with(true, false);

    let plain = mem
        .load_symbolic(RustBV::concrete(ADDR as u128, 64), 8, &ctx, &concretizer)
        .expect("concrete-address load must succeed");
    assert_eq!(plain.as_u64(), Some(BACKER as u64));

    let unified = mem
        .load_symbolic_unified(RustBV::concrete(ADDR as u128, 64), 8, &ctx, &concretizer)
        .expect("concrete-address unified load must succeed");
    assert_eq!(unified.as_u64(), Some(BACKER as u64));
}

// ---------------------------------------------------------------------------
// Writes
// ---------------------------------------------------------------------------

/// Reads [`ADDR`] back concretely — the store paths are what is under test, so
/// the verification read deliberately avoids them.
#[cfg(feature = "vex-engine-z3")]
fn read_back(mem: &SymbolicMemory, ctx: &SymContext) -> Option<u64> {
    mem.load_concrete(ADDR, 8, ctx)
        .expect("concrete read-back must succeed")
        .as_u64()
}

/// `store_symbolic` with AVOID_MULTIVALUED_WRITES off writes through the
/// pinned address; with it on the write is silently dropped.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn store_symbolic_drops_the_write_under_avoid_multivalued_writes() {
    let ctx = SymContext::new_mock();

    let mut off = backed_memory();
    off.store_symbolic(
        pinned_symbolic_addr(&ctx, "wr_off_addr"),
        RustBV::concrete(STORED, 64),
        &ctx,
        &concretizer_with(false, false),
    )
    .expect("store with the option off must succeed");
    assert_eq!(
        read_back(&off, &ctx),
        Some(STORED as u64),
        "option off: the pinned address must concretize and take the write"
    );

    let mut on = backed_memory();
    on.store_symbolic(
        pinned_symbolic_addr(&ctx, "wr_on_addr"),
        RustBV::concrete(STORED, 64),
        &ctx,
        &concretizer_with(false, true),
    )
    .expect("store with the option on must report success, not an error");
    assert_eq!(
        read_back(&on, &ctx),
        Some(BACKER as u64),
        "option on: the write must be dropped, leaving the backer intact"
    );
}

/// The unified store signals the drop through its return value (`Ok(None)`)
/// as well as by leaving memory untouched.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn store_symbolic_unified_reports_none_and_drops_the_write_under_avoid_multivalued_writes() {
    let ctx = SymContext::new_mock();

    let mut off = backed_memory();
    let off_result = off
        .store_symbolic_unified(
            pinned_symbolic_addr(&ctx, "wru_off_addr"),
            RustBV::concrete(STORED, 64),
            &ctx,
            &concretizer_with(false, false),
        )
        .expect("unified store with the option off must succeed");
    assert!(
        matches!(off_result, Some(ConcretizationResult::Single(a)) if a == ADDR),
        "option off: expected Single({ADDR:#x}), got {off_result:?}"
    );
    assert_eq!(read_back(&off, &ctx), Some(STORED as u64));

    let mut on = backed_memory();
    let on_result = on
        .store_symbolic_unified(
            pinned_symbolic_addr(&ctx, "wru_on_addr"),
            RustBV::concrete(STORED, 64),
            &ctx,
            &concretizer_with(false, true),
        )
        .expect("unified store with the option on must report success");
    assert!(
        on_result.is_none(),
        "option on: the dropped write must report no concretization, got {on_result:?}"
    );
    assert_eq!(
        read_back(&on, &ctx),
        Some(BACKER as u64),
        "option on: the write must be dropped, leaving the backer intact"
    );
}

/// The multiwrite entry point carries its own copy of the short-circuit ahead
/// of `concretize_write_multiwrite`, so it needs its own coverage.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn store_symbolic_unified_multi_reports_none_and_drops_the_write_under_avoid_multivalued_writes() {
    let ctx = SymContext::new_mock();

    let mut off = backed_memory();
    let off_result = off
        .store_symbolic_unified_multi(
            pinned_symbolic_addr(&ctx, "wrm_off_addr"),
            RustBV::concrete(STORED, 64),
            &ctx,
            &concretizer_with(false, false),
        )
        .expect("multiwrite store with the option off must succeed");
    assert!(
        matches!(off_result, Some(ConcretizationResult::Single(a)) if a == ADDR),
        "option off: expected Single({ADDR:#x}), got {off_result:?}"
    );
    assert_eq!(read_back(&off, &ctx), Some(STORED as u64));

    let mut on = backed_memory();
    let on_result = on
        .store_symbolic_unified_multi(
            pinned_symbolic_addr(&ctx, "wrm_on_addr"),
            RustBV::concrete(STORED, 64),
            &ctx,
            &concretizer_with(false, true),
        )
        .expect("multiwrite store with the option on must report success");
    assert!(
        on_result.is_none(),
        "option on: the dropped write must report no concretization, got {on_result:?}"
    );
    assert_eq!(
        read_back(&on, &ctx),
        Some(BACKER as u64),
        "option on: the write must be dropped, leaving the backer intact"
    );
}

/// Concrete-address stores are past the fast path before the gate, so all
/// three write entry points must still land with the option on.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn avoid_multivalued_writes_does_not_fire_for_concrete_addresses() {
    let ctx = SymContext::new_mock();
    let concretizer = concretizer_with(false, true);
    let addr = || RustBV::concrete(ADDR as u128, 64);
    let value = || RustBV::concrete(STORED, 64);

    let mut plain = backed_memory();
    plain
        .store_symbolic(addr(), value(), &ctx, &concretizer)
        .expect("concrete-address store must succeed");
    assert_eq!(read_back(&plain, &ctx), Some(STORED as u64));

    let mut unified = backed_memory();
    unified
        .store_symbolic_unified(addr(), value(), &ctx, &concretizer)
        .expect("concrete-address unified store must succeed");
    assert_eq!(read_back(&unified, &ctx), Some(STORED as u64));

    let mut multi = backed_memory();
    multi
        .store_symbolic_unified_multi(addr(), value(), &ctx, &concretizer)
        .expect("concrete-address multiwrite store must succeed");
    assert_eq!(read_back(&multi, &ctx), Some(STORED as u64));
}
