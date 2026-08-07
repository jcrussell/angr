//! Tests for [`super::super::simprocedures`] — SimProcedure return-address
//! extraction out of the in-flight store buffers.

use super::*;

fn new_interp(arch: VexArch, ctx: &SymContext) -> VEXInterpreter<'_> {
    VEXInterpreter::new(arch, ctx)
}

/// Point `sp` at `addr` without going through a VEX block.
fn set_sp(interp: &mut VEXInterpreter<'_>, addr: u64) {
    let ptr_bits = interp.registers.arch().bytes() * 8;
    interp
        .registers
        .set_sp(RustBV::concrete(addr as u128, ptr_bits));
}

#[test]
fn get_return_addr_reads_pending_concrete_store() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(VexArch::AMD64, &ctx);
    set_sp(&mut interp, 0x7fff_0000);
    interp
        .pending_stores
        .push(0x7fff_0000, 0x0040_1122_u64.to_le_bytes().to_vec());
    assert_eq!(interp.get_return_addr(), Some(0x0040_1122));
}

#[test]
fn get_return_addr_reads_flushed_concrete_store() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(VexArch::AMD64, &ctx);
    set_sp(&mut interp, 0x7fff_0000);
    interp
        .all_flushed_stores
        .insert(0x7fff_0000, 0x0040_1122_u64.to_le_bytes().to_vec());
    assert_eq!(interp.get_return_addr(), Some(0x0040_1122));
}

// angr-9ke6b.87: a symbolic store at `[sp]` pushes no placeholder bytes into
// the concrete buffers (angr-ofyh), so before the fix `get_return_addr` walked
// straight past it and answered with whatever *stale* concrete bytes were still
// buffered at the same slot from before the call. Now the symbolic maps are
// consulted in `load_concrete_addr`'s precedence order, and a genuinely
// symbolic return address declines with `None` rather than lying.
#[test]
fn get_return_addr_pending_symbolic_store_shadows_stale_concrete_bytes() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(VexArch::AMD64, &ctx);
    set_sp(&mut interp, 0x7fff_0000);
    // Stale pre-call bytes still sitting in both concrete buffers.
    interp
        .pending_stores
        .push(0x7fff_0000, 0xdead_beefu64.to_le_bytes().to_vec());
    interp
        .all_flushed_stores
        .insert(0x7fff_0000, 0xdead_beefu64.to_le_bytes().to_vec());
    // The write that actually landed last is symbolic.
    interp
        .pending_symbolic_stores
        .insert(0x7fff_0000, RustBV::symbolic(&ctx, "ret", 64));
    assert_eq!(
        interp.get_return_addr(),
        None,
        "symbolic [sp] must decline, not report the stale concrete address"
    );
}

#[test]
fn get_return_addr_flushed_symbolic_store_shadows_stale_concrete_bytes() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(VexArch::AMD64, &ctx);
    set_sp(&mut interp, 0x7fff_0000);
    interp
        .all_flushed_stores
        .insert(0x7fff_0000, 0xdead_beefu64.to_le_bytes().to_vec());
    interp
        .all_flushed_symbolic_stores
        .insert(0x7fff_0000, RustBV::symbolic(&ctx, "ret", 64));
    assert_eq!(interp.get_return_addr(), None);
}

// angr-sqfj8.62: every call site of `get_return_addr()` must route through
// `get_return_addr_or_log` instead of a bare `.unwrap_or(0)` — this pins the
// helper's own None-handling contract (substitute 0 without panicking, the
// same fallback value the old unwrap_or(0) sites used) so a symbolic/
// unavailable return address doesn't crash the caller. The "every call site
// actually uses the helper" half of the invariant is enforced by code review
// / grep (`grep -rn 'get_return_addr().unwrap_or' native/angr/src/interpreter/`
// should return nothing), not by this test — a compiled call site can't
// observe whether a *different* call site regressed.
#[test]
fn get_return_addr_or_log_substitutes_zero_for_symbolic() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(VexArch::AMD64, &ctx);
    set_sp(&mut interp, 0x7fff_0000);
    interp
        .all_flushed_symbolic_stores
        .insert(0x7fff_0000, RustBV::symbolic(&ctx, "ret", 64));
    assert_eq!(interp.get_return_addr_or_log("test"), 0);
}

// A pending *concrete* store is the post-`call` write and still wins over an
// older flushed symbolic shadow — precedence must match `load_concrete_addr`.
#[test]
fn get_return_addr_pending_concrete_beats_flushed_symbolic() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(VexArch::AMD64, &ctx);
    set_sp(&mut interp, 0x7fff_0000);
    interp
        .all_flushed_symbolic_stores
        .insert(0x7fff_0000, RustBV::symbolic(&ctx, "old", 64));
    interp
        .pending_stores
        .push(0x7fff_0000, 0x0040_1122_u64.to_le_bytes().to_vec());
    assert_eq!(interp.get_return_addr(), Some(0x0040_1122));
}

// On a link-register ABI nothing is pushed at the call, so `[sp]` holds an
// unrelated caller local. Peeking at the store buffers there would return that
// local as the return address; the convention's LR read is the only answer.
#[test]
fn get_return_addr_ignores_stack_buffers_on_link_register_abi() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(VexArch::ARM, &ctx);
    assert!(
        !interp.calling_convention.pops_return_addr(),
        "ARM must be a link-register ABI for this test to mean anything"
    );
    let lr = interp
        .calling_convention
        .link_register()
        .expect("ARM names its link register");
    set_sp(&mut interp, 0x7fff_0000);
    interp.registers.put(lr, RustBV::concrete(0x8004, 32));
    // A caller local that happens to sit at [sp].
    interp
        .pending_stores
        .push(0x7fff_0000, 0xdead_beefu32.to_le_bytes().to_vec());
    assert_eq!(interp.get_return_addr(), Some(0x8004));
}
