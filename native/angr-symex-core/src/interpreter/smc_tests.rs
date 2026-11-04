use super::*;

fn make_irsb(addr: u64, len_bytes: u32) -> IRSB {
    let mut irsb = IRSB::new(addr, VexArch::AMD64);
    irsb.statements.push(IRStmt::IMark {
        addr,
        len: len_bytes,
        delta: 0,
    });
    irsb
}

#[test]
fn invalidate_removes_overlapping_block_and_marks_page_dirty() {
    let ctx = SymContext::new_mock();
    let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);
    interp.add_concrete_memory(0x1000, vec![0u8; 0x1000]);
    // Block at 0x1010 covering 8 bytes -> [0x1010, 0x1018).
    interp.cache_block(0x1010, make_irsb(0x1010, 8));
    assert!(interp.has_cached_block(0x1010));
    // Write a single byte at 0x1014 (inside the block range).
    interp.invalidate_code_at(0x1014, 1);
    assert!(!interp.has_cached_block(0x1010));
    assert!(interp.is_code_page_dirtied(0x1014));
    assert!(interp.is_code_page_dirtied(0x1010));
}

#[test]
fn invalidate_skips_non_overlapping_blocks() {
    let ctx = SymContext::new_mock();
    let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);
    interp.add_concrete_memory(0x1000, vec![0u8; 0x2000]);
    interp.cache_block(0x1010, make_irsb(0x1010, 8));
    // Write a byte at 0x1100 — different bytes, but same page (0x1).
    interp.invalidate_code_at(0x1100, 1);
    // The block at 0x1010 doesn't overlap the write range, so it stays.
    assert!(interp.has_cached_block(0x1010));
    // But the page is now marked dirty (0x1100 >> 12 == 0x1).
    assert!(interp.is_code_page_dirtied(0x1100));
    assert!(interp.is_code_page_dirtied(0x1010));
}

#[test]
fn invalidate_handles_multi_page_writes() {
    let ctx = SymContext::new_mock();
    let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);
    interp.add_concrete_memory(0x1000, vec![0u8; 0x4000]);
    // 16-byte write straddles 0x1ff8..0x2008 — pages 0x1 and 0x2.
    interp.invalidate_code_at(0x1ff8, 16);
    assert!(interp.is_code_page_dirtied(0x1ff8));
    assert!(interp.is_code_page_dirtied(0x2000));
}

#[test]
fn is_code_range_dirtied_spans_pages() {
    let ctx = SymContext::new_mock();
    let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);
    interp.add_concrete_memory(0x1000, vec![0u8; 0x4000]);
    // Dirty just page 0x2.
    interp.invalidate_code_at(0x2000, 1);
    // A lift window at 0x1ff0 size 32 crosses pages 0x1 and 0x2.
    assert!(interp.is_code_range_dirtied(0x1ff0, 32));
    // A lift window at 0x1000 size 16 stays within page 0x1.
    assert!(!interp.is_code_range_dirtied(0x1000, 16));
}

#[test]
fn fork_inherits_dirtied_pages() {
    let ctx = SymContext::new_mock();
    let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);
    interp.add_concrete_memory(0x1000, vec![0u8; 0x1000]);
    interp.invalidate_code_at(0x1500, 1);
    let child = interp.fork();
    assert!(child.is_code_page_dirtied(0x1500));
}
