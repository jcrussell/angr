//! Auxiliary value/metadata types (call stack, heap, history, changes).

use super::*;

/// Entry in the function call stack.
///
/// Tracks call/return pairs during symbolic execution. Pushed on `Ijk_Call`,
/// popped on `Ijk_Ret`. Cloned on state fork.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CallStackEntry {
    /// Address of the call instruction (caller site).
    pub call_site_addr: u64,
    /// Address of the callee function entry.
    pub callee_addr: u64,
    /// Expected return address (instruction after the call).
    pub return_addr: u64,
    /// Stack pointer value at call time.
    pub stack_ptr: u64,
}

/// Heap metadata tracking for malloc/free/calloc/realloc.
///
/// Tracks allocated regions and freed addresses for heap exploitation
/// analysis. Cloned on fork so each exploration path has its own heap state.
///
/// ## Merge coverage (angr-91vj9.13)
///
/// `RustSimState` labels `heap_metadata` `#[merge_policy = "delegate"]`, which
/// ends the top-level derive's guarantee here; deriving
/// [`angr_macros::MergePolicy`] extends it, so a new field does not compile
/// until it declares how [`Self::union_from`] treats it. `union_from` merges in
/// place and both fields are combined by hand (the `allocated` union reads its
/// `freed` sibling), so both are `joint` rather than the mechanical `union` —
/// that policy generates a `bool`-only `||` body.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, angr_macros::MergePolicy)]
pub struct HeapMetadata {
    /// Currently allocated regions: address -> size in bytes.
    ///
    /// Merged as a `self`-preferring union gated on `freed` — see
    /// [`Self::union_from`].
    #[merge_policy = "joint"]
    pub allocated: FxHashMap<u64, u64>,
    /// Freed addresses, as a **set** in first-free order.
    ///
    /// Membership, not multiplicity: freeing the same pointer twice on one
    /// path records one entry, matching what `union_from` does when both
    /// merged branches inherited the same free. A `Vec` (not a hash set)
    /// because export order must stay deterministic; see `record_free` for
    /// why the linear membership scan is acceptable.
    ///
    /// Merged as a deduplicated set union — see [`Self::union_from`].
    #[merge_policy = "joint"]
    pub freed: Vec<u64>,
}

impl HeapMetadata {
    /// Record a new allocation.
    pub fn record_alloc(&mut self, addr: u64, size: u64) {
        self.allocated.insert(addr, size);
    }

    /// Record a free. Returns the size of the freed region, or None if not
    /// tracked (never allocated here, or already freed).
    ///
    /// `freed` is a set: a double free of the same pointer on one path records
    /// one entry, so `free_count` means the same thing whether the state was
    /// merged or not. Nothing derives double-free detection from a duplicate
    /// entry — the `None` return already distinguishes the second free, and
    /// under the bump allocator (`RustSimState::heap_alloc`) an address is
    /// never handed out twice, so a repeat free is always the same allocation.
    ///
    /// The membership check is a linear scan, matching `union_from`; `freed`
    /// holds one entry per distinct freed pointer on a path, which stays small
    /// enough that a hash set would cost more (allocation, nondeterministic
    /// export order) than it saves.
    pub fn record_free(&mut self, addr: u64) -> Option<u64> {
        let size = self.allocated.remove(&addr);
        if addr != 0 && !self.freed.contains(&addr) {
            self.freed.push(addr);
        }
        size
    }

    /// Check if an address was allocated.
    pub fn is_allocated(&self, addr: u64) -> bool {
        self.allocated.contains_key(&addr)
    }

    /// Get the size of an allocation.
    pub fn alloc_size(&self, addr: u64) -> Option<u64> {
        self.allocated.get(&addr).copied()
    }

    /// Get the number of active allocations.
    pub fn alloc_count(&self) -> usize {
        self.allocated.len()
    }

    /// Get the number of distinct freed addresses (see `freed`: a set, so a
    /// double free of one pointer counts once).
    pub fn free_count(&self) -> usize {
        self.freed.len()
    }

    /// Union another branch's heap bookkeeping into `self` during a state merge.
    ///
    /// A merged state's `heap_brk` is the max across branches (fork.rs), so an
    /// allocation made only on a dropped branch stays reachable in the merged
    /// memory — but if its `allocated` entry lived only on that branch,
    /// `alloc_size` would return `None` and `NativeRealloc` would default to
    /// copying the *full* new size, over-reading past the true old allocation
    /// (angr-n0irt.3). Unioning every branch's allocations closes that gap.
    ///
    /// On address collision the existing (earlier-state) size is kept: a bump
    /// allocator makes same-address collisions across branches vanishingly
    /// rare, and preferring `self` keeps the merge deterministic. `freed`
    /// addresses are unioned as a set so a pointer freed on any branch stays
    /// recorded without double-counting a free both branches inherited.
    /// angr-sqfj8.85: an address `self` already freed must not be resurrected
    /// into `self.allocated` just because `other` never freed it on its
    /// branch — a freed-then-reallocated address on `self`'s path can also
    /// legitimately reappear in `other.allocated` at a different size, so
    /// the freed check must run before the `or_insert`, not after.
    pub fn union_from(&mut self, other: &HeapMetadata) {
        for (&addr, &size) in &other.allocated {
            if !self.freed.contains(&addr) {
                self.allocated.entry(addr).or_insert(size);
            }
        }
        for &addr in &other.freed {
            if !self.freed.contains(&addr) {
                self.freed.push(addr);
            }
        }
    }
}
/// Entry in the execution history trace.
///
/// Records block-level execution events with jumpkind and jump target.
/// Appended at each block execution, cloned on state fork.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HistoryEntry {
    /// Block address that was executed.
    pub addr: u64,
    /// Jump kind at block exit (0=Boring, 1=Call, 2=Ret, 3=Syscall, 4=Other).
    pub jumpkind: u8,
    /// Target address of the jump (where execution went after this block).
    pub jump_target: u64,
}

impl HistoryEntry {
    /// Jumpkind constants matching VEX conventions.
    pub const JK_BORING: u8 = 0;
    pub const JK_CALL: u8 = 1;
    pub const JK_RET: u8 = 2;
    pub const JK_SYSCALL: u8 = 3;
    pub const JK_OTHER: u8 = 4;

    /// Create from JumpKind enum.
    pub fn jumpkind_from_vex(jk: &crate::vex::JumpKind) -> u8 {
        if jk.is_call() {
            Self::JK_CALL
        } else if jk.is_ret() {
            Self::JK_RET
        } else if jk.is_syscall() {
            Self::JK_SYSCALL
        } else {
            Self::JK_BORING
        }
    }

    /// Convert jumpkind byte to string (for Python API).
    ///
    /// Delegates to `JumpKind::ijk_name` so the `Ijk_*` spellings live in one
    /// place (angr-sqfj8.60). `JK_OTHER` is the lossy catch-all this byte
    /// encoding adds on top of the enum — no `JumpKind` variant maps to it, so
    /// its string stays local.
    pub fn jumpkind_str(jk: u8) -> &'static str {
        use crate::vex::JumpKind;
        match jk {
            Self::JK_BORING => JumpKind::Boring.ijk_name(),
            Self::JK_CALL => JumpKind::Call.ijk_name(),
            Self::JK_RET => JumpKind::Ret.ijk_name(),
            Self::JK_SYSCALL => JumpKind::Sys_syscall.ijk_name(),
            _ => "Ijk_Other",
        }
    }
}
/// Incremental state changes for efficient sync.
///
/// Instead of syncing entire state, we track only what changed.
#[derive(Debug, Clone, Default)]
pub struct StateChanges {
    /// Register changes: (offset, size, value_bytes).
    pub register_writes: Vec<(u32, u32, Vec<u8>)>,
    /// Memory writes: (addr, value_bytes).
    pub memory_writes: Vec<(u64, Vec<u8>)>,
    /// Updated PC.
    pub new_pc: Option<u64>,
}

impl StateChanges {
    pub fn new() -> Self {
        Self::default()
    }
}
