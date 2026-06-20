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
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct HeapMetadata {
    /// Currently allocated regions: address -> size in bytes.
    pub allocated: FxHashMap<u64, u64>,
    /// Freed addresses (in order of free calls).
    pub freed: Vec<u64>,
}

impl HeapMetadata {
    /// Record a new allocation.
    pub fn record_alloc(&mut self, addr: u64, size: u64) {
        self.allocated.insert(addr, size);
    }

    /// Record a free. Returns the size of the freed region, or None if not tracked.
    pub fn record_free(&mut self, addr: u64) -> Option<u64> {
        let size = self.allocated.remove(&addr);
        if addr != 0 {
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

    /// Get the number of free calls.
    pub fn free_count(&self) -> usize {
        self.freed.len()
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
    pub fn jumpkind_str(jk: u8) -> &'static str {
        match jk {
            Self::JK_BORING => "Ijk_Boring",
            Self::JK_CALL => "Ijk_Call",
            Self::JK_RET => "Ijk_Ret",
            Self::JK_SYSCALL => "Ijk_Sys_syscall",
            _ => "Ijk_Other",
        }
    }
}
/// Execution event from stepping a state.
///
/// `#[non_exhaustive]` per angr-irwe: minor versions may add new
/// variants as new stepping outcomes emerge; match sites must include
/// a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ExecutionEvent {
    /// Reached end of a basic block, continuing to next address.
    BlockEnd { next_addr: u64 },
    /// Encountered a symbolic branch condition.
    SymbolicBranch {
        condition: RustBV,
        true_target: u64,
        false_target: u64,
    },
    /// Hit a hook address (SimProcedure).
    Hook { addr: u64 },
    /// Syscall instruction.
    Syscall { num: u64 },
    /// Unmapped memory access - need Python callback.
    UnmappedMemory { addr: u64, size: u64 },
    /// Error during execution.
    Error { message: String },
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
    /// New constraints added (as Z3 AST indices or serialized form).
    pub new_constraints: Vec<u64>,
    /// Updated PC.
    pub new_pc: Option<u64>,
}

impl StateChanges {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.register_writes.is_empty()
            && self.memory_writes.is_empty()
            && self.new_constraints.is_empty()
            && self.new_pc.is_none()
    }
}
