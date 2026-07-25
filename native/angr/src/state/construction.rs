//! Constructor family for `RustSimState`.
//!
//! Every entry point that builds a fresh `RustSimState` lives here: the
//! arch-name constructors, the shared-solver variants used when
//! forking, and the private `new_state_memory` helper that registers the heap
//! as a lazy region so the native SimProcedure heap fast path is consistent
//! regardless of which constructor created the state. Split out of `mod.rs`
//! per the god-object decomposition (angr-0mqkc.5); mirrors the
//! `fork.rs` / `snapshot.rs` / `migration.rs` extension-impl pattern.

use super::*;

impl RustSimState {
    /// Create a new state for the given architecture.
    ///
    /// # Arguments
    /// * `arch_name` - Architecture name (e.g., "amd64", "x86", "arm")
    /// * `little_endian` - Override endianness (None = use arch default)
    ///
    /// # Returns
    /// New state with default initialization.
    pub fn new(arch_name: &str) -> Result<Self, String> {
        Self::new_with_endian(arch_name, None)
    }

    /// Heap region `[heap_base, mmap_base)` — the bump-allocator range. Registered
    /// as a lazy region in every constructor so native SimProcedures that allocate
    /// a struct on the heap and immediately write to it (e.g. `fopen` writing the
    /// fd into a fresh `_IO_FILE`) auto-map the backing page through
    /// `memory_store` instead of erroring `Unmapped` and falling back to Python.
    /// Unwritten heap pages stay unmapped, so reads of uninitialized heap still
    /// surface `Unmapped` and fall back to Python — preserving angr's
    /// symbolic-fill semantics rather than reading a zero page.
    const HEAP_REGION_START: u64 = 0xC000_0000;
    const HEAP_REGION_SIZE: u64 = 0x0100_0000; // [0xC0000000, 0xC1000000) == [heap_base, mmap_base)

    /// Build a fresh `SymbolicMemory` with the heap registered as a lazy region.
    /// Shared by every `RustSimState` constructor so the heap fast path is
    /// consistent regardless of which entry point created the state.
    fn new_state_memory(endness: Endness) -> SymbolicMemory {
        let mut memory = SymbolicMemory::new(endness);
        memory.add_lazy_region(Self::HEAP_REGION_START, Self::HEAP_REGION_SIZE);
        memory
    }

    /// Create a new state with explicit endianness override.
    pub fn new_with_endian(arch_name: &str, little_endian: Option<bool>) -> Result<Self, String> {
        let arch = arch_from_name(arch_name)
            .ok_or_else(|| format!("unknown architecture: {arch_name}"))?;
        let vex_arch = arch.vex_arch();
        let is_le = little_endian.unwrap_or_else(|| arch.is_little_endian());
        let endness = if is_le { Endness::Little } else { Endness::Big };

        Ok(RustSimState {
            vex_arch,
            registers: RegisterFile::new(arch.clone()),
            memory: Self::new_state_memory(endness),
            solver: Rc::new(RefCell::new(SymContext::new())),
            pc: 0,
            state_id: next_state_id(),
            parent_id: None,
            history: VecDeque::new(),
            detailed_history: VecDeque::new(),
            max_history: 1000,
            hooks: Arc::new(HashSet::new()),
            concretizer: AddressConcretizer::default(),
            track_history: true,
            arch,
            fs: FileSystem::default(),
            heap_brk: 0xC000_0000,
            posix_brk: 0x1B0_0000,
            mmap_base: 0xC100_0000,
            getopt_optind: 1,
            getopt_optchar: 0,
            getopt_extern: GetoptExternAddrs::default(),
            native_resume_stack: Vec::new(),
            ctype_loc: CtypeLocPtrs::default(),
            stdin_symbols: Vec::new(),
            call_stack: Vec::new(),
            heap_metadata: HeapMetadata::default(),
            inspection: InspectionManager::default(),
            environment: Arc::new(HashMap::new()),
            symbolic_pages: HashMap::new(),
            hook_symbolic_memory: HashMap::new(),
            addr_to_ast: HashMap::new(),
            last_time: None,
            no_ip_concretization: false,
            no_symbolic_jump_resolution: false,
            keep_ip_symbolic: false,
            force_eager_forks: false,
            cgc_allocation_base: 0xB800_0000,
            cgc_sinkholes: Vec::new(),
            sim_options: Arc::new(HashSet::new()),
        })
    }

    /// Create a state with a shared solver context.
    ///
    /// This is used when forking to share constraints across states.
    pub fn with_solver(arch_name: &str, solver: Rc<RefCell<SymContext>>) -> Result<Self, String> {
        Self::with_solver_endian(arch_name, solver, None)
    }

    /// Create a state with a shared solver context and explicit endianness.
    pub fn with_solver_endian(
        arch_name: &str,
        solver: Rc<RefCell<SymContext>>,
        little_endian: Option<bool>,
    ) -> Result<Self, String> {
        let arch = arch_from_name(arch_name)
            .ok_or_else(|| format!("unknown architecture: {arch_name}"))?;
        let vex_arch = arch.vex_arch();
        let is_le = little_endian.unwrap_or_else(|| arch.is_little_endian());
        let endness = if is_le { Endness::Little } else { Endness::Big };

        Ok(RustSimState {
            vex_arch,
            registers: RegisterFile::new(arch.clone()),
            memory: Self::new_state_memory(endness),
            solver,
            pc: 0,
            state_id: next_state_id(),
            parent_id: None,
            history: VecDeque::new(),
            detailed_history: VecDeque::new(),
            max_history: 1000,
            hooks: Arc::new(HashSet::new()),
            concretizer: AddressConcretizer::default(),
            track_history: true,
            arch,
            fs: FileSystem::default(),
            heap_brk: 0xC000_0000,
            posix_brk: 0x1B0_0000,
            mmap_base: 0xC100_0000,
            getopt_optind: 1,
            getopt_optchar: 0,
            getopt_extern: GetoptExternAddrs::default(),
            native_resume_stack: Vec::new(),
            ctype_loc: CtypeLocPtrs::default(),
            stdin_symbols: Vec::new(),
            call_stack: Vec::new(),
            heap_metadata: HeapMetadata::default(),
            inspection: InspectionManager::default(),
            environment: Arc::new(HashMap::new()),
            symbolic_pages: HashMap::new(),
            hook_symbolic_memory: HashMap::new(),
            addr_to_ast: HashMap::new(),
            last_time: None,
            no_ip_concretization: false,
            no_symbolic_jump_resolution: false,
            keep_ip_symbolic: false,
            force_eager_forks: false,
            cgc_allocation_base: 0xB800_0000,
            cgc_sinkholes: Vec::new(),
            sim_options: Arc::new(HashSet::new()),
        })
    }
}
