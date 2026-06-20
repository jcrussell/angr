//! Snapshot serialization (serde wire format) for `RustSimState`.

use super::*;

// =============================================================================
// Snapshot / Serialization (angr-x04s.1.3)
// =============================================================================

/// Format-version byte at the head of every [`RustSimState::to_serialized`]
/// envelope. Bump on any breaking shape change to [`RustSimStateSnapshot`]
/// so a stale snapshot fails fast with `SnapshotError::VersionMismatch`
/// instead of silently producing a wrong-shaped state.
pub const SNAPSHOT_VERSION: u8 = 1;

/// Errors raised by [`RustSimState::from_serialized`] /
/// `StashManager::load_snapshot`.
#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("empty snapshot envelope")]
    EmptyEnvelope,
    #[error("snapshot version mismatch: have {found}, expected {expected}")]
    VersionMismatch { found: u8, expected: u8 },
    #[error("decode error: {0}")]
    Decode(String),
}

/// Snapshot of a [`RustSimState`]'s persistable state (angr-x04s.1.3).
///
/// Covers all bucket A/B/C fields per the `rustsimstate-field-buckets` bd
/// memory:
///
/// * **Bucket A (trivials)** — pc, state_id, parent_id, history,
///   detailed_history, max_history, heap_brk, posix_brk, mmap_base,
///   stdin_symbols, call_stack, heap_metadata, no_ip_concretization,
///   no_symbolic_jump_resolution, keep_ip_symbolic, vex_arch,
///   inspection, concretizer, fs, track_history, drop_terminal flag
///   (carried on StashManager side).
/// * **Bucket B (concrete + symbolic overlay)** — registers
///   ([`RegisterFile`] serde), memory (`SymbolicMemorySnapshot`).
/// * **Bucket C (Arc-shared collapse)** — hooks (`Vec<u64>`), environment
///   (BTreeMap<bytes, bytes>).
/// * **SymContext** — captured via `SymContextSnapshot` (replays
///   `assumed_constraints` into a fresh Z3 solver on restore).
///
/// **Bucket D (`Py<PyAny>` overlays)** — symbolic_pages,
/// hook_symbolic_memory, addr_to_ast, last_time — deferred per the task
/// acceptance. After restore these fields are empty / None; Python-side
/// integration tests (.1.4) will route them through claripy.dumps/loads.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RustSimStateSnapshot {
    pub arch_name: String,
    pub vex_arch: crate::vex::VexArch,
    pub registers: crate::arch::RegisterFile,
    pub memory: crate::memory::SymbolicMemorySnapshot,
    pub solver: crate::symbolic::SymContextSnapshot,
    pub pc: u64,
    pub state_id: u64,
    pub parent_id: Option<u64>,
    pub history: Vec<u64>,
    pub detailed_history: Vec<HistoryEntry>,
    pub max_history: usize,
    pub hooks: Vec<u64>,
    pub concretizer: AddressConcretizer,
    pub track_history: bool,
    pub fs: FileSystem,
    pub heap_brk: u64,
    pub posix_brk: u64,
    pub mmap_base: u64,
    pub stdin_symbols: Vec<(String, u32)>,
    pub call_stack: Vec<CallStackEntry>,
    pub heap_metadata: HeapMetadata,
    pub inspection: InspectionManager,
    /// `(key, value)` byte pairs sorted by key for deterministic ordering.
    /// Not a `BTreeMap<Vec<u8>, Vec<u8>>` because `serde_json` only allows
    /// string-shaped map keys; the angr environment is byte-keyed.
    pub environment: Vec<(Vec<u8>, Vec<u8>)>,
    pub no_ip_concretization: bool,
    pub no_symbolic_jump_resolution: bool,
    pub keep_ip_symbolic: bool,
    /// angr-027h: eager-fork override. `#[serde(default)]` keeps pre-027h
    /// snapshots forward-compatible (restores to `false`).
    #[serde(default)]
    pub force_eager_forks: bool,
    /// CGC `state.cgc.allocation_base` mirror. `#[serde(default)]` keeps
    /// pre-CGC snapshots forward-compatible — restoration defaults to the
    /// canonical 0xB800_0000 bump start used by fresh CGC states.
    #[serde(default = "default_cgc_allocation_base")]
    pub cgc_allocation_base: u64,
    /// CGC `state.cgc.sinkholes` mirror. `#[serde(default)]` keeps pre-CGC
    /// snapshots forward-compatible — restoration defaults to an empty
    /// freelist.
    #[serde(default)]
    pub cgc_sinkholes: Vec<(u64, u64)>,
}

fn default_cgc_allocation_base() -> u64 {
    0xB800_0000
}

impl RustSimState {
    /// Build a serializable snapshot of this state (angr-x04s.1.3).
    ///
    /// Bucket-D `Py<PyAny>` overlays (symbolic_pages, hook_symbolic_memory,
    /// addr_to_ast, last_time) are NOT captured here — see
    /// [`RustSimStateSnapshot`].
    pub fn to_snapshot(&self) -> RustSimStateSnapshot {
        let mut hooks: Vec<u64> = self.hooks.iter().copied().collect();
        hooks.sort_unstable();
        let mut environment: Vec<(Vec<u8>, Vec<u8>)> = self
            .environment
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        environment.sort_by(|a, b| a.0.cmp(&b.0));
        RustSimStateSnapshot {
            arch_name: self.arch.name().to_string(),
            vex_arch: self.vex_arch,
            registers: self.registers.clone(),
            memory: self.memory.to_snapshot(),
            solver: self.solver.borrow().to_snapshot(),
            pc: self.pc,
            state_id: self.state_id,
            parent_id: self.parent_id,
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks,
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            stdin_symbols: self.stdin_symbols.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment,
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
            force_eager_forks: self.force_eager_forks,
            cgc_allocation_base: self.cgc_allocation_base,
            cgc_sinkholes: self.cgc_sinkholes.clone(),
        }
    }

    /// Restore a snapshot into a fresh [`RustSimState`]. Replays solver
    /// constraints via [`SymContext::restore_from_snapshot`] so the Z3
    /// solver, sat/model caches, and `assumed_constraints` log all rebuild
    /// consistently. Bucket-D `Py<PyAny>` overlays restore to empty (see
    /// [`RustSimStateSnapshot`]).
    pub fn from_snapshot(snap: RustSimStateSnapshot) -> Result<Self, String> {
        let arch = arch_from_name(&snap.arch_name)
            .ok_or_else(|| format!("unknown architecture: {}", snap.arch_name))?;
        let solver = Rc::new(RefCell::new(SymContext::new()));
        solver.borrow().restore_from_snapshot(&snap.solver);
        let memory = SymbolicMemory::from_snapshot(snap.memory);
        let environment: HashMap<Vec<u8>, Vec<u8>> = snap.environment.into_iter().collect();
        let hooks: HashSet<u64> = snap.hooks.into_iter().collect();
        Ok(RustSimState {
            arch,
            vex_arch: snap.vex_arch,
            registers: snap.registers,
            memory,
            solver,
            pc: snap.pc,
            state_id: snap.state_id,
            parent_id: snap.parent_id,
            history: snap.history,
            detailed_history: snap.detailed_history,
            max_history: snap.max_history,
            hooks: Arc::new(hooks),
            concretizer: snap.concretizer,
            track_history: snap.track_history,
            fs: snap.fs,
            heap_brk: snap.heap_brk,
            posix_brk: snap.posix_brk,
            mmap_base: snap.mmap_base,
            stdin_symbols: snap.stdin_symbols,
            call_stack: snap.call_stack,
            heap_metadata: snap.heap_metadata,
            inspection: snap.inspection,
            environment: Arc::new(environment),
            symbolic_pages: HashMap::new(),
            hook_symbolic_memory: HashMap::new(),
            addr_to_ast: HashMap::new(),
            last_time: None,
            no_ip_concretization: snap.no_ip_concretization,
            no_symbolic_jump_resolution: snap.no_symbolic_jump_resolution,
            keep_ip_symbolic: snap.keep_ip_symbolic,
            force_eager_forks: snap.force_eager_forks,
            cgc_allocation_base: snap.cgc_allocation_base,
            cgc_sinkholes: snap.cgc_sinkholes,
        })
    }

    /// Serialize this state to a versioned envelope:
    /// `[SNAPSHOT_VERSION: u8] ++ serde_json(RustSimStateSnapshot)`.
    /// The format-version byte lets [`Self::from_serialized`] reject a
    /// stale on-disk snapshot fast. `serde_json` was chosen over postcard
    /// for the prototype because the inner [`RustBV`] op-tree carries
    /// `Arc<...>` boxed enums whose postcard schema would lock the format
    /// to today's `crate::symbolic::value::BVOp` layout; JSON tolerates
    /// minor variant churn without a breaking change.
    pub fn to_serialized(&self) -> Vec<u8> {
        let snap = self.to_snapshot();
        let body = serde_json::to_vec(&snap).expect("snapshot encode");
        let mut out = Vec::with_capacity(1 + body.len());
        out.push(SNAPSHOT_VERSION);
        out.extend_from_slice(&body);
        out
    }

    /// Inverse of [`Self::to_serialized`]. Rejects an empty envelope or a
    /// version-byte mismatch with [`SnapshotError`].
    pub fn from_serialized(bytes: &[u8]) -> Result<Self, SnapshotError> {
        if bytes.is_empty() {
            return Err(SnapshotError::EmptyEnvelope);
        }
        let version = bytes[0];
        if version != SNAPSHOT_VERSION {
            return Err(SnapshotError::VersionMismatch {
                found: version,
                expected: SNAPSHOT_VERSION,
            });
        }
        // serde_json's default recursion limit (128) is hit by deep
        // RustBV op-trees that real benches accumulate (per-byte memory
        // loads nest store/load chains hundreds of levels deep).
        // `disable_recursion_limit()` lifts the cap; the on-disk envelope
        // is trusted (written by our own `to_serialized`) so the DoS
        // hardening the limit provides is not load-bearing here.
        let mut de = serde_json::Deserializer::from_slice(&bytes[1..]);
        de.disable_recursion_limit();
        let snap: RustSimStateSnapshot = serde::Deserialize::deserialize(&mut de)
            .map_err(|e| SnapshotError::Decode(e.to_string()))?;
        Self::from_snapshot(snap).map_err(SnapshotError::Decode)
    }
}

// =============================================================================
// Python Bindings
// =============================================================================
