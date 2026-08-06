//! Snapshot serialization (serde wire format) for `RustSimState`.
//!
//! **Panic policy (angr-9ke6b.212, angr-9ke6b.126):** every function that takes
//! bytes — [`RustSimState::from_serialized`] and the
//! [`RustSimState::bench_decode_snapshot`] benchmark hook alike — is
//! `Result`-typed over [`SnapshotError`]; nothing on that direction panics. The
//! one surviving `expect` is the encode direction, where `serde_json::to_vec`
//! over a derived `Serialize` writing into a `Vec` has no reachable `Err`; it is
//! documented at its `#[allow]`.
//!
//! **Enforcement (angr-qwyti.11):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

// =============================================================================
// Snapshot / Serialization (angr-x04s.1.3)
// =============================================================================

/// Format-version byte at the head of every [`RustSimState::to_serialized`]
/// envelope. Bump on any breaking shape change to [`RustSimStateSnapshot`]
/// so a stale snapshot fails fast with `SnapshotError::VersionMismatch`
/// instead of silently producing a wrong-shaped state.
// v2 (angr-t3l5o Phase 1): `SymContextSnapshot` switched from a full-solver
// `solver_smtlib2` dump to the two-class `residual_smtlib2` + `reassert_assumed`
// shape. A v1 envelope replayed under v2 would double-assert the assume class
// (full text dump re-asserted AND `assumed_constraints` re-asserted), so reject
// it via the version gate rather than silently mixing the formats.
pub const SNAPSHOT_VERSION: u8 = 2;

/// Errors raised by [`RustSimState::from_serialized`] /
/// `StashManager::load_snapshot`.
#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("empty snapshot envelope")]
    EmptyEnvelope,
    #[error("snapshot version mismatch: have {found}, expected {expected}")]
    VersionMismatch { found: u8, expected: u8 },
    #[error("unknown architecture: {name}")]
    UnknownArch { name: String },
    #[error("decode error: {0}")]
    Decode(String),
    /// `reattach` was called while `target_ctx` was not the active thread-local
    /// Z3 context — rebuilding would mint ASTs in the wrong context (angr-1ilq.1).
    #[error("reattach target_ctx is not the active thread-local Z3 context")]
    ContextMismatch,
}

/// Snapshot of a [`RustSimState`]'s persistable state (angr-x04s.1.3).
///
/// Covers all bucket A/B/C fields:
///
/// * **Bucket A (trivials)** — pc, state_id, parent_id, history,
///   detailed_history, max_history, heap_brk, posix_brk, mmap_base,
///   tsc_counter, getopt_optind, getopt_optchar, getopt_extern,
///   stdin_symbols, call_stack, heap_metadata, no_ip_concretization,
///   no_symbolic_jump_resolution, keep_ip_symbolic, vex_arch,
///   inspection, concretizer, fs, track_history, drop_terminal flag
///   (carried on StashManager side).
/// * **Bucket B (concrete + symbolic overlay)** — registers
///   (`RegisterFile` serde), memory (`SymbolicMemorySnapshot`).
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
    pub history: VecDeque<u64>,
    pub detailed_history: VecDeque<HistoryEntry>,
    pub max_history: usize,
    pub hooks: Vec<u64>,
    /// Hook addresses explicitly removed since this state's last fork point
    /// (angr-9ke6b.121 bug fix follow-up; see `RustSimState::removed_hooks`).
    /// `#[serde(default)]` keeps pre-this-fix snapshots forward-compatible —
    /// restoration defaults to an empty tombstone set.
    #[serde(default)]
    pub removed_hooks: Vec<u64>,
    pub concretizer: AddressConcretizer,
    pub track_history: bool,
    pub fs: FileSystem,
    pub heap_brk: u64,
    pub posix_brk: u64,
    pub mmap_base: u64,
    /// Simulated RDTSC counter (angr-9ke6b.173). `#[serde(default)]` keeps
    /// pre-.173 snapshots loadable; they restore to 0, which is harmless — the
    /// counter only has to be monotonic within a state, not start at
    /// `TSC_INITIAL`.
    #[serde(default)]
    pub tsc_counter: u64,
    pub getopt_optind: u32,
    pub getopt_optchar: u32,
    pub getopt_extern: crate::state::GetoptExternAddrs,
    /// Suspended native sub-call continuations (bead angr-pn3w8). `#[serde(default)]`
    /// keeps pre-pn3w8 snapshots forward-compatible (restores to an empty stack).
    #[serde(default)]
    pub native_resume_stack: Vec<crate::state::NativeResumeFrame>,
    pub ctype_loc: crate::state::CtypeLocPtrs,
    pub stdin_symbols: Vec<(String, u32)>,
    pub call_stack: Vec<CallStackEntry>,
    pub heap_metadata: HeapMetadata,
    pub inspection: InspectionManager,
    /// `(key, value)` byte pairs sorted by key for deterministic ordering.
    /// Not a `BTreeMap<Vec<u8>, Vec<u8>>` because `serde_json` only allows
    /// string-shaped map keys; the angr environment is byte-keyed.
    pub environment: Vec<(Vec<u8>, Vec<u8>)>,
    /// Environment keys explicitly removed since this state's last fork point
    /// (angr-9ke6b.121 bug fix follow-up; see `RustSimState::removed_env_keys`).
    /// `#[serde(default)]` keeps pre-this-fix snapshots forward-compatible.
    #[serde(default)]
    pub removed_env_keys: Vec<Vec<u8>>,
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
    /// Symex-relevant SimOption mirror (angr-kzjv6). Stored as a sorted `Vec`
    /// for deterministic serialization (the live state holds an
    /// `Arc<HashSet<String>>`). `#[serde(default)]` keeps pre-kzjv6 snapshots
    /// forward-compatible — restoration defaults to an empty option set.
    #[serde(default)]
    pub sim_options: Vec<String>,
    /// SimOption names explicitly disabled since this state's last fork point
    /// (angr-9ke6b.121 bug fix follow-up; see `RustSimState::removed_sim_options`).
    /// `#[serde(default)]` keeps pre-this-fix snapshots forward-compatible.
    #[serde(default)]
    pub removed_sim_options: Vec<String>,
}

fn default_cgc_allocation_base() -> u64 {
    0xB800_0000
}

impl RustSimState {
    /// Flush this state's memory (Multi cells + any queued lazy-write ITEs)
    /// and return proof of it, for any caller about to build a
    /// snapshot/export or otherwise read `symbolic_objects`/pages directly
    /// and that must see fully-materialized memory (angr-sqfj8.71,
    /// angr-sqfj8.26, angr-sqfj8.52). Mirrors [`Self::flush_and_export_full`]'s
    /// flush step.
    pub(crate) fn flush_memory(&mut self) -> crate::memory::MultiFlushed {
        let ctx = self.solver.borrow();
        // SILENT(cat-b): same rationale as `flush_and_export_full` — a
        // lazy-write materialization failure costs the rest of the queue
        // either way, so log it rather than losing the snapshot outright.
        // `flush_multi_cells` (the part the proof certifies) runs
        // unconditionally before any pending-write processing inside
        // `flush_pending_writes`, so the proof is valid even on this error
        // path.
        match self.memory.flush_pending_writes(&ctx, &self.concretizer) {
            Ok(proof) => proof,
            Err(e) => {
                log::warn!(
                    "flush_memory: pending-write flush failed ({e:?}); \
                     remaining deferred stores are not reflected in this snapshot"
                );
                self.memory.flush_multi_cells(&ctx)
            }
        }
    }

    /// Build a serializable snapshot of this state (angr-x04s.1.3).
    ///
    /// Bucket-D `Py<PyAny>` overlays (symbolic_pages, hook_symbolic_memory,
    /// addr_to_ast, last_time) are NOT captured here — see
    /// [`RustSimStateSnapshot`].
    ///
    /// `&mut self`: flushes memory first (angr-sqfj8.71) so Multi-covered
    /// bytes from the lazy symbolic-address store path are never silently
    /// absent from the snapshot.
    pub fn to_snapshot(&mut self) -> RustSimStateSnapshot {
        let mem_proof = self.flush_memory();
        let mut hooks: Vec<u64> = self.hooks.iter().copied().collect();
        hooks.sort_unstable();
        let mut removed_hooks: Vec<u64> = self.removed_hooks.iter().copied().collect();
        removed_hooks.sort_unstable();
        let mut environment: Vec<(Vec<u8>, Vec<u8>)> = self
            .environment
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        environment.sort_by(|a, b| a.0.cmp(&b.0));
        let mut removed_env_keys: Vec<Vec<u8>> = self.removed_env_keys.iter().cloned().collect();
        removed_env_keys.sort();
        let mut sim_options: Vec<String> = self.sim_options.iter().cloned().collect();
        sim_options.sort_unstable();
        let mut removed_sim_options: Vec<String> =
            self.removed_sim_options.iter().cloned().collect();
        removed_sim_options.sort_unstable();
        RustSimStateSnapshot {
            arch_name: self.arch.name().to_string(),
            vex_arch: self.vex_arch,
            registers: self.registers.clone(),
            memory: self.memory.to_snapshot(&mem_proof),
            solver: self.solver.borrow().to_snapshot(),
            pc: self.pc,
            state_id: self.state_id,
            parent_id: self.parent_id,
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks,
            removed_hooks,
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            tsc_counter: self.tsc_counter,
            getopt_optind: self.getopt_optind,
            getopt_optchar: self.getopt_optchar,
            getopt_extern: self.getopt_extern,
            native_resume_stack: self.native_resume_stack.clone(),
            ctype_loc: self.ctype_loc,
            stdin_symbols: self.stdin_symbols.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment,
            removed_env_keys,
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
            force_eager_forks: self.force_eager_forks,
            cgc_allocation_base: self.cgc_allocation_base,
            cgc_sinkholes: self.cgc_sinkholes.clone(),
            sim_options,
            removed_sim_options,
        }
    }

    /// Restore a snapshot into a fresh [`RustSimState`]. Replays solver
    /// constraints via [`SymContext::restore_from_snapshot`] so the Z3
    /// solver, sat/model caches, and `assumed_constraints` log all rebuild
    /// consistently. Bucket-D `Py<PyAny>` overlays restore to empty (see
    /// [`RustSimStateSnapshot`]).
    pub fn from_snapshot(snap: RustSimStateSnapshot) -> Result<Self, SnapshotError> {
        let arch = arch_from_name(&snap.arch_name).ok_or_else(|| SnapshotError::UnknownArch {
            name: snap.arch_name.clone(),
        })?;
        // The restored ID was minted by another process/epoch; teach the local
        // counter about it so later forks cannot re-mint it (`state-id-never-reused`).
        crate::state::reserve_state_id(snap.state_id);
        let solver = Rc::new(RefCell::new(SymContext::new()));
        solver.borrow().restore_from_snapshot(&snap.solver);
        // angr-t3l5o Phase 0b: time the symbolic-memory page rebuild (the
        // non-solver "leaf rebuild" phase) when migration phase timers are on.
        let memory = crate::migrate_phase_timers::time_phase(
            &crate::migrate_phase_timers::MIGRATE_LEAF_REBUILD_NS,
            || SymbolicMemory::from_snapshot(snap.memory),
        );
        let environment: HashMap<Vec<u8>, Vec<u8>> = snap.environment.into_iter().collect();
        let hooks: HashSet<u64> = snap.hooks.into_iter().collect();
        let removed_hooks: HashSet<u64> = snap.removed_hooks.into_iter().collect();
        let removed_env_keys: HashSet<Vec<u8>> = snap.removed_env_keys.into_iter().collect();
        let removed_sim_options: HashSet<String> = snap.removed_sim_options.into_iter().collect();
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
            removed_hooks: Arc::new(removed_hooks),
            concretizer: snap.concretizer,
            track_history: snap.track_history,
            fs: snap.fs,
            heap_brk: snap.heap_brk,
            posix_brk: snap.posix_brk,
            mmap_base: snap.mmap_base,
            tsc_counter: snap.tsc_counter,
            getopt_optind: snap.getopt_optind,
            getopt_optchar: snap.getopt_optchar,
            getopt_extern: snap.getopt_extern,
            native_resume_stack: snap.native_resume_stack,
            ctype_loc: snap.ctype_loc,
            stdin_symbols: snap.stdin_symbols,
            call_stack: snap.call_stack,
            heap_metadata: snap.heap_metadata,
            inspection: snap.inspection,
            environment: Arc::new(environment),
            removed_env_keys: Arc::new(removed_env_keys),
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
            sim_options: Arc::new(snap.sim_options.into_iter().collect()),
            removed_sim_options: Arc::new(removed_sim_options),
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
    #[allow(
        clippy::expect_used,
        reason = "`serde_json::to_vec` over `RustSimStateSnapshot`, whose derived `Serialize` has no fallible arm and writes into a `Vec` (so no io error). Left as a panic rather than propagated because `to_serialized` returns `Vec<u8>` across the PyO3 surface and the migration payload path; widening it to `Result` would ripple into every caller — out of scope for angr-9ke6b.212"
    )]
    pub fn to_serialized(&mut self) -> Vec<u8> {
        crate::migrate_phase_timers::time_roundtrip_half(|| {
            // angr-t3l5o Phase 0b: arm the migration-serialize guard so the
            // SMT-LIB2 emit timer inside `to_snapshot` fires ONLY for migration
            // transport (not for unrelated stash `to_snapshot` calls).
            let prev = crate::migrate_phase_timers::set_serializing(true);
            let snap = self.to_snapshot();
            // Time the pure serde-json encode (the SMT-LIB2 emit already
            // happened inside `to_snapshot` and is timed separately).
            let body = crate::migrate_phase_timers::time_phase(
                &crate::migrate_phase_timers::MIGRATE_SERDE_NS,
                || serde_json::to_vec(&snap).expect("snapshot encode"),
            );
            crate::migrate_phase_timers::set_serializing(prev);
            let mut out = Vec::with_capacity(1 + body.len());
            out.push(SNAPSHOT_VERSION);
            out.extend_from_slice(&body);
            out
        })
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
        crate::migrate_phase_timers::time_roundtrip_half(|| {
            // angr-t3l5o Phase 0b: arm the migration-serialize guard so the
            // serde-decode / SMT-LIB2-parse / leaf-rebuild sub-timers inside
            // `from_snapshot` fire ONLY for migration transport (not for stash
            // `from_snapshot` calls). Restored before returning either arm.
            let prev = crate::migrate_phase_timers::set_serializing(true);
            // Time the pure serde-json decode of the snapshot struct (the
            // SMT-LIB2 parse and memory leaf-rebuild happen later in
            // `from_snapshot` and are timed separately).
            let decoded: Result<RustSimStateSnapshot, _> = crate::migrate_phase_timers::time_phase(
                &crate::migrate_phase_timers::MIGRATE_SERDE_NS,
                || {
                    let mut de = serde_json::Deserializer::from_slice(&bytes[1..]);
                    de.disable_recursion_limit();
                    serde::Deserialize::deserialize(&mut de)
                },
            );
            let snap: RustSimStateSnapshot = match decoded {
                Ok(s) => s,
                Err(e) => {
                    crate::migrate_phase_timers::set_serializing(prev);
                    return Err(SnapshotError::Decode(e.to_string()));
                }
            };
            let out = Self::from_snapshot(snap);
            crate::migrate_phase_timers::set_serializing(prev);
            out
        })
    }

    /// angr-t3l5o Phase 0a bench hook: serde-json decode a snapshot byte
    /// buffer into the [`RustSimStateSnapshot`] struct (no `from_snapshot`
    /// rebuild), isolating the serde-decode cost. Uses the same
    /// `disable_recursion_limit` as [`Self::from_serialized`].
    ///
    /// Takes the *body* only — no version byte — unlike
    /// [`Self::from_serialized`]. Mirrors that sibling's error typing
    /// (angr-9ke6b.126) so this `pub` entry point cannot panic on malformed
    /// bytes even though its only in-repo callers are the benches.
    #[doc(hidden)]
    pub fn bench_decode_snapshot(bytes: &[u8]) -> Result<RustSimStateSnapshot, SnapshotError> {
        let mut de = serde_json::Deserializer::from_slice(bytes);
        de.disable_recursion_limit();
        serde::Deserialize::deserialize(&mut de).map_err(|e| SnapshotError::Decode(e.to_string()))
    }

    /// angr-t3l5o Phase 0a bench hook: round-trip just the symbolic-memory
    /// pages through `to_snapshot` / `from_snapshot`, isolating the
    /// memory-page copy cost. Returns the rebuilt page count so the result
    /// cannot be optimized away.
    #[doc(hidden)]
    pub fn bench_memory_snapshot_roundtrip(&mut self) -> usize {
        let ctx = self.solver.borrow();
        let proof = self.memory.flush_multi_cells(&ctx);
        let snap = self.memory.to_snapshot(&proof);
        let mut rebuilt = crate::memory::SymbolicMemory::from_snapshot(snap);
        let rebuilt_proof = rebuilt.flush_multi_cells(&ctx);
        rebuilt.to_snapshot(&rebuilt_proof).pages.len()
    }
}

// =============================================================================
// Python Bindings
// =============================================================================
