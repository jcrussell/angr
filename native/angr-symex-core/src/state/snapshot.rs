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
//! The `deny` below follows the crate [Panic policy](crate#panic-policy)
//! (angr-qwyti.11).
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
///   detailed_history, max_history, heap_brk, posix_brk, mmap_base, process_ids,
///   pending_syscall, dirtied_code_pages, tracked_variables, exit_code, exit_pending, getopt_cursor, getopt_extern,
///   stdin_symbols, stdin_read_lengths, stdin_host_packets, call_stack, heap_metadata, no_ip_concretization,
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
    pub vex_arch: angr_vex::VexArch,
    pub registers: crate::arch::RegisterFile,
    pub memory: crate::memory::SymbolicMemorySnapshot,
    pub solver: crate::symbolic::SymContextSnapshot,
    pub pc: u64,
    pub state_id: u64,
    pub parent_id: Option<u64>,
    pub history: VecDeque<u64>,
    pub detailed_history: VecDeque<HistoryEntry>,
    pub max_history: usize,
    /// See `RustSimState::block_count` (angr-vmi2l.31).
    #[serde(default)]
    pub block_count: u64,
    /// See `RustSimState::evicted_visits` (angr-vmi2l.31).
    #[serde(default)]
    pub evicted_visits: HashMap<u64, u32>,
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
    /// See `RustSimState::process_ids` (angr-vmi2l.149).
    #[serde(default)]
    pub process_ids: crate::state::ProcessIds,
    /// See `RustSimState::pending_syscall` (angr-mx00p.132.12).
    #[serde(default)]
    pub pending_syscall: Option<angr_vex::JumpKind>,
    /// See `RustSimState::dirtied_code_pages` (angr-vmi2l.163).
    #[serde(default)]
    pub dirtied_code_pages: Vec<crate::memory::PageIndex>,
    /// See `RustSimState::tracked_variables` (angr-vmi2l.126).
    #[serde(default)]
    pub tracked_variables: Vec<crate::state::TrackedVariable>,
    /// See `RustSimState::exit_code` (angr-mx00p.132.8).
    #[serde(default)]
    pub exit_code: Option<RustBV>,
    /// See `RustSimState::exit_pending` (angr-mx00p.136.30).
    #[serde(default)]
    pub exit_pending: bool,
    pub getopt_cursor: crate::state::GetoptCursor,
    pub getopt_extern: crate::state::GetoptExternAddrs,
    /// Suspended native sub-call continuations (bead angr-pn3w8). `#[serde(default)]`
    /// keeps pre-pn3w8 snapshots forward-compatible (restores to an empty stack).
    #[serde(default)]
    pub native_resume_stack: Vec<crate::state::NativeResumeFrame>,
    pub ctype_loc: crate::state::CtypeLocPtrs,
    pub stdin_symbols: Vec<(String, u32)>,
    /// See `RustSimState::stdin_read_lengths` (angr-mx00p.132.3).
    #[serde(default)]
    pub stdin_read_lengths: Vec<crate::state::StdinReadLength>,
    /// See `RustSimState::stdin_host_packets` (angr-azp2b).
    #[serde(default)]
    pub stdin_host_packets: Vec<crate::state::StdinHostPacket>,
    pub call_stack: Vec<CallStackEntry>,
    pub heap_metadata: HeapMetadata,
    // angr-vmi2l.197: the dead per-state `InspectionManager` and its
    // `inspection` field are gone. No version bump: the envelope is JSON
    // without `deny_unknown_fields`, so an older snapshot's `inspection` key
    // is ignored on decode.
    /// `(key, value)` byte pairs sorted by key for deterministic ordering.
    /// Not a `BTreeMap<Vec<u8>, Vec<u8>>` because `serde_json` only allows
    /// string-shaped map keys; the angr environment is byte-keyed.
    pub environment: Vec<(Vec<u8>, Vec<u8>)>,
    /// Environment keys explicitly removed since this state's last fork point
    /// (angr-9ke6b.121 bug fix follow-up; see `RustSimState::removed_env_keys`).
    /// `#[serde(default)]` keeps pre-this-fix snapshots forward-compatible.
    #[serde(default)]
    pub removed_env_keys: Vec<Vec<u8>>,
    /// `(key, addr)` for each [`environment`](Self::environment) entry with a
    /// guest value address ([`EnvVar::addr`]), sorted by key. A separate
    /// field, with `#[serde(default)]`, so older snapshots still decode.
    #[serde(default)]
    pub env_value_addrs: Vec<(Vec<u8>, u64)>,
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
    /// Flush this state's Multi cells into symbolic objects and return proof
    /// of it, for any caller about to build a snapshot/export or otherwise
    /// read `symbolic_objects`/pages directly and that must see
    /// fully-materialized memory (angr-sqfj8.71, angr-sqfj8.26,
    /// angr-sqfj8.52). [`Self::flush_and_export_full`] flushes through it.
    pub fn flush_memory(&mut self) -> crate::memory::MultiFlushed {
        self.memory.flush_multi_cells()
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
            .map(|(k, v)| (k.clone(), v.value.clone()))
            .collect();
        environment.sort_by(|a, b| a.0.cmp(&b.0));
        let mut env_value_addrs: Vec<(Vec<u8>, u64)> = self
            .environment
            .iter()
            .filter_map(|(k, v)| v.addr.map(|addr| (k.clone(), addr)))
            .collect();
        env_value_addrs.sort_by(|a, b| a.0.cmp(&b.0));
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
            block_count: self.block_count,
            evicted_visits: (*self.evicted_visits).clone(),
            hooks,
            removed_hooks,
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            process_ids: self.process_ids,
            pending_syscall: self.pending_syscall,
            dirtied_code_pages: {
                let mut pages: Vec<_> = self.dirtied_code_pages.iter().copied().collect();
                pages.sort_unstable();
                pages
            },
            tracked_variables: self.tracked_variables.iter().cloned().collect(),
            exit_code: self.exit_code.clone(),
            exit_pending: self.exit_pending,
            getopt_cursor: self.getopt_cursor,
            getopt_extern: self.getopt_extern,
            native_resume_stack: self.native_resume_stack.clone(),
            ctype_loc: self.ctype_loc,
            stdin_symbols: self.stdin_symbols.clone(),
            stdin_read_lengths: self.stdin_read_lengths.clone(),
            stdin_host_packets: self.stdin_host_packets.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            environment,
            removed_env_keys,
            env_value_addrs,
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
        let memory = SymbolicMemory::from_snapshot(snap.memory);
        let mut environment: HashMap<Vec<u8>, EnvVar> = snap
            .environment
            .into_iter()
            .map(|(key, value)| (key, EnvVar { value, addr: None }))
            .collect();
        for (key, addr) in snap.env_value_addrs {
            if let Some(var) = environment.get_mut(&key) {
                var.addr = Some(addr);
            }
        }
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
            block_count: snap.block_count,
            evicted_visits: Arc::new(snap.evicted_visits),
            hooks: Arc::new(hooks),
            removed_hooks: Arc::new(removed_hooks),
            concretizer: snap.concretizer,
            track_history: snap.track_history,
            fs: snap.fs,
            heap_brk: snap.heap_brk,
            posix_brk: snap.posix_brk,
            mmap_base: snap.mmap_base,
            process_ids: snap.process_ids,
            pending_syscall: snap.pending_syscall,
            dirtied_code_pages: Arc::new(snap.dirtied_code_pages.into_iter().collect()),
            tracked_variables: snap.tracked_variables.into_iter().collect(),
            exit_code: snap.exit_code,
            exit_pending: snap.exit_pending,
            getopt_cursor: snap.getopt_cursor,
            getopt_extern: snap.getopt_extern,
            native_resume_stack: snap.native_resume_stack,
            ctype_loc: snap.ctype_loc,
            stdin_symbols: snap.stdin_symbols,
            stdin_read_lengths: snap.stdin_read_lengths,
            stdin_host_packets: snap.stdin_host_packets,
            call_stack: snap.call_stack,
            heap_metadata: snap.heap_metadata,
            environment: Arc::new(environment),
            removed_env_keys: Arc::new(removed_env_keys),
            symbolic_pages: OverlayMap::default(),
            hook_symbolic_memory: OverlayMap::default(),
            addr_to_ast: OverlayMap::default(),
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
        reason = "`serde_json::to_vec` over `RustSimStateSnapshot`, whose derived `Serialize` has no fallible arm and writes into a `Vec` (so no io error). Left as a panic rather than propagated because `to_serialized` returns `Vec<u8>` across the PyO3 surface and the migration payload path; widening it to `Result` would ripple into every caller — out of scope here (angr-9ke6b.212)"
    )]
    pub fn to_serialized(&mut self) -> Vec<u8> {
        let snap = self.to_snapshot();
        let body = serde_json::to_vec(&snap).expect("snapshot encode");
        // overflow-ok: `body` is a live `Vec`, so its length is at most
        // `isize::MAX`; the version byte cannot push it past `usize::MAX`.
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
        Self::from_snapshot(snap)
    }

    /// Bench hook (angr-t3l5o Phase 0a): serde-json decode a snapshot byte
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

    /// Bench hook (angr-t3l5o Phase 0a): round-trip just the symbolic-memory
    /// pages through `to_snapshot` / `from_snapshot`, isolating the
    /// memory-page copy cost. Returns the rebuilt page count so the result
    /// cannot be optimized away.
    #[doc(hidden)]
    pub fn bench_memory_snapshot_roundtrip(&mut self) -> usize {
        let proof = self.memory.flush_multi_cells();
        let snap = self.memory.to_snapshot(&proof);
        let mut rebuilt = crate::memory::SymbolicMemory::from_snapshot(snap);
        let rebuilt_proof = rebuilt.flush_multi_cells();
        rebuilt.to_snapshot(&rebuilt_proof).pages.len()
    }
}

// =============================================================================
// Python Bindings
// =============================================================================
