//! Persistence for [`FileSystem`]: the serde shadow form ([`FileSystemData`])
//! and the cross-Z3-context [`FileSystem::translate_into`] twin of `clone`.

use super::*;

/// Serde shadow form for [`FileSystem`].
///
/// Collapses `Arc<HashMap<u32, FileDescriptor>>` to a deterministic
/// `BTreeMap<u32, FileDescriptor>` on the wire and carries `next_fd` /
/// `cwd` explicitly. Mirrors the `MemoryPage` / `RegisterFile` snapshot
/// shadow pattern (angr-x04s.1.2).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileSystemData {
    pub fds: std::collections::BTreeMap<u32, FileDescriptor>,
    pub next_fd: u32,
    pub cwd: Vec<u8>,
    /// `#[serde(default)]` keeps pre-angr-k3ol.2 snapshots loadable —
    /// the field reconstitutes to an empty set, matching the previous
    /// behavior (no native access lookups would succeed).
    #[serde(default)]
    pub known_paths: std::collections::BTreeSet<String>,
    /// `#[serde(default)]` keeps pre-angr-11djq.6.2 snapshots loadable —
    /// reconstitutes to an empty map (no symlinks, `readlink` → `-1`).
    #[serde(default)]
    pub symlinks: std::collections::BTreeMap<String, Vec<u8>>,
    /// Symbolic-content registry. The `Arc`s go on the wire as their
    /// inner `Vec<RustBV>` via serde's `rc` feature (symbolic leaves
    /// rebuild by (name, width) via the `RustBVData` shadow); each load
    /// rebuilds fresh `Arc`s, so fd/registry sharing is not preserved
    /// through serde. `#[serde(default)]` keeps pre-angr-0xyq2 snapshots
    /// loadable — reconstitutes to an empty registry. BTreeMap keeps the
    /// wire ordering deterministic.
    #[serde(default)]
    pub file_contents: std::collections::BTreeMap<String, Arc<Vec<RustBV>>>,
    /// Accumulated demoted-path set (angr-qluof). `#[serde(default)]` keeps
    /// pre-angr-qluof snapshots loadable — reconstitutes to an empty set
    /// (no lineage-demotion memory, matching the previous behavior).
    /// BTreeSet keeps the wire ordering deterministic.
    #[serde(default)]
    pub demoted_paths: std::collections::BTreeSet<String>,
}

impl From<FileSystem> for FileSystemData {
    fn from(fs: FileSystem) -> Self {
        let fds: std::collections::BTreeMap<u32, FileDescriptor> =
            fs.fds.iter().map(|(k, v)| (*k, v.clone())).collect();
        let known_paths: std::collections::BTreeSet<String> =
            fs.known_paths.iter().cloned().collect();
        let symlinks: std::collections::BTreeMap<String, Vec<u8>> = fs
            .symlinks
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let file_contents: std::collections::BTreeMap<String, Arc<Vec<RustBV>>> = fs
            .file_contents
            .iter()
            .map(|(k, v)| (k.clone(), Arc::clone(v)))
            .collect();
        let demoted_paths: std::collections::BTreeSet<String> =
            fs.demoted_paths.iter().cloned().collect();
        FileSystemData {
            fds,
            next_fd: fs.next_fd,
            cwd: fs.cwd,
            known_paths,
            symlinks,
            file_contents,
            demoted_paths,
        }
    }
}

impl From<FileSystemData> for FileSystem {
    // RustBV's Z3 AST makes FileDescriptor !Send; the Arcs here are
    // per-state CoW handles that never cross threads (states migrate
    // between workers via the serde snapshot, not by moving Arcs) — hence
    // the !Send Arcs go through `crate::arc_shared` (see its doc comment).
    fn from(d: FileSystemData) -> Self {
        let fds: HashMap<u32, FileDescriptor> = d.fds.into_iter().collect();
        let symlinks: HashMap<String, Vec<u8>> = d.symlinks.into_iter().collect();
        let file_contents: HashMap<String, Arc<Vec<RustBV>>> =
            d.file_contents.into_iter().collect();
        // demoted_paths keys are always normalized at insertion, so no
        // re-normalization pass is needed (unlike known_paths below).
        let demoted_paths: HashSet<String> = d.demoted_paths.into_iter().collect();
        let mut fs = FileSystem {
            fds: crate::arc_shared(fds),
            next_fd: d.next_fd,
            cwd: d.cwd,
            known_paths: Arc::new(HashSet::new()),
            symlinks: Arc::new(symlinks),
            file_contents: crate::arc_shared(file_contents),
            demoted_paths: Arc::new(demoted_paths),
        };
        // Re-normalize known_paths on load: the key space is normalized at
        // insertion (see `open`), but pre-normalization snapshots may carry
        // raw relative entries. Absolute canonical entries are fixed points,
        // so this is a no-op for current-format snapshots.
        let known_paths: HashSet<String> =
            d.known_paths.iter().map(|p| fs.normalize_path(p)).collect();
        fs.known_paths = Arc::new(known_paths);
        fs
    }
}

impl FileSystem {
    /// Cross-context twin of `clone` for `RustSimState::translate_state`
    /// (angr-ahypj): every context-bound `RustBV` — each fd's
    /// `content_sym` vec and each `file_contents` registry value — is
    /// `Z3_translate`d into `target_ctx` via [`RustBV::translate_into`];
    /// every other field is context-independent and cloned. Like the
    /// serde path, Arc *sharing* between an fd and the registry is not
    /// preserved through translation (each gets a fresh translated Arc) —
    /// acceptable, sharing is only a fork-time perf optimization.
    // !Send Arcs go through `crate::arc_shared` (see its doc comment).
    #[cfg(feature = "vex-engine-z3")]
    pub fn translate_into(&self, target_ctx: &z3::Context) -> Self {
        // Fast path: no symbolic file content anywhere → nothing is
        // context-bound, so the O(1) Arc-sharing clone (the pre-angr-0xyq2
        // behavior) is correct. All production states today take this.
        if self.file_contents.is_empty() && self.fds.values().all(|d| d.content_sym.is_none()) {
            return self.clone();
        }
        let translate_vec = |v: &Arc<Vec<RustBV>>| -> Arc<Vec<RustBV>> {
            crate::arc_shared(v.iter().map(|bv| bv.translate_into(target_ctx)).collect())
        };
        let fds: HashMap<u32, FileDescriptor> = self
            .fds
            .iter()
            .map(|(k, d)| {
                let mut d = d.clone();
                d.content_sym = d.content_sym.as_ref().map(&translate_vec);
                (*k, d)
            })
            .collect();
        let file_contents: HashMap<String, Arc<Vec<RustBV>>> = self
            .file_contents
            .iter()
            .map(|(k, v)| (k.clone(), translate_vec(v)))
            .collect();
        FileSystem {
            fds: crate::arc_shared(fds),
            next_fd: self.next_fd,
            cwd: self.cwd.clone(),
            known_paths: Arc::clone(&self.known_paths),
            symlinks: Arc::clone(&self.symlinks),
            file_contents: crate::arc_shared(file_contents),
            demoted_paths: Arc::clone(&self.demoted_paths),
        }
    }
}
