//! Process / OS-environment state accessors for `RustSimState`.
//!
//! The per-state POSIX and OS-layout surface: file-descriptor output buffers
//! (stdout/fd), recorded stdin symbols, the environment map, the malloc
//! (`heap_brk`) and POSIX (`posix_brk`) break pointers, the getopt(3) cursor
//! and extern-global addresses, the mmap base, the locale ctype table
//! pointers, the native sub-call resume stack, the CGC allocator/sinkhole
//! state, the last time(2) return value, and the heap bump-allocator. Split
//! out of `mod.rs` per the god-object decomposition (angr-0mqkc.5); mirrors
//! the `registers.rs` / `history.rs` / `options.rs` extension-impl pattern.

use super::*;

impl RustSimState {
    /// Get the stdout buffer (fd=1).
    pub fn stdout_buffer(&self) -> &[u8] {
        self.fd_buffer(1)
    }

    /// Append bytes to the stdout buffer (fd=1). Same refusal contract as
    /// [`write_fd`](Self::write_fd) — `false` only when fd 1 was rebound
    /// (`dup2`) onto a bounded-symbolic-content fd, which is then demoted.
    #[must_use = "false means the write was refused (symbolic content demoted); bounce to Python"]
    pub fn write_stdout(&mut self, data: &[u8]) -> bool {
        self.write_fd(1, data)
    }

    /// Check if stdout has been written to.
    pub fn has_stdout(&self) -> bool {
        !self.fs.fd_content(1).is_empty()
    }

    /// Get the output buffer for a file descriptor.
    pub fn fd_buffer(&self, fd: u32) -> &[u8] {
        self.fs.fd_content(fd)
    }

    /// Append bytes to a file descriptor's output buffer.
    ///
    /// Choke-point contract (angr-0xyq2 Phase 2, see `FileSystem::write`):
    /// returns `false` — with the fd's bounded symbolic content demoted and
    /// NOTHING written — when the fd carried `content_sym`. Also returns
    /// `false` — without demoting — when the fd is tracked but closed
    /// (angr-9ke6b.118), and when the write would grow the fd's content past
    /// `MAX_FS_FILE_SIZE` (angr-c7xno.67). The caller must convert any of
    /// these into its own Python-fallback error (never a hard/state-killing
    /// error). Zero-length writes are a no-demotion no-op (`true`).
    #[must_use = "false means the write was refused (closed fd, symbolic content demoted, or past MAX_FS_FILE_SIZE); bounce to Python"]
    pub fn write_fd(&mut self, fd: u32, data: &[u8]) -> bool {
        self.fs.write(fd, data)
    }

    /// Get a mutable reference to the file system state.
    pub fn file_system(&mut self) -> &mut FileSystem {
        &mut self.fs
    }

    /// Get a read-only reference to the file system state.
    pub fn file_system_ref(&self) -> &FileSystem {
        &self.fs
    }

    /// Record a symbolic variable that was read from stdin.
    /// Used by native fgets/fgetc/getchar to track stdin reads for posix.dumps(0).
    pub fn record_stdin_symbol(&mut self, name: String, bits: u32) {
        self.stdin_symbols.push((name, bits));
    }

    /// Get the list of symbolic variables read from stdin.
    /// Returns (name, bit_width) tuples in read order.
    pub fn stdin_symbols(&self) -> &[(String, u32)] {
        &self.stdin_symbols
    }

    /// Check if any stdin symbols have been recorded.
    pub fn has_stdin_symbols(&self) -> bool {
        !self.stdin_symbols.is_empty()
    }

    /// Get an environment variable value by key.
    pub fn getenv(&self, key: &[u8]) -> Option<&[u8]> {
        self.environment.get(key).map(std::vec::Vec::as_slice)
    }

    /// Set an environment variable.
    pub fn setenv(&mut self, key: Vec<u8>, value: Vec<u8>) {
        // No longer "removed since fork" if it was — see `removed_env_keys`.
        if self.removed_env_keys.contains(&key) {
            Arc::make_mut(&mut self.removed_env_keys).remove(&key);
        }
        Arc::make_mut(&mut self.environment).insert(key, value);
    }

    /// Remove an environment variable. Returns true if the key was present.
    pub fn unsetenv(&mut self, key: &[u8]) -> bool {
        let env = Arc::make_mut(&mut self.environment);
        if env.remove(key).is_some() {
            Arc::make_mut(&mut self.removed_env_keys).insert(key.to_vec());
            true
        } else {
            false
        }
    }

    /// Clear all environment variables.
    pub fn clearenv(&mut self) {
        if !self.environment.is_empty() {
            let cleared: Vec<Vec<u8>> = self.environment.keys().cloned().collect();
            Arc::make_mut(&mut self.environment).clear();
            Arc::make_mut(&mut self.removed_env_keys).extend(cleared);
        }
    }

    /// Get the environment map (for export).
    pub fn environment(&self) -> &HashMap<Vec<u8>, Vec<u8>> {
        &self.environment
    }

    /// Get the current heap brk pointer.
    pub fn heap_brk(&self) -> u64 {
        self.heap_brk
    }

    /// Set the heap brk (malloc bump allocator) pointer.
    ///
    /// **Invariant I7 (cross-mixin sync):** like `set_posix_brk`, both
    /// engines mutate the malloc bump allocator — Rust on native
    /// malloc/calloc/realloc/strdup/fopen (`heap_alloc`), Python on a
    /// fallback heap-allocating SimProcedure (which bumps
    /// `state.heap.heap_location`). The Python export path computes
    /// `max(rust_value, python_value)` so neither side hands out an address
    /// the other already allocated. This setter is the commanded path: it
    /// takes whatever value the caller (FFI cross-sync) supplies, without
    /// enforcing monotonicity locally. Regression test:
    /// `TestHeapBrkSync.test_export_path_syncs_rust_heap_brk_into_state_heap`.
    pub fn set_heap_brk(&mut self, addr: u64) {
        self.heap_brk = addr;
    }

    /// Get the POSIX brk pointer (mirrors `state.posix.brk` for the brk(2)
    /// syscall). Distinct from `heap_brk`, which is the malloc bump allocator.
    ///
    /// **Invariant I7 (cross-mixin sync):** see `set_posix_brk`.
    pub fn posix_brk(&self) -> u64 {
        self.posix_brk
    }

    /// Set the POSIX brk pointer.
    ///
    /// **Invariant I7 (cross-mixin sync):** Python and Rust both mutate
    /// `posix_brk` independently — Rust on native brk(2), Python on
    /// fallback syscall handlers and user mutation of `state.posix.brk`.
    /// The Python export path computes `max(rust_value, python_value)` so
    /// neither side is silently rewound by a stale value. This Rust setter
    /// is the commanded path: it takes whatever value the caller (native
    /// syscall handler or the FFI cross-sync) supplies, without enforcing
    /// monotonicity locally. Directionality lives at the Python export
    /// site. Regression test:
    /// `TestPosixBrkSync.test_export_path_syncs_rust_posix_brk_into_state_posix`.
    pub fn set_posix_brk(&mut self, addr: u64) {
        self.posix_brk = addr;
    }

    /// getopt(3) cursor pair `(optind, optchar)` — mirrors Python's
    /// `state.libc.getopt_optind` / `getopt_optchar`. Per-state, fork- and
    /// snapshot-carried. Consumed by the native getopt proc (bead
    /// angr-bhk0a.3).
    pub fn getopt_cursor(&self) -> (u32, u32) {
        (self.getopt_optind, self.getopt_optchar)
    }

    /// Set the getopt(3) cursor pair `(optind, optchar)`. See `getopt_cursor`.
    pub fn set_getopt_cursor(&mut self, optind: u32, optchar: u32) {
        self.getopt_optind = optind;
        self.getopt_optchar = optchar;
    }

    /// Get the mmap base pointer (mirrors `state.heap.mmap_base` for the
    /// mmap(2) syscall; advances when addr=0 native mmap allocates).
    ///
    /// **Invariant I7 (cross-mixin sync):** see `set_mmap_base`.
    pub fn mmap_base(&self) -> u64 {
        self.mmap_base
    }

    /// Set the mmap base pointer.
    ///
    /// **Invariant I7 (cross-mixin sync):** Python and Rust both mutate
    /// `mmap_base` independently — Rust on native mmap(2) with addr=0,
    /// Python on fallback syscall handlers and user mutation of
    /// `state.heap.mmap_base`. The Python export path computes
    /// `max(rust_value, python_value)` so a Python-side advance survives
    /// even if Rust still holds a stale lower value. This Rust setter is
    /// the commanded path: it takes whatever value the caller supplies,
    /// without enforcing monotonicity locally. Directionality lives at
    /// the Python export site. Regression test:
    /// `TestMmapBaseSync.test_export_path_does_not_clobber_higher_python_mmap_base`.
    pub fn set_mmap_base(&mut self, addr: u64) {
        self.mmap_base = addr;
    }

    /// Next value the `RDTSC` dirty helper will return for this state.
    ///
    /// Per-state rather than process-wide (angr-9ke6b.173): seeded into
    /// `VEXInterpreter::dirty_helper_state` by `run_interpreter_step_core`
    /// and written back by `apply_interpreter_step_result`, so the Nth RDTSC
    /// along a path is reproducible regardless of what other states — or
    /// other parallel workers — executed first.
    pub fn tsc_counter(&self) -> u64 {
        self.tsc_counter
    }

    /// Set the simulated timestamp counter. See [`Self::tsc_counter`].
    pub fn set_tsc_counter(&mut self, tsc: u64) {
        self.tsc_counter = tsc;
    }

    /// Locale ctype table pointers (see [`CtypeLocPtrs`]). Read by the native
    /// `__ctype_b_loc` / `__ctype_tolower_loc` / `__ctype_toupper_loc` procs.
    pub fn ctype_loc(&self) -> CtypeLocPtrs {
        self.ctype_loc
    }

    /// Push the locale ctype table pointers from Python's `__libc_start_main`
    /// init pass into Rust at seed-state creation.
    pub fn set_ctype_loc(&mut self, ptrs: CtypeLocPtrs) {
        self.ctype_loc = ptrs;
    }

    /// Guest addresses of the getopt(3) extern globals (see
    /// [`GetoptExternAddrs`]). Consumed by the native getopt proc (bhk0a.2).
    pub fn getopt_extern(&self) -> GetoptExternAddrs {
        self.getopt_extern
    }

    /// Push the loader-resolved getopt(3) extern-global addresses from Python
    /// into Rust at seed-state creation.
    pub fn set_getopt_extern(&mut self, addrs: GetoptExternAddrs) {
        self.getopt_extern = addrs;
    }

    /// Read-only view of the native sub-call resume stack (LIFO). Empty unless
    /// the S2 dispatcher has suspended a native proc mid sub-call. See
    /// [`NativeResumeFrame`].
    pub fn native_resume_stack(&self) -> &[NativeResumeFrame] {
        &self.native_resume_stack
    }

    /// Push a suspended-continuation frame onto the native sub-call resume
    /// stack. Used by the S2 dispatcher when a native proc calls a guest
    /// function and needs to resume afterwards.
    pub fn push_native_resume_frame(&mut self, frame: NativeResumeFrame) {
        self.native_resume_stack.push(frame);
    }

    /// Pop the most-recently-pushed continuation frame, or `None` if the stack
    /// is empty. Used by the S2 dispatcher at the resume sentinel.
    pub fn pop_native_resume_frame(&mut self) -> Option<NativeResumeFrame> {
        self.native_resume_stack.pop()
    }

    /// CGC `state.cgc.allocation_base` — current high-water bump pointer
    /// used by the native CGC `allocate(5)` syscall. Inert for non-CGC
    /// binaries. Mirror of `state_plugins/cgc.py::allocation_base`.
    pub fn cgc_allocation_base(&self) -> u64 {
        self.cgc_allocation_base
    }

    /// Update the CGC bump-pointer high-water. Called from the native
    /// `allocate` handler after a fresh bump and (in principle) from the
    /// Python→Rust state sync path when a Python-side allocate ran.
    pub fn set_cgc_allocation_base(&mut self, base: u64) {
        self.cgc_allocation_base = base;
    }

    /// CGC `state.cgc.sinkholes` — `(addr, length)` freelist of
    /// previously-deallocated regions, candidates for reuse by `allocate`.
    pub fn cgc_sinkholes(&self) -> &[(u64, u64)] {
        &self.cgc_sinkholes
    }

    /// Add a region to the CGC sinkhole freelist. Mirrors
    /// `SimStateCGC.add_sinkhole`. Duplicate `(addr, length)` pairs are
    /// silently merged into a single entry (the Python plugin uses a `set`).
    pub fn cgc_add_sinkhole(&mut self, addr: u64, length: u64) {
        if !self
            .cgc_sinkholes
            .iter()
            .any(|&(a, l)| a == addr && l == length)
        {
            self.cgc_sinkholes.push((addr, length));
        }
    }

    /// CGC first-fit allocator over the sinkhole freelist. Walks sinkholes
    /// in descending-address order (matching `SimStateCGC.get_max_sinkhole`)
    /// and returns the first one big enough to fit `length` bytes. The
    /// chosen sinkhole is split if larger than `length`: the leftover at
    /// the LOW end stays in the freelist, the HIGH end is returned.
    /// Returns `None` if no sinkhole fits — caller bumps `allocation_base`.
    pub fn cgc_take_max_sinkhole(&mut self, length: u64) -> Option<u64> {
        // Find index of highest-address sinkhole that fits.
        let mut best: Option<usize> = None;
        let mut best_addr: u64 = 0;
        for (i, &(addr, sz)) in self.cgc_sinkholes.iter().enumerate() {
            if sz >= length && (best.is_none() || addr > best_addr) {
                best = Some(i);
                best_addr = addr;
            }
        }
        let idx = best?;
        let (addr, sz) = self.cgc_sinkholes.swap_remove(idx);
        let remaining = sz - length;
        let chosen = addr + remaining;
        if remaining > 0 {
            self.cgc_sinkholes.push((addr, remaining));
        }
        Some(chosen)
    }

    /// Most recent symbolic value returned by the time(2) syscall, used to
    /// chain monotonic constraints across consecutive calls. Mirrors
    /// `state.globals['sys_last_time']` in Python's
    /// `procedures/linux_kernel/time.py`.
    pub fn last_time(&self) -> Option<&RustBV> {
        self.last_time.as_ref()
    }

    /// Record the most recent time(2) return value (called by the native time
    /// syscall handler).
    pub fn set_last_time(&mut self, bv: RustBV) {
        self.last_time = Some(bv);
    }

    /// Bump-allocate from the heap. Returns the address of the allocation.
    /// Aligns size up to 16 bytes (matching angr's SimHeapBrk).
    pub fn heap_alloc(&mut self, size: u64) -> u64 {
        let aligned = (size + 15) & !15; // round up to 16
        let addr = self.heap_brk;
        self.heap_brk = addr.wrapping_add(aligned);
        self.heap_metadata.record_alloc(addr, size);
        addr
    }

    /// Bump-allocate `size` bytes with the returned address aligned to
    /// `alignment` (must be a power of 2). The size bump is rounded up to 16,
    /// matching `heap_alloc`, so consecutive allocations remain aligned.
    /// Falls back to `heap_alloc` semantics when `alignment` is 0 or 1.
    pub fn heap_alloc_aligned(&mut self, size: u64, alignment: u64) -> u64 {
        if alignment <= 1 {
            return self.heap_alloc(size);
        }
        let mask = alignment - 1;
        let addr = self.heap_brk.wrapping_add(mask) & !mask;
        let aligned = (size + 15) & !15;
        self.heap_brk = addr.wrapping_add(aligned);
        self.heap_metadata.record_alloc(addr, size);
        addr
    }

    /// Record a heap free. Returns the original allocation size if tracked.
    pub fn heap_free(&mut self, addr: u64) -> Option<u64> {
        self.heap_metadata.record_free(addr)
    }

    /// Get heap metadata (for analysis/export).
    pub fn heap_metadata(&self) -> &HeapMetadata {
        &self.heap_metadata
    }
}
