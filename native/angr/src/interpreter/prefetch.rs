//! Guest-page fetching and the prefetch heuristics layered on top of it.
//!
//! `fetch_page` / `fetch_pages_batch` pull page bytes across the Python
//! callback boundary; `fetch_page_with_prefetch` wraps them with the two
//! speculation policies — an eager list around the stack pointer
//! (`get_eager_prefetch_list` + `is_stack_region`) and a nearby-page window
//! sized by `set_page_prefetch_count` (`get_nearby_prefetch_list`) — so a
//! sequential access pattern pays one callback per batch instead of one per
//! page. The stack-pointer accessors the stack heuristic needs live here too,
//! including the `SILENT(cat-c)` `get_stack_pointer_or_log`.

use super::*;

impl<'a> VEXInterpreter<'a> {
    /// Set the number of pages to prefetch when fetching a page.
    ///
    /// When a page needs to be fetched from Python, this many additional pages
    /// will be fetched in each direction (before and after) to improve locality.
    /// Set to 0 to disable page prefetching.
    ///
    /// Production sets `page_prefetch_count` once at construction from
    /// `ExecutionConfig`; only `prefetch_tests` re-tunes it (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn set_page_prefetch_count(&mut self, count: u32) {
        self.page_prefetch_count = count;
    }

    /// Fetch a page from Python and map it in Rust memory.
    ///
    /// This is called when a load/store encounters an unmapped page in a lazy region.
    /// The page is fetched via Python callback and added to rust_memory.
    ///
    /// Returns true if the page was successfully fetched and mapped.
    pub(crate) fn fetch_page(
        &mut self,
        callbacks: &PythonCallbacks,
        page_addr: u64,
    ) -> Result<bool, CbExecutionError> {
        // Check if we have the callback
        if !callbacks.has_fetch_page() {
            return Ok(false);
        }

        // angr-gorvf.4.6: Python told us at setup which pages it can serve.
        // A page outside that set would come back declined, so skip the GIL
        // attach and decline it here.
        if !callbacks.python_can_serve_page(page_addr) {
            return Ok(false);
        }

        // Call Python to fetch the page
        let (data, permissions, is_mapped) = callbacks
            .call_fetch_page(page_addr)
            .map_err(|e| CbExecutionError::Callback(format!("fetch_page failed: {e}")))?;

        if !is_mapped {
            // Page doesn't exist in Python memory either
            return Ok(false);
        }

        // Ensure we have Rust memory enabled
        if let Some(ref mut rust_mem) = self.rust_memory {
            // Convert permission bits to Permission struct
            let perm = Permission::from_bits(permissions);

            // Map the page in Rust memory
            rust_mem.map_page(page_addr, data, perm);

            Ok(true)
        } else {
            // Rust memory not enabled - shouldn't happen but handle gracefully
            Ok(false)
        }
    }

    /// Fetch multiple pages from Python in a batch.
    ///
    /// This is more efficient than fetching pages one at a time.
    /// Returns the number of pages successfully fetched.
    pub(crate) fn fetch_pages_batch(
        &mut self,
        callbacks: &PythonCallbacks,
        page_addrs: &[u64],
    ) -> Result<usize, CbExecutionError> {
        if page_addrs.is_empty() {
            return Ok(0);
        }

        // Same presence guard as the single-page `fetch_page` above: with no
        // `fetch_page` callback there is nothing to fetch, so decline every
        // page instead of falling into `call_batch_fetch_pages`' unset-batch
        // fallback and hard-erroring on `fetch_page callback not set`. Keeping
        // this in step with `fetch_page` is what lets `PythonCallbacks::is_ready`
        // leave `fetch_page` out of its check — see its doc comment.
        if !callbacks.has_fetch_page() {
            return Ok(0);
        }

        // angr-gorvf.4.6: drop the pages Python told us at setup it cannot
        // serve. When nothing is left there is no crossing at all — this is
        // what takes the run-loop `batch_fetch_pages` GIL to a literal zero on
        // the benches whose every fetched page is a declined lazy-stack page.
        let servable: Vec<u64> = page_addrs
            .iter()
            .copied()
            .filter(|&pa| callbacks.python_can_serve_page(pa))
            .collect();
        if servable.is_empty() {
            return Ok(0);
        }

        // Call Python to fetch pages in batch
        let results = callbacks
            .call_batch_fetch_pages(&servable)
            .map_err(|e| CbExecutionError::Callback(format!("batch_fetch_pages failed: {e}")))?;

        let mut fetched = 0;

        if let Some(ref mut rust_mem) = self.rust_memory {
            // `zip` rather than `enumerate` + `servable[i]`: `call_list_batch`
            // already rejects a length mismatch (see its "one result per item
            // is a hard contract" section), and pairing positionally here means
            // a future callback path that skips that check truncates instead of
            // panicking the process — `panic = "abort"` (angr-03vl4.5).
            for ((data, permissions, is_mapped), &page_addr) in
                results.into_iter().zip(servable.iter())
            {
                if is_mapped {
                    let perm = Permission::from_bits(permissions);
                    rust_mem.map_page(page_addr, data, perm);
                    fetched += 1;
                }
            }
        }

        Ok(fetched)
    }

    /// Fetch a page and prefetch nearby pages for better locality.
    ///
    /// This is an optimization that reduces future FFI calls by speculatively
    /// fetching pages around the accessed address. Useful for sequential access
    /// patterns (like stack frames, arrays, etc.).
    ///
    /// When `enable_eager_prefetch` is set in config, this will fetch all
    /// unmapped pages in the lazy region containing the page. Otherwise,
    /// it fetches `prefetch_count` pages in each direction.
    ///
    /// Args:
    ///     prefetch_count: Number of pages to prefetch in each direction (0 = disabled)
    ///
    /// Returns true if the main page was successfully fetched.
    pub(crate) fn fetch_page_with_prefetch(
        &mut self,
        callbacks: &PythonCallbacks,
        page_addr: u64,
        prefetch_count: u32,
    ) -> Result<bool, CbExecutionError> {
        if prefetch_count == 0 && !self.config.enable_eager_prefetch {
            // No prefetching, just fetch the single page
            return self.fetch_page(callbacks, page_addr);
        }

        // Build list of pages to fetch
        let pages_to_fetch = if self.config.enable_eager_prefetch {
            // Eager region prefetch: fetch all unmapped pages in the region
            self.get_eager_prefetch_list(page_addr)
        } else {
            // Nearby prefetch: fetch pages before/after the trigger
            self.get_nearby_prefetch_list(page_addr, prefetch_count)
        };

        if pages_to_fetch.is_empty() {
            // No pages to fetch (shouldn't happen, but handle gracefully)
            return self.fetch_page(callbacks, page_addr);
        }

        // Fetch all pages in one batch
        let fetched = self.fetch_pages_batch(callbacks, &pages_to_fetch)?;

        // Return true if at least the main page was fetched
        if let Some(ref rust_mem) = self.rust_memory {
            Ok(rust_mem.is_mapped(page_addr))
        } else {
            Ok(fetched > 0)
        }
    }

    /// Get pages to fetch for eager region prefetch.
    ///
    /// Returns all unmapped pages in the lazy region containing `page_addr`,
    /// up to `max_prefetch_batch` pages.
    fn get_eager_prefetch_list(&self, page_addr: u64) -> Vec<u64> {
        if let Some(ref rust_mem) = self.rust_memory
            && let Some(pages) =
                rust_mem.get_region_prefetch_list(page_addr, self.config.max_prefetch_batch)
        {
            return pages;
        }
        // Fallback to just the main page
        vec![page_addr]
    }

    /// Get the current stack pointer value (architecture-aware).
    ///
    /// The inner value is `None` when the stack pointer is symbolic or
    /// otherwise unreadable; see [`AddrOrSymbolic`] for why the raw
    /// `Option<u64>` isn't returned directly.
    pub(crate) fn get_stack_pointer(&self) -> AddrOrSymbolic {
        let offset = self.registers.arch().sp_offset();
        self.registers.get_offset_u64(offset, self.ctx).into()
    }

    /// [`Self::get_stack_pointer`], substituting `0` and logging when the
    /// answer is symbolic/unavailable.
    ///
    /// `context` should identify the call site so the warning is actionable.
    // SILENT(cat-c): a symbolic/unreadable stack pointer collapsing to the
    // literal 0 is a wrong-answer risk — the value is exported verbatim as
    // `CallStackEntry.stack_ptr` (see `state/export.rs::get_call_stack`), so
    // downstream call-stack logic and Python callers cannot distinguish it
    // from a genuine SP of 0. Callers must go through this single logged
    // fallback rather than a bare `.unwrap_or(0)` (angr-sqfj8.63).
    pub(crate) fn get_stack_pointer_or_log(&self, context: &str) -> u64 {
        self.get_stack_pointer()
            .or_log("get_stack_pointer", context)
    }

    /// Check if an address is in the stack region (near current RSP).
    /// Stack typically grows downward, so we check if addr is below RSP + some margin.
    fn is_stack_region(&self, addr: u64) -> bool {
        if let Some(sp) = self.get_stack_pointer().concrete() {
            // Stack region: addresses from RSP - 1MB to RSP + 64KB
            // (stack grows down, but we allow some upward margin for locals)
            let stack_base = sp.saturating_sub(1024 * 1024); // 1MB below RSP
            let stack_limit = sp.saturating_add(64 * 1024); // 64KB above RSP
            addr >= stack_base && addr <= stack_limit
        } else {
            false
        }
    }

    /// Get pages to fetch for nearby prefetch (stack-aware).
    ///
    /// Returns `prefetch_count` unmapped pages, prioritizing stack growth direction
    /// (downward) when the access is in the stack region.
    fn get_nearby_prefetch_list(&self, page_addr: u64, prefetch_count: u32) -> Vec<u64> {
        let page_size = crate::memory::PAGE_SIZE;
        let mut pages_to_fetch = Vec::with_capacity(1 + 2 * prefetch_count as usize);

        // Add main page first
        pages_to_fetch.push(page_addr);

        // Determine if this is a stack access
        let is_stack = self.is_stack_region(page_addr);

        // For stack accesses, prioritize downward prefetch (stack grows down)
        // For non-stack, use balanced bidirectional prefetch
        let (down_count, up_count) = if is_stack {
            // Stack: 3x more pages downward than upward
            let down = (prefetch_count * 3).min(16);
            let up = prefetch_count.min(4);
            (down, up)
        } else {
            // Non-stack: equal in both directions
            (prefetch_count, prefetch_count)
        };

        // Add pages before (lower addresses - stack growth direction)
        for i in 1..=down_count {
            if let Some(addr) = page_addr.checked_sub(i as u64 * page_size) {
                // Check if not already mapped
                if let Some(ref rust_mem) = self.rust_memory
                    && !rust_mem.is_mapped(addr)
                    && rust_mem.is_addr_in_lazy_region(addr)
                {
                    pages_to_fetch.push(addr);
                }
            }
        }

        // Add pages after (higher addresses)
        for i in 1..=up_count {
            if let Some(addr) = page_addr.checked_add(i as u64 * page_size) {
                // Check if not already mapped
                if let Some(ref rust_mem) = self.rust_memory
                    && !rust_mem.is_mapped(addr)
                    && rust_mem.is_addr_in_lazy_region(addr)
                {
                    pages_to_fetch.push(addr);
                }
            }
        }

        pages_to_fetch
    }
}

test_submod!("prefetch_tests.rs" => prefetch_tests);
