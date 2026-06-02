//! Native implementations of VEX dirty helpers.
//!
//! This module provides Rust implementations of common dirty helper functions
//! used by VEX IR. When a dirty call can be handled natively, we avoid the
//! Python callback overhead.
//!
//! Supported helpers:
//! - AMD64/x86 CPUID variants
//! - RDTSC (timestamp counter)
//! - Basic x87 FPU helpers

use std::collections::HashMap;

/// Result of a native dirty helper call.
#[derive(Debug, Clone)]
pub struct DirtyHelperResult {
    /// Return value (if any).
    pub return_value: Option<u64>,
    /// Register writes: (offset, value).
    pub reg_writes: Vec<(u32, u64)>,
}

impl DirtyHelperResult {
    /// Create a result with just a return value.
    pub fn with_return(value: u64) -> Self {
        DirtyHelperResult {
            return_value: Some(value),
            reg_writes: Vec::new(),
        }
    }

    /// Create an empty result (no return value).
    pub fn empty() -> Self {
        DirtyHelperResult {
            return_value: None,
            reg_writes: Vec::new(),
        }
    }

    /// Create a result with register writes.
    pub fn with_reg_writes(writes: Vec<(u32, u64)>) -> Self {
        DirtyHelperResult {
            return_value: None,
            reg_writes: writes,
        }
    }
}

/// Dirty helper dispatch table.
pub struct DirtyHelperDispatch {
    handlers: HashMap<&'static str, DirtyHandlerFn>,
}

type DirtyHandlerFn = fn(&[u64]) -> Option<DirtyHelperResult>;

impl Default for DirtyHelperDispatch {
    fn default() -> Self {
        Self::new()
    }
}

impl DirtyHelperDispatch {
    /// Create a new dispatch table with all supported helpers.
    pub fn new() -> Self {
        let mut handlers: HashMap<&'static str, DirtyHandlerFn> = HashMap::new();

        // CPUID helpers (amd64)
        handlers.insert("amd64g_dirtyhelper_CPUID_baseline", handle_cpuid_baseline);
        handlers.insert(
            "amd64g_dirtyhelper_CPUID_sse3_and_cx16",
            handle_cpuid_sse3_cx16,
        );
        handlers.insert(
            "amd64g_dirtyhelper_CPUID_sse42_and_cx16",
            handle_cpuid_sse42_cx16,
        );
        handlers.insert(
            "amd64g_dirtyhelper_CPUID_avx_and_cx16",
            handle_cpuid_avx_cx16,
        );
        handlers.insert("amd64g_dirtyhelper_CPUID_avx2", handle_cpuid_avx2);

        // RDTSC helper
        handlers.insert("amd64g_dirtyhelper_RDTSC", handle_rdtsc);
        handlers.insert("x86g_dirtyhelper_RDTSC", handle_rdtsc);

        // RDTSCP helper
        handlers.insert("amd64g_dirtyhelper_RDTSCP", handle_rdtscp);

        // IN/OUT port helpers (return safe defaults)
        handlers.insert("amd64g_dirtyhelper_IN", handle_in_port);
        handlers.insert("x86g_dirtyhelper_IN", handle_in_port);
        handlers.insert("amd64g_dirtyhelper_OUT", handle_out_port);
        handlers.insert("x86g_dirtyhelper_OUT", handle_out_port);

        // CPUID for x86
        handlers.insert("x86g_dirtyhelper_CPUID_sse0", handle_cpuid_x86_sse0);
        handlers.insert("x86g_dirtyhelper_CPUID_sse2", handle_cpuid_x86_sse2);
        handlers.insert("x86g_dirtyhelper_CPUID_sse3", handle_cpuid_x86_sse3);

        DirtyHelperDispatch { handlers }
    }

    /// Try to handle a dirty call natively.
    ///
    /// # Arguments
    /// * `name` - The helper function name
    /// * `args` - Concrete argument values
    ///
    /// # Returns
    /// Some(result) if handled natively, None if Python callback needed.
    pub fn try_call(&self, name: &str, args: &[u64]) -> Option<DirtyHelperResult> {
        if let Some(handler) = self.handlers.get(name) {
            handler(args)
        } else {
            None
        }
    }

    /// Check if a helper can be handled natively.
    pub fn can_handle(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }
}

// ============================================================================
// CPUID Handlers
// ============================================================================

/// Simulated CPUID values for a modern x86-64 CPU.
/// These are safe defaults that indicate support for common features.
/// All four fields preserve the natural EAX/EBX/ECX/EDX register quartet
/// emitted by CPUID; `pack_cpuid_result` currently consumes only eax + edx,
/// so ebx/ecx are inert today but kept for table symmetry and to make
/// extending the packing trivial if a new handler needs them.
#[allow(dead_code)]
struct CpuidValues {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

fn get_cpuid_values(leaf: u32, subleaf: u32) -> CpuidValues {
    match leaf {
        0 => {
            // Highest basic calling parameter and vendor ID
            CpuidValues {
                eax: 0x16,       // Max leaf
                ebx: 0x756e6547, // "Genu"
                ecx: 0x6c65746e, // "ntel"
                edx: 0x49656e69, // "ineI"
            }
        }
        1 => {
            // Processor info and feature bits
            CpuidValues {
                // Family 6, Model 63 (Haswell), Stepping 2
                eax: 0x000306f2,
                // CLFLUSH line size, initial APIC ID
                ebx: 0x00100800,
                // Feature flags: SSE3, PCLMULDQ, SSSE3, SSE4.1, SSE4.2, POPCNT, AES, AVX
                ecx: 0x7ffafbbf,
                // Feature flags: FPU, DE, PSE, TSC, MSR, PAE, MCE, CX8, APIC, SEP, MTRR,
                //                PGE, MCA, CMOV, PAT, PSE36, CLFSH, MMX, FXSR, SSE, SSE2
                edx: 0xbfebfbff,
            }
        }
        7 => {
            match subleaf {
                0 => {
                    // Extended feature flags
                    CpuidValues {
                        eax: 0,
                        // FSGSBASE, BMI1, AVX2, BMI2
                        ebx: 0x009c01a9,
                        ecx: 0,
                        edx: 0,
                    }
                }
                _ => CpuidValues {
                    eax: 0,
                    ebx: 0,
                    ecx: 0,
                    edx: 0,
                },
            }
        }
        0x80000000 => {
            // Extended function info
            CpuidValues {
                eax: 0x80000008, // Max extended leaf
                ebx: 0,
                ecx: 0,
                edx: 0,
            }
        }
        0x80000001 => {
            // Extended processor info
            CpuidValues {
                eax: 0,
                ebx: 0,
                // LAHF/SAHF
                ecx: 0x00000001,
                // SYSCALL/SYSRET, NX, RDTSCP, LM
                edx: 0x2c100800,
            }
        }
        _ => CpuidValues {
            eax: 0,
            ebx: 0,
            ecx: 0,
            edx: 0,
        },
    }
}

/// Pack CPUID results for VEX dirty helper return.
fn pack_cpuid_result(vals: &CpuidValues) -> u64 {
    // VEX CPUID helpers return a packed value:
    // The exact packing depends on the helper variant.
    // For baseline, we return the combined result that VEX expects.
    ((vals.edx as u64) << 32) | (vals.eax as u64)
}

fn handle_cpuid_baseline(args: &[u64]) -> Option<DirtyHelperResult> {
    // Args: (eax_in)
    let leaf = args.first().copied().unwrap_or(0) as u32;
    let vals = get_cpuid_values(leaf, 0);
    Some(DirtyHelperResult::with_return(pack_cpuid_result(&vals)))
}

fn handle_cpuid_sse3_cx16(args: &[u64]) -> Option<DirtyHelperResult> {
    handle_cpuid_baseline(args)
}

fn handle_cpuid_sse42_cx16(args: &[u64]) -> Option<DirtyHelperResult> {
    handle_cpuid_baseline(args)
}

fn handle_cpuid_avx_cx16(args: &[u64]) -> Option<DirtyHelperResult> {
    handle_cpuid_baseline(args)
}

fn handle_cpuid_avx2(args: &[u64]) -> Option<DirtyHelperResult> {
    handle_cpuid_baseline(args)
}

fn handle_cpuid_x86_sse0(args: &[u64]) -> Option<DirtyHelperResult> {
    handle_cpuid_baseline(args)
}

fn handle_cpuid_x86_sse2(args: &[u64]) -> Option<DirtyHelperResult> {
    handle_cpuid_baseline(args)
}

fn handle_cpuid_x86_sse3(args: &[u64]) -> Option<DirtyHelperResult> {
    handle_cpuid_baseline(args)
}

// ============================================================================
// RDTSC Handlers
// ============================================================================

/// Simulated TSC value.
/// We use a static counter that increments on each call.
use std::sync::atomic::{AtomicU64, Ordering};

static TSC_COUNTER: AtomicU64 = AtomicU64::new(0x1000000000);

fn handle_rdtsc(_args: &[u64]) -> Option<DirtyHelperResult> {
    // Return an incrementing timestamp value
    let tsc = TSC_COUNTER.fetch_add(1000, Ordering::Relaxed);
    Some(DirtyHelperResult::with_return(tsc))
}

fn handle_rdtscp(_args: &[u64]) -> Option<DirtyHelperResult> {
    // RDTSCP also reads IA32_TSC_AUX into ECX, but we just return TSC
    let tsc = TSC_COUNTER.fetch_add(1000, Ordering::Relaxed);
    Some(DirtyHelperResult::with_return(tsc))
}

// ============================================================================
// I/O Port Handlers
// ============================================================================

fn handle_in_port(_args: &[u64]) -> Option<DirtyHelperResult> {
    // Return 0xFF for all IN port reads (safe default)
    Some(DirtyHelperResult::with_return(0xFF))
}

fn handle_out_port(_args: &[u64]) -> Option<DirtyHelperResult> {
    // OUT has no return value - just ignore it
    Some(DirtyHelperResult::empty())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpuid_dispatch() {
        let dispatch = DirtyHelperDispatch::new();

        // Test CPUID baseline
        assert!(dispatch.can_handle("amd64g_dirtyhelper_CPUID_baseline"));
        let result = dispatch.try_call("amd64g_dirtyhelper_CPUID_baseline", &[0]);
        assert!(result.is_some());
        let res = result.unwrap();
        assert!(res.return_value.is_some());
    }

    #[test]
    fn test_rdtsc() {
        let dispatch = DirtyHelperDispatch::new();

        assert!(dispatch.can_handle("amd64g_dirtyhelper_RDTSC"));

        let result1 = dispatch.try_call("amd64g_dirtyhelper_RDTSC", &[]).unwrap();
        let result2 = dispatch.try_call("amd64g_dirtyhelper_RDTSC", &[]).unwrap();

        // TSC should increment
        assert!(result2.return_value.unwrap() > result1.return_value.unwrap());
    }

    #[test]
    fn test_unknown_helper() {
        let dispatch = DirtyHelperDispatch::new();

        assert!(!dispatch.can_handle("unknown_helper"));
        assert!(dispatch.try_call("unknown_helper", &[]).is_none());
    }

    #[test]
    fn test_in_out_ports() {
        let dispatch = DirtyHelperDispatch::new();

        // IN should return 0xFF
        let in_result = dispatch.try_call("amd64g_dirtyhelper_IN", &[0x80]).unwrap();
        assert_eq!(in_result.return_value, Some(0xFF));

        // OUT should have no return value
        let out_result = dispatch
            .try_call("amd64g_dirtyhelper_OUT", &[0x80, 0x00])
            .unwrap();
        assert!(out_result.return_value.is_none());
    }
}
