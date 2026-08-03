/// Icicle bindings
///
/// This module provides Python bindings for the Icicle emulator, allowing
/// interaction with the emulator's CPU, memory, and execution state.
///
/// This module is adapted from the `icicle-python` project, which can be found at:
/// https://github.com/icicle-emu/icicle-python
use std::{collections::HashMap, path::PathBuf, pin::Pin};

use icicle_fuzzing::coverage::register_afl_hit_counts_all;
use icicle_vm::{
    cpu::{
        Cpu, ValueSource,
        mem::{Mapping, perm},
    },
    injector::{PathTracerRef, add_path_tracer},
};

use pyo3::{
    exceptions::{PyKeyError, PyRuntimeError},
    prelude::*,
};
use send_wrapper::SendWrapper;
use target_lexicon::Architecture;

struct X86FlagsRegHandler {
    pub eflags: pcode::VarNode,
}

impl icicle_vm::cpu::RegHandler for X86FlagsRegHandler {
    fn read(&mut self, cpu: &mut Cpu) {
        let eflags = icicle_vm::x86::eflags(cpu);
        cpu.write_var::<u32>(self.eflags, eflags);
    }

    fn write(&mut self, cpu: &mut Cpu) {
        let eflags = cpu.read_var::<u32>(self.eflags);
        icicle_vm::x86::set_eflags(cpu, eflags);
    }
}

/// VmExit is the result of a VM execution. Borrowed directly from icicle.
#[pyclass(module = "angr.rustylib.icicle", from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmExit {
    /// The VM is still running.
    Running,
    /// The VM exited because it reached instruction count limit.
    InstructionLimit,
    /// The VM exited because it reached a breakpoint.
    Breakpoint,
    /// The VM exited because the interrupt flag was set.
    Interrupted,
    /// The VM has halted.
    Halt,
    /// Killed by an environment specific mechanism.
    Killed,
    /// A deadlock was detected.
    Deadlock,
    /// MMU was unable to allocate memory for an operation.
    OutOfMemory,
    /// Internal error where the emulator reached unimplemented code.
    Unimplemented,
    /// The VM exited due to a unhandled exception.
    UnhandledException,
}

#[pymethods]
impl VmExit {
    pub fn __eq__(&self, other: &Self) -> bool {
        *self == *other
    }
}

impl From<icicle_vm::VmExit> for VmExit {
    fn from(exit: icicle_vm::VmExit) -> Self {
        match exit {
            icicle_vm::VmExit::Running => VmExit::Running,
            icicle_vm::VmExit::InstructionLimit => VmExit::InstructionLimit,
            icicle_vm::VmExit::Breakpoint => VmExit::Breakpoint,
            icicle_vm::VmExit::Interrupted => VmExit::Interrupted,
            icicle_vm::VmExit::Halt => VmExit::Halt,
            icicle_vm::VmExit::Killed => VmExit::Killed,
            icicle_vm::VmExit::Deadlock => VmExit::Deadlock,
            icicle_vm::VmExit::OutOfMemory => VmExit::OutOfMemory,
            icicle_vm::VmExit::Unimplemented => VmExit::Unimplemented,
            icicle_vm::VmExit::UnhandledException(..) => VmExit::UnhandledException,
        }
    }
}

#[pyclass(module = "angr.rustylib.icicle", from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum ExceptionCode {
    #[pyo3(name = "NoException")]
    None,
    InstructionLimit,
    Halt,
    Sleep,
    SoftwareBreakpoint,
    Syscall,
    CpuStateChanged,
    DivisionException,
    ReadUnmapped,
    ReadPerm,
    ReadUnaligned,
    ReadWatch,
    ReadUninitialized,
    WriteUnmapped,
    WritePerm,
    WriteWatch,
    WriteUnaligned,
    ExecViolation,
    SelfModifyingCode,
    ExecUnaligned,
    OutOfMemory,
    AddressOverflow,
    InvalidInstruction,
    UnknownInterrupt,
    UnknownCpuID,
    InvalidOpSize,
    InvalidFloatSize,
    CodeNotTranslated,
    ShadowStackOverflow,
    ShadowStackInvalid,
    InvalidTarget,
    UnimplementedOp,
    ExternalAddr,
    Environment,
    JitError,
    InternalError,
    UnmappedRegister,
    UnknownError,
}

impl ExceptionCode {
    pub(crate) fn from_code(code: u32) -> Self {
        icicle_vm::cpu::ExceptionCode::from_u32(code).into()
    }
}

#[pymethods]
impl ExceptionCode {
    pub fn __eq__(&self, other: &Self) -> bool {
        *self == *other
    }
}

impl From<icicle_vm::cpu::ExceptionCode> for ExceptionCode {
    fn from(value: icicle_vm::cpu::ExceptionCode) -> Self {
        use icicle_vm::cpu::ExceptionCode::*;
        match value {
            None => ExceptionCode::None,
            InstructionLimit => ExceptionCode::InstructionLimit,
            Halt => ExceptionCode::Halt,
            Sleep => ExceptionCode::Sleep,
            SoftwareBreakpoint => ExceptionCode::SoftwareBreakpoint,
            Syscall => ExceptionCode::Syscall,
            CpuStateChanged => ExceptionCode::CpuStateChanged,
            DivisionException => ExceptionCode::DivisionException,
            ReadUnmapped => ExceptionCode::ReadUnmapped,
            ReadPerm => ExceptionCode::ReadPerm,
            ReadUnaligned => ExceptionCode::ReadUnaligned,
            ReadWatch => ExceptionCode::ReadWatch,
            ReadUninitialized => ExceptionCode::ReadUninitialized,
            WriteUnmapped => ExceptionCode::WriteUnmapped,
            WritePerm => ExceptionCode::WritePerm,
            WriteWatch => ExceptionCode::WriteWatch,
            WriteUnaligned => ExceptionCode::WriteUnaligned,
            ExecViolation => ExceptionCode::ExecViolation,
            SelfModifyingCode => ExceptionCode::SelfModifyingCode,
            ExecUnaligned => ExceptionCode::ExecUnaligned,
            OutOfMemory => ExceptionCode::OutOfMemory,
            AddressOverflow => ExceptionCode::AddressOverflow,
            InvalidInstruction => ExceptionCode::InvalidInstruction,
            UnknownInterrupt => ExceptionCode::UnknownInterrupt,
            UnknownCpuID => ExceptionCode::UnknownCpuID,
            InvalidOpSize => ExceptionCode::InvalidOpSize,
            InvalidFloatSize => ExceptionCode::InvalidFloatSize,
            CodeNotTranslated => ExceptionCode::CodeNotTranslated,
            ShadowStackOverflow => ExceptionCode::ShadowStackOverflow,
            ShadowStackInvalid => ExceptionCode::ShadowStackInvalid,
            InvalidTarget => ExceptionCode::InvalidTarget,
            UnimplementedOp => ExceptionCode::UnimplementedOp,
            ExternalAddr => ExceptionCode::ExternalAddr,
            Environment => ExceptionCode::Environment,
            JitError => ExceptionCode::JitError,
            InternalError => ExceptionCode::InternalError,
            UnmappedRegister => ExceptionCode::UnmappedRegister,
            UnknownError => ExceptionCode::UnknownError,
        }
    }
}

struct Hitmap {
    inner: Pin<Box<[u8]>>,
    _pin: std::marker::PhantomPinned,
}

impl Hitmap {
    pub(crate) fn new(size: usize) -> Self {
        let hitmap = Pin::from(vec![0u8; size].into_boxed_slice());
        Hitmap {
            inner: hitmap,
            _pin: std::marker::PhantomPinned,
        }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.inner.as_mut_ptr()
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.inner
    }

    pub(crate) fn as_slice_mut(&mut self) -> &mut [u8] {
        &mut self.inner
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass(unsendable, module = "angr.rustylib.icicle")]
struct Icicle {
    #[pyo3(get)]
    architecture: String,
    vm: icicle_vm::Vm,
    path_tracer: Option<PathTracerRef>,
    edge_count_hitmap: Option<Hitmap>,
    snapshot: Option<icicle_vm::Snapshot>,
}

#[pymethods]
impl Icicle {
    #[new]
    pub(crate) fn new(
        architecture: String,
        processors_path: String,
        enable_tracing: bool,
        enable_edge_count: bool,
    ) -> PyResult<Self> {
        let mut config =
            icicle_vm::cpu::Config::from_target_triple(format!("{architecture}-none").as_str());
        config.enable_shadow_stack = false;
        let mut vm = icicle_vm::build_with_path(&config, &PathBuf::from(processors_path))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build VM: {e}")))?;

        // Populate the lowercase register map
        let mut regs = HashMap::new();
        let sleigh = &vm.cpu.arch.sleigh;
        for reg in &sleigh.named_registers {
            let name = sleigh.get_str(reg.name);
            regs.insert(name.to_lowercase(), reg.clone());
        }

        // Special handling for x86 flags
        match config.triple.architecture {
            Architecture::X86_32(_) | Architecture::X86_64 | Architecture::X86_64h => {
                let eflags = sleigh
                    .get_reg("eflags")
                    .ok_or(PyKeyError::new_err(
                        "Could not find eflags register in the architecture",
                    ))?
                    .get_var()
                    .ok_or(PyKeyError::new_err(
                        "Eflags register does not have a variable node",
                    ))?;
                let reg_handler = X86FlagsRegHandler { eflags };
                vm.cpu.add_reg_handler(eflags.id, Box::new(reg_handler));
            }
            _ => {}
        }

        let path_tracer =
            if enable_tracing {
                Some(add_path_tracer(&mut vm).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to add path tracer: {e}"))
                })?)
            } else {
                None
            };

        let edge_count_hitmap = if enable_edge_count {
            let mut hitmap = Hitmap::new(65536);
            register_afl_hit_counts_all(&mut vm, hitmap.as_mut_ptr(), hitmap.len() as u32);
            Some(hitmap)
        } else {
            None
        };

        Ok(Self {
            architecture,
            vm,
            path_tracer,
            edge_count_hitmap,
            snapshot: None,
        })
    }

    // Basic state accessors

    pub(crate) fn reg_read(&mut self, name: String) -> PyResult<u64> {
        Ok(self.vm.cpu.read_reg(get_reg_varnode(&self.vm, &name)?))
    }

    pub(crate) fn reg_write(&mut self, reg: String, value: u64) -> PyResult<()> {
        self.vm
            .cpu
            .write_reg(get_reg_varnode(&self.vm, &reg)?, value);
        Ok(())
    }

    pub(crate) fn mem_map(&mut self, addr: u64, size: u64, perm: u8) -> PyResult<()> {
        if !self.vm.cpu.mem.map_memory_len(
            addr,
            size,
            Mapping {
                perm: perms_to_icicle(perm),
                value: 0,
            },
        ) {
            return Err(PyRuntimeError::new_err(format!(
                "Failed to map memory at {addr:#x} with size {size}"
            )));
        }
        Ok(())
    }

    pub(crate) fn mem_unmap(&mut self, addr: u64, size: u64) -> PyResult<()> {
        self.invalidate_code_range(addr, size);
        if !self.vm.cpu.mem.unmap_memory_len(addr, size) {
            return Err(PyRuntimeError::new_err(format!(
                "Failed to unmap memory at {addr:#x} with size {size}"
            )));
        }
        Ok(())
    }

    pub(crate) fn mem_protect(&mut self, addr: u64, size: u64, perms: u8) -> PyResult<()> {
        self.invalidate_code_range(addr, size);
        self.vm
            .cpu
            .mem
            .update_perm(addr, size, perms_to_icicle(perms))
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to protect memory at {addr:#x} with size {size}: {e}"
                ))
            })?;
        Ok(())
    }

    pub(crate) fn mem_read(&mut self, addr: u64, size: u64) -> PyResult<Vec<u8>> {
        let mut buf = vec![0; size as usize];
        self.vm
            .cpu
            .mem
            .read_bytes(addr, &mut buf, perm::NONE)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read memory: {e}")))?;
        Ok(buf)
    }

    pub(crate) fn mem_write(&mut self, addr: u64, data: Vec<u8>) -> PyResult<()> {
        self.invalidate_code_range(addr, data.len() as u64);
        // The cache invalidation above makes this write safe; suppress the
        // mmu's SMC guard for just this call so it doesn't reject sync
        // writes into bytes that were previously executed. Guest writes
        // during `emu.run()` still see the guard.
        let prev_smc = self.vm.cpu.mem.detect_self_modifying_code;
        self.vm.cpu.mem.detect_self_modifying_code = false;
        let result = self
            .vm
            .cpu
            .mem
            .write_bytes(addr, &data, perm::NONE)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write memory: {e}")));
        self.vm.cpu.mem.detect_self_modifying_code = prev_smc;
        result
    }

    /// Invalidate any lifted/JIT code whose extent overlaps `[addr, addr+size)`.
    ///
    /// Removes affected entries from the block map and drops their JIT
    /// compilations so subsequent execution re-lifts from current memory.
    /// Also clears the cached per-address disassembly strings — the lifter
    /// compares freshly-lifted disassembly against this cache and raises
    /// `SelfModifyingCode` on mismatch, which would otherwise fire whenever
    /// the newly-written instruction differs from the one previously at the
    /// same address.
    fn invalidate_code_range(&mut self, addr: u64, size: u64) {
        let Some((addr, end)) = written_range(addr, size) else {
            return;
        };

        let mut affected_keys = Vec::new();
        let mut affected_blocks = Vec::new();
        for (key, group) in &self.vm.code.map {
            if code_group_overlaps(group.start, group.end, addr, end) {
                affected_keys.push(*key);
                affected_blocks.extend(group.range());
            }
        }

        for key in affected_keys {
            self.vm.code.map.remove(&key);
        }
        for id in affected_blocks {
            self.vm.jit.invalidate(id);
        }
        self.vm
            .code
            .disasm
            .retain(|&vaddr, _| disasm_addr_retained(vaddr, addr, end));
    }

    // Specialized state accessors

    #[getter]
    pub(crate) fn get_pc(&self) -> u64 {
        self.vm.cpu.read_pc()
    }

    #[setter]
    pub(crate) fn set_pc(&mut self, pc: u64) -> PyResult<()> {
        self.vm.cpu.write_pc(pc);
        Ok(())
    }

    #[getter]
    pub(crate) fn get_isa_mode(&self) -> u8 {
        self.vm.cpu.isa_mode()
    }

    #[setter]
    pub(crate) fn set_isa_mode(&mut self, mode: u8) {
        // https://github.com/icicle-emu/icicle-emu/issues/70#issuecomment-2857265222
        self.vm.cpu.set_isa_mode(mode);
        let _ = self.set_pc(self.get_pc());
    }

    // Execution

    pub(crate) fn add_breakpoint(&mut self, addr: u64) -> bool {
        self.vm.add_breakpoint(addr)
    }

    pub(crate) fn remove_breakpoint(&mut self, addr: u64) -> PyResult<()> {
        if !self.vm.remove_breakpoint(addr) {
            return Err(PyRuntimeError::new_err(format!(
                "Failed to remove breakpoint at {addr:#x}"
            )));
        }
        Ok(())
    }

    #[setter]
    pub(crate) fn set_icount_limit(&mut self, limit: u64) {
        self.vm.icount_limit = limit;
    }

    #[getter]
    pub(crate) fn get_icount_limit(&self) -> u64 {
        self.vm.icount_limit
    }

    #[getter]
    pub(crate) fn get_cpu_icount(&self) -> u64 {
        self.vm.cpu.icount
    }

    pub(crate) fn run(&mut self, py: Python) -> VmExit {
        // By calling `py.detach`, we allow Python to release the GIL and
        // allow other threads to run while the VM is executing. This allows
        // using multiple engines in parallel within a single Python process.
        let mut wrapped = SendWrapper::new(&mut self.vm);
        py.detach(|| (*wrapped).run().into())
    }

    #[getter]
    pub(crate) fn get_exception_code(&self) -> ExceptionCode {
        ExceptionCode::from_code(self.vm.cpu.exception.code)
    }

    #[getter]
    pub(crate) fn get_exception_value(&self) -> u64 {
        self.vm.cpu.exception.value
    }

    // Tracing

    #[getter]
    pub(crate) fn get_recent_blocks(&mut self) -> Vec<(u64, u64)> {
        if let Some(path_tracer) = self.path_tracer {
            path_tracer.get_last_blocks(&mut self.vm)
        } else {
            Vec::new()
        }
    }

    #[getter]
    pub(crate) fn get_edge_hitmap(&mut self) -> Option<&[u8]> {
        self.edge_count_hitmap
            .as_ref()
            .map(|hitmap| hitmap.as_slice())
    }

    #[setter]
    pub(crate) fn set_edge_hitmap(&mut self, new_hitmap: &[u8]) -> PyResult<()> {
        if let Some(hitmap) = &mut self.edge_count_hitmap {
            if hitmap.len() != new_hitmap.len() {
                return Err(PyRuntimeError::new_err("Hitmap size mismatch"));
            }
            hitmap.as_slice_mut().copy_from_slice(new_hitmap);
        } else {
            return Err(PyRuntimeError::new_err("Edge hitmap is not enabled"));
        }
        Ok(())
    }

    // Snapshot/restore

    pub(crate) fn save_snapshot(&mut self) {
        self.snapshot = Some(self.vm.snapshot());
    }

    pub(crate) fn restore_snapshot(&mut self) -> PyResult<()> {
        let snapshot = self
            .snapshot
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("No snapshot saved"))?;
        // Code/JIT/ISA-mode caches aren't in the snapshot; reset clears
        // them (and memory), then restore puts memory back.
        self.vm.reset();
        self.vm.restore(snapshot);
        if let Some(path_tracer) = self.path_tracer {
            path_tracer.clear(&mut self.vm);
        }
        if let Some(hitmap) = &mut self.edge_count_hitmap {
            hitmap.as_slice_mut().fill(0);
        }
        Ok(())
    }

    pub(crate) fn has_snapshot(&self) -> bool {
        self.snapshot.is_some()
    }

    pub(crate) fn clear_path_tracer(&mut self) {
        if let Some(path_tracer) = self.path_tracer {
            path_tracer.clear(&mut self.vm);
        }
    }

    // Dirty page tracking

    /// Get the set of page-aligned virtual addresses that have been modified
    /// since the last call to reset_page_modification_tracking.
    #[getter]
    pub(crate) fn get_modified_pages(&self) -> Vec<u64> {
        self.vm.cpu.mem.modified.iter().copied().collect()
    }

    /// Reset page modification tracking so that only writes occurring after
    /// this call are recorded.  For each given page address, the per-page
    /// `modified` flag on the underlying physical page is cleared.  Then the
    /// global modified-address set and TLB write cache are flushed.
    pub(crate) fn reset_page_modification_tracking(&mut self, page_addresses: Vec<u64>) {
        for addr in page_addresses {
            if let Some(index) = self.vm.cpu.mem.get_physical_index(addr) {
                self.vm.cpu.mem.get_physical_mut(index).modified = false;
            }
        }
        self.vm.cpu.mem.clear_page_modification_log();
    }
}

fn get_reg_varnode(vm: &icicle_vm::Vm, name: &str) -> PyResult<pcode::VarNode> {
    // Try original name first, then uppercase for case-insensitive matching
    let lookup = vm
        .cpu
        .arch
        .sleigh
        .get_reg(name)
        .or_else(|| vm.cpu.arch.sleigh.get_reg(&name.to_uppercase()));
    let reg =
        lookup.ok_or_else(|| PyKeyError::new_err(format!("Could not find register {name}")))?;
    reg.get_var().ok_or(PyKeyError::new_err(format!(
        "Register {name} does not have a variable node"
    )))
}

/// Converts a permission byte to an icicle permission byte.
///
/// The input follows the cle/angr convention — bit 0 read, bit 1 write, bit 2
/// exec — so the three literals below are written at a uniform 3-bit width to
/// keep the bit positions countable at a glance.
fn perms_to_icicle(perm: u8) -> u8 {
    let mut icicle_perm = perm::INIT; // Always mark as initialized
    if perm & 0b001 != 0 {
        icicle_perm |= perm::READ;
    }
    if perm & 0b010 != 0 {
        icicle_perm |= perm::WRITE;
    }
    if perm & 0b100 != 0 {
        icicle_perm |= perm::EXEC;
    }
    icicle_perm
}

/// The half-open range `[addr, addr + size)` a write touches, or `None` for an
/// empty write. `size` is clamped with `saturating_add` so a write running off
/// the top of the address space ends at `u64::MAX` instead of wrapping to 0 and
/// silently invalidating nothing.
fn written_range(addr: u64, size: u64) -> Option<(u64, u64)> {
    if size == 0 {
        return None;
    }
    Some((addr, addr.saturating_add(size)))
}

/// Does a lifted code group overlap the written range?
///
/// A group covers the **inclusive** byte range `[group_start, group_end]`,
/// while the written range is **half-open** `[addr, end)` (`end` is
/// `addr.saturating_add(size)` and is never touched by the write). Mixing the
/// two conventions is where off-by-ones hide, so the predicate lives here with
/// its own tests rather than inline in `invalidate_code_range`.
fn code_group_overlaps(group_start: u64, group_end: u64, addr: u64, end: u64) -> bool {
    group_start < end && group_end >= addr
}

/// Should a cached disassembly entry at `vaddr` survive a write to `[addr, end)`?
///
/// Entries are keyed by the address of a single instruction, so only an entry
/// *inside* the half-open written range is stale.
fn disasm_addr_retained(vaddr: u64, addr: u64, end: u64) -> bool {
    vaddr < addr || vaddr >= end
}

#[cfg(test)]
#[path = "icicle_tests.rs"]
mod tests;

#[pymodule]
pub(crate) fn icicle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VmExit>()?;
    m.add_class::<ExceptionCode>()?;
    m.add_class::<Icicle>()?;
    Ok(())
}
