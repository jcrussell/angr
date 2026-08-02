//! Inspection / breakpoint event subsystem for `RustSimState`.

use std::collections::VecDeque;

/// Types of inspection events that can be tracked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum InspectEvent {
    /// Memory read: (addr, size)
    MemRead = 0,
    /// Memory write: (addr, size)
    MemWrite = 1,
    /// Register read: (offset, size)
    RegRead = 2,
    /// Register write: (offset, size)
    RegWrite = 3,
    /// State fork (branch)
    Fork = 4,
    /// State exit/deadend
    Exit = 5,
}

impl InspectEvent {
    /// Number of event types.
    pub const COUNT: usize = 6;

    /// Convert from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(InspectEvent::MemRead),
            1 => Some(InspectEvent::MemWrite),
            2 => Some(InspectEvent::RegRead),
            3 => Some(InspectEvent::RegWrite),
            4 => Some(InspectEvent::Fork),
            5 => Some(InspectEvent::Exit),
            _ => None,
        }
    }

    /// Convert to string name.
    pub fn name(&self) -> &'static str {
        match self {
            InspectEvent::MemRead => "mem_read",
            InspectEvent::MemWrite => "mem_write",
            InspectEvent::RegRead => "reg_read",
            InspectEvent::RegWrite => "reg_write",
            InspectEvent::Fork => "fork",
            InspectEvent::Exit => "exit",
        }
    }
}

/// A recorded inspection event with address/offset and size.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InspectRecord {
    /// Event type.
    pub event: InspectEvent,
    /// Address (for mem events) or register offset (for reg events).
    pub addr: u64,
    /// Size in bytes.
    pub size: u32,
    /// Block address where the event occurred.
    pub block_addr: u64,
}

/// Inspection/breakpoint manager for state events.
///
/// Tracks which event types are enabled for logging and maintains a
/// ring buffer of the `max_events` most recent events. Designed for
/// minimal overhead when no inspections are registered (single bool
/// check); once the buffer is full, `record` evicts the oldest entry in
/// O(1) via the `VecDeque`'s head, so a hot mem_read/mem_write path pays
/// no per-event memmove.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InspectionManager {
    /// Bitmask of enabled event types (bit N = InspectEvent with value N).
    enabled: u8,
    /// Ring buffer of recent events (capacity = max_events). A `VecDeque`
    /// so eviction of the oldest event is O(1); serializes as a plain
    /// sequence, same wire shape as the `Vec` it replaced.
    events: VecDeque<InspectRecord>,
    /// Maximum number of events to retain (ring buffer capacity). Zero
    /// means "count events but retain none".
    max_events: usize,
    /// Total event count per type (never reset, for statistics).
    event_counts: [u64; InspectEvent::COUNT],
}

impl Default for InspectionManager {
    fn default() -> Self {
        InspectionManager {
            enabled: 0,
            events: VecDeque::new(),
            max_events: 1024,
            event_counts: [0; InspectEvent::COUNT],
        }
    }
}

impl InspectionManager {
    /// Check if any inspections are enabled. O(1).
    #[inline(always)]
    pub fn is_active(&self) -> bool {
        self.enabled != 0
    }

    /// Check if a specific event type is enabled.
    #[inline(always)]
    pub fn is_enabled(&self, event: InspectEvent) -> bool {
        self.enabled & (1 << event as u8) != 0
    }

    /// Enable tracking for an event type.
    pub fn enable(&mut self, event: InspectEvent) {
        self.enabled |= 1 << event as u8;
    }

    /// Disable tracking for an event type.
    pub fn disable(&mut self, event: InspectEvent) {
        self.enabled &= !(1 << event as u8);
    }

    /// Enable all event types.
    pub fn enable_all(&mut self) {
        self.enabled = (1 << InspectEvent::COUNT) - 1;
    }

    /// Disable all event types.
    pub fn disable_all(&mut self) {
        self.enabled = 0;
    }

    /// Set the maximum number of events to retain. Shrinking drops the
    /// oldest events so the newest `max` survive.
    pub fn set_max_events(&mut self, max: usize) {
        self.max_events = max;
        while self.events.len() > max {
            self.events.pop_front();
        }
    }

    /// Record an event. Only called when the event type is enabled.
    ///
    /// The per-type count always ticks; retention is capped at
    /// `max_events`, evicting the oldest entry in O(1).
    pub fn record(&mut self, event: InspectEvent, addr: u64, size: u32, block_addr: u64) {
        self.event_counts[event as usize] += 1;
        if self.max_events == 0 {
            return;
        }
        if self.events.len() >= self.max_events {
            self.events.pop_front();
        }
        self.events.push_back(InspectRecord {
            event,
            addr,
            size,
            block_addr,
        });
    }

    /// Get all recorded events, oldest first.
    pub fn events(&self) -> &VecDeque<InspectRecord> {
        &self.events
    }

    /// Get event counts per type.
    pub fn event_counts(&self) -> &[u64; InspectEvent::COUNT] {
        &self.event_counts
    }

    /// Get the enabled bitmask (for serialization).
    pub fn enabled_mask(&self) -> u8 {
        self.enabled
    }

    /// Set the enabled bitmask (for deserialization).
    pub fn set_enabled_mask(&mut self, mask: u8) {
        self.enabled = mask;
    }
}

use super::*;

impl RustSimState {
    /// Get the inspection manager (read-only).
    pub fn inspection(&self) -> &InspectionManager {
        &self.inspection
    }

    /// Get the inspection manager (mutable).
    pub fn inspection_mut(&mut self) -> &mut InspectionManager {
        &mut self.inspection
    }

    /// Record a memory read event (if mem_read inspection is enabled).
    #[inline(always)]
    pub fn inspect_mem_read(&mut self, addr: u64, size: u32) {
        if self.inspection.is_enabled(InspectEvent::MemRead) {
            self.inspection
                .record(InspectEvent::MemRead, addr, size, self.pc);
        }
    }

    /// Record a memory write event (if mem_write inspection is enabled).
    #[inline(always)]
    pub fn inspect_mem_write(&mut self, addr: u64, size: u32) {
        if self.inspection.is_enabled(InspectEvent::MemWrite) {
            self.inspection
                .record(InspectEvent::MemWrite, addr, size, self.pc);
        }
    }

    /// Record a fork event (if fork inspection is enabled).
    #[inline(always)]
    pub fn inspect_fork(&mut self) {
        if self.inspection.is_enabled(InspectEvent::Fork) {
            self.inspection.record(InspectEvent::Fork, 0, 0, self.pc);
        }
    }

    /// Record an exit event (if exit inspection is enabled).
    #[inline(always)]
    pub fn inspect_exit(&mut self) {
        if self.inspection.is_enabled(InspectEvent::Exit) {
            self.inspection.record(InspectEvent::Exit, 0, 0, self.pc);
        }
    }
}
