//! Inspection / breakpoint event subsystem for `RustSimState`.

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
/// ring buffer of recent events. Designed for minimal overhead when
/// no inspections are registered (single bool check).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InspectionManager {
    /// Bitmask of enabled event types (bit N = InspectEvent with value N).
    enabled: u8,
    /// Ring buffer of recent events (capacity = max_events).
    events: Vec<InspectRecord>,
    /// Maximum number of events to retain (ring buffer capacity).
    max_events: usize,
    /// Total event count per type (never reset, for statistics).
    event_counts: [u64; InspectEvent::COUNT],
}

impl Default for InspectionManager {
    fn default() -> Self {
        InspectionManager {
            enabled: 0,
            events: Vec::new(),
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

    /// Set the maximum number of events to retain.
    pub fn set_max_events(&mut self, max: usize) {
        self.max_events = max;
        if self.events.len() > max {
            let drain = self.events.len() - max;
            self.events.drain(0..drain);
        }
    }

    /// Record an event. Only called when the event type is enabled.
    pub fn record(&mut self, event: InspectEvent, addr: u64, size: u32, block_addr: u64) {
        self.event_counts[event as usize] += 1;
        if self.events.len() >= self.max_events {
            self.events.remove(0);
        }
        self.events.push(InspectRecord {
            event,
            addr,
            size,
            block_addr,
        });
    }

    /// Get all recorded events.
    pub fn events(&self) -> &[InspectRecord] {
        &self.events
    }

    /// Get event counts per type.
    pub fn event_counts(&self) -> &[u64; InspectEvent::COUNT] {
        &self.event_counts
    }

    /// Get events filtered by type.
    pub fn events_of_type(&self, event: InspectEvent) -> Vec<&InspectRecord> {
        self.events.iter().filter(|e| e.event == event).collect()
    }

    /// Clear all recorded events (keeps enabled state and counts).
    pub fn clear_events(&mut self) {
        self.events.clear();
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
