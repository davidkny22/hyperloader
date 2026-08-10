//! Atomic hot-path counters and event-driven summary storage.

use super::histogram::LatencyHistogram;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// One inspectable controller decision.
#[derive(Clone, Debug, PartialEq)]
pub struct ControllerRecord {
    pub previous_width: u32,
    pub width: u32,
    pub reason: String,
    pub starvation: bool,
    pub binding: Option<String>,
    pub resource_loss: f64,
}

/// One completed epoch's bounded summary.
#[derive(Clone, Debug, PartialEq)]
pub struct EpochSummary {
    pub epoch: u64,
    pub delivered_samples: u64,
    pub delivered_batches: u64,
    pub delivered_bytes: u64,
    pub delivery_interval_ns: u64,
    pub delivery_latency_ns: [u64; 3],
    pub stall_events: u64,
    pub controller_decisions: Vec<ControllerRecord>,
    pub ceiling_binds: u64,
}

/// A read-only current snapshot plus the most recently completed epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct TelemetrySnapshot {
    pub startup_ns: u64,
    pub current: EpochSummary,
    pub last_epoch: Option<EpochSummary>,
}

/// Lock-free hot-path metrics with a cadence-only controller event lock.
pub struct Telemetry {
    startup_ns: AtomicU64,
    delivered_samples: AtomicU64,
    delivered_batches: AtomicU64,
    delivered_bytes: AtomicU64,
    delivery_interval_ns: AtomicU64,
    delivery_latency: LatencyHistogram,
    stall_events: AtomicU64,
    ceiling_binds: AtomicU64,
    controller_decisions: Mutex<Vec<ControllerRecord>>,
    last_epoch: Mutex<Option<EpochSummary>>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl Telemetry {
    /// Construct an empty recorder.
    pub fn new() -> Self {
        Self {
            startup_ns: AtomicU64::new(0),
            delivered_samples: AtomicU64::new(0),
            delivered_batches: AtomicU64::new(0),
            delivered_bytes: AtomicU64::new(0),
            delivery_interval_ns: AtomicU64::new(0),
            delivery_latency: LatencyHistogram::new(),
            stall_events: AtomicU64::new(0),
            ceiling_binds: AtomicU64::new(0),
            controller_decisions: Mutex::new(Vec::new()),
            last_epoch: Mutex::new(None),
        }
    }

    /// Record the construction-to-first-delivery duration once.
    pub fn record_startup(&self, nanoseconds: u64) {
        let _ = self.startup_ns.compare_exchange(
            0,
            nanoseconds.max(1),
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    /// Record one successful delivery using atomic hot-path operations.
    pub fn record_delivery(&self, samples: u64, bytes: u64, latency_ns: u64, interval_ns: u64) {
        self.record_deliveries(samples, 1, bytes, latency_ns, interval_ns);
    }

    /// Record one event-sampled group of successful deliveries.
    pub fn record_deliveries(
        &self,
        samples: u64,
        batches: u64,
        bytes: u64,
        latency_ns: u64,
        interval_ns: u64,
    ) {
        self.delivered_samples.fetch_add(samples, Ordering::Relaxed);
        self.delivered_batches.fetch_add(batches, Ordering::Relaxed);
        self.delivered_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.delivery_interval_ns
            .fetch_add(interval_ns, Ordering::Relaxed);
        self.delivery_latency.observe(latency_ns);
    }

    /// Record exact delivery counters without adding a latency sample.
    pub fn record_counts(&self, samples: u64, batches: u64, bytes: u64, interval_ns: u64) {
        self.delivered_samples.fetch_add(samples, Ordering::Relaxed);
        self.delivered_batches.fetch_add(batches, Ordering::Relaxed);
        self.delivered_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.delivery_interval_ns
            .fetch_add(interval_ns, Ordering::Relaxed);
    }

    /// Record one event-driven delivery stall.
    pub fn record_stall(&self) {
        self.stall_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one low-cadence controller decision.
    pub fn record_controller(&self, decision: ControllerRecord) {
        if decision.binding.is_some() {
            self.ceiling_binds.fetch_add(1, Ordering::Relaxed);
        }
        self.controller_decisions
            .lock()
            .expect("controller telemetry mutex poisoned")
            .push(decision);
    }

    /// Seal the current counters as the latest epoch summary and reset them.
    pub fn finish_epoch(&self, epoch: u64) {
        let summary = self.current_summary(epoch);
        *self
            .last_epoch
            .lock()
            .expect("epoch telemetry mutex poisoned") = Some(summary);
        self.reset_epoch();
    }

    /// Return a bounded point-in-time view.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            startup_ns: self.startup_ns.load(Ordering::Relaxed),
            current: self.current_summary(0),
            last_epoch: self
                .last_epoch
                .lock()
                .expect("epoch telemetry mutex poisoned")
                .clone(),
        }
    }

    fn current_summary(&self, epoch: u64) -> EpochSummary {
        EpochSummary {
            epoch,
            delivered_samples: self.delivered_samples.load(Ordering::Relaxed),
            delivered_batches: self.delivered_batches.load(Ordering::Relaxed),
            delivered_bytes: self.delivered_bytes.load(Ordering::Relaxed),
            delivery_interval_ns: self.delivery_interval_ns.load(Ordering::Relaxed),
            delivery_latency_ns: self.delivery_latency.percentiles(),
            stall_events: self.stall_events.load(Ordering::Relaxed),
            controller_decisions: self
                .controller_decisions
                .lock()
                .expect("controller telemetry mutex poisoned")
                .clone(),
            ceiling_binds: self.ceiling_binds.load(Ordering::Relaxed),
        }
    }

    fn reset_epoch(&self) {
        self.delivered_samples.store(0, Ordering::Relaxed);
        self.delivered_batches.store(0, Ordering::Relaxed);
        self.delivered_bytes.store(0, Ordering::Relaxed);
        self.delivery_interval_ns.store(0, Ordering::Relaxed);
        self.delivery_latency.reset();
        self.stall_events.store(0, Ordering::Relaxed);
        self.ceiling_binds.store(0, Ordering::Relaxed);
        self.controller_decisions
            .lock()
            .expect("controller telemetry mutex poisoned")
            .clear();
    }
}
