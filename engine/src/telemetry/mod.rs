//! Stable instruments and read-only telemetry snapshots live in this module.

mod histogram;
mod recorder;
mod registry;

pub use recorder::{ControllerRecord, EpochSummary, Telemetry, TelemetrySnapshot};
pub use registry::{INSTRUMENTS, InstrumentDefinition};

#[cfg(test)]
mod tests;
