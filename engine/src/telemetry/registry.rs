//! Stable instrument definitions shared by snapshots and documentation.

/// One versioned telemetry instrument definition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstrumentDefinition {
    pub id: &'static str,
    pub version: u16,
    pub unit: &'static str,
    pub description: &'static str,
}

/// The total registry of instrument identifiers named by the product contract.
pub const INSTRUMENTS: &[InstrumentDefinition] = &[
    instrument("bytes_per_sample", "bytes", "Measured sample payload size."),
    instrument(
        "nontensor_fraction",
        "ratio",
        "Fraction of non-tensor leaves.",
    ),
    instrument("t_shim", "nanoseconds", "Per-sample Python shim time."),
    instrument("t_seed", "nanoseconds", "Per-sample RNG install time."),
    instrument("hold", "events", "Consumer-held arena recycle events."),
    instrument(
        "growth_events",
        "events",
        "Arena or frontier growth events.",
    ),
    instrument(
        "view_export_copy",
        "bytes",
        "Bytes copied while exporting views.",
    ),
    instrument("overflow_events", "events", "Overflow-slab sample events."),
    instrument(
        "staged_copy_transients",
        "bytes",
        "Transient bytes copied before stable layout selection.",
    ),
    instrument(
        "stall_events",
        "events",
        "Delivery waits for unavailable work.",
    ),
    instrument(
        "hung_position",
        "position",
        "Position named by a liveness timeout.",
    ),
    instrument(
        "ceiling_binds",
        "events",
        "Controller decisions bound by user ceilings.",
    ),
    instrument(
        "gil_restore_events",
        "events",
        "Thread-tier GIL restoration events.",
    ),
    instrument(
        "delivery_rate",
        "samples_per_second",
        "Delivered sample rate.",
    ),
    instrument(
        "delivery_latency",
        "nanoseconds",
        "Event-sampled successful next-batch latency.",
    ),
    instrument(
        "startup",
        "nanoseconds",
        "Construction-to-first-delivery latency.",
    ),
    instrument(
        "controller_decisions",
        "events",
        "Inspectable controller decisions.",
    ),
];

const fn instrument(
    id: &'static str,
    unit: &'static str,
    description: &'static str,
) -> InstrumentDefinition {
    InstrumentDefinition {
        id,
        version: 1,
        unit,
        description,
    }
}
