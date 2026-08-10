//! Python binding for native atomic telemetry.

use crate::telemetry::{ControllerRecord, EpochSummary, INSTRUMENTS, Telemetry};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::time::Instant;

#[pyclass(name = "_Telemetry")]
pub(crate) struct PyTelemetry {
    recorder: Telemetry,
    constructed_at: Instant,
    last_delivery_at: Instant,
}

#[pymethods]
impl PyTelemetry {
    #[new]
    fn new() -> Self {
        let now = Instant::now();
        Self {
            recorder: Telemetry::new(),
            constructed_at: now,
            last_delivery_at: now,
        }
    }

    #[staticmethod]
    fn registry<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let entries = PyList::empty(py);
        for definition in INSTRUMENTS {
            let entry = PyDict::new(py);
            entry.set_item("description", definition.description)?;
            entry.set_item("id", definition.id)?;
            entry.set_item("unit", definition.unit)?;
            entry.set_item("version", definition.version)?;
            entries.append(entry)?;
        }
        Ok(entries)
    }

    fn record_startup(&self, nanoseconds: u64) {
        self.recorder.record_startup(nanoseconds);
    }

    fn record_startup_now(&mut self) {
        let now = Instant::now();
        self.recorder
            .record_startup(duration_ns(now.duration_since(self.constructed_at)));
        self.last_delivery_at = now;
    }

    fn record_delivery(&mut self, samples: u64, bytes: u64, latency_ns: u64) {
        self.record_deliveries(samples, 1, bytes, latency_ns);
    }

    fn record_deliveries(&mut self, samples: u64, batches: u64, bytes: u64, latency_ns: u64) {
        let delivered_at = Instant::now();
        self.recorder.record_startup(duration_ns(
            delivered_at.duration_since(self.constructed_at),
        ));
        self.recorder.record_deliveries(
            samples,
            batches,
            bytes,
            latency_ns,
            duration_ns(delivered_at.duration_since(self.last_delivery_at)),
        );
        self.last_delivery_at = delivered_at;
    }

    fn record_counts(&mut self, samples: u64, batches: u64, bytes: u64) {
        let delivered_at = Instant::now();
        self.recorder.record_startup(duration_ns(
            delivered_at.duration_since(self.constructed_at),
        ));
        self.recorder.record_counts(
            samples,
            batches,
            bytes,
            duration_ns(delivered_at.duration_since(self.last_delivery_at)),
        );
        self.last_delivery_at = delivered_at;
    }

    fn record_stall(&self) {
        self.recorder.record_stall();
    }

    #[pyo3(signature = (previous_width, width, reason, starvation, resource_loss, binding=None))]
    fn record_controller(
        &self,
        previous_width: u32,
        width: u32,
        reason: String,
        starvation: bool,
        resource_loss: f64,
        binding: Option<String>,
    ) {
        self.recorder.record_controller(ControllerRecord {
            previous_width,
            width,
            reason,
            starvation,
            binding,
            resource_loss,
        });
    }

    fn finish_epoch(&self, epoch: u64) {
        self.recorder.finish_epoch(epoch);
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = self.recorder.snapshot();
        let result = PyDict::new(py);
        result.set_item("current", summary_dict(py, &snapshot.current)?)?;
        result.set_item("enabled", true)?;
        result.set_item("registry", Self::registry(py)?)?;
        result.set_item("startup_ns", snapshot.startup_ns)?;
        match snapshot.last_epoch {
            Some(summary) => result.set_item("last_epoch", summary_dict(py, &summary)?)?,
            None => result.set_item("last_epoch", py.None())?,
        }
        Ok(result)
    }
}

fn duration_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn summary_dict<'py>(py: Python<'py>, summary: &EpochSummary) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let latency = PyDict::new(py);
    latency.set_item("p50", summary.delivery_latency_ns[0])?;
    latency.set_item("p95", summary.delivery_latency_ns[1])?;
    latency.set_item("p99", summary.delivery_latency_ns[2])?;
    let decisions = PyList::empty(py);
    for decision in &summary.controller_decisions {
        let item = PyDict::new(py);
        item.set_item("binding", decision.binding.as_deref())?;
        item.set_item("previous_width", decision.previous_width)?;
        item.set_item("reason", &decision.reason)?;
        item.set_item("resource_loss", decision.resource_loss)?;
        item.set_item("starvation", decision.starvation)?;
        item.set_item("width", decision.width)?;
        decisions.append(item)?;
    }
    let delivery_rate = if summary.delivery_interval_ns == 0 {
        0.0
    } else {
        summary.delivered_samples as f64 * 1_000_000_000.0 / summary.delivery_interval_ns as f64
    };
    result.set_item("ceiling_binds", summary.ceiling_binds)?;
    result.set_item("controller_decisions", decisions)?;
    result.set_item("delivered_batches", summary.delivered_batches)?;
    result.set_item("delivered_bytes", summary.delivered_bytes)?;
    result.set_item("delivered_samples", summary.delivered_samples)?;
    result.set_item("delivery_latency_ns", latency)?;
    result.set_item("delivery_rate", delivery_rate)?;
    result.set_item("epoch", summary.epoch)?;
    result.set_item("stall_events", summary.stall_events)?;
    Ok(result)
}
