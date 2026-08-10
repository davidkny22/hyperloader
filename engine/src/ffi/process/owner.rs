//! Loader-owner allocation, dispatch, completion, and delivery endpoint.

use super::sizing::process_slab_specs;
use crate::arena::{
    ArenaAllocator, DeliveryView, GrowthPolicy, RegionRegistry, RegionToken, SlotRef,
};
use crate::exec::{CommandTransport, DispatchMessage, TransportError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

const MAX_WORKERS: u32 = 1022;

#[derive(Clone, Copy)]
pub(super) struct PendingSlots {
    pub(super) primary: SlotRef,
    pub(super) exception: SlotRef,
}

/// Native resources owned by one Python process pool.
#[pyclass(name = "_ProcessResources")]
pub(crate) struct ProcessResources {
    token: RegionToken,
    registry: RegionRegistry,
    pub(super) allocator: ArenaAllocator,
    pub(super) transports: Vec<Option<CommandTransport>>,
    queue_capacity: usize,
    payload_capacity: usize,
    exception_capacity: usize,
    pub(super) pending: HashMap<(u32, u64), PendingSlots>,
}

#[pymethods]
impl ProcessResources {
    #[new]
    #[pyo3(signature = (worker_count, queue_capacity=2, payload_capacity=262_144, exception_capacity=65_536, registry_path=None))]
    fn new(
        worker_count: u32,
        queue_capacity: usize,
        payload_capacity: usize,
        exception_capacity: usize,
        registry_path: Option<PathBuf>,
    ) -> PyResult<Self> {
        if worker_count == 0 || worker_count > MAX_WORKERS {
            return Err(PyValueError::new_err(format!(
                "worker_count must be between 1 and {MAX_WORKERS}"
            )));
        }
        if payload_capacity == 0 || exception_capacity == 0 {
            return Err(PyValueError::new_err(
                "payload and exception capacities must be positive",
            ));
        }
        let registry = match registry_path {
            Some(path) => RegionRegistry::new(path),
            None => {
                RegionRegistry::prepare_current_user()
                    .map_err(runtime_error)?
                    .0
            }
        };
        let token = RegionToken::random().map_err(runtime_error)?;
        let mut transports = Vec::with_capacity(worker_count as usize);
        for worker in 0..worker_count {
            transports.push(Some(
                CommandTransport::create(
                    registry.clone(),
                    token,
                    worker as u16,
                    queue_capacity,
                    queue_capacity,
                )
                .map_err(runtime_error)?,
            ));
        }
        let specs = process_slab_specs(
            worker_count,
            queue_capacity,
            payload_capacity,
            exception_capacity,
        )
        .map_err(runtime_error)?;
        let allocator = ArenaAllocator::new_at_sequence(
            registry.clone(),
            token,
            &specs,
            GrowthPolicy::Safe,
            worker_count as u16,
        )
        .map_err(runtime_error)?;
        Ok(Self {
            token,
            registry,
            allocator,
            transports,
            queue_capacity,
            payload_capacity,
            exception_capacity,
            pending: HashMap::new(),
        })
    }

    /// Return attachment inputs for one persistent worker.
    fn descriptor(&self, worker: u32) -> PyResult<(Vec<u8>, u16, usize, usize)> {
        self.transport(worker)?;
        Ok((
            self.token.as_bytes().to_vec(),
            worker as u16,
            self.queue_capacity,
            self.queue_capacity,
        ))
    }

    /// Reserve payload and exception slots and attempt one targeted dispatch.
    #[pyo3(signature = (epoch, position, index, stage_plan, worker, batch_len=0))]
    fn try_submit(
        &mut self,
        epoch: u64,
        position: u64,
        index: u64,
        stage_plan: u32,
        worker: u32,
        batch_len: u32,
    ) -> PyResult<bool> {
        if self.pending.contains_key(&(worker, position)) {
            return Err(PyValueError::new_err(
                "worker already has this position pending",
            ));
        }
        self.transport(worker)?;
        let primary = self
            .allocator
            .reserve(self.payload_capacity, worker)
            .map_err(runtime_error)?;
        let exception = match self.allocator.reserve(self.exception_capacity, worker) {
            Ok(slot) => slot,
            Err(error) => {
                let _ = self.allocator.cancel(primary, worker);
                return Err(runtime_error(error));
            }
        };
        let message = DispatchMessage {
            position,
            epoch,
            stage_plan,
            index,
            worker,
            batch_len,
            slot: primary,
            exception_slot: exception,
        };
        match self.transport(worker)?.try_send_dispatch(message) {
            Ok(()) => {
                self.pending
                    .insert((worker, position), PendingSlots { primary, exception });
                Ok(true)
            }
            Err(TransportError::DispatchFull) => {
                self.allocator
                    .cancel(primary, worker)
                    .map_err(runtime_error)?;
                self.allocator
                    .cancel(exception, worker)
                    .map_err(runtime_error)?;
                Ok(false)
            }
            Err(error) => {
                let _ = self.allocator.cancel(primary, worker);
                let _ = self.allocator.cancel(exception, worker);
                Err(runtime_error(error))
            }
        }
    }

    /// Attempt one completion and return `(position, status, encoded_bytes)`.
    fn try_receive(
        &mut self,
        py: Python<'_>,
        worker: u32,
    ) -> PyResult<Option<(u64, u8, Py<PyAny>)>> {
        self.transport(worker)?;
        super::delivery::try_receive(self, py, worker)
    }

    /// Reclaim reservations only after the operating-system worker is confirmed dead.
    fn reclaim_dead_worker(&mut self, worker: u32) -> PyResult<Vec<u64>> {
        self.transport(worker)?;
        self.reclaim_dead_slots(worker)
    }

    /// Reclaim a proven-dead writer and replace its command region.
    fn restart_worker(&mut self, worker: u32) -> PyResult<Vec<u64>> {
        let positions = self.reclaim_dead_worker(worker)?;
        let index = worker as usize;
        let old = self.transports[index]
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("worker transport is unavailable"))?;
        drop(old);
        let transport = CommandTransport::create(
            self.registry.clone(),
            self.token,
            worker as u16,
            self.queue_capacity,
            self.queue_capacity,
        )
        .map_err(runtime_error)?;
        self.transports[index] = Some(transport);
        Ok(positions)
    }
}

impl ProcessResources {
    fn reclaim_dead_slots(&mut self, worker: u32) -> PyResult<Vec<u64>> {
        let positions: Vec<u64> = self
            .pending
            .keys()
            .filter_map(|(owner, position)| (*owner == worker).then_some(*position))
            .collect();
        let poisoned = self
            .allocator
            .poison_writer(worker)
            .map_err(runtime_error)?;
        for slot in poisoned {
            self.allocator
                .reclaim_poisoned(slot)
                .map_err(runtime_error)?;
        }
        self.pending.retain(|(owner, _), _| *owner != worker);
        Ok(positions)
    }

    pub(super) fn transport(&self, worker: u32) -> PyResult<&CommandTransport> {
        self.transports
            .get(worker as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| PyValueError::new_err(format!("worker {worker} is out of range")))
    }

    pub(super) fn read_slot(&self, slot: SlotRef) -> PyResult<Vec<u8>> {
        let mut views = self
            .allocator
            .deliver(
                slot,
                std::num::NonZeroU32::new(1).expect("one view is nonzero"),
            )
            .map_err(runtime_error)?;
        views
            .pop()
            .expect("one requested view is returned")
            .to_vec()
            .map_err(runtime_error)
    }

    pub(super) fn deliver_slot(&self, slot: SlotRef) -> PyResult<DeliveryView> {
        self.allocator
            .deliver(
                slot,
                std::num::NonZeroU32::new(1).expect("one view is nonzero"),
            )
            .map_err(runtime_error)?
            .pop()
            .ok_or_else(|| PyRuntimeError::new_err("arena delivery returned no view"))
    }
}

pub(super) fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
