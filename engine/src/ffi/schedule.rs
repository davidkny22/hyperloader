//! Private Python binding for the fixed-frontier scheduler.

use crate::sched::{Dispatch, StaticSchedule};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Native bounded frontier and reorder state machine.
#[pyclass(name = "_StaticSchedule")]
pub(crate) struct PyStaticSchedule {
    schedule: StaticSchedule,
}

#[pymethods]
impl PyStaticSchedule {
    #[new]
    fn new(start: u64, end: u64, depth: usize, worker_count: u32) -> PyResult<Self> {
        Ok(Self {
            schedule: StaticSchedule::new(start, end, depth, worker_count).map_err(value_error)?,
        })
    }

    fn next_dispatch(&self) -> Option<(u64, u32)> {
        self.schedule
            .next_dispatch()
            .map(|dispatch| (dispatch.position, dispatch.worker))
    }

    fn dispatch_candidates(&self) -> Vec<u64> {
        self.schedule.dispatch_candidates().collect()
    }

    fn dispatch_at(&self, position: u64) -> Option<(u64, u32)> {
        self.schedule
            .dispatch_at(position)
            .map(|dispatch| (dispatch.position, dispatch.worker))
    }

    fn mark_dispatched(&mut self, position: u64, worker: u32) -> PyResult<()> {
        self.schedule
            .mark_dispatched(Dispatch { position, worker })
            .map_err(value_error)
    }

    fn mark_completed(&mut self, position: u64, worker: u32) -> PyResult<()> {
        self.schedule
            .mark_completed(Dispatch { position, worker })
            .map_err(value_error)
    }

    fn try_commit(&mut self) -> Option<u64> {
        self.schedule.try_commit()
    }

    fn try_commit_ready(&mut self, position: u64) -> Option<u64> {
        self.schedule.try_commit_ready(position)
    }

    fn delivered_positions(&self) -> Vec<u64> {
        let mut positions = self.schedule.delivered_positions().collect::<Vec<_>>();
        positions.sort_unstable();
        positions
    }

    fn seed_delivered(&mut self, position: u64) -> PyResult<()> {
        self.schedule.seed_delivered(position).map_err(value_error)
    }

    fn is_complete(&self) -> bool {
        self.schedule.is_complete()
    }

    fn occupied(&self) -> usize {
        self.schedule.occupied()
    }

    fn set_depth(&mut self, depth: usize) -> PyResult<()> {
        self.schedule.set_depth(depth).map_err(value_error)
    }

    fn set_worker_count(&mut self, worker_count: u32) -> PyResult<()> {
        self.schedule
            .set_worker_count(worker_count)
            .map_err(value_error)
    }
}

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}
