//! Completion validation and arena delivery for process-owner resources.

use super::owner::{ProcessResources, runtime_error};
use crate::exec::{CompletionStatus, TransportError};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub(super) fn try_receive(
    resources: &mut ProcessResources,
    py: Python<'_>,
    worker: u32,
) -> PyResult<Option<(u64, u8, Py<PyBytes>)>> {
    let completion = match resources.transport(worker)?.try_recv_completion() {
        Ok(message) => message,
        Err(TransportError::CompletionEmpty) => return Ok(None),
        Err(error) => return Err(runtime_error(error)),
    };
    if completion.worker != worker {
        return Err(PyRuntimeError::new_err(
            "completion worker does not match its transport",
        ));
    }
    let pending = resources
        .pending
        .get(&(worker, completion.position))
        .copied()
        .ok_or_else(|| PyRuntimeError::new_err("completion has no pending dispatch"))?;
    if completion.slot != pending.primary {
        return Err(PyRuntimeError::new_err(
            "completion primary slot does not match its dispatch",
        ));
    }
    let (status, payload) = match completion.status {
        CompletionStatus::Ready => {
            if completion.exception.is_some() {
                return Err(PyRuntimeError::new_err(
                    "ready completion unexpectedly references an exception",
                ));
            }
            resources
                .allocator
                .publish(
                    pending.primary,
                    worker,
                    usize::try_from(completion.produced_length)
                        .map_err(|_| PyRuntimeError::new_err("produced length is too large"))?,
                )
                .map_err(runtime_error)?;
            resources
                .allocator
                .cancel(pending.exception, worker)
                .map_err(runtime_error)?;
            (0, resources.read_slot(pending.primary)?)
        }
        CompletionStatus::Exception => {
            let exception = completion.exception.ok_or_else(|| {
                PyRuntimeError::new_err("exception completion has no side-slab reference")
            })?;
            if exception.slot != pending.exception {
                return Err(PyRuntimeError::new_err(
                    "exception slot does not match its dispatch",
                ));
            }
            resources
                .allocator
                .cancel(pending.primary, worker)
                .map_err(runtime_error)?;
            resources
                .allocator
                .publish(
                    pending.exception,
                    worker,
                    usize::try_from(exception.length)
                        .map_err(|_| PyRuntimeError::new_err("exception length is too large"))?,
                )
                .map_err(runtime_error)?;
            (1, resources.read_slot(pending.exception)?)
        }
    };
    resources.pending.remove(&(worker, completion.position));
    Ok(Some((
        completion.position,
        status,
        PyBytes::new(py, &payload).unbind(),
    )))
}
