//! Worker attachment, command receipt, direct writes, and completion publication.

use crate::arena::{ArenaWriter, RegionToken};
use crate::exec::{
    CommandTransport, CompletionMessage, CompletionStatus, DispatchMessage, ExceptionRef,
    TransportError,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// One decoded dispatch retained while Python executes user code.
#[pyclass(name = "_WorkerCommand", frozen)]
#[derive(Clone)]
pub(crate) struct WorkerCommand {
    message: DispatchMessage,
}

#[pymethods]
impl WorkerCommand {
    #[getter]
    const fn position(&self) -> u64 {
        self.message.position
    }

    #[getter]
    const fn epoch(&self) -> u64 {
        self.message.epoch
    }

    #[getter]
    const fn index(&self) -> u64 {
        self.message.index
    }

    #[getter]
    const fn stage_plan(&self) -> u32 {
        self.message.stage_plan
    }

    #[getter]
    const fn worker(&self) -> u32 {
        self.message.worker
    }

    #[getter]
    const fn batch_len(&self) -> u32 {
        self.message.batch_len
    }
}

/// Native endpoint attached inside one persistent Python worker.
#[pyclass(name = "_WorkerEndpoint", unsendable)]
pub(crate) struct WorkerEndpoint {
    transport: CommandTransport,
    writer: ArenaWriter,
}

#[pymethods]
impl WorkerEndpoint {
    #[new]
    fn new(
        token: &[u8],
        sequence: u16,
        dispatch_capacity: usize,
        completion_capacity: usize,
    ) -> PyResult<Self> {
        let token = parse_token(token)?;
        Ok(Self {
            transport: CommandTransport::attach(
                token,
                sequence,
                dispatch_capacity,
                completion_capacity,
            )
            .map_err(runtime_error)?,
            writer: ArenaWriter::new(token),
        })
    }

    /// Attempt to receive one targeted dispatch.
    fn try_recv(&self) -> PyResult<Option<WorkerCommand>> {
        match self.transport.try_recv_dispatch() {
            Ok(message) => Ok(Some(WorkerCommand { message })),
            Err(TransportError::DispatchEmpty) => Ok(None),
            Err(error) => Err(runtime_error(error)),
        }
    }

    /// Write a successful payload and attempt to publish its completion.
    fn try_complete_ready(&mut self, command: &WorkerCommand, payload: &[u8]) -> PyResult<bool> {
        self.writer
            .write(command.message.slot, payload)
            .map_err(runtime_error)?;
        self.try_complete(CompletionMessage {
            position: command.message.position,
            worker: command.message.worker,
            status: CompletionStatus::Ready,
            slot: command.message.slot,
            produced_length: payload.len() as u64,
            exception: None,
        })
    }

    /// Write an encoded exception and attempt to publish its side-slab reference.
    fn try_complete_exception(
        &mut self,
        command: &WorkerCommand,
        payload: &[u8],
    ) -> PyResult<bool> {
        self.writer
            .write(command.message.exception_slot, payload)
            .map_err(runtime_error)?;
        self.try_complete(CompletionMessage {
            position: command.message.position,
            worker: command.message.worker,
            status: CompletionStatus::Exception,
            slot: command.message.slot,
            produced_length: 0,
            exception: Some(ExceptionRef {
                slot: command.message.exception_slot,
                length: payload.len() as u64,
            }),
        })
    }
}

impl WorkerEndpoint {
    fn try_complete(&self, message: CompletionMessage) -> PyResult<bool> {
        match self.transport.try_send_completion(message) {
            Ok(()) => Ok(true),
            Err(TransportError::CompletionFull) => Ok(false),
            Err(error) => Err(runtime_error(error)),
        }
    }
}

fn parse_token(token: &[u8]) -> PyResult<RegionToken> {
    let bytes: [u8; 16] = token
        .try_into()
        .map_err(|_| PyValueError::new_err("loader token must contain exactly 16 bytes"))?;
    Ok(RegionToken::from_bytes(bytes))
}

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
