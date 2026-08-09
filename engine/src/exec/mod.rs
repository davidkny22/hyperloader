//! Native, Python-thread, and persistent-process executors live in this module.

mod transport;

pub use transport::{
    CommandTransport, CompletionMessage, CompletionStatus, DispatchMessage, ExceptionRef,
    TransportError,
};
