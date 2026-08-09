//! Fixed-frontier scheduling and strict-order commit coordination.

mod static_frontier;

pub use static_frontier::{Dispatch, ScheduleError, StaticSchedule};
