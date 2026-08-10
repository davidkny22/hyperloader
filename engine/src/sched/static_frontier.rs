//! Bounded FIFO dispatch with out-of-order completion buffering.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// One position and its deterministic worker route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dispatch {
    /// Sampler-stream position.
    pub position: u64,
    /// Process worker selected in FIFO round-robin order.
    pub worker: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PositionState {
    Dispatched(u32),
    Ready(u32),
}

/// Invalid scheduler state transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduleError(&'static str);

impl Display for ScheduleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl Error for ScheduleError {}

/// Fixed-depth FIFO frontier with strict sampler-order commit.
#[derive(Debug)]
pub struct StaticSchedule {
    end: u64,
    depth: u64,
    worker_count: u32,
    next_dispatch: u64,
    next_commit: u64,
    dispatch_ordinal: u64,
    positions: HashMap<u64, PositionState>,
}

impl StaticSchedule {
    /// Create a schedule over the half-open position range `[start, end)`.
    pub fn new(
        start: u64,
        end: u64,
        depth: usize,
        worker_count: u32,
    ) -> Result<Self, ScheduleError> {
        if end < start {
            return Err(ScheduleError("schedule end precedes its start"));
        }
        if depth == 0 {
            return Err(ScheduleError("frontier depth must be positive"));
        }
        if worker_count == 0 {
            return Err(ScheduleError("worker count must be positive"));
        }
        let depth = u64::try_from(depth)
            .map_err(|_| ScheduleError("frontier depth does not fit the position domain"))?;
        Ok(Self {
            end,
            depth,
            worker_count,
            next_dispatch: start,
            next_commit: start,
            dispatch_ordinal: 0,
            positions: HashMap::with_capacity(depth as usize),
        })
    }

    /// Return the next FIFO dispatch while the bounded frontier has room.
    pub fn next_dispatch(&self) -> Option<Dispatch> {
        if self.next_dispatch >= self.end
            || self.next_dispatch.saturating_sub(self.next_commit) >= self.depth
        {
            return None;
        }
        Some(Dispatch {
            position: self.next_dispatch,
            worker: (self.dispatch_ordinal % u64::from(self.worker_count)) as u32,
        })
    }

    /// Confirm that the current candidate entered its worker transport.
    pub fn mark_dispatched(&mut self, dispatch: Dispatch) -> Result<(), ScheduleError> {
        if Some(dispatch) != self.next_dispatch() {
            return Err(ScheduleError(
                "dispatch does not match the next FIFO candidate",
            ));
        }
        self.positions.insert(
            dispatch.position,
            PositionState::Dispatched(dispatch.worker),
        );
        self.next_dispatch += 1;
        self.dispatch_ordinal += 1;
        Ok(())
    }

    /// Record a completion, allowing arrival in any execution order.
    pub fn mark_completed(&mut self, dispatch: Dispatch) -> Result<(), ScheduleError> {
        let Some(state) = self.positions.get_mut(&dispatch.position) else {
            return Err(ScheduleError("completion is outside the active frontier"));
        };
        match *state {
            PositionState::Dispatched(worker) if worker == dispatch.worker => {
                *state = PositionState::Ready(dispatch.worker);
                Ok(())
            }
            PositionState::Dispatched(_) => {
                Err(ScheduleError("completion came from the wrong worker"))
            }
            PositionState::Ready(_) => Err(ScheduleError("position completed twice")),
        }
    }

    /// Commit the next sampler position only when that exact position is ready.
    pub fn try_commit(&mut self) -> Option<u64> {
        if !matches!(
            self.positions.get(&self.next_commit),
            Some(PositionState::Ready(_))
        ) {
            return None;
        }
        let position = self.next_commit;
        self.positions.remove(&position);
        self.next_commit += 1;
        Some(position)
    }

    /// Report whether every position has committed and the frontier is empty.
    pub fn is_complete(&self) -> bool {
        self.next_commit == self.end && self.positions.is_empty()
    }

    /// Return the number of dispatched, uncommitted positions.
    pub fn occupied(&self) -> usize {
        self.positions.len()
    }
}
