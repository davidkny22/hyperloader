//! Bounded dispatch selection with out-of-order completion buffering.

use std::collections::{HashMap, HashSet};
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

/// Bounded frontier with selectable dispatch and strict sampler-order commit.
#[derive(Debug)]
pub struct StaticSchedule {
    end: u64,
    depth: u64,
    worker_count: u32,
    worker_ceiling: u32,
    next_commit: u64,
    dispatch_ordinal: u64,
    positions: HashMap<u64, PositionState>,
    delivered: HashSet<u64>,
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
            worker_ceiling: worker_count,
            next_commit: start,
            dispatch_ordinal: 0,
            positions: HashMap::with_capacity(depth as usize),
            delivered: HashSet::with_capacity(depth as usize),
        })
    }

    /// Return the next FIFO dispatch while the bounded frontier has room.
    pub fn next_dispatch(&self) -> Option<Dispatch> {
        let position = self.dispatch_candidates().next()?;
        self.dispatch_at(position)
    }

    /// Return every unsubmitted position admitted by the current frontier.
    pub fn dispatch_candidates(&self) -> impl Iterator<Item = u64> + '_ {
        let stop = self.next_commit.saturating_add(self.depth).min(self.end);
        (self.next_commit..stop).filter(|position| {
            !self.positions.contains_key(position) && !self.delivered.contains(position)
        })
    }

    /// Build the next worker route for one eligible frontier position.
    pub fn dispatch_at(&self, position: u64) -> Option<Dispatch> {
        let stop = self.next_commit.saturating_add(self.depth).min(self.end);
        if position < self.next_commit
            || position >= stop
            || self.positions.contains_key(&position)
            || self.delivered.contains(&position)
        {
            return None;
        }
        Some(Dispatch {
            position,
            worker: (self.dispatch_ordinal % u64::from(self.worker_count)) as u32,
        })
    }

    /// Confirm that the current candidate entered its worker transport.
    pub fn mark_dispatched(&mut self, dispatch: Dispatch) -> Result<(), ScheduleError> {
        if Some(dispatch) != self.dispatch_at(dispatch.position) {
            return Err(ScheduleError(
                "dispatch is not an eligible frontier candidate",
            ));
        }
        self.positions.insert(
            dispatch.position,
            PositionState::Dispatched(dispatch.worker),
        );
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
        self.try_commit_ready(self.next_commit)
    }

    /// Commit one ready position immediately and advance the contiguous base when possible.
    pub fn try_commit_ready(&mut self, position: u64) -> Option<u64> {
        if !matches!(self.positions.get(&position), Some(PositionState::Ready(_))) {
            return None;
        }
        self.positions.remove(&position);
        if position == self.next_commit {
            self.next_commit += 1;
            while self.delivered.remove(&self.next_commit) {
                self.next_commit += 1;
            }
        } else {
            self.delivered.insert(position);
        }
        Some(position)
    }

    /// Return committed positions beyond the contiguous delivery prefix.
    pub fn delivered_positions(&self) -> impl Iterator<Item = u64> + '_ {
        self.delivered.iter().copied()
    }

    /// Report whether every position has committed and the frontier is empty.
    pub fn is_complete(&self) -> bool {
        self.next_commit == self.end && self.positions.is_empty() && self.delivered.is_empty()
    }

    /// Return the number of dispatched, uncommitted positions.
    pub fn occupied(&self) -> usize {
        self.positions.len()
    }

    /// Change the frontier depth without evicting active positions.
    pub fn set_depth(&mut self, depth: usize) -> Result<(), ScheduleError> {
        let depth = u64::try_from(depth)
            .map_err(|_| ScheduleError("frontier depth does not fit the position domain"))?;
        if depth == 0 {
            return Err(ScheduleError("frontier depth must be positive"));
        }
        let stop = self.next_commit.saturating_add(depth);
        if self.positions.keys().any(|position| *position >= stop) {
            return Err(ScheduleError("frontier depth excludes an active position"));
        }
        self.depth = depth;
        Ok(())
    }

    /// Change the live routing width without changing the spawned worker ceiling.
    pub fn set_worker_count(&mut self, worker_count: u32) -> Result<(), ScheduleError> {
        if worker_count == 0 || worker_count > self.worker_ceiling {
            return Err(ScheduleError(
                "live worker count is outside the plan ceiling",
            ));
        }
        self.worker_count = worker_count;
        self.dispatch_ordinal %= u64::from(worker_count);
        Ok(())
    }
}
