//! Absolute monotonic timing for periodic machine-keeping activity.

#[cfg(target_os = "linux")]
pub(super) struct PeriodicPulse {
    period_ns: u64,
    next_start_ns: u64,
}

#[cfg(target_os = "linux")]
impl PeriodicPulse {
    pub(super) fn new(period_ns: u64) -> Self {
        Self {
            period_ns,
            next_start_ns: monotonic_ns(),
        }
    }

    pub(super) fn reset(&mut self) {
        self.next_start_ns = monotonic_ns();
    }

    pub(super) fn active_until(&self, active_ns: u64) -> u64 {
        self.next_start_ns.saturating_add(active_ns)
    }

    pub(super) fn before(&self, deadline_ns: u64) -> bool {
        monotonic_ns() < deadline_ns
    }

    pub(super) fn wait_next(&mut self) {
        self.next_start_ns = self.next_start_ns.saturating_add(self.period_ns);
        let deadline = libc::timespec {
            tv_sec: (self.next_start_ns / 1_000_000_000) as libc::time_t,
            tv_nsec: (self.next_start_ns % 1_000_000_000) as libc::c_long,
        };
        loop {
            // SAFETY: deadline is a fully initialized CLOCK_MONOTONIC absolute timestamp,
            // and a null remainder is valid when TIMER_ABSTIME is selected.
            let result = unsafe {
                libc::clock_nanosleep(
                    libc::CLOCK_MONOTONIC,
                    libc::TIMER_ABSTIME,
                    &deadline,
                    std::ptr::null_mut(),
                )
            };
            if result != libc::EINTR {
                break;
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn monotonic_ns() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: now points to writable initialized storage for one timespec.
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    if result != 0 {
        return 0;
    }
    (now.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(now.tv_nsec as u64)
}

#[cfg(not(target_os = "linux"))]
pub(super) struct PeriodicPulse {
    period: std::time::Duration,
    next_start: std::time::Instant,
}

#[cfg(not(target_os = "linux"))]
impl PeriodicPulse {
    pub(super) fn new(period_ns: u64) -> Self {
        Self {
            period: std::time::Duration::from_nanos(period_ns),
            next_start: std::time::Instant::now(),
        }
    }

    pub(super) fn reset(&mut self) {
        self.next_start = std::time::Instant::now();
    }

    pub(super) fn active_until(&self, active_ns: u64) -> std::time::Instant {
        self.next_start + std::time::Duration::from_nanos(active_ns)
    }

    pub(super) fn before(&self, deadline: std::time::Instant) -> bool {
        std::time::Instant::now() < deadline
    }

    pub(super) fn wait_next(&mut self) {
        self.next_start += self.period;
        if let Some(remaining) = self
            .next_start
            .checked_duration_since(std::time::Instant::now())
        {
            std::thread::sleep(remaining);
        }
    }
}

#[cfg(test)]
mod tests;
