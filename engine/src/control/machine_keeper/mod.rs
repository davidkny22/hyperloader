//! Native consumer activity that prevents measured idle-state wake taxes.

mod idle;
mod pulse;
mod tuner;

use idle::IdleEntryMonitor;
use pulse::PeriodicPulse;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tuner::DutyTuner;

const DUTY_SCALE: u32 = 1_000_000;
const TUNING_WINDOW: Duration = Duration::from_millis(500);

struct SharedState {
    active: AtomicBool,
    stop: AtomicBool,
    duty: AtomicU32,
    wake: Condvar,
    parked: Mutex<()>,
}

/// One engine-owned low-duty thread for the consumer's active CPU set.
pub struct MachineKeeper {
    shared: Arc<SharedState>,
    worker: Option<JoinHandle<()>>,
    minimum_gap_ns: u64,
}

impl MachineKeeper {
    /// Start a parked activity thread and wait until its affinity is applied.
    pub fn new(
        cpus: Vec<usize>,
        maximum_duty: f64,
        initial_duty: f64,
        minimum_gap_ns: u64,
    ) -> Result<Self, String> {
        if cpus.is_empty() {
            return Err("machine keeping requires a nonempty consumer CPU set".to_owned());
        }
        if !(0.0 < maximum_duty && maximum_duty <= 1.0) {
            return Err("machine-keeping maximum duty must be in (0, 1]".to_owned());
        }
        if !(0.0 < initial_duty && initial_duty <= maximum_duty) {
            return Err("machine-keeping initial duty must be in (0, maximum]".to_owned());
        }
        if minimum_gap_ns == 0 {
            return Err("machine keeping requires a positive gap threshold".to_owned());
        }
        let initial = duty_units(initial_duty);
        let maximum = duty_units(maximum_duty);
        let shared = Arc::new(SharedState {
            active: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            duty: AtomicU32::new(initial),
            wake: Condvar::new(),
            parked: Mutex::new(()),
        });
        let worker_shared = Arc::clone(&shared);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let period_ns = (minimum_gap_ns / 2).clamp(100_000, 1_000_000);
        let worker = thread::Builder::new()
            .name("hyperloader-machine-keeper".to_owned())
            .spawn(move || {
                if let Err(error) = apply_affinity(&cpus) {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
                let mut monitor = IdleEntryMonitor::discover(&cpus);
                let mut tuner = DutyTuner::new(initial, maximum);
                let _ = ready_tx.send(Ok(()));
                run_loop(worker_shared, period_ns, &mut monitor, &mut tuner);
            })
            .map_err(|error| format!("machine-keeping thread could not start: {error}"))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                shared,
                worker: Some(worker),
                minimum_gap_ns,
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(error) => {
                let _ = worker.join();
                Err(format!(
                    "machine-keeping thread did not initialize: {error}"
                ))
            }
        }
    }

    /// Activate only when the observed consumer gap reaches the calibrated regime.
    pub fn observe_gap(&self, nanoseconds: u64) {
        let active = nanoseconds >= self.minimum_gap_ns;
        self.shared.active.store(active, Ordering::Release);
        if active {
            self.shared.wake.notify_one();
        }
    }

    /// Park activity immediately at exhaustion or invalidation.
    pub fn park(&self) {
        self.shared.active.store(false, Ordering::Release);
    }

    /// Return the currently applied duty, or zero while parked.
    pub fn duty(&self) -> f64 {
        if !self.shared.active.load(Ordering::Acquire) {
            return 0.0;
        }
        f64::from(self.shared.duty.load(Ordering::Relaxed)) / f64::from(DUTY_SCALE)
    }

    /// Stop the native thread and release its process resource.
    pub fn close(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.active.store(false, Ordering::Release);
        self.shared.wake.notify_one();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for MachineKeeper {
    fn drop(&mut self) {
        self.close();
    }
}

fn run_loop(
    shared: Arc<SharedState>,
    period_ns: u64,
    monitor: &mut Option<IdleEntryMonitor>,
    tuner: &mut DutyTuner,
) {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut window_started = Instant::now();
    let mut pulse = PeriodicPulse::new(period_ns);
    while !shared.stop.load(Ordering::Acquire) {
        if !shared.active.load(Ordering::Acquire) {
            let guard = shared
                .parked
                .lock()
                .expect("machine-keeping mutex poisoned");
            let _guard = shared
                .wake
                .wait_while(guard, |_| {
                    !shared.active.load(Ordering::Acquire) && !shared.stop.load(Ordering::Acquire)
                })
                .expect("machine-keeping mutex poisoned");
            window_started = Instant::now();
            pulse.reset();
            if let Some(entries) = monitor.as_mut() {
                entries.reset();
            }
            continue;
        }
        let duty = shared.duty.load(Ordering::Relaxed);
        let active_ns = period_ns.saturating_mul(u64::from(duty)) / u64::from(DUTY_SCALE);
        let active_until = pulse.active_until(active_ns.max(1));
        while pulse.before(active_until) && shared.active.load(Ordering::Relaxed) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state = black_box(state.wrapping_mul(0xd134_2543_de82_ef95));
        }
        if window_started.elapsed() >= TUNING_WINDOW {
            if let Some(entries) = monitor.as_mut() {
                let next = tuner.observe(entries.delta());
                shared.duty.store(next, Ordering::Relaxed);
            }
            window_started = Instant::now();
        }
        pulse.wait_next();
    }
    black_box(state);
}

fn duty_units(value: f64) -> u32 {
    (value * f64::from(DUTY_SCALE)).round() as u32
}

#[cfg(target_os = "linux")]
fn apply_affinity(cpus: &[usize]) -> Result<(), String> {
    // SAFETY: cpu_set_t is initialized before use, every index is checked against CPU_SETSIZE,
    // and sched_setaffinity receives the exact initialized object size.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in cpus {
            if cpu >= libc::CPU_SETSIZE as usize {
                return Err(format!(
                    "consumer CPU {cpu} exceeds the affinity mask limit"
                ));
            }
            libc::CPU_SET(cpu, &mut set);
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(format!(
                "machine-keeping affinity failed: {}",
                std::io::Error::last_os_error()
            ));
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn apply_affinity(_cpus: &[usize]) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests;
