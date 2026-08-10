//! Fixed-memory latency histogram with power-of-two upper bounds.

use std::array;
use std::sync::atomic::{AtomicU64, Ordering};

const BUCKETS: usize = 65;

pub(crate) struct LatencyHistogram {
    counts: [AtomicU64; BUCKETS],
}

impl LatencyHistogram {
    pub(crate) fn new() -> Self {
        Self {
            counts: array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    pub(crate) fn observe(&self, value: u64) {
        self.counts[bucket(value)].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn percentiles(&self) -> [u64; 3] {
        let counts = self
            .counts
            .iter()
            .map(|count| count.load(Ordering::Relaxed))
            .collect::<Vec<_>>();
        let total = counts.iter().sum::<u64>();
        [50, 95, 99].map(|percent| percentile(&counts, total, percent))
    }

    pub(crate) fn reset(&self) {
        for count in &self.counts {
            count.store(0, Ordering::Relaxed);
        }
    }
}

fn bucket(value: u64) -> usize {
    if value == 0 {
        0
    } else {
        (u64::BITS - value.leading_zeros()) as usize
    }
}

fn percentile(counts: &[u64], total: u64, percent: u64) -> u64 {
    if total == 0 {
        return 0;
    }
    let target = total.saturating_mul(percent).div_ceil(100);
    let mut cumulative = 0;
    for (index, count) in counts.iter().enumerate() {
        cumulative += count;
        if cumulative >= target {
            return bucket_upper_bound(index);
        }
    }
    u64::MAX
}

fn bucket_upper_bound(index: usize) -> u64 {
    match index {
        0 => 0,
        64 => u64::MAX,
        _ => (1_u64 << index) - 1,
    }
}
