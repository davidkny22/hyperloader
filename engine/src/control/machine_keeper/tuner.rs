//! Bounded minimum-duty search over powered-down idle entries.

const DUTY_STEP: u32 = 10_000;

pub(super) struct DutyTuner {
    current: u32,
    maximum: u32,
    lowest_zero: Option<u32>,
    settled: bool,
}

impl DutyTuner {
    pub(super) fn new(initial: u32, maximum: u32) -> Self {
        Self {
            current: initial.min(maximum),
            maximum,
            lowest_zero: None,
            settled: false,
        }
    }

    pub(super) fn observe(&mut self, powered_down_entries: u64) -> u32 {
        if self.settled {
            if powered_down_entries == 0 {
                return self.current;
            }
            self.lowest_zero = None;
            self.settled = false;
            self.current = (self.current + DUTY_STEP).min(self.maximum);
            return self.current;
        }
        if powered_down_entries == 0 {
            self.lowest_zero = Some(self.current);
            if self.current > DUTY_STEP {
                self.current -= DUTY_STEP;
            } else {
                self.settled = true;
            }
        } else if let Some(lowest_zero) = self.lowest_zero {
            self.current = lowest_zero;
            self.settled = true;
        } else if self.current < self.maximum {
            self.current = (self.current + DUTY_STEP).min(self.maximum);
        } else {
            self.settled = true;
        }
        self.current
    }
}
