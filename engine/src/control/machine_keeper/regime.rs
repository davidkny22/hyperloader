//! Activation decision after a complete passive counter window.

pub(super) struct ActivityProbe {
    primed: bool,
}

impl ActivityProbe {
    pub(super) fn new() -> Self {
        Self { primed: false }
    }

    pub(super) fn observe(&mut self, powered_down_entries: Option<u64>) -> bool {
        let Some(entries) = powered_down_entries else {
            return true;
        };
        if !self.primed {
            self.primed = true;
            return false;
        }
        entries > 0
    }
}

#[cfg(test)]
mod tests;
