//! Activation decision for passive powered-down entry observations.

pub(super) fn needs_activity(powered_down_entries: Option<u64>) -> bool {
    powered_down_entries.is_none_or(|entries| entries > 0)
}

#[cfg(test)]
mod tests;
