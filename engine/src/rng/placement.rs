//! Pure distributed placement over the contract permutation.

use super::{FEISTEL_THRESHOLD, feistel_permute, materialized_permutation};

/// One delivered dataset index and the position that owns its RNG stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacedSample {
    /// The global position, including positions introduced by padding.
    pub position: u64,
    /// The dataset index selected by the epoch permutation.
    pub index: u64,
}

/// Inputs that determine one rank's native-sampler placement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementRequest {
    /// The user seed that determines the epoch permutation.
    pub root_seed: u64,
    /// The epoch counter that determines the epoch permutation.
    pub epoch: u64,
    /// The number of samples in the dataset.
    pub dataset_len: u64,
    /// The per-rank batch size.
    pub batch_size: u64,
    /// The distributed world size.
    pub world_size: u64,
    /// The rank receiving the returned positions.
    pub rank: u64,
    /// Whether an incomplete global tail is discarded.
    pub drop_last: bool,
    /// Whether an incomplete global tail avoids padding and duplication.
    pub exact_count: bool,
}

/// Invalid placement inputs that would make the stream ambiguous or unreachable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementError {
    /// Per-rank batches must contain at least one sample.
    ZeroBatchSize,
    /// A distributed world must contain at least one rank.
    ZeroWorldSize,
    /// The selected rank must be a member of the world.
    RankOutOfRange,
    /// The global batch or padded epoch length exceeds 64-bit coordinates.
    CoordinateOverflow,
}

fn padded_len(dataset_len: u64, global_batch: u64) -> Result<u64, PlacementError> {
    if dataset_len == 0 {
        return Ok(0);
    }
    dataset_len
        .div_ceil(global_batch)
        .checked_mul(global_batch)
        .ok_or(PlacementError::CoordinateOverflow)
}

/// Return the new per-rank batch size when an elastic world preserves a global batch.
pub const fn elastic_batch_size(global_batch: u64, new_world_size: u64) -> Option<u64> {
    if new_world_size == 0 || !global_batch.is_multiple_of(new_world_size) {
        return None;
    }
    Some(global_batch / new_world_size)
}

/// Compute one rank's positions and dataset indices without cross-rank communication.
pub fn rank_placements(request: PlacementRequest) -> Result<Vec<PlacedSample>, PlacementError> {
    if request.batch_size == 0 {
        return Err(PlacementError::ZeroBatchSize);
    }
    if request.world_size == 0 {
        return Err(PlacementError::ZeroWorldSize);
    }
    if request.rank >= request.world_size {
        return Err(PlacementError::RankOutOfRange);
    }
    if request.dataset_len == 0 {
        return Ok(Vec::new());
    }
    let global_batch = request
        .batch_size
        .checked_mul(request.world_size)
        .ok_or(PlacementError::CoordinateOverflow)?;
    let full_end = (request.dataset_len / global_batch) * global_batch;
    let regular_end = if request.drop_last || request.exact_count {
        full_end
    } else {
        padded_len(request.dataset_len, global_batch)?
    };
    let small_permutation = if request.dataset_len < FEISTEL_THRESHOLD {
        Some(
            materialized_permutation(request.root_seed, request.epoch, request.dataset_len as u32)
                .expect("small dataset length must select the materialized regime"),
        )
    } else {
        None
    };
    let map_index = |permutation_position: u64| -> u64 {
        if let Some(permutation) = &small_permutation {
            permutation[permutation_position as usize] as u64
        } else {
            feistel_permute(
                request.root_seed,
                request.epoch,
                request.dataset_len,
                permutation_position,
            )
            .expect("position reduced into the Feistel domain")
        }
    };
    let mut output = Vec::new();
    let mut batch_start = 0_u64;
    while batch_start < regular_end {
        let rank_start = batch_start + request.rank * request.batch_size;
        let rank_end = rank_start + request.batch_size;
        for position in rank_start..rank_end {
            let permutation_position = if position < request.dataset_len {
                position
            } else {
                (position - request.dataset_len) % request.dataset_len
            };
            output.push(PlacedSample {
                position,
                index: map_index(permutation_position),
            });
        }
        batch_start += global_batch;
    }
    if request.exact_count && !request.drop_last {
        let tail_size = request.dataset_len - full_end;
        let rank_start = full_end + request.rank * tail_size / request.world_size;
        let rank_end = full_end + (request.rank + 1) * tail_size / request.world_size;
        for position in rank_start..rank_end {
            output.push(PlacedSample {
                position,
                index: map_index(position),
            });
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{PlacementError, PlacementRequest, elastic_batch_size, rank_placements};

    #[test]
    fn default_tail_pads_to_equal_rank_counts() {
        let mut per_rank = Vec::new();
        for rank in 0..3 {
            per_rank.push(
                rank_placements(PlacementRequest {
                    root_seed: 7,
                    epoch: 2,
                    dataset_len: 10,
                    batch_size: 2,
                    world_size: 3,
                    rank,
                    drop_last: false,
                    exact_count: false,
                })
                .unwrap(),
            );
        }
        assert!(per_rank.iter().all(|placements| placements.len() == 4));
        let mut positions: Vec<_> = per_rank
            .into_iter()
            .flatten()
            .map(|item| item.position)
            .collect();
        positions.sort_unstable();
        assert_eq!(positions, (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn exact_count_partitions_tail_without_duplication() {
        let mut positions = Vec::new();
        let mut counts = Vec::new();
        for rank in 0..3 {
            let placed = rank_placements(PlacementRequest {
                root_seed: 7,
                epoch: 2,
                dataset_len: 10,
                batch_size: 2,
                world_size: 3,
                rank,
                drop_last: false,
                exact_count: true,
            })
            .unwrap();
            counts.push(placed.len());
            positions.extend(placed.into_iter().map(|item| item.position));
        }
        positions.sort_unstable();
        assert_eq!(positions, (0..10).collect::<Vec<_>>());
        assert_eq!(counts, vec![3, 3, 4]);
    }

    #[test]
    fn invalid_topologies_are_rejected() {
        let base = PlacementRequest {
            root_seed: 0,
            epoch: 0,
            dataset_len: 1,
            batch_size: 1,
            world_size: 1,
            rank: 0,
            drop_last: false,
            exact_count: false,
        };
        assert_eq!(
            rank_placements(PlacementRequest {
                batch_size: 0,
                ..base
            }),
            Err(PlacementError::ZeroBatchSize)
        );
        assert_eq!(
            rank_placements(PlacementRequest {
                world_size: 0,
                ..base
            }),
            Err(PlacementError::ZeroWorldSize)
        );
        assert_eq!(elastic_batch_size(48, 8), Some(6));
        assert_eq!(elastic_batch_size(48, 7), None);
    }
}
