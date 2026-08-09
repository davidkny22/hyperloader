use _hyperloader::rng::{PlacementError, PlacementRequest, elastic_batch_size, rank_placements};

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
