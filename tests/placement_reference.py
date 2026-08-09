"""Independent Python reference for native distributed placement."""

from rng_reference import FEISTEL_THRESHOLD, feistel_permute, materialized_permutation


def rank_placements(
    root_seed: int,
    epoch: int,
    dataset_len: int,
    batch_size: int,
    world_size: int,
    rank: int,
    drop_last: bool = False,
    exact_count: bool = False,
) -> list[tuple[int, int]]:
    """Return one rank's global positions and permuted dataset indices."""
    if batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid placement configuration")
    if dataset_len == 0:
        return []
    small_permutation = (
        materialized_permutation(root_seed, epoch, dataset_len)[0]
        if dataset_len < FEISTEL_THRESHOLD
        else None
    )

    def map_index(position: int) -> int:
        if small_permutation is not None:
            return small_permutation[position]
        return feistel_permute(root_seed, epoch, dataset_len, position)

    global_batch = batch_size * world_size
    full_end = dataset_len // global_batch * global_batch
    if drop_last or exact_count:
        regular_end = full_end
    else:
        regular_end = (dataset_len + global_batch - 1) // global_batch * global_batch
    output: list[tuple[int, int]] = []
    for batch_start in range(0, regular_end, global_batch):
        rank_start = batch_start + rank * batch_size
        for position in range(rank_start, rank_start + batch_size):
            perm_position = position if position < dataset_len else (position - dataset_len) % dataset_len
            output.append((position, map_index(perm_position)))
    if exact_count and not drop_last:
        tail_size = dataset_len - full_end
        rank_start = full_end + rank * tail_size // world_size
        rank_end = full_end + (rank + 1) * tail_size // world_size
        for position in range(rank_start, rank_end):
            output.append((position, map_index(position)))
    return output
