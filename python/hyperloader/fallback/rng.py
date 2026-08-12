"""Pure-Python implementation of the sealed RNG and placement contracts."""

from __future__ import annotations

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
FEISTEL_THRESHOLD = 1 << 17


def splitmix64(value: int) -> int:
    """Apply the bare SplitMix64 finalizer with unsigned arithmetic."""
    value &= MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & MASK64
    value ^= value >> 31
    return value


def key64(root_seed: int, epoch: int) -> int:
    """Derive the epoch-specific contract key."""
    epoch_tag = ((epoch << 32) | 0x9E37) & MASK64
    return splitmix64(root_seed) ^ splitmix64(epoch_tag)


def philox4x32_10(
    counter: tuple[int, int, int, int], key: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Evaluate ten Random123 Philox4x32 rounds."""
    words = tuple(word & MASK32 for word in counter)
    key0, key1 = (word & MASK32 for word in key)
    for round_index in range(10):
        product0 = PHILOX_M0 * words[0]
        product1 = PHILOX_M1 * words[2]
        lo0, hi0 = product0 & MASK32, product0 >> 32
        lo1, hi1 = product1 & MASK32, product1 >> 32
        words = (
            (hi1 ^ words[1] ^ key0) & MASK32,
            lo1,
            (hi0 ^ words[3] ^ key1) & MASK32,
            lo0,
        )
        if round_index != 9:
            key0 = (key0 + PHILOX_W0) & MASK32
            key1 = (key1 + PHILOX_W1) & MASK32
    return words


def rng_block(
    root_seed: int, epoch: int, coord: int, draw_index: int, stream_id: int
) -> tuple[int, int, int, int]:
    """Derive one block from a root seed and epoch."""
    return rng_block_from_key(key64(root_seed, epoch), coord, draw_index, stream_id)


def rng_block_from_key(
    key: int, coord: int, draw_index: int, stream_id: int
) -> tuple[int, int, int, int]:
    """Derive one block from an already resolved key."""
    return philox4x32_10(
        (coord & MASK32, draw_index, stream_id, (coord >> 32) & MASK32),
        (key & MASK32, key >> 32),
    )


def sample_rng_context(root_seed: int, epoch: int, coord: int) -> tuple[int, int, int]:
    """Return the sample torch seed, epoch key, and coordinate."""
    key = key64(root_seed, epoch)
    words = rng_block_from_key(key, coord, 0, 0)
    return words[0] | (words[1] << 32), key, coord


def feistel_permute(root_seed: int, epoch: int, domain: int, position: int) -> int:
    """Cycle-walk the eight-round unbalanced Feistel permutation."""
    if domain < FEISTEL_THRESHOLD or not 0 <= position < domain:
        raise ValueError(
            "the Feistel permutation requires a domain of at least 131072 "
            "and a position inside it"
        )
    bits = (domain - 1).bit_length()
    lower_width = bits // 2
    permutation_key = splitmix64(key64(root_seed, epoch) ^ 5)
    key = (permutation_key & MASK32, permutation_key >> 32)
    candidate = position
    while True:
        high_width = bits - lower_width
        high = candidate >> lower_width
        low = candidate & ((1 << lower_width) - 1)
        for round_index in range(8):
            mask = (1 << high_width) - 1
            function = philox4x32_10((low, round_index, 3, 0), key)[0] & mask
            high, low = low, (high + function) & mask
            high_width = bits - high_width
        candidate = (high << lower_width) | low
        if candidate < domain:
            return candidate


def materialized_permutation(root_seed: int, epoch: int, domain: int) -> list[int]:
    """Build the exact-uniform Fisher-Yates permutation."""
    if not 0 <= domain < FEISTEL_THRESHOLD:
        raise ValueError(
            "the materialized permutation requires a domain smaller than 131072"
        )
    permutation_key = splitmix64(key64(root_seed, epoch) ^ 5)
    key = (permutation_key & MASK32, permutation_key >> 32)
    permutation = list(range(domain))
    draw_ordinal = 0
    for upper in range(domain, 1, -1):
        limit = (1 << 32) - ((1 << 32) % upper)
        while True:
            word = philox4x32_10((draw_ordinal, 8, 3, 0), key)[0]
            draw_ordinal += 1
            if word < limit:
                selected = word % upper
                break
        permutation[upper - 1], permutation[selected] = (
            permutation[selected],
            permutation[upper - 1],
        )
    return permutation


def permutation_index(root_seed: int, epoch: int, domain: int, position: int) -> int:
    """Evaluate either sealed permutation regime."""
    if not 0 <= position < domain:
        raise ValueError("the permutation position must be inside its domain")
    if domain < FEISTEL_THRESHOLD:
        return materialized_permutation(root_seed, epoch, domain)[position]
    return feistel_permute(root_seed, epoch, domain, position)


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
    """Return one rank's global positions and dataset indices."""
    if batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid placement configuration")
    if dataset_len == 0:
        return []
    permutation = (
        materialized_permutation(root_seed, epoch, dataset_len)
        if dataset_len < FEISTEL_THRESHOLD
        else None
    )

    def map_index(position: int) -> int:
        if permutation is not None:
            return permutation[position]
        return feistel_permute(root_seed, epoch, dataset_len, position)

    global_batch = batch_size * world_size
    full_end = dataset_len // global_batch * global_batch
    regular_end = (
        full_end
        if drop_last or exact_count
        else (dataset_len + global_batch - 1) // global_batch * global_batch
    )
    output: list[tuple[int, int]] = []
    for batch_start in range(0, regular_end, global_batch):
        rank_start = batch_start + rank * batch_size
        for position in range(rank_start, rank_start + batch_size):
            source = (
                position
                if position < dataset_len
                else (position - dataset_len) % dataset_len
            )
            output.append((position, map_index(source)))
    if exact_count and not drop_last:
        tail_size = dataset_len - full_end
        rank_start = full_end + rank * tail_size // world_size
        rank_end = full_end + (rank + 1) * tail_size // world_size
        output.extend(
            (position, map_index(position)) for position in range(rank_start, rank_end)
        )
    return output
