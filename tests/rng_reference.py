"""Clear Python reference for hyperloader's counter-based RNG contract."""

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85


def splitmix64(value: int) -> int:
    """Apply the bare SplitMix64 finalizer using unsigned 64-bit arithmetic."""
    value &= MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & MASK64
    value ^= value >> 31
    return value


def key64(root_seed: int, epoch: int) -> int:
    """Derive the epoch-specific key defined by the contract."""
    epoch_tag = ((epoch << 32) | 0x9E37) & MASK64
    return splitmix64(root_seed) ^ splitmix64(epoch_tag)


def philox4x32_10(counter: tuple[int, int, int, int], key: tuple[int, int]) -> tuple[int, int, int, int]:
    """Evaluate the ten Random123 Philox4x32 rounds without mutable global state."""
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


def block(root_seed: int, epoch: int, coord: int, draw_index: int, stream_id: int) -> tuple[int, int, int, int]:
    """Derive one block using hyperloader's key and counter layout."""
    key = key64(root_seed, epoch)
    return philox4x32_10(
        (coord & MASK32, draw_index, stream_id, (coord >> 32) & MASK32),
        (key & MASK32, key >> 32),
    )


def sample_seed_words(root_seed: int, epoch: int, coord: int) -> tuple[int, int, tuple[int, int, int, int]]:
    """Return the torch, Python random, and NumPy seed material."""
    globals_block = block(root_seed, epoch, coord, 0, 0)
    numpy_block = block(root_seed, epoch, coord, 1, 0)
    torch_seed = globals_block[0] | (globals_block[1] << 32)
    random_seed = globals_block[2] | (globals_block[3] << 32)
    return torch_seed, random_seed, numpy_block
