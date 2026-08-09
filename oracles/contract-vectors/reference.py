"""Pure Python reference engine for hyperloader's deterministic contracts."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
FEISTEL_THRESHOLD = 1 << 17
PERM_STREAM = 2
PERM_FUNCTION_STREAM = 3
FISHER_YATES_ROUND_WORD = 8
REQUIRED_DOMAINS = (
    1,
    2,
    3,
    65_536,
    (1 << 17) - 1,
    1 << 17,
    (1 << 17) + 1,
    300_000,
    1 << 20,
    1_000_000_007,
)
SEED_EPOCHS = (
    (0, 0),
    (1, 0),
    (1, 7),
    (MASK64, MASK32),
)


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
    """Derive the epoch-specific key."""
    return splitmix64(root_seed) ^ splitmix64(((epoch << 32) | 0x9E37) & MASK64)


def philox4x32_10(
    counter: tuple[int, int, int, int], key: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Evaluate Philox4x32-10 following the Random123 word order."""
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


def block(
    root_seed: int, epoch: int, coord: int, draw_index: int, stream_id: int
) -> tuple[int, int, int, int]:
    """Return one stream-separated block."""
    derived_key = key64(root_seed, epoch)
    return philox4x32_10(
        (coord & MASK32, draw_index, stream_id, (coord >> 32) & MASK32),
        (derived_key & MASK32, derived_key >> 32),
    )


def permutation_key(root_seed: int, epoch: int) -> tuple[int, int]:
    """Return the two words of the permutation key."""
    folded = splitmix64(key64(root_seed, epoch) ^ ((PERM_STREAM << 1) | 1))
    return folded & MASK32, folded >> 32


def feistel_permute(root_seed: int, epoch: int, domain: int, position: int) -> int:
    """Cycle-walk the eight-round unbalanced Feistel permutation."""
    if domain < FEISTEL_THRESHOLD or not 0 <= position < domain:
        raise ValueError("invalid Feistel domain or position")
    bits = (domain - 1).bit_length()
    lower_width = bits // 2
    key = permutation_key(root_seed, epoch)
    candidate = position
    while True:
        high_width = bits - lower_width
        low_width = lower_width
        high = candidate >> lower_width
        low = candidate & ((1 << lower_width) - 1)
        for round_index in range(8):
            mask = (1 << high_width) - 1
            function = philox4x32_10(
                (low, round_index, PERM_FUNCTION_STREAM, 0), key
            )[0] & mask
            high, low = low, (high + function) & mask
            high_width, low_width = low_width, high_width
        candidate = (high << lower_width) | low
        if candidate < domain:
            return candidate


def materialized_permutation(
    root_seed: int, epoch: int, domain: int
) -> tuple[list[int], int]:
    """Build the exact-uniform backward Fisher-Yates permutation."""
    if not 0 <= domain < FEISTEL_THRESHOLD:
        raise ValueError("invalid materialized domain")
    key = permutation_key(root_seed, epoch)
    permutation = list(range(domain))
    draw_ordinal = 0
    for upper in range(domain, 1, -1):
        limit = (1 << 32) - ((1 << 32) % upper)
        while True:
            word = philox4x32_10(
                (draw_ordinal, FISHER_YATES_ROUND_WORD, PERM_FUNCTION_STREAM, 0),
                key,
            )[0]
            draw_ordinal += 1
            if word < limit:
                selected = word % upper
                break
        permutation[upper - 1], permutation[selected] = (
            permutation[selected],
            permutation[upper - 1],
        )
    return permutation, draw_ordinal


def permutation_index(
    root_seed: int, epoch: int, domain: int, position: int
) -> int:
    """Evaluate a permutation position in either domain regime."""
    if not 0 <= position < domain:
        raise ValueError("invalid permutation position")
    if domain < FEISTEL_THRESHOLD:
        return materialized_permutation(root_seed, epoch, domain)[0][position]
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
    """Return one rank's positions and dataset indices."""
    if batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid placement configuration")
    if dataset_len == 0:
        return []
    small = (
        materialized_permutation(root_seed, epoch, dataset_len)[0]
        if dataset_len < FEISTEL_THRESHOLD
        else None
    )

    def map_index(position: int) -> int:
        if small is not None:
            return small[position]
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
            perm_position = (
                position
                if position < dataset_len
                else (position - dataset_len) % dataset_len
            )
            output.append((position, map_index(perm_position)))
    if exact_count and not drop_last:
        tail_size = dataset_len - full_end
        rank_start = full_end + rank * tail_size // world_size
        rank_end = full_end + (rank + 1) * tail_size // world_size
        for position in range(rank_start, rank_end):
            output.append((position, map_index(position)))
    return output


def _positions(domain: int) -> list[int]:
    candidates = {0, 1, domain // 2, domain - 2, domain - 1}
    return sorted(position for position in candidates if 0 <= position < domain)


def _permutation_digest(permutation: list[int]) -> str:
    digest = hashlib.sha256()
    for value in permutation:
        digest.update(value.to_bytes(8, "little"))
    return digest.hexdigest()


def _philox_vectors() -> list[dict[str, Any]]:
    inputs = itertools.product(
        (0, 1, MASK64),
        (0, 1, MASK32),
        (0, 1, MASK32, 1 << 32, MASK64),
        (0, 1, 2, MASK32),
        (0, 1, 4, 5, 6),
    )
    return [
        {
            "root_seed": root_seed,
            "epoch": epoch,
            "coord": coord,
            "draw_index": draw_index,
            "stream_id": stream_id,
            "words": list(block(root_seed, epoch, coord, draw_index, stream_id)),
        }
        for root_seed, epoch, coord, draw_index, stream_id in inputs
    ]


def _permutation_vectors() -> list[dict[str, Any]]:
    vectors = []
    for domain, (root_seed, epoch) in itertools.product(REQUIRED_DOMAINS, SEED_EPOCHS):
        positions = _positions(domain)
        vector: dict[str, Any] = {
            "root_seed": root_seed,
            "epoch": epoch,
            "domain": domain,
            "regime": "materialized" if domain < FEISTEL_THRESHOLD else "feistel",
        }
        if domain < FEISTEL_THRESHOLD:
            permutation, draws = materialized_permutation(root_seed, epoch, domain)
            vector["draw_count"] = draws
            vector["digest"] = _permutation_digest(permutation)
            vector["points"] = [[position, permutation[position]] for position in positions]
        else:
            vector["points"] = [
                [position, feistel_permute(root_seed, epoch, domain, position)]
                for position in positions
            ]
        vectors.append(vector)
    return vectors


def _placement_vectors() -> list[dict[str, Any]]:
    cases = (
        ("empty", 0, 1, 1, False, False),
        ("padded-tail", 5, 4, 8, False, False),
        ("exact-count-tail", 103, 4, 8, False, True),
        ("dropped-tail", 103, 4, 8, True, False),
        ("tiny-large-world", 7, 2, 48, False, False),
        ("single-rank", 17, 4, 1, False, False),
    )
    vectors = []
    for name, dataset_len, batch_size, world_size, drop_last, exact_count in cases:
        ranks = [
            {
                "rank": rank,
                "items": [
                    list(item)
                    for item in rank_placements(
                        11,
                        3,
                        dataset_len,
                        batch_size,
                        world_size,
                        rank,
                        drop_last,
                        exact_count,
                    )
                ],
            }
            for rank in range(world_size)
        ]
        vectors.append(
            {
                "name": name,
                "root_seed": 11,
                "epoch": 3,
                "dataset_len": dataset_len,
                "batch_size": batch_size,
                "world_size": world_size,
                "drop_last": drop_last,
                "exact_count": exact_count,
                "ranks": ranks,
            }
        )
    return vectors


def build_document() -> dict[str, Any]:
    """Build the complete deterministic contract-vector document."""
    return {
        "contract_version": 1,
        "bindings": {
            "fisher_yates_counter": ["draw_ordinal", 8, 3, 0],
            "fisher_yates_rejections_advance": True,
            "feistel_rounds": 8,
            "feistel_threshold": FEISTEL_THRESHOLD,
            "philox_rounds": 10,
        },
        "philox": {
            "random123_zero": [
                0x6627E8D5,
                0xE169C58D,
                0xBC57AC4C,
                0x9B00DBD8,
            ],
            "vectors": _philox_vectors(),
        },
        "permutations": _permutation_vectors(),
        "placements": _placement_vectors(),
    }


def validate_document(document: dict[str, Any]) -> None:
    """Reject incomplete or structurally ambiguous vector documents."""
    if set(document) != {
        "contract_version",
        "bindings",
        "philox",
        "permutations",
        "placements",
    }:
        raise ValueError("contract-vector top-level fields do not match the schema")
    if document["contract_version"] != 1:
        raise ValueError("unsupported contract version")
    bindings = document["bindings"]
    if bindings.get("fisher_yates_counter") != ["draw_ordinal", 8, 3, 0]:
        raise ValueError("the Fisher-Yates counter binding is not normative")
    if bindings.get("fisher_yates_rejections_advance") is not True:
        raise ValueError("Fisher-Yates rejections must advance the draw ordinal")
    philox_vectors = document["philox"].get("vectors")
    if not isinstance(philox_vectors, list) or len(philox_vectors) != 900:
        raise ValueError("the Philox boundary matrix must contain 900 vectors")
    domains = {vector.get("domain") for vector in document["permutations"]}
    if domains != set(REQUIRED_DOMAINS):
        raise ValueError("the permutation vectors do not cover every required domain")
    names = {vector.get("name") for vector in document["placements"]}
    if names != {
        "empty",
        "padded-tail",
        "exact-count-tail",
        "dropped-tail",
        "tiny-large-world",
        "single-rank",
    }:
        raise ValueError("the placement vectors do not cover every required mode")


def serialize(document: dict[str, Any]) -> str:
    """Return the canonical UTF-8 JSON representation."""
    validate_document(document)
    return json.dumps(document, indent=2, sort_keys=True) + "\n"
