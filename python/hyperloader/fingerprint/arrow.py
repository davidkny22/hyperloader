"""Arrow table and Parquet dataset fingerprint elements."""

from __future__ import annotations

import hashlib
from typing import Any

from .callable import stable_value
from .files import file_elements
from .model import FingerprintElement


def arrow_elements(dataset: Any, mode: str, prefix: str) -> list[FingerprintElement]:
    """Describe a Hugging Face Arrow dataset's schema and logical rows."""
    data = dataset._data
    schema = getattr(data, "schema", None)
    elements = [
        FingerprintElement(f"{prefix}.schema", str(schema)),
        FingerprintElement(f"{prefix}.row_count", len(dataset)),
        FingerprintElement(f"{prefix}.format_type", stable_value(dataset._format_type)),
        FingerprintElement(
            f"{prefix}.format_columns", stable_value(dataset._format_columns)
        ),
        FingerprintElement(
            f"{prefix}.output_all_columns", bool(dataset._output_all_columns)
        ),
        FingerprintElement(
            f"{prefix}.format_kwargs", stable_value(dataset._format_kwargs or {})
        ),
    ]
    cache_files = [row["filename"] for row in getattr(dataset, "cache_files", ())]
    if cache_files:
        elements.extend(file_elements(cache_files, mode, f"{prefix}.files"))
    if mode == "strict":
        elements.append(
            FingerprintElement(f"{prefix}.content_sha256", _arrow_digest(data))
        )
    return elements


def parquet_elements(dataset: Any, mode: str, prefix: str) -> list[FingerprintElement]:
    """Describe a Parquet dataset's schema, rows, and ordered fragments."""
    try:
        fragments = tuple(dataset.get_fragments())
    except (AttributeError, TypeError):
        fragments = ()
    elements = [
        FingerprintElement(f"{prefix}.schema", str(getattr(dataset, "schema", None))),
        FingerprintElement(f"{prefix}.row_count", int(dataset.count_rows())),
    ]
    elements.extend(
        file_elements(
            [fragment.path for fragment in fragments], mode, f"{prefix}.files"
        )
    )
    return elements


def _arrow_digest(data: Any) -> str:
    digest = hashlib.sha256()
    table = getattr(data, "table", data)
    for column in table.columns:
        for chunk in column.chunks:
            for buffer in chunk.buffers():
                if buffer is not None:
                    digest.update(memoryview(buffer))
    return digest.hexdigest()
