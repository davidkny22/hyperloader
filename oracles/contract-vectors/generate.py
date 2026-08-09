"""Create the normative hyperloader contract-vector artifact once."""

from __future__ import annotations

import argparse
from pathlib import Path

from reference import build_document, serialize


def write_new(output: Path) -> None:
    """Write a new artifact while refusing to replace an existing contract."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialize(build_document()))


def main() -> None:
    """Parse the destination and create the vector file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("vectors.json"),
    )
    arguments = parser.parse_args()
    write_new(arguments.output)


if __name__ == "__main__":
    main()
