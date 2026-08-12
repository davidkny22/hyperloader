"""Build the universal pure-Python fallback wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path

import tomllib


def _digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}"


def build_wheel(root: Path, output: Path) -> Path:
    """Create one deterministic py3-none-any wheel from the fallback package."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    name = str(project["name"]).replace("-", "_")
    version = str(project["version"])
    dist_info = f"{name}-{version}.dist-info"
    filename = output / f"{name}-{version}-py3-none-any.whl"
    package = root / "python" / "hyperloader"
    entries: dict[str, bytes] = {}
    for path in sorted(package.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in {".so", ".pyd", ".dylib"}:
            continue
        relative = path.relative_to(package).as_posix()
        entries[f"hyperloader/{relative}"] = path.read_bytes()
    metadata = [
        "Metadata-Version: 2.4",
        f"Name: {project['name']}",
        f"Version: {version}",
        f"Summary: {project['description']}",
        f"Requires-Python: {project['requires-python']}",
        "License-Expression: MIT",
        "Description-Content-Type: text/markdown",
    ]
    metadata.extend(f"Requires-Dist: {item}" for item in project["dependencies"])
    description = (root / str(project["readme"])).read_text(encoding="utf-8")
    entries[f"{dist_info}/METADATA"] = ("\n".join(metadata) + f"\n\n{description}\n").encode()
    entries[f"{dist_info}/WHEEL"] = (
        b"Wheel-Version: 1.0\n"
        b"Generator: hyperloader fallback builder\n"
        b"Root-Is-Purelib: true\n"
        b"Tag: py3-none-any\n\n"
    )
    entries[f"{dist_info}/licenses/LICENSE"] = (root / "LICENSE").read_bytes()
    entries[f"{dist_info}/licenses/NOTICE"] = (root / "NOTICE").read_bytes()
    record_path = f"{dist_info}/RECORD"
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for path, data in sorted(entries.items()):
        writer.writerow((path, _digest(data), len(data)))
    writer.writerow((record_path, "", ""))
    entries[record_path] = record.getvalue().encode()
    output.mkdir(parents=True, exist_ok=True)
    timestamp = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as wheel:
        for path, data in sorted(entries.items()):
            info = zipfile.ZipInfo(path, timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            wheel.writestr(info, data)
    return filename


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    print(build_wheel(arguments.root.resolve(), arguments.output.resolve()))


if __name__ == "__main__":
    main()
