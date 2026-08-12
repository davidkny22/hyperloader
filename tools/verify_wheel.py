"""Verify release wheel contents, metadata, size, and package build roots."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path, PurePosixPath

import tomllib

WHEEL_SIZE_LIMIT = 5 * 1024 * 1024
REQUIRED_DEPENDENCIES = {
    "numpy": frozenset({">=2.0"}),
    "torch": frozenset({">=2.10", "<2.14"}),
}
NATIVE_SUFFIXES = (".so", ".pyd", ".dylib")


def verify_build_graph(root: Path) -> set[str]:
    """Return package files reachable from the declared Python build root."""
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    maturin = config["tool"]["maturin"]
    source_root = (root / str(maturin["python-source"])).resolve()
    package_name = str(maturin["module-name"]).split(".", maxsplit=1)[0]
    package_root = (source_root / package_name).resolve()
    tests_root = (root / "tests").resolve()
    assert not tests_root.is_relative_to(package_root)
    assert package_root.is_dir()
    return {
        path.relative_to(source_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not path.name.endswith(NATIVE_SUFFIXES)
    }


def _requirements(metadata: str) -> dict[str, frozenset[str]]:
    requirements: dict[str, frozenset[str]] = {}
    for line in metadata.splitlines():
        if not line.startswith("Requires-Dist: "):
            continue
        requirement = line.removeprefix("Requires-Dist: ")
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*(.*)", requirement)
        assert match is not None
        name, constraints = match.groups()
        requirements[name] = frozenset(item.strip() for item in constraints.split(","))
    return requirements


def verify_wheel(path: Path, *, kind: str, root: Path) -> dict[str, int]:
    """Verify one native or fallback wheel against the release contract."""
    assert path.stat().st_size <= WHEEL_SIZE_LIMIT
    reachable = verify_build_graph(root)
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        parts = [PurePosixPath(name).parts for name in names]
        assert all("tests" not in item for item in parts)
        assert all(
            item[0] == "hyperloader" or item[0].endswith(".dist-info")
            for item in parts
        )
        assert reachable.issubset(set(names))
        dist_info = {item[0] for item in parts if item[0].endswith(".dist-info")}
        assert len(dist_info) == 1
        metadata = wheel.read(f"{next(iter(dist_info))}/METADATA").decode("utf-8")
        assert _requirements(metadata) == REQUIRED_DEPENDENCIES
        for document in ("LICENSE", "NOTICE"):
            assert any(
                name.endswith(f".dist-info/licenses/{document}") for name in names
            )
        native_count = sum(name.endswith(NATIVE_SUFFIXES) for name in names)
        assert native_count == (1 if kind == "native" else 0)
        if kind == "fallback":
            assert "hyperloader/_hyperloader.py" in names
    return {"bytes": path.stat().st_size, "files": len(names), "native_files": native_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("native", "fallback"), required=True)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    print(verify_wheel(arguments.wheel, kind=arguments.kind, root=arguments.root.resolve()))


if __name__ == "__main__":
    main()
