from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RELEASE_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "LICENSE",
    ROOT / "NOTICE",
    ROOT / "CHANGELOG.md",
    ROOT / "CONTRIBUTING.md",
)
BENCH_REGISTRY = ROOT / "docs" / "provenance" / "bench_registry.jsonl"


def _registry() -> dict[str, dict[str, object]]:
    records = [json.loads(line) for line in BENCH_REGISTRY.read_text(encoding="utf-8").splitlines()]
    return {str(record["id"]): record for record in records}


def _identity_result(record: dict[str, object]) -> tuple[str, str]:
    value = record["value"]
    interval = record["interval"]
    assert isinstance(value, dict)
    assert isinstance(interval, dict)
    mean = abs(float(value["mean_penalty_percent"]))
    lower = abs(float(interval["upper_percent"]))
    upper = abs(float(interval["lower_percent"]))
    return f"{mean:.3f}% faster", f"[{lower:.3f}%, {upper:.3f}%]"


def _overhead_result(record: dict[str, object]) -> tuple[str, str]:
    value = record["value"]
    interval = record["interval"]
    assert isinstance(value, dict)
    assert isinstance(interval, dict)
    mean = float(value["mean_penalty_percent"])
    lower = float(interval["lower_percent"])
    upper = float(interval["upper_percent"])
    return f"{mean:.3f}%", f"[{lower:.3f}%, {upper:.3f}%]"


def _assert_claim(readme: str, record_id: str, *, identity: bool) -> None:
    record = _registry()[record_id]
    assert record["status"] == "verified"
    result, interval = (
        _identity_result(record) if identity else _overhead_result(record)
    )
    assert result in readme
    assert interval in readme


def _assert_local_links_resolve(text: str, source: Path) -> None:
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        assert (source.parent / target).is_file(), target


def test_release_document_set_is_complete_and_product_facing() -> None:
    for path in RELEASE_DOCUMENTS:
        assert path.is_file(), path.name
        text = path.read_text(encoding="utf-8")
        assert "\N{EM DASH}" not in text
        assert not re.search(r"\bT\d{3}\b|task\.md|SPEC §|Phase [1-4]", text)
        _assert_local_links_resolve(text, path)

    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "hyperloader" in contributing
    assert "hypertok" not in contributing


@pytest.mark.skipif(not BENCH_REGISTRY.is_file(), reason="operational registry is unavailable")
def test_readme_numbers_trace_to_verified_registry_records() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for record_id in (
        "identity-fixed-text-spark",
        "identity-numpy-array-spark",
        "identity-arrow-tabular-spark",
    ):
        _assert_claim(readme, record_id, identity=True)
    for record_id in (
        "overhead-fixed-text-compute-spark",
        "overhead-fixed-text-bandwidth-spark",
    ):
        _assert_claim(readme, record_id, identity=False)


def test_notice_covers_the_documented_mechanism_lineage() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for source in (
        "PyTorch",
        "Philox",
        "Black and Rogaway",
        "NIST SP 800-38G",
        "Grain",
        "Graham's LPT",
        "SPDL",
        "FFCV",
        "StatefulDataLoader",
        "MosaicML Streaming",
        "MinatoLoader",
        "tf.data AUTOTUNE",
        "Plumber",
        "Cachew",
        "Pecan",
        "RINAS",
    ):
        assert source in notice


@pytest.mark.skipif(not BENCH_REGISTRY.is_file(), reason="operational registry is unavailable")
def test_claim_trace_rejects_a_changed_interval() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changed = readme.replace("[5.376%, 5.604%]", "[5.377%, 5.604%]")
    with pytest.raises(AssertionError):
        _assert_claim(changed, "identity-fixed-text-spark", identity=True)


def test_local_link_check_rejects_a_missing_artifact() -> None:
    with pytest.raises(AssertionError, match="missing-file"):
        _assert_local_links_resolve("[missing](missing-file)", ROOT / "README.md")
