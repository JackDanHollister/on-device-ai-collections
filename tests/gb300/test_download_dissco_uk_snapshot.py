from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "gb300"
    / "download_dissco_uk_snapshot.py"
)
SPEC = importlib.util.spec_from_file_location("download_dissco_uk_snapshot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_parse_organization_keys_filters_deleted_and_sorts() -> None:
    payload = {
        "results": [
            {"key": "b"},
            {"key": "a"},
            {"key": "deleted", "deleted": "2020-01-01"},
            {"key": "a"},
        ]
    }
    assert MODULE.parse_organization_keys(payload) == ["a", "b"]


def test_select_uses_portal_basis_types_and_snapshot_proxies() -> None:
    sql = MODULE.build_select_sql("s3://example/occurrence.parquet/*")
    assert "publishingorgkey IN" in sql
    assert "institutioncode" in sql
    assert "collectioncode" in sql
    assert "PRESERVED_SPECIMEN" in sql
    assert "LIVING_SPECIMEN" in sql
    assert "s3://example/occurrence.parquet/*" in sql
