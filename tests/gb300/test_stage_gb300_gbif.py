import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "gb300"
sys.path.insert(0, str(SCRIPTS))
import stage_gb300_gbif as stage


def xml(key="occurrence/2026-07-01/occurrence.parquet/000001", truncated="false", token=""):
    return (f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f'<IsTruncated>{truncated}</IsTruncated><NextContinuationToken>{token}</NextContinuationToken>'
            f'<Contents><Key>{key}</Key><Size>10</Size><ETag>abc</ETag>'
            '<LastModified>today</LastModified></Contents></ListBucketResult>').encode()


def test_inventory(monkeypatch):
    monkeypatch.setattr(stage, "request", lambda url: (xml(), {}))
    entries = stage.inventory("2026-07-01")
    assert entries == [{"key": "occurrence/2026-07-01/occurrence.parquet/000001", "bytes": 10,
                        "etag": "abc", "last_modified": "today"}]


@pytest.mark.parametrize("body", [xml(key="wrong"), xml(truncated="true")])
def test_inventory_rejects_invalid(monkeypatch, body):
    monkeypatch.setattr(stage, "request", lambda url: (body, {}))
    with pytest.raises(ValueError):
        stage.inventory("2026-07-01")


def test_registry_pagination(monkeypatch):
    calls = []
    def fetch(url):
        calls.append(url)
        i = len(calls) - 1
        return json.dumps({"offset": i, "endOfRecords": bool(i),
                           "results": [{"key": str(i)}]}).encode(), {}
    monkeypatch.setattr(stage, "request", fetch)
    assert stage.organizations()["active_keys"] == ["0", "1"]
    assert "offset=1" in calls[1]


def test_remote_mutation(monkeypatch):
    monkeypatch.setattr(stage, "request", lambda *a, **kw: (b"", {"ETag": "changed", "Content-Length": "10"}))
    with pytest.raises(ValueError, match="Source changed"):
        stage.check_remote({"key": "key", "etag": "expected", "bytes": 10})


def test_resume_receipt_detects_corruption(tmp_path):
    (tmp_path / "parts").mkdir()
    (tmp_path / "receipts").mkdir()
    part = tmp_path / "parts/000000.parquet"
    part.write_bytes(b"good")
    source = {"key": "example"}
    receipt = {"source": source, "identity_sha256": "identity", "part": part.name,
               "bytes": 4, "sha256": stage.sha256_file(part), "records": 1}
    (tmp_path / "receipts/000000.json").write_text(json.dumps(receipt))
    assert stage.validate_receipt(tmp_path, 0, source, "identity") == receipt
    part.write_bytes(b"evil")
    with pytest.raises(ValueError, match="checksum"):
        stage.validate_receipt(tmp_path, 0, source, "identity")
