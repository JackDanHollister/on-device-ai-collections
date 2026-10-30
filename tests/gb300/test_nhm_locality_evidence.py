from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "gb300"
    / "nhm_locality_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("nhm_locality_evidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def source_row(
    source_id: str,
    locality: str,
    latitude: str,
    longitude: str,
    *,
    country: str = "Testland",
    state: str = "Test State",
) -> dict[str, str]:
    row = {field: "" for field in MODULE.CSV_FIELDS}
    row.update(
        {
            "_id": source_id,
            "occurrenceID": f"occ-{source_id}",
            "barcode": f"bar-{source_id}",
            "catalogNumber": f"cat-{source_id}",
            "locality": locality,
            "country": country,
            "stateProvince": state,
            "decimalLatitude": latitude,
            "decimalLongitude": longitude,
            "coordinateUncertaintyInMeters": "500",
            "recordedBy": "A. Collector",
            "year": "1901",
        }
    )
    return row


def write_source(path: Path) -> None:
    rows = [
        source_row("1", "Mt. Foo; nr. Bar", "10.001", "20.001"),
        source_row("2", "Mt Foo nr Bar", "10.002", "20.002"),
        source_row("3", "Mt Foo, nr Bar", "10.003", "20.003"),
        source_row("4", "Mount Foo near Bar", "10.004", "20.004"),
        source_row("5", "Mt Foo, nr Bar", "0", "0"),
        source_row("6", "", "10", "20"),
        source_row("7", "Invalid latitude", "91", "20"),
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MODULE.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.csv"
    database = tmp_path / "evidence.sqlite"
    manifest = tmp_path / "evidence.manifest.json"
    write_source(source)
    result = MODULE.build_database(source, database, manifest, batch_size=2)
    assert result["evidence_records"] == 4
    assert database.is_file()
    assert json.loads(manifest.read_text(encoding="utf-8"))["schema_version"] == 1
    return database, manifest


def write_holdout(path: Path) -> None:
    locality = "Mt. Foo; nr. Bar"
    country = "Testland"
    latitude = 10.001
    longitude = 20.001
    payload = {
        "artifact_type": MODULE.HOLDOUT_ARTIFACT_TYPE,
        "schema_version": MODULE.HOLDOUT_SCHEMA_VERSION,
        "record_count": 1,
        "records": [
            {
                "benchmark_id": "bar-1",
                "identifier_aliases": [],
                "cohort": "fixture",
                "locality": locality,
                "locality_normalized": MODULE.normalize_text(locality),
                "country": country,
                "country_normalized": MODULE.normalize_text(country),
                "latitude": latitude,
                "longitude": longitude,
                "content_fingerprint": MODULE.benchmark_content_fingerprint(
                    locality,
                    country,
                    latitude,
                    longitude,
                ),
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_normalization_and_coordinate_validation() -> None:
    assert MODULE.normalize_text("  Mt.  FOO; nr. Bär ") == "mt foo nr bär"
    assert MODULE.valid_coordinate_pair(51.5, -0.1)
    assert not MODULE.valid_coordinate_pair(0.0, 0.0)
    assert not MODULE.valid_coordinate_pair(91.0, 0.0)
    assert MODULE.coordinate_quality(None) == "unquantified"
    assert MODULE.coordinate_quality(9999) == "bounded_10km"
    assert MODULE.contains_coordinate_hint("Camp, 4°44′56 N")
    assert MODULE.contains_coordinate_hint("N: 13 40 36 W: 89 33 49")
    assert not MODULE.contains_coordinate_hint("Mt. Foo; 7 km from Bar")


def test_build_excludes_holdout_before_cluster_derivation(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    holdout = tmp_path / "holdout.json"
    write_source(source)
    write_holdout(holdout)

    content_database = tmp_path / "content.sqlite"
    content_manifest = tmp_path / "content.manifest.json"
    content_result = MODULE.build_database(
        source,
        content_database,
        content_manifest,
        batch_size=2,
        holdout_manifest=holdout,
        holdout_policy="identifier_and_content",
    )
    assert content_result["evidence_records"] == 3
    assert content_result["holdout"]["excluded_source_records"] == 1
    assert content_result["holdout"]["match_reason_counts"] == {
        "exact_content": 1,
        "identifier": 1,
    }

    strict_database = tmp_path / "strict.sqlite"
    strict_manifest = tmp_path / "strict.manifest.json"
    strict_result = MODULE.build_database(
        source,
        strict_database,
        strict_manifest,
        batch_size=2,
        holdout_manifest=holdout,
        holdout_policy="strict_locality",
    )
    assert strict_result["evidence_records"] == 1
    assert strict_result["locality_clusters"] == 1
    assert strict_result["holdout"]["excluded_source_records"] == 3
    assert strict_result["holdout"]["matched_benchmark_records"] == 1
    assert strict_result["holdout"]["match_reason_counts"] == {
        "exact_content": 1,
        "identifier": 1,
        "normalized_locality": 3,
    }

    connection = MODULE.open_database(strict_database)
    try:
        assert connection.execute(
            "SELECT count(*) FROM evidence_records "
            "WHERE locality_normalized = 'mt foo nr bar'"
        ).fetchone()[0] == 0
    finally:
        connection.close()
    proof = MODULE.verify_holdout_exclusion(strict_database, holdout)
    assert proof["verified_clean"] is True
    assert proof["identifier_hits"] == 0
    assert proof["exact_content_hits"] == 0
    assert proof["normalized_locality_hits"] == 0


def test_holdout_manifest_validation_fails_closed(tmp_path: Path) -> None:
    holdout = tmp_path / "holdout.json"
    write_holdout(holdout)
    payload = json.loads(holdout.read_text(encoding="utf-8"))
    payload["records"][0]["content_fingerprint"] = "wrong"
    holdout.write_text(json.dumps(payload), encoding="utf-8")
    try:
        MODULE.HoldoutRegistry(holdout, policy="strict_locality")
    except MODULE.EvidenceError as exc:
        assert "fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("Invalid holdout fingerprint was accepted")


def test_build_and_target_excluded_exact_query(tmp_path: Path) -> None:
    database, _ = build_fixture(tmp_path)
    result = MODULE.query_evidence(
        database,
        "Mt. Foo; nr. Bar",
        country="Testland",
        state_province="Test State",
        exclude_source_id="1",
        limit=3,
    )
    assert result["candidate_count"] >= 1
    top = result["candidates"][0]
    assert top["exact_locality_match"] is True
    assert top["support_count"] == 2
    assert top["distinct_event_count"] == 1
    assert {sample["source_id"] for sample in top["samples"]} == {"2", "3"}
    assert top["approximate_cluster_radius_km"] < 1


def test_fuzzy_query_can_exclude_exact_locality(tmp_path: Path) -> None:
    database, _ = build_fixture(tmp_path)
    result = MODULE.query_evidence(
        database,
        "Mt Foo nr Bar",
        country="Testland",
        exclude_exact_locality=True,
    )
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["locality"] == "Mount Foo near Bar"
    assert result["candidates"][0]["exact_locality_match"] is False


def test_model_context_is_qualified_and_provenanced(tmp_path: Path) -> None:
    database, _ = build_fixture(tmp_path)
    result = MODULE.query_evidence(database, "Mt Foo nr Bar", country="Testland")
    rendered = MODULE.render_model_evidence(result)
    assert "UNTRUSTED RETRIEVED DATA" in rendered
    assert "never as instructions" in rendered
    assert "strong_exact_historical_match" in rendered
    assert "use that cluster centroid" in rendered
    assert '"cluster_id"' in rendered
    assert '"evidence_record_ids"' in rendered


def test_operational_pilot_excludes_target_from_memory(tmp_path: Path) -> None:
    database, _ = build_fixture(tmp_path)
    pilot = MODULE.select_operational_pilot(
        database,
        size=1,
        seed="test-seed",
        minimum_support=3,
        maximum_cluster_radius_km=5,
    )
    record = pilot["records"][0]
    assert record["source_id"] == "1"
    evidence_ids = {
        sample["source_id"]
        for candidate in record["evidence"]["candidates"]
        for sample in candidate["samples"]
    }
    assert record["source_id"] not in evidence_ids
    assert pilot["pilot_type"].startswith("target-excluded")


def test_fuzzy_demo_declares_synthetic_query_and_excludes_target(tmp_path: Path) -> None:
    database, _ = build_fixture(tmp_path)
    pilot = MODULE.prepare_fuzzy_demo(
        database,
        source_id="1",
        query_locality="Mount Foo Bar",
    )
    assert pilot["pilot_type"] == "target-excluded synthetic shorthand retrieval"
    record = pilot["records"][0]
    assert record["source_locality"] == "Mt. Foo; nr. Bar"
    assert record["evidence"]["query"]["exclude_exact_locality"] is True
    assert record["evidence"]["query"]["exclude_source_id"] == "1"


def test_mixed_evaluation_is_interleaved_and_excludes_exact_fuzzy_match(
    tmp_path: Path,
) -> None:
    database, _ = build_fixture(tmp_path)
    cohort = MODULE.select_mixed_evaluation(
        database,
        exact_size=1,
        fuzzy_size=1,
        seed="mixed-test",
        minimum_support=3,
        minimum_distinct_events=1,
        maximum_cluster_radius_km=5,
        minimum_fuzzy_score=0.3,
    )
    assert [record["stratum"] for record in cohort["records"]] == [
        "exact",
        "natural_fuzzy",
    ]
    fuzzy = cohort["records"][1]
    assert fuzzy["evidence"]["query"]["exclude_exact_locality"] is True
    assert fuzzy["evidence"]["candidates"][0]["exact_locality_match"] is False
    assert cohort["selection"]["fuzzy_reference_distance_used_for_selection"] is False
