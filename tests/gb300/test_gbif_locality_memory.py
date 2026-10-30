"""Adversarial tests for the benchmark-clean GBIF specimen-memory importer.

Fixtures use actual Parquet files and the public stager's identity/receipt
format.  They never depend on the in-progress production download.
"""
from __future__ import annotations

import csv
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import duckdb
import pytest


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_standalone_completed_verifier_rejects_mutable_sidecars(tmp_path, holdout, suffix):
    source = source_fixture(tmp_path, [[raw_record()]])
    output = tmp_path / "memory"
    run_build(source, output, holdout)
    (output / (memory.DB_FILE + suffix)).write_bytes(b"")
    with pytest.raises(ValueError, match="sidecar"):
        memory.verify_output(output)


def test_standalone_completed_verifier_handles_uri_characters(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record()]])
    output = tmp_path / "memory # question?"
    run_build(source, output, holdout)
    assert memory.verify_output(output)["records"] == 1

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "gb300"
sys.path.insert(0, str(SCRIPTS))
import build_benchmark_holdout_manifest as holdouts
import gbif_locality_memory as memory


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def reference(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=(
            "Barcode", "label_verbatim", "Country", "lat_manual", "long_manual"))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def holdout(tmp_path):
    full = tmp_path / "reference.tsv"
    alternate = tmp_path / "alternate.tsv"
    one = {"Barcode": "HOLDOUT-001", "label_verbatim": "1st label: Île Test; 2nd label: Banks Is.",
           "Country": "GB", "lat_manual": "51.9", "long_manual": "-1.9"}
    reference(full, [one])
    reference(alternate, [{**one, "label_verbatim": "Old specimen site spelling"}])
    manifest = holdouts.build_manifest(
        [holdouts.SourceSpec("odonata", p, holdouts.REFERENCE_KIND) for p in (full, alternate)],
        expected_record_count=1, allow_multiple_references=True, allow_content_variants=True)
    path = tmp_path / "holdout.json"
    path.write_bytes(holdouts.manifest_bytes(manifest))
    return path, digest(path)


@pytest.fixture
def exclusions(holdout):
    return memory.Exclusions(*holdout)


def raw_record(**changes):
    row = {
        "gbifid": "10001", "occurrenceid": "https://example.invalid/specimen/10001",
        "datasetkey": "11111111-1111-4111-8111-111111111111",
        "publishingorgkey": "22222222-2222-4222-8222-222222222222",
        "catalognumber": "SPECIMEN-10001", "recordnumber": "FIELD-10001",
        "locality": "Independent specimen site", "countrycode": "GB",
        "stateprovince": "Somerset", "decimallatitude": 51.2,
        "decimallongitude": -2.5, "coordinateuncertaintyinmeters": 150.0,
        "license": "CC_BY_4_0", "rightsholder": "Fixture museum",
        "basisofrecord": "PRESERVED_SPECIMEN", "institutioncode": "FIXTURE",
        "collectioncode": "TEST", "issue": [],
    }
    row.update(changes)
    return row


def parquet(path: Path, rows: list[dict]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    real_fields = {"decimallatitude", "decimallongitude", "coordinateuncertaintyinmeters"}
    definitions = [f'"{field}" ' + ("DOUBLE" if field in real_fields else
                   "VARCHAR[]" if field == "issue" else "VARCHAR") for field in fields]
    with duckdb.connect(":memory:") as db:
        db.execute("CREATE TABLE raw (" + ",".join(definitions) + ")")
        db.executemany("INSERT INTO raw VALUES (" + ",".join("?" for _ in fields) + ")",
                       [[row.get(field) for field in fields] for row in rows])
        db.execute("COPY raw TO ? (FORMAT PARQUET)", [str(path)])


def source_fixture(tmp_path, groups, *, complete=True, missing_indices=()):
    root = tmp_path / "source"
    (root / "parts").mkdir(parents=True)
    (root / "receipts").mkdir()
    write_json(root / "organizations.json", {"active_keys": ["fixture-publisher"], "pages": []})
    (root / "citation.txt").write_text("Public GBIF fixture citation.\n")
    sources = [{"key": f"occurrence/2026-07-01/occurrence.parquet/{i + 1:06d}",
                "bytes": 1000 + i, "etag": f'"remote-etag-{i}"',
                "last_modified": "2026-07-09T12:00:00Z"} for i in range(len(groups))]
    identity = {
        "schema": "placebot-gbif-raw-proxy-v1", "snapshot": "2026-07-01",
        "bucket": "https://gbif-open-data-eu-central-1.s3.eu-central-1.amazonaws.com/",
        "sources": sources, "registry_sha256": digest(root / "organizations.json"),
        "citation_url": "https://example.invalid/citation.txt",
        "citation_sha256": digest(root / "citation.txt"),
        "active_organization_keys": ["fixture-publisher"],
        "predicate_sql": "publishingorgkey IN (SELECT key FROM uk_organization_keys)",
        "live_portal_predicate": "fixture-only", "qualification": "RAW fixture; not evidence",
        "duckdb_version": duckdb.__version__, "code_sha256": {"fixture": "0" * 64},
    }
    write_json(root / "identity.json", identity)
    receipts = []
    for index, rows in enumerate(groups):
        part = root / "parts" / f"{index:06d}.parquet"
        parquet(part, rows)
        receipt = {"source": sources[index], "identity_sha256": digest(root / "identity.json"),
                   "part": part.name, "bytes": part.stat().st_size, "sha256": digest(part),
                   "records": len(rows), "nhm_records": 0, "seconds": 0.1,
                   "completed_at": "2026-09-05T12:00:00Z"}
        if index not in missing_indices:
            write_json(root / "receipts" / f"{index:06d}.json", receipt)
            receipts.append(receipt)
    if complete:
        write_json(root / "manifest.json", {
            "identity_sha256": digest(root / "identity.json"), "qualification": identity["qualification"],
            "evaluation_exclusions_applied": False, "completed_at": "2026-09-05T12:00:00Z",
            "records": sum(r["records"] for r in receipts), "nhm_records": 0,
            "parts": receipts, "read_glob": str(root / "parts" / "*.parquet"),
        })
        write_json(root / "COMPLETE.json", {
            "status": "complete", "manifest_sha256": digest(root / "manifest.json")})
    return root


def run_build(source, output, holdout, **kwargs):
    return memory.build(source, output, holdout[0], holdout[1], **kwargs)


def stored_rows(output, table="records"):
    with closing(sqlite3.connect(output / memory.DB_FILE)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(f'SELECT * FROM "{table}"')]


def test_accepts_valid_row_without_inventing_fields(exclusions):
    record, reason = memory.normalize_record(raw_record(), exclusions)
    assert reason == "accepted"
    assert record["latitude"] == 51.2
    assert record["longitude"] == -2.5
    assert record["uncertainty_m"] == 150
    assert record["gbif_id"] == "10001"


@pytest.mark.parametrize("locality", [
    "1st label: Île Test; 2nd label: Banks Is.", "I\u0302LE TEST", "Banks Is.",
    "old SPECIMEN-site spelling",
])
def test_strict_holdout_locality_and_label_variants_excluded(exclusions, locality):
    row = raw_record(locality=locality, countrycode="AU", decimallatitude=-33.0)
    assert exclusions.match(row), "Holdout locality matching must not depend on country/coordinates"
    normalized, reason = memory.normalize_record(row, exclusions)
    assert normalized is None
    assert reason


@pytest.mark.parametrize("field", ["catalognumber", "occurrenceid", "recordnumber"])
def test_known_identifier_excluded_even_when_locality_changes(exclusions, field):
    assert exclusions.match(raw_record(**{field: "HOLDOUT-001"}))


def test_gbif_identifier_namespace_is_not_mistaken_for_benchmark_barcode(exclusions):
    assert exclusions.match(raw_record(gbifid="HOLDOUT-001")) == set()


def test_identifier_normalization_is_case_and_whitespace_stable():
    assert memory.normalize_identifier("  Holdout-001  ") == memory.normalize_identifier("HOLDOUT-001")
    assert memory.normalize_identifier(None) == ""
    assert memory.normalize_identifier("000123") != memory.normalize_identifier("123")


def test_numeric_nhm_aliases_require_nhm_context(tmp_path):
    path = tmp_path / "numeric.tsv"
    reference(path, [
        {"Barcode": "99999", "label_verbatim": "One holdout locality", "Country": "GB",
         "lat_manual": "55", "long_manual": "-1"},
        {"Barcode": "00123", "label_verbatim": "Another holdout locality", "Country": "GB",
         "lat_manual": "55", "long_manual": "-2"},
    ])
    manifest = holdouts.build_manifest([holdouts.SourceSpec("bombus", path, holdouts.REFERENCE_KIND)],
                                       expected_record_count=2)
    manifest_path = tmp_path / "numeric-holdout.json"
    manifest_path.write_bytes(holdouts.manifest_bytes(manifest))
    exclusions = memory.Exclusions(manifest_path, digest(manifest_path))
    assert exclusions.match(raw_record(catalognumber="NHMUK000123"))
    assert exclusions.match(raw_record(catalognumber="123", institutioncode="NHMUK"))
    assert exclusions.match(raw_record(catalognumber="123", datasetkey=memory.NHM_DATASET))
    assert not exclusions.match(raw_record(catalognumber="123")), "Generic institution IDs keep their namespace"
    assert not exclusions.match(raw_record(gbifid="123", institutioncode="NHMUK"))


def test_pinned_holdout_hash_required(holdout):
    with pytest.raises((ValueError, RuntimeError), match="(?i)(sha|hash|digest|checksum)"):
        memory.Exclusions(holdout[0], "0" * 64)


def test_holdout_internal_record_hash_checked_even_if_file_repinned(holdout):
    path, _ = holdout
    payload = json.loads(path.read_text())
    payload["records"][0]["locality"] = "Changed without recomputing internal hashes"
    write_json(path, payload)
    with pytest.raises((ValueError, RuntimeError)):
        memory.Exclusions(path, digest(path))


@pytest.mark.parametrize("uncertainty", [None, "", " "])
def test_unknown_uncertainty_stays_unknown(exclusions, uncertainty):
    record, reason = memory.normalize_record(raw_record(coordinateuncertaintyinmeters=uncertainty), exclusions)
    assert reason == "accepted"
    assert record["uncertainty_m"] is None


@pytest.mark.parametrize("uncertainty", [0, -1, float("nan"), float("inf"), "not-a-number"])
def test_invalid_uncertainty_is_not_silently_promoted(exclusions, uncertainty):
    record, reason = memory.normalize_record(raw_record(coordinateuncertaintyinmeters=uncertainty), exclusions)
    assert record is None
    assert reason


def test_coarse_uncertainty_boundary_is_explicit(exclusions):
    accepted, _ = memory.normalize_record(raw_record(coordinateuncertaintyinmeters=100000), exclusions)
    assert accepted["uncertainty_m"] == 100000
    assert memory.normalize_record(raw_record(coordinateuncertaintyinmeters=100000.01), exclusions)[0] is None


@pytest.mark.parametrize("license", ["CC0_1_0", "CC_BY_4_0"])
def test_supported_public_licenses_preserved(exclusions, license):
    record, reason = memory.normalize_record(raw_record(license=license), exclusions)
    assert reason == "accepted"
    assert record["license"] == license


@pytest.mark.parametrize("license", ["CC_BY_NC_4_0", "", None, "UNKNOWN"])
def test_nc_and_unknown_licenses_excluded_by_default(exclusions, license):
    assert memory.normalize_record(raw_record(license=license), exclusions)[0] is None


@pytest.mark.parametrize("issue", ["COORDINATE_INVALID", "COORDINATE_OUT_OF_RANGE",
                                   "ZERO_COORDINATE", "COUNTRY_COORDINATE_MISMATCH",
                                   "COORDINATE_UNCERTAINTY_METERS_INVALID", "COORDINATE_ACCURACY_INVALID",
                                   "COORDINATE_PRECISION_UNCERTAINTY_MISMATCH"])
def test_risky_coordinate_issues_excluded(exclusions, issue):
    assert memory.normalize_record(raw_record(issue=[issue]), exclusions)[0] is None


def test_taxon_issue_is_not_used_as_geographic_evidence(exclusions):
    record, reason = memory.normalize_record(raw_record(issue=["TAXON_MATCH_FUZZY"]), exclusions)
    assert reason == "accepted"
    assert record is not None


@pytest.mark.parametrize("issues", ["COORDINATE_INVALID", {"COORDINATE_INVALID": True}, [None], [42]])
def test_malformed_issue_collection_is_not_silently_ignored(exclusions, issues):
    assert memory.normalize_record(raw_record(issue=issues), exclusions)[0] is None


@pytest.mark.parametrize("latitude,longitude", [(91, 0), (0, 181), (0, 0),
                                                (None, -2.5), (float("nan"), -2.5)])
def test_invalid_coordinates_rejected(exclusions, latitude, longitude):
    assert memory.normalize_record(raw_record(decimallatitude=latitude,
        decimallongitude=longitude), exclusions)[0] is None


def test_whitelist_prevents_reference_and_taxon_persistence(tmp_path, holdout):
    sentinel = "FORBIDDEN_REFERENCE_OR_TAXON_SENTINEL"
    forbidden = {field: sentinel for field in ("lat_manual", "long_manual", "radius_km_manual",
                 "ai_processed_result", "scientificname", "family", "genus", "species", "taxonkey")}
    source = source_fixture(tmp_path, [[raw_record(**forbidden)]])
    output = tmp_path / "memory"
    manifest = run_build(source, output, holdout)
    rows = stored_rows(output)
    assert manifest["records"] == 1
    assert len(rows) == 1
    assert not (set(forbidden) & set(rows[0]))
    assert sentinel not in json.dumps(rows)
    assert sentinel.encode() not in (output / memory.DB_FILE).read_bytes()


def test_exact_coordinate_groups_do_not_round_together(tmp_path, holdout):
    rows = [raw_record(), raw_record(gbifid="10002", occurrenceid="urn:10002",
             decimallatitude=51.2000001), raw_record(gbifid="10003", occurrenceid="urn:10003")]
    source = source_fixture(tmp_path, [rows])
    output = tmp_path / "memory"
    manifest = run_build(source, output, holdout)
    assert manifest["records"] == 3
    assert manifest["places"] == 2
    places = stored_rows(output, "places")
    assert sorted(p["support_count"] for p in places) == [1, 2]
    assert {p["latitude"] for p in places} == {51.2, 51.2000001}


def test_group_uncertainty_never_hides_unknown_support(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record(), raw_record(gbifid="10002",
        occurrenceid="urn:10002", coordinateuncertaintyinmeters=None)]])
    output = tmp_path / "memory"
    run_build(source, output, holdout)
    place, = stored_rows(output, "places")
    assert place["support_count"] == 2
    assert place["uncertainty_m"] is None


def test_support_counts_distinguish_republished_occurrence(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record(), raw_record(gbifid="10002")]])
    output = tmp_path / "memory"
    run_build(source, output, holdout)
    place, = stored_rows(output, "places")
    assert place["support_count"] == 2
    assert place["distinct_occurrence_count"] == 1


def test_distinct_occurrence_count_tags_identifier_namespace(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record(occurrenceid="10002"),
                                       raw_record(gbifid="10002", occurrenceid="")]])
    output = tmp_path / "memory"
    run_build(source, output, holdout)
    place, = stored_rows(output, "places")
    assert place["support_count"] == 2
    assert place["distinct_occurrence_count"] == 2


def test_same_place_retains_both_per_record_licenses(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record(), raw_record(gbifid="10002", license="CC0_1_0")]])
    output = tmp_path / "memory"
    run_build(source, output, holdout)
    place, = stored_rows(output, "places")
    assert set(json.loads(place["license_set_json"])) == {"CC_BY_4_0", "CC0_1_0"}
    assert {row["license"] for row in stored_rows(output)} == {"CC_BY_4_0", "CC0_1_0"}


def test_identical_gbif_records_are_deduplicated_across_parts(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record()], [raw_record()]])
    output = tmp_path / "memory"
    manifest = run_build(source, output, holdout)
    assert manifest["records"] == 1
    assert len(stored_rows(output)) == 1
    assert stored_rows(output, "places")[0]["support_count"] == 1


def test_conflicting_duplicate_id_fails_transaction(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record(), raw_record(gbifid="10002"),
                                      raw_record(decimallongitude=-2.6)]])
    output = tmp_path / "memory"
    with pytest.raises((ValueError, RuntimeError, sqlite3.IntegrityError), match="(?i)(conflict|duplicate)"):
        run_build(source, output, holdout)
    assert not (output / "COMPLETE.json").exists()
    if (output / memory.DB_FILE).exists():
        assert stored_rows(output) == []


def test_conflicting_duplicate_is_detected_before_quality_filter(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record(), raw_record(license="CC_BY_NC_4_0")]])
    output = tmp_path / "memory"
    with pytest.raises((ValueError, RuntimeError, sqlite3.IntegrityError), match="(?i)(conflict|duplicate)"):
        run_build(source, output, holdout)
    assert not (output / "COMPLETE.json").exists()


def test_foreign_taxon_changes_do_not_make_geographic_duplicate_conflict(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record(species="Taxon A"), raw_record(species="Taxon B")]])
    manifest = run_build(source, tmp_path / "memory", holdout)
    assert manifest["records"] == 1


def test_incomplete_source_cannot_build_full_memory(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record()]], complete=False)
    with pytest.raises((ValueError, RuntimeError), match="(?i)(complete|full|incomplete)"):
        run_build(source, tmp_path / "memory", holdout)


def test_pilot_is_explicit_and_never_reads_unreceipted_part(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record()], [raw_record(gbifid="10002")]],
                            complete=False, missing_indices={1})
    output = tmp_path / "memory"
    manifest = run_build(source, output, holdout, mode="pilot", pilot_parts=1)
    assert manifest["mode"] == "pilot"
    assert manifest["production_approved"] is False
    assert manifest["records"] == 1
    assert {row["gbif_id"] for row in stored_rows(output)} == {"10001"}


def test_pilot_does_not_glob_orphaned_parquet_files(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record()]], complete=False)
    (source / "parts/999999.parquet").write_bytes(b"not parquet and not receipted")
    manifest = run_build(source, tmp_path / "memory", holdout, mode="pilot", pilot_parts=1)
    assert manifest["records"] == 1


@pytest.mark.parametrize("mutation", ["part_bytes", "receipt_hash", "receipt_source", "receipt_part",
                                     "receipt_count", "identity", "registry", "citation", "terminal"])
def test_source_corruption_fails_closed(tmp_path, mutation):
    source = source_fixture(tmp_path, [[raw_record()]])
    receipt_path = source / "receipts/000000.json"
    receipt = json.loads(receipt_path.read_text())
    if mutation == "part_bytes":
        with (source / "parts/000000.parquet").open("ab") as stream:
            stream.write(b"corrupt")
    elif mutation.startswith("receipt_"):
        if mutation == "receipt_hash":
            receipt["sha256"] = "0" * 64
        elif mutation == "receipt_source":
            receipt["source"]["etag"] = "changed"
        elif mutation == "receipt_part":
            receipt["part"] = "../outside.parquet"
        else:
            receipt["records"] = -1
        write_json(receipt_path, receipt)
    elif mutation == "identity":
        payload = json.loads((source / "identity.json").read_text())
        payload["snapshot"] = "2026-08-01"
        write_json(source / "identity.json", payload)
    elif mutation in {"registry", "citation"}:
        path = source / ("organizations.json" if mutation == "registry" else "citation.txt")
        path.write_text("changed")
    else:
        write_json(source / "COMPLETE.json", {"status": "complete", "manifest_sha256": "0" * 64})
    with pytest.raises((ValueError, RuntimeError)):
        memory.validate_source(source, "full", 16)


def test_source_with_complete_marker_but_missing_receipt_fails(tmp_path):
    source = source_fixture(tmp_path, [[raw_record()], [raw_record(gbifid="10002")]], missing_indices={1})
    with pytest.raises((ValueError, RuntimeError)):
        memory.validate_source(source, "full", 16)


@pytest.mark.parametrize("indices", [[-1], [1], [0, 0]])
def test_frozen_selection_requires_distinct_in_range_indices(tmp_path, indices):
    source = source_fixture(tmp_path, [[raw_record()]], complete=False)
    with pytest.raises((ValueError, RuntimeError)):
        memory.validate_source(source, "pilot", 1, selected_indices=indices)


def test_receipt_row_count_must_equal_actual_parquet_count(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record()]], complete=False)
    path = source / "receipts/000000.json"
    receipt = json.loads(path.read_text())
    receipt["records"] = 999
    write_json(path, receipt)
    output = tmp_path / "memory"
    with pytest.raises((ValueError, RuntimeError), match="(?i)(count|rows|records)"):
        run_build(source, output, holdout, mode="pilot", pilot_parts=1)
    assert not (output / "COMPLETE.json").exists()


def test_pilot_resume_rejects_changed_selection(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record()], [raw_record(gbifid="10002")]], complete=False)
    output = tmp_path / "memory"
    initial = run_build(source, output, holdout, mode="pilot", pilot_parts=1)
    with pytest.raises((ValueError, RuntimeError), match="(?i)(identity|resume|selection|complete|changed)"):
        run_build(source, output, holdout, mode="pilot", pilot_parts=2)
    assert digest(output / memory.DB_FILE) == initial["database_sha256"]


def test_completed_build_revalidates_database(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record()]])
    output = tmp_path / "memory"
    run_build(source, output, holdout)
    with closing(sqlite3.connect(output / memory.DB_FILE)) as db:
        db.execute("UPDATE records SET locality='tampered'")
        db.commit()
    with pytest.raises((ValueError, RuntimeError), match="(?i)(hash|sha|checksum|database|attestation)"):
        run_build(source, output, holdout)


def test_completed_build_revalidates_source_and_holdout(tmp_path, holdout):
    source = source_fixture(tmp_path, [[raw_record()]])
    output = tmp_path / "memory"
    run_build(source, output, holdout)
    original_db = digest(output / memory.DB_FILE)
    payload = json.loads(holdout[0].read_text())
    source_reference = Path(payload["source_files"][0]["path"])
    source_reference.write_text(source_reference.read_text() + "\n")
    with pytest.raises((ValueError, RuntimeError), match="(?i)(source|holdout|changed)"):
        run_build(source, output, holdout)
    assert digest(output / memory.DB_FILE) == original_db


def test_resume_after_aggregation_crash_preserves_counts(tmp_path, holdout, monkeypatch):
    source = source_fixture(tmp_path, [[raw_record()], [raw_record(gbifid="10002")]])
    output = tmp_path / "memory"
    original = memory.aggregate_places

    def crash(db):
        raise RuntimeError("simulated aggregation interruption")

    monkeypatch.setattr(memory, "aggregate_places", crash)
    with pytest.raises(RuntimeError, match="simulated aggregation interruption"):
        run_build(source, output, holdout)
    assert len(stored_rows(output)) == 2
    assert not (output / "COMPLETE.json").exists()
    with pytest.raises((ValueError, RuntimeError)):
        memory.verify_output(output)
    monkeypatch.setattr(memory, "aggregate_places", original)
    manifest = run_build(source, output, holdout)
    assert manifest["records"] == manifest["source_records"] == 2
    assert manifest["counters"]["accepted"] == 2
    assert manifest["counters"].get("duplicate_accepted_rows", 0) == 0
    assert stored_rows(output, "places")[0]["support_count"] == 2
    assert run_build(source, output, holdout) == manifest


def test_resume_after_partial_ingestion_skips_committed_part(tmp_path, holdout, monkeypatch):
    source = source_fixture(tmp_path, [[raw_record()], [raw_record(gbifid="10002")]])
    output = tmp_path / "memory"
    original = memory.normalize_record

    def crash_second(raw, exclusions):
        if raw["gbifid"] == "10002":
            raise RuntimeError("simulated second-part interruption")
        return original(raw, exclusions)

    monkeypatch.setattr(memory, "normalize_record", crash_second)
    with pytest.raises(RuntimeError, match="simulated second-part interruption"):
        run_build(source, output, holdout)
    assert {row["gbif_id"] for row in stored_rows(output)} == {"10001"}
    with closing(sqlite3.connect(output / memory.DB_FILE)) as db:
        assert db.execute("SELECT part_index FROM completed_parts").fetchall() == [(0,)]
    assert not (output / "COMPLETE.json").exists()
    observed_ids = []

    def observe(raw, exclusions):
        observed_ids.append(raw["gbifid"])
        return original(raw, exclusions)

    monkeypatch.setattr(memory, "normalize_record", observe)
    manifest = run_build(source, output, holdout)
    # The first row is read only by the final independent audit, not re-ingested.
    assert observed_ids.count("10001") == 1
    assert observed_ids.count("10002") == 2
    assert manifest["records"] == manifest["source_records"] == 2
    assert manifest["counters"]["accepted"] == 2
    assert manifest["counters"].get("duplicate_accepted_rows", 0) == 0
