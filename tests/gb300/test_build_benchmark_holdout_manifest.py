from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "gb300"
    / "build_benchmark_holdout_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_benchmark_holdout_manifest",
    SCRIPT,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_record(
    benchmark_id: str,
    *,
    locality: str = "St. John's, Île Test",
    country: str = "Test Country",
    latitude: str = "51.12345649",
    longitude: str = "-3.98765451",
) -> dict[str, object]:
    return {
        "original_barcode": benchmark_id,
        "original_locality": locality,
        "original_country": country,
        "original_lat": latitude,
        "original_long": longitude,
        "ai_processed_result": {
            "latitude": 1.0,
            "longitude": 2.0,
            "notes": "must never enter the holdout manifest",
        },
        "success": True,
    }


def write_archive(
    path: Path,
    records: list[dict[str, object]],
    *,
    second_model_records: list[dict[str, object]] | None = None,
) -> None:
    payload = {
        "test_metadata": {"irrelevant": True},
        "model_results": {
            "model_b": {
                "processed_records": (
                    second_model_records
                    if second_model_records is not None
                    else records
                ),
                "performance_metrics": {"ignored": True},
            },
            "model_a": {
                "processed_records": records,
                "performance_metrics": {"ignored": True},
            },
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def write_reference(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Barcode",
                "label_verbatim",
                "Country",
                "lat_manual",
                "long_manual",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def test_multiple_reference_union_is_explicit_and_preserves_variants(tmp_path):
    first, second = tmp_path / "full.tsv", tmp_path / "subset.tsv"
    record = {"Barcode": "O001", "label_verbatim": "First label", "Country": "GB",
              "lat_manual": "51", "long_manual": "0"}
    write_reference(first, [record])
    write_reference(second, [{**record, "label_verbatim": "Alternate label"}])
    sources = [MODULE.SourceSpec("odonata", p, MODULE.REFERENCE_KIND) for p in (first, second)]
    with pytest.raises(MODULE.HoldoutManifestError, match="duplicate"):
        MODULE.build_manifest(sources, expected_record_count=1, allow_content_variants=True)
    with pytest.raises(MODULE.HoldoutManifestError, match="content differs"):
        MODULE.build_manifest(sources, expected_record_count=1, allow_multiple_references=True)
    manifest = MODULE.build_manifest(sources, expected_record_count=1,
                                    allow_multiple_references=True, allow_content_variants=True)
    assert manifest["record_count"] == 1
    assert len(manifest["source_files"]) == 2
    record = manifest["records"][0]
    assert len(record["content_variants"]) == 1
    assert {record["locality"], record["content_variants"][0]["locality"]} == {"First label", "Alternate label"}


def test_manifest_is_deterministic_and_contains_no_model_outputs(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_archive(first_path, [make_record("B002")])
    write_archive(
        second_path,
        [
            make_record(
                "O001",
                locality="Banks Is.",
                country="",
                latitude="-10",
                longitude="142",
            )
        ],
    )
    sources = [
        MODULE.SourceSpec("odonata", second_path),
        MODULE.SourceSpec("bombus", first_path),
    ]

    first = MODULE.build_manifest(sources, expected_record_count=2)
    second = MODULE.build_manifest(reversed(sources), expected_record_count=2)

    assert MODULE.manifest_bytes(first) == MODULE.manifest_bytes(second)
    assert first["schema_version"] == 1
    assert first["artifact_type"] == "placebot_benchmark_holdout"
    assert first["cohort_counts"] == {"bombus": 1, "odonata": 1}
    assert first["record_count"] == 2
    assert [record["benchmark_id"] for record in first["records"]] == [
        "B002",
        "O001",
    ]
    serialized = MODULE.manifest_bytes(first)
    assert b"ai_processed_result" not in serialized
    assert b"notes" not in serialized
    assert b"model_a" not in serialized
    assert b"model_b" not in serialized


def test_inconsistent_model_inputs_fail_closed(tmp_path: Path) -> None:
    archive_path = tmp_path / "changed.json"
    reference = [make_record("B001"), make_record("B002")]
    changed = [
        make_record("B001"),
        make_record("B002", locality="A different locality"),
    ]
    write_archive(
        archive_path,
        reference,
        second_model_records=changed,
    )

    with pytest.raises(MODULE.HoldoutManifestError, match="record 1 differs"):
        MODULE.build_manifest(
            [MODULE.SourceSpec("bombus", archive_path)],
            expected_record_count=2,
        )


def test_duplicate_benchmark_ids_fail_across_cohorts(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_archive(first_path, [make_record("DUPLICATE")])
    write_archive(second_path, [make_record("DUPLICATE")])

    with pytest.raises(
        MODULE.HoldoutManifestError,
        match="duplicate benchmark_id",
    ):
        MODULE.build_manifest(
            [
                MODULE.SourceSpec("bombus", first_path),
                MODULE.SourceSpec("odonata", second_path),
            ],
            expected_record_count=2,
        )


def test_ground_truth_coordinates_and_content_are_never_omitted(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "archive.json"
    write_archive(archive_path, [make_record("B001")])

    manifest = MODULE.build_manifest(
        [MODULE.SourceSpec("bombus", archive_path)],
        expected_record_count=1,
    )
    record = manifest["records"][0]

    assert record == {
        "cohort": "bombus",
        "benchmark_id": "B001",
        "locality": "St. John's, Île Test",
        "locality_normalized": "st john s île test",
        "locality_holdout_keys": ["st john s île test"],
        "country": "Test Country",
        "country_normalized": "test country",
        "latitude": 51.123456,
        "longitude": -3.987655,
        "content_fingerprint": MODULE.content_fingerprint(
            {
                "benchmark_id": "B001",
                "locality": "St. John's, Île Test",
                "country": "Test Country",
                "latitude": 51.123456,
                "longitude": -3.987655,
            }
        ),
        "source_memberships": [
            {
                "source_index": 0,
                "record_index": 0,
                "raw_benchmark_id": "B001",
                "id_method": "exact",
            }
        ],
        "content_variants": [],
    }
    ids_payload = MODULE.canonical_json_bytes(["B001"])
    assert manifest["benchmark_ids_sha256"] == hashlib.sha256(
        ids_payload
    ).hexdigest()

    missing_coordinate = make_record("B002")
    del missing_coordinate["original_lat"]
    write_archive(archive_path, [missing_coordinate])
    with pytest.raises(MODULE.HoldoutManifestError, match="original_lat"):
        MODULE.build_manifest(
            [MODULE.SourceSpec("bombus", archive_path)],
            expected_record_count=1,
        )


def test_atomic_writer_refuses_to_replace_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "holdout.json"
    MODULE.write_fresh_atomic(output, b"first\n")
    with pytest.raises(MODULE.HoldoutManifestError, match="refusing to replace"):
        MODULE.write_fresh_atomic(output, b"second\n")
    assert output.read_bytes() == b"first\n"


def test_reference_tsv_merges_with_archive_and_preserves_raw_leading_zero(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "archive.json"
    reference_path = tmp_path / "reference.tsv"
    write_archive(
        archive_path,
        [
            make_record(
                "010810986",
                locality="10 Linden Grove",
                latitude="51.50883",
                longitude="-3.12996",
            )
        ],
    )
    write_reference(
        reference_path,
        [
            {
                "Barcode": "10810986",
                "label_verbatim": "10 Linden Grove",
                "Country": "Test Country",
                "lat_manual": "51.5088300",
                "long_manual": "-3.1299600",
            },
            {
                "Barcode": "001234567",
                "label_verbatim": "Leading Zero Place",
                "Country": "",
                "lat_manual": "1",
                "long_manual": "2",
            },
        ],
    )

    manifest = MODULE.build_manifest(
        [
            MODULE.SourceSpec(
                "bombus_uk",
                archive_path,
                MODULE.ARCHIVE_KIND,
            ),
            MODULE.SourceSpec(
                "bombus_uk",
                reference_path,
                MODULE.REFERENCE_KIND,
            ),
        ],
        expected_record_count=2,
    )

    assert [record["benchmark_id"] for record in manifest["records"]] == [
        "010810986",
        "001234567",
    ]
    assert manifest["records"][0]["source_memberships"] == [
        {
            "source_index": 0,
            "record_index": 0,
            "raw_benchmark_id": "10810986",
            "id_method": "bombus_reference_numeric_zfill_9",
        },
        {
            "source_index": 1,
            "record_index": 0,
            "raw_benchmark_id": "010810986",
            "id_method": "exact",
        },
    ]
    assert manifest["records"][1]["source_memberships"][0][
        "raw_benchmark_id"
    ] == "001234567"


def test_reference_archive_content_mismatch_fails_closed(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.json"
    reference_path = tmp_path / "reference.tsv"
    write_archive(archive_path, [make_record("010810986", locality="Archive")])
    write_reference(
        reference_path,
        [
            {
                "Barcode": "10810986",
                "label_verbatim": "Reference changed",
                "Country": "Test Country",
                "lat_manual": "51.12345649",
                "long_manual": "-3.98765451",
            }
        ],
    )

    with pytest.raises(MODULE.HoldoutManifestError, match="content differs"):
        MODULE.build_manifest(
            [
                MODULE.SourceSpec(
                    "bombus",
                    archive_path,
                    MODULE.ARCHIVE_KIND,
                ),
                MODULE.SourceSpec(
                    "bombus",
                    reference_path,
                    MODULE.REFERENCE_KIND,
                ),
            ],
            expected_record_count=1,
        )


def test_declared_content_variants_are_preserved_with_provenance(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "archive.json"
    reference_path = tmp_path / "reference.tsv"
    write_archive(archive_path, [make_record("ODO001", locality="Short label")])
    write_reference(
        reference_path,
        [
            {
                "Barcode": "ODO001",
                "label_verbatim": "Short label; second label",
                "Country": "Test Country",
                "lat_manual": "51.12345649",
                "long_manual": "-3.98765451",
            }
        ],
    )

    manifest = MODULE.build_manifest(
        [
            MODULE.SourceSpec(
                "odonata",
                archive_path,
                MODULE.ARCHIVE_KIND,
            ),
            MODULE.SourceSpec(
                "odonata",
                reference_path,
                MODULE.REFERENCE_KIND,
            ),
        ],
        expected_record_count=1,
        allow_content_variants=True,
    )

    assert manifest["content_variant_count"] == 1
    record = manifest["records"][0]
    assert record["locality"] == "Short label; second label"
    assert record["locality_holdout_keys"] == [
        "short label second label",
        "short label",
        "second label",
    ]
    assert record["content_variants"][0]["locality"] == "Short label"
    assert record["content_variants"][0]["source_memberships"][0][
        "id_method"
    ] == "exact"


def test_numbered_label_prefix_is_also_held_out() -> None:
    assert MODULE.locality_holdout_keys(
        "1st label: Uganda, Gulu; 2nd label: Mt. Elgon"
    ) == [
        "1st label uganda gulu 2nd label mt elgon",
        "1st label uganda gulu",
        "2nd label mt elgon",
        "uganda gulu",
        "mt elgon",
    ]
