from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "gb300"
    / "analyze_dissco_uk_snapshot.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_dissco_uk_snapshot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_core_institution_mapping_is_stable() -> None:
    institutions = {item.slug: item for item in MODULE.CORE_INSTITUTIONS}
    assert set(institutions) == {"nhm", "kew", "rbge", "mba"}
    assert institutions["nhm"].publisher_key == (
        "19456090-b49a-11d8-abeb-b8a03c50a862"
    )
    sql = MODULE.cohort_case_sql()
    assert "other_publishers" in sql
    for item in institutions.values():
        assert item.publisher_key in sql
        assert item.slug in sql


def test_percent_handles_empty_denominator() -> None:
    assert MODULE.percent(25, 100) == 25.0
    assert MODULE.percent(1, 0) == 0.0
    assert MODULE.percent(None, 100) == 0.0


def test_recovery_lanes_are_mutually_exclusive_in_sql() -> None:
    sql = MODULE.aggregate_sql("read_parquet('input.parquet')")
    assert "unresolved_with_locality" in sql
    assert "unresolved_structured_geography_only" in sql
    assert "unresolved_no_location_evidence" in sql
    assert "NOT (" in sql
    assert "publishingorgkey" in sql
    assert "normalized_locality_country_keys" in sql


def test_array_breakdowns_count_each_type_once_per_record() -> None:
    sql = MODULE.array_breakdown_sql("read_parquet('input.parquet')", "mediatype", 8)
    assert "list_distinct(items)" in sql


def test_markdown_table_escapes_no_content() -> None:
    rendered = MODULE.markdown_table(("A", "B"), (("x", 1),))
    assert rendered.splitlines() == [
        "| A | B |",
        "| --- | --- |",
        "| x | 1 |",
    ]


def test_report_explains_unavailable_nhm_routes() -> None:
    metrics = {
        "records": 10,
        "datasets": 1,
        "publisher_organizations": 1,
        "institution_codes": 1,
        "collection_codes": 1,
        "species_keys": 1,
        "unresolved_distinct_normalized_locality_country_keys": 2,
        "minimum_year": 1900,
        "maximum_year": 2020,
        "valid_coordinates": 2,
        "coordinate_and_locality": 1,
        "coordinate_only": 1,
        "locality_text": 4,
        "unresolved_records": 8,
        "unresolved_with_locality": 3,
        "unresolved_structured_geography_only": 4,
        "unresolved_no_location_evidence": 1,
        "coordinate_pair_populated_but_invalid": 0,
        "partial_coordinate_pair": 0,
        "zero_zero_coordinates": 0,
        "country_code": 6,
        "state_province": 2,
        "any_media_type": 1,
        "still_image_type": 1,
        "any_issue": 2,
        "scientific_name": 9,
        "species_name": 8,
        "event_date": 7,
        "year_plausible": 7,
        "occurrence_id": 10,
        "catalog_number": 9,
        "locality_with_detail_punctuation": 2,
        "locality_over_80_characters": 1,
    }
    report = {
        "generated_at": "2026-07-21T00:00:00+00:00",
        "source": {
            "path": "/tmp/input.parquet",
            "snapshot": "2026-07-01",
            "columns": 50,
            "row_groups": 1,
            "bytes": 100,
            "sha256": "abc",
            "manifest_checks_passed": True,
            "live_portal_records": 9,
            "proxy_minus_live_portal_records": 1,
        },
        "derived": {"core_institution_records": 8},
        "cohorts": {
            cohort: {
                "metrics": metrics,
                "breakdowns": {
                    name: []
                    for name in (
                        "basis_of_record",
                        "country_code",
                        "kingdom",
                        "order",
                        "institution_code",
                        "collection_code",
                        "media_type",
                        "issue",
                    )
                },
            }
            for cohort in MODULE.COHORT_ORDER
        },
    }
    rendered = MODULE.render_markdown(report)
    assert "Executive summary" in rendered
    assert "Scribe OCR" in rendered
    assert "Not measurable" in rendered
    assert "Associated publications" in rendered
