#!/usr/bin/env python3
"""Profile the DiSSCo UK snapshot overall and by four target publishers.

The profiler is read-only with respect to the source Parquet. It produces a
machine-readable JSON artifact and a human-readable Markdown report without
calling a model or a network service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(
    "data/private/dissco-uk/2026-07-01/occurrence.parquet"
)
DEFAULT_MANIFEST = DEFAULT_INPUT.with_name("manifest.json")
DEFAULT_REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "dissco_uk_georeferencing_opportunity_assessment_20260721"
)


@dataclass(frozen=True)
class Institution:
    """A core DiSSCo UK publisher represented in the compact snapshot."""

    slug: str
    name: str
    publisher_key: str


CORE_INSTITUTIONS = (
    Institution(
        "nhm",
        "Natural History Museum, London",
        "19456090-b49a-11d8-abeb-b8a03c50a862",
    ),
    Institution(
        "kew",
        "Royal Botanic Gardens, Kew",
        "061b4f20-f241-11da-a328-b8a03c50a862",
    ),
    Institution(
        "rbge",
        "Royal Botanic Garden Edinburgh",
        "98e934b0-5f31-11de-b67e-b8a03c50a862",
    ),
    Institution(
        "mba",
        "Marine Biological Association, Plymouth",
        "f6b48504-1651-4a49-a88d-c2bc6178694d",
    ),
)

COHORT_LABELS = {
    "all_snapshot": "All snapshot records",
    **{institution.slug: institution.name for institution in CORE_INSTITUTIONS},
    "other_publishers": "Other GB publishers admitted by the proxy",
}
COHORT_ORDER = (
    "all_snapshot",
    "nhm",
    "kew",
    "rbge",
    "mba",
    "other_publishers",
)

REQUIRED_COLUMNS = {
    "gbifid",
    "datasetkey",
    "publishingorgkey",
    "decimallatitude",
    "decimallongitude",
    "coordinateuncertaintyinmeters",
    "locality",
    "stateprovince",
    "countrycode",
    "eventdate",
    "year",
    "basisofrecord",
    "institutioncode",
    "collectioncode",
    "occurrenceid",
    "catalognumber",
    "scientificname",
    "species",
    "taxonkey",
    "specieskey",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "occurrencestatus",
    "mediatype",
    "issue",
    "typestatus",
    "recordedby",
    "identifiedby",
    "license",
    "rightsholder",
    "elevation",
    "depth",
}

VALID_COORDINATE_SQL = """(
    decimallatitude IS NOT NULL
    AND decimallongitude IS NOT NULL
    AND isfinite(decimallatitude)
    AND isfinite(decimallongitude)
    AND decimallatitude BETWEEN -90.0 AND 90.0
    AND decimallongitude BETWEEN -180.0 AND 180.0
)"""
HAS_LOCALITY_SQL = "nullif(trim(locality), '') IS NOT NULL"
HAS_COUNTRY_SQL = "nullif(trim(countrycode), '') IS NOT NULL"
HAS_STATE_SQL = "nullif(trim(stateprovince), '') IS NOT NULL"
HAS_STRUCTURED_GEOGRAPHY_SQL = f"({HAS_COUNTRY_SQL} OR {HAS_STATE_SQL})"
HAS_MEDIA_SQL = "coalesce(array_length(mediatype), 0) > 0"
HAS_ISSUE_SQL = "coalesce(array_length(issue), 0) > 0"


def sql_literal(value: str | Path) -> str:
    """Return a single-quoted DuckDB string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def cohort_case_sql() -> str:
    """Return the stable publisher-key-to-cohort SQL expression."""
    clauses = " ".join(
        f"WHEN {sql_literal(item.publisher_key)} THEN {sql_literal(item.slug)}"
        for item in CORE_INSTITUTIONS
    )
    return f"CASE publishingorgkey {clauses} ELSE 'other_publishers' END"


def percent(count: int | float | None, denominator: int | float | None) -> float:
    """Return a percentage, using zero for an empty denominator."""
    if count is None or not denominator:
        return 0.0
    return float(count) * 100.0 / float(denominator)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON atomically."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    """Write text atomically."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def connect_duckdb(threads: int, memory_limit: str, temp_directory: Path) -> Any:
    """Open an in-memory DuckDB connection with bounded local spill."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "duckdb is required; run this with the PlaceBot GBIF tools venv"
        ) from exc

    connection = duckdb.connect(database=":memory:")
    connection.execute(f"SET threads = {int(threads)}")
    connection.execute(f"SET memory_limit = {sql_literal(memory_limit)}")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = {sql_literal(temp_directory)}")
    return connection


def source_relation(path: Path) -> str:
    return f"read_parquet({sql_literal(path)})"


def validate_schema(connection: Any, relation: str) -> list[tuple[str, str]]:
    """Validate the compact snapshot schema and return its columns."""
    rows = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    columns = [(str(row[0]), str(row[1])) for row in rows]
    missing = sorted(REQUIRED_COLUMNS - {name for name, _ in columns})
    if missing:
        raise ValueError(f"input is missing required columns: {', '.join(missing)}")
    return columns


def aggregate_sql(relation: str) -> str:
    """Build the single-pass overall and institution metrics query."""
    valid = VALID_COORDINATE_SQL
    locality = HAS_LOCALITY_SQL
    structured = HAS_STRUCTURED_GEOGRAPHY_SQL
    normalized_locality = "lower(regexp_replace(trim(locality), '\\s+', ' ', 'g'))"
    normalized_country = "lower(coalesce(nullif(trim(countrycode), ''), ''))"
    metrics = {
        "records": "count(*)",
        "datasets": "count(DISTINCT nullif(trim(datasetkey), ''))",
        "publisher_organizations": (
            "count(DISTINCT nullif(trim(publishingorgkey), ''))"
        ),
        "institution_codes": (
            "count(DISTINCT nullif(trim(institutioncode), ''))"
        ),
        "collection_codes": (
            "count(DISTINCT nullif(trim(collectioncode), ''))"
        ),
        "species_keys": "count(DISTINCT specieskey)",
        "taxon_keys": "count(DISTINCT taxonkey)",
        "valid_coordinates": f"count(*) FILTER (WHERE {valid})",
        "coordinate_and_locality": (
            f"count(*) FILTER (WHERE {valid} AND {locality})"
        ),
        "coordinate_only": (
            f"count(*) FILTER (WHERE {valid} AND NOT ({locality}))"
        ),
        "locality_text": f"count(*) FILTER (WHERE {locality})",
        "unresolved_with_locality": (
            f"count(*) FILTER (WHERE NOT ({valid}) AND {locality})"
        ),
        "neither_coordinate_nor_locality": (
            f"count(*) FILTER (WHERE NOT ({valid}) AND NOT ({locality}))"
        ),
        "unresolved_structured_geography_only": (
            f"count(*) FILTER (WHERE NOT ({valid}) AND NOT ({locality}) "
            f"AND {structured})"
        ),
        "unresolved_no_location_evidence": (
            f"count(*) FILTER (WHERE NOT ({valid}) AND NOT ({locality}) "
            f"AND NOT ({structured}))"
        ),
        "unresolved_records": f"count(*) FILTER (WHERE NOT ({valid}))",
        "unresolved_distinct_normalized_locality_country_keys": (
            f"count(DISTINCT ({normalized_locality}, {normalized_country})) FILTER "
            f"(WHERE NOT ({valid}) AND {locality})"
        ),
        "coordinate_pair_populated_but_invalid": (
            "count(*) FILTER (WHERE decimallatitude IS NOT NULL "
            "AND decimallongitude IS NOT NULL AND NOT " + valid + ")"
        ),
        "partial_coordinate_pair": (
            "count(*) FILTER (WHERE (decimallatitude IS NULL) <> "
            "(decimallongitude IS NULL))"
        ),
        "zero_zero_coordinates": (
            f"count(*) FILTER (WHERE {valid} AND decimallatitude = 0.0 "
            "AND decimallongitude = 0.0)"
        ),
        "coordinate_uncertainty_present": (
            "count(*) FILTER (WHERE coordinateuncertaintyinmeters IS NOT NULL)"
        ),
        "coordinate_uncertainty_nonnegative": (
            "count(*) FILTER (WHERE coordinateuncertaintyinmeters >= 0.0)"
        ),
        "coordinate_uncertainty_median_m": (
            "median(coordinateuncertaintyinmeters) FILTER "
            "(WHERE coordinateuncertaintyinmeters >= 0.0)"
        ),
        "coordinate_uncertainty_p90_m": (
            "quantile_cont(coordinateuncertaintyinmeters, 0.9) FILTER "
            "(WHERE coordinateuncertaintyinmeters >= 0.0)"
        ),
        "locality_with_detail_punctuation": (
            "count(*) FILTER (WHERE " + locality
            + " AND regexp_matches(locality, '[;,:]'))"
        ),
        "locality_over_80_characters": (
            f"count(*) FILTER (WHERE {locality} AND length(locality) > 80)"
        ),
        "country_code": f"count(*) FILTER (WHERE {HAS_COUNTRY_SQL})",
        "state_province": f"count(*) FILTER (WHERE {HAS_STATE_SQL})",
        "any_media_type": f"count(*) FILTER (WHERE {HAS_MEDIA_SQL})",
        "still_image_type": (
            "count(*) FILTER (WHERE list_contains(mediatype, 'StillImage'))"
        ),
        "any_issue": f"count(*) FILTER (WHERE {HAS_ISSUE_SQL})",
        "no_reported_issue": f"count(*) FILTER (WHERE NOT ({HAS_ISSUE_SQL}))",
        "type_status": (
            "count(*) FILTER (WHERE coalesce(array_length(typestatus), 0) > 0)"
        ),
        "event_date": "count(*) FILTER (WHERE eventdate IS NOT NULL)",
        "year_present": "count(*) FILTER (WHERE year IS NOT NULL)",
        "year_plausible": (
            "count(*) FILTER (WHERE year BETWEEN 1000 AND 2026)"
        ),
        "year_pre_1800": "count(*) FILTER (WHERE year BETWEEN 1000 AND 1799)",
        "year_1800_1899": "count(*) FILTER (WHERE year BETWEEN 1800 AND 1899)",
        "year_1900_1949": "count(*) FILTER (WHERE year BETWEEN 1900 AND 1949)",
        "year_1950_1999": "count(*) FILTER (WHERE year BETWEEN 1950 AND 1999)",
        "year_2000_2026": "count(*) FILTER (WHERE year BETWEEN 2000 AND 2026)",
        "minimum_year": "min(year) FILTER (WHERE year BETWEEN 1000 AND 2026)",
        "maximum_year": "max(year) FILTER (WHERE year BETWEEN 1000 AND 2026)",
        "scientific_name": (
            "count(*) FILTER (WHERE nullif(trim(scientificname), '') IS NOT NULL)"
        ),
        "species_name": (
            "count(*) FILTER (WHERE nullif(trim(species), '') IS NOT NULL)"
        ),
        "family_name": (
            "count(*) FILTER (WHERE nullif(trim(family), '') IS NOT NULL)"
        ),
        "occurrence_id": (
            "count(*) FILTER (WHERE nullif(trim(occurrenceid), '') IS NOT NULL)"
        ),
        "catalog_number": (
            "count(*) FILTER (WHERE nullif(trim(catalognumber), '') IS NOT NULL)"
        ),
        "recorded_by": (
            "count(*) FILTER (WHERE coalesce(array_length(recordedby), 0) > 0)"
        ),
        "identified_by": (
            "count(*) FILTER (WHERE coalesce(array_length(identifiedby), 0) > 0)"
        ),
        "rights_holder": (
            "count(*) FILTER (WHERE nullif(trim(rightsholder), '') IS NOT NULL)"
        ),
        "license_present": (
            "count(*) FILTER (WHERE nullif(trim(license), '') IS NOT NULL)"
        ),
        "elevation_present": "count(*) FILTER (WHERE elevation IS NOT NULL)",
        "depth_present": "count(*) FILTER (WHERE depth IS NOT NULL)",
    }
    projections = ",\n        ".join(
        f"{expression} AS {name}" for name, expression in metrics.items()
    )
    return f"""
WITH base AS (
    SELECT *, {cohort_case_sql()} AS cohort
    FROM {relation}
)
SELECT
    CASE WHEN GROUPING(cohort) = 1 THEN 'all_snapshot' ELSE cohort END AS cohort,
    {projections}
FROM base
GROUP BY GROUPING SETS ((cohort), ())
"""


def query_metrics(connection: Any, relation: str) -> dict[str, dict[str, Any]]:
    cursor = connection.execute(aggregate_sql(relation))
    columns = [description[0] for description in cursor.description]
    results: dict[str, dict[str, Any]] = {}
    for row in cursor.fetchall():
        values = dict(zip(columns, row))
        cohort = str(values.pop("cohort"))
        results[cohort] = values
    return results


def scalar_breakdown_sql(
    relation: str,
    expression: str,
    limit: int,
) -> str:
    value = f"coalesce(nullif(trim(cast({expression} AS VARCHAR)), ''), '[blank]')"
    return f"""
WITH base AS (
    SELECT {cohort_case_sql()} AS cohort, {value} AS value
    FROM {relation}
), counts AS (
    SELECT
        CASE WHEN GROUPING(cohort) = 1 THEN 'all_snapshot' ELSE cohort END AS cohort,
        value,
        count(*) AS records
    FROM base
    GROUP BY GROUPING SETS ((cohort, value), (value))
), ranked AS (
    SELECT *, row_number() OVER (
        PARTITION BY cohort ORDER BY records DESC, value
    ) AS rank
    FROM counts
)
SELECT cohort, value, records
FROM ranked
WHERE rank <= {int(limit)}
ORDER BY cohort, rank
"""


def array_breakdown_sql(relation: str, column: str, limit: int) -> str:
    return f"""
WITH base AS (
    SELECT {cohort_case_sql()} AS cohort, {column} AS items
    FROM {relation}
), expanded AS (
    SELECT cohort, trim(cast(value AS VARCHAR)) AS value
    FROM base, unnest(list_distinct(items)) AS item(value)
), counts AS (
    SELECT
        CASE WHEN GROUPING(cohort) = 1 THEN 'all_snapshot' ELSE cohort END AS cohort,
        value,
        count(*) AS records
    FROM expanded
    WHERE value <> ''
    GROUP BY GROUPING SETS ((cohort, value), (value))
), ranked AS (
    SELECT *, row_number() OVER (
        PARTITION BY cohort ORDER BY records DESC, value
    ) AS rank
    FROM counts
)
SELECT cohort, value, records
FROM ranked
WHERE rank <= {int(limit)}
ORDER BY cohort, rank
"""


def query_breakdown(
    connection: Any,
    sql: str,
) -> dict[str, list[dict[str, Any]]]:
    results = {cohort: [] for cohort in COHORT_ORDER}
    for cohort, value, records in connection.execute(sql).fetchall():
        results[str(cohort)].append(
            {"value": str(value), "records": int(records)}
        )
    return results


def query_breakdowns(connection: Any, relation: str) -> dict[str, Any]:
    scalar_specs = {
        "basis_of_record": ("basisofrecord", 8),
        "country_code": ("countrycode", 12),
        "kingdom": ("kingdom", 8),
        "phylum": ("phylum", 10),
        "class": ("class", 10),
        "order": ('"order"', 12),
        "family": ("family", 12),
        "institution_code": ("institutioncode", 12),
        "collection_code": ("collectioncode", 12),
        "occurrence_status": ("occurrencestatus", 8),
        "license": ("license", 8),
    }
    results = {
        name: query_breakdown(
            connection,
            scalar_breakdown_sql(relation, expression, limit),
        )
        for name, (expression, limit) in scalar_specs.items()
    }
    results["media_type"] = query_breakdown(
        connection,
        array_breakdown_sql(relation, "mediatype", 8),
    )
    results["issue"] = query_breakdown(
        connection,
        array_breakdown_sql(relation, "issue", 12),
    )
    results["type_status"] = query_breakdown(
        connection,
        array_breakdown_sql(relation, "typestatus", 12),
    )
    return results


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def metric_rows(
    metrics: dict[str, Any],
    definitions: Sequence[tuple[str, str]],
) -> list[tuple[str, str, str]]:
    total = int(metrics["records"])
    rows = []
    for label, key in definitions:
        value = metrics.get(key)
        if isinstance(value, float):
            formatted = f"{value:,.2f}"
            share = "—"
        elif value is None:
            formatted = "—"
            share = "—"
        else:
            formatted = f"{int(value):,}"
            share = f"{percent(value, total):.3f}%"
        rows.append((label, formatted, share))
    return rows


def render_breakdown(
    title: str,
    values: Sequence[dict[str, Any]],
    denominator: int,
) -> str:
    if not values:
        return f"##### {title}\n\nNo populated values."
    rows = [
        (
            item["value"].replace("|", "\\|"),
            f"{int(item['records']):,}",
            f"{percent(item['records'], denominator):.3f}%",
        )
        for item in values
    ]
    return (
        f"##### {title}\n\n"
        + markdown_table(("Value", "Records", "Share of cohort"), rows)
    )


def render_markdown(report: dict[str, Any]) -> str:
    cohorts = report["cohorts"]
    total = int(cohorts["all_snapshot"]["metrics"]["records"])
    all_metrics = cohorts["all_snapshot"]["metrics"]
    unresolved_total = int(all_metrics["unresolved_records"])
    core_total = int(report["derived"]["core_institution_records"])
    lines = [
        "# DiSSCo UK Collection – Georeferencing Opportunity Assessment",
        "",
        f"Generated: **{report['generated_at']}**.",
        "",
        "This is a read-only statistical profile of the compact GBIF monthly",
        "snapshot proxy used by PlaceBot. It covers the complete file and then",
        "applies the same definitions to the four target institution publishers.",
        "No model or network service was called.",
        "",
        "## Essential qualification",
        "",
        "The compact Parquet omits GBIF's interpreted `institutionKey` and",
        "`collectionKey`. Institution sections are therefore assigned by the",
        "retained `publishingorgkey`, not by exact live-portal collection keys.",
        "It also has media-type indicators but no media URLs or label/register",
        "categories, and it lacks the NHM CSV's verbatim-coordinate and richer",
        "geography fields. Media counts below are not label-image counts.",
        "",
        "## Executive summary",
        "",
        f"The compact snapshot contains **{total:,} specimen records**. Of these,",
        f"**{int(cohorts['all_snapshot']['metrics']['valid_coordinates']):,}**",
        "already contain a finite, in-range decimal coordinate pair. Among the",
        f"remaining **{int(cohorts['all_snapshot']['metrics']['unresolved_records']):,}**",
        "records,",
        f"**{int(cohorts['all_snapshot']['metrics']['unresolved_with_locality']):,}**",
        "have locality text suitable for direct PlaceBot processing,",
        f"**{int(cohorts['all_snapshot']['metrics']['unresolved_structured_geography_only']):,}**",
        "have only country/state-level constraints, and",
        f"**{int(cohorts['all_snapshot']['metrics']['unresolved_no_location_evidence']):,}**",
        "have no coordinate, locality, country or state/province evidence in this",
        "compact schema.",
        "",
        "A separate Scribe/OCR candidate count cannot be derived from this file:",
        "it reports broad media types but omits media URLs and label/register",
        "categories. Verbatim-coordinate parsing and publication-link counts are",
        "also unavailable because their source fields are absent.",
        "",
        "### Headline figures",
        "",
        markdown_table(
            ("Processing route", "Records", "Percentage"),
            (
                (
                    "Already contain valid coordinates",
                    f"{int(all_metrics['valid_coordinates']):,}",
                    f"{percent(all_metrics['valid_coordinates'], total):.2f}% of total",
                ),
                (
                    "PlaceBot from existing locality text",
                    f"{int(all_metrics['unresolved_with_locality']):,}",
                    f"{percent(all_metrics['unresolved_with_locality'], unresolved_total):.2f}% of records without coordinates",
                ),
                (
                    "Structured geography only",
                    f"{int(all_metrics['unresolved_structured_geography_only']):,}",
                    f"{percent(all_metrics['unresolved_structured_geography_only'], unresolved_total):.2f}% of records without coordinates",
                ),
                (
                    "No location evidence in compact schema",
                    f"{int(all_metrics['unresolved_no_location_evidence']):,}",
                    f"{percent(all_metrics['unresolved_no_location_evidence'], unresolved_total):.2f}% of records without coordinates",
                ),
                (
                    "Scribe OCR from label/register images",
                    "Not measurable",
                    "Media identity and URLs absent",
                ),
                (
                    "Verbatim coordinate-string parsing",
                    "Not measurable",
                    "Source fields absent",
                ),
            ),
        ),
        "",
        f"The unresolved PlaceBot lane contains **{int(all_metrics['unresolved_distinct_normalized_locality_country_keys']):,}**",
        "distinct normalised locality-country keys before any deeper semantic",
        "deduplication, gazetteer routing or model calls.",
        "",
        "## Source and integrity",
        "",
        markdown_table(
            ("Property", "Value"),
            (
                ("Parquet", f"`{report['source']['path']}`"),
                ("Snapshot", report["source"].get("snapshot", "—")),
                ("Records", f"{total:,}"),
                (
                    "Live portal records at comparison",
                    f"{int(report['source']['live_portal_records']):,}",
                ),
                (
                    "Proxy minus live portal",
                    f"{int(report['source']['proxy_minus_live_portal_records']):+,}",
                ),
                ("Columns", f"{report['source']['columns']:,}"),
                ("Row groups", f"{report['source']['row_groups']:,}"),
                ("Bytes", f"{report['source']['bytes']:,}"),
                ("SHA-256", f"`{report['source']['sha256']}`"),
                (
                    "Manifest checks",
                    "passed" if report["source"]["manifest_checks_passed"] else "FAILED",
                ),
            ),
        ),
        "",
        "## Executive comparison",
        "",
    ]

    comparison_rows = []
    for cohort in COHORT_ORDER:
        metrics = cohorts[cohort]["metrics"]
        records = int(metrics["records"])
        comparison_rows.append(
            (
                COHORT_LABELS[cohort],
                f"{records:,}",
                f"{percent(records, total):.3f}%",
                f"{percent(metrics['valid_coordinates'], records):.2f}%",
                f"{percent(metrics['locality_text'], records):.2f}%",
                f"{percent(metrics['unresolved_with_locality'], records):.2f}%",
                f"{percent(metrics['unresolved_structured_geography_only'], records):.2f}%",
                f"{percent(metrics['unresolved_no_location_evidence'], records):.2f}%",
            )
        )
    lines.extend(
        [
            markdown_table(
                (
                    "Cohort",
                    "Records",
                    "Snapshot share",
                    "Valid coords",
                    "Locality",
                    "PlaceBot-ready unresolved",
                    "Structured-only unresolved",
                    "No location evidence",
                ),
                comparison_rows,
            ),
            "",
            f"The four target publishers account for **{core_total:,} records",
            f"({percent(core_total, total):.3f}%)**. The remaining",
            f"{total - core_total:,} rows are proxy-admitted records from other",
            "GB publishers and are kept visible rather than silently discarded.",
            "",
            "## Recovery-lane definitions",
            "",
            "The lanes are mutually exclusive and sum to each cohort total:",
            "",
            "1. valid finite, in-range decimal coordinates already exist;",
            "2. no valid pair, but nonblank locality text is directly PlaceBot-ready;",
            "3. neither of the above, but country or state/province provides only a",
            "   coarse constraint;",
            "4. none of those location signals is present in this compact schema.",
            "",
            "Unlike the native NHM audit, this snapshot cannot identify an OCR-first",
            "label/register lane because it retains media type but not asset category,",
            "title or URL.",
            "",
            "## NHM-report comparability",
            "",
            markdown_table(
                ("NHM assessment route", "Cross-institution status"),
                (
                    ("Existing valid coordinates", "Measured directly"),
                    ("PlaceBot from existing locality text", "Measured directly"),
                    (
                        "Scribe OCR from label/register images",
                        "Not measurable: media identity and URLs are absent",
                    ),
                    (
                        "Verbatim latitude/longitude parsing",
                        "Not measurable: verbatim-coordinate fields are absent",
                    ),
                    (
                        "Structured geography only",
                        "Partially measured using country and state/province only",
                    ),
                    (
                        "Associated publications",
                        "Not measurable: DOI and citation fields are absent",
                    ),
                ),
            ),
            "",
        ]
    )

    detailed_cohorts = ("all_snapshot", "nhm", "kew", "rbge", "mba")
    location_metrics = (
        ("Valid coordinates", "valid_coordinates"),
        ("Valid coordinates and locality", "coordinate_and_locality"),
        ("Valid coordinates only", "coordinate_only"),
        ("Unresolved with locality text", "unresolved_with_locality"),
        (
            "Unresolved with structured geography only",
            "unresolved_structured_geography_only",
        ),
        ("Unresolved with no location evidence", "unresolved_no_location_evidence"),
        ("Coordinate pair populated but invalid", "coordinate_pair_populated_but_invalid"),
        ("Partial coordinate pair", "partial_coordinate_pair"),
        ("Syntactically valid (0, 0)", "zero_zero_coordinates"),
    )
    coverage_metrics = (
        ("Nonblank locality", "locality_text"),
        ("Country code", "country_code"),
        ("State/province", "state_province"),
        ("Any media-type indicator", "any_media_type"),
        ("Still-image indicator", "still_image_type"),
        ("Any GBIF issue flag", "any_issue"),
        ("Scientific name", "scientific_name"),
        ("Species-level name", "species_name"),
        ("Event date", "event_date"),
        ("Plausible year (1000–2026)", "year_plausible"),
        ("Occurrence ID", "occurrence_id"),
        ("Catalogue number", "catalog_number"),
    )
    breakdown_titles = (
        ("basis_of_record", "Basis of record"),
        ("country_code", "Top country codes"),
        ("kingdom", "Top kingdoms"),
        ("order", "Top taxonomic orders"),
        ("institution_code", "Top institution codes"),
        ("collection_code", "Top collection codes"),
        ("media_type", "Media types"),
        ("issue", "Top GBIF issue flags"),
    )

    for cohort in detailed_cohorts:
        entry = cohorts[cohort]
        metrics = entry["metrics"]
        records = int(metrics["records"])
        unresolved = int(metrics["unresolved_records"])
        if unresolved:
            recovery_summary = (
                f"Of {unresolved:,} records without valid coordinates, "
                f"{percent(metrics['unresolved_with_locality'], unresolved):.2f}% "
                "are directly PlaceBot-ready, "
                f"{percent(metrics['unresolved_structured_geography_only'], unresolved):.2f}% "
                "have country/state constraints only, and "
                f"{percent(metrics['unresolved_no_location_evidence'], unresolved):.2f}% "
                "have none of those location signals."
            )
        else:
            recovery_summary = (
                "All records in this publisher cohort already contain a finite, "
                "in-range decimal coordinate pair."
            )
        lines.extend(
            [
                f"## {COHORT_LABELS[cohort]}",
                "",
                markdown_table(
                    ("Source structure", "Value"),
                    (
                        ("Records", f"{records:,}"),
                        ("Distinct datasets", f"{int(metrics['datasets']):,}"),
                        (
                            "Publishing organisations",
                            f"{int(metrics['publisher_organizations']):,}",
                        ),
                        (
                            "Institution codes",
                            f"{int(metrics['institution_codes']):,}",
                        ),
                        (
                            "Collection codes",
                            f"{int(metrics['collection_codes']):,}",
                        ),
                        ("Distinct species keys", f"{int(metrics['species_keys']):,}"),
                        (
                            "Distinct unresolved normalised locality-country keys",
                            f"{int(metrics['unresolved_distinct_normalized_locality_country_keys']):,}",
                        ),
                        (
                            "Plausible year range",
                            f"{metrics['minimum_year']}–{metrics['maximum_year']}",
                        ),
                    ),
                ),
                "",
                "### Location coverage and recovery lanes",
                "",
                recovery_summary,
                "",
                markdown_table(
                    ("Measure", "Records", "Share of cohort"),
                    metric_rows(metrics, location_metrics),
                ),
                "",
                "### Other coverage indicators",
                "",
                markdown_table(
                    ("Measure", "Records", "Share of cohort"),
                    metric_rows(metrics, coverage_metrics),
                ),
                "",
                f"Among {int(metrics['locality_text']):,} locality strings,",
                f"{int(metrics['locality_with_detail_punctuation']):,} contain a",
                "semicolon, comma or colon and",
                f"{int(metrics['locality_over_80_characters']):,} exceed 80 characters.",
                "",
            ]
        )
        for breakdown_key, title in breakdown_titles:
            lines.extend(
                [
                    render_breakdown(
                        title,
                        entry["breakdowns"][breakdown_key],
                        records,
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This report measures what is present in the compact snapshot. It does",
            "not prove that every publisher-owned row belongs to the corresponding",
            "physical institution, that every still image is a specimen label, or",
            "that country/state-only rows can safely be converted to point coordinates.",
            "The valid-coordinate definition includes 356 syntactically in-range",
            "(0, 0) pairs in the whole snapshot; these are reported separately and",
            "should be treated as data-quality exceptions before use.",
            "The Marine Biological Association publisher group spans 114 datasets",
            "and includes DASSH records, so it is a publisher-grouped marine data",
            "cohort rather than evidence of one physically held collection.",
            "The native 143-field NHM CSV remains the richer source for NHM media and",
            "recovery analysis; this report is the matched cross-institution baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("manifest must contain a JSON object")
    return value


def build_report(
    connection: Any,
    input_path: Path,
    manifest_path: Path,
    verify_checksum: bool,
) -> dict[str, Any]:
    relation = source_relation(input_path)
    columns = validate_schema(connection, relation)
    manifest = load_manifest(manifest_path)
    metrics = query_metrics(connection, relation)
    breakdowns = query_breakdowns(connection, relation)

    if set(metrics) != set(COHORT_ORDER):
        raise ValueError(
            f"unexpected cohorts: expected {sorted(COHORT_ORDER)}, "
            f"found {sorted(metrics)}"
        )

    metadata = connection.execute(
        f"SELECT sum(num_row_groups) AS row_groups, sum(num_rows) AS rows "
        f"FROM parquet_file_metadata({sql_literal(input_path)})"
    ).fetchone()
    checksum = sha256_file(input_path) if verify_checksum else manifest["output"]["sha256"]
    source_records = int(metrics["all_snapshot"]["records"])
    manifest_records = int(manifest["output"]["records"])
    manifest_checksum = str(manifest["output"]["sha256"])
    manifest_checks_passed = (
        source_records == manifest_records
        and int(metadata[1]) == source_records
        and checksum == manifest_checksum
    )
    if not manifest_checks_passed:
        raise ValueError("source row count or checksum does not match the manifest")

    core_total = sum(int(metrics[item.slug]["records"]) for item in CORE_INSTITUTIONS)
    cohort_entries = {}
    for cohort in COHORT_ORDER:
        institution = next(
            (item for item in CORE_INSTITUTIONS if item.slug == cohort),
            None,
        )
        cohort_entries[cohort] = {
            "label": COHORT_LABELS[cohort],
            "publisher_key": institution.publisher_key if institution else None,
            "metrics": metrics[cohort],
            "breakdowns": {
                name: values[cohort] for name, values in breakdowns.items()
            },
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "title": "DiSSCo UK snapshot profile",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "path": str(input_path),
            "manifest_path": str(manifest_path),
            "snapshot": manifest.get("snapshot"),
            "bytes": input_path.stat().st_size,
            "sha256": checksum,
            "checksum_recomputed": verify_checksum,
            "columns": len(columns),
            "column_schema": [
                {"name": name, "type": data_type} for name, data_type in columns
            ],
            "row_groups": int(metadata[0]),
            "metadata_rows": int(metadata[1]),
            "manifest_checks_passed": manifest_checks_passed,
            "qualification": manifest.get("qualification"),
            "live_portal_records": int(
                manifest["live_portal_comparison"]["portal_records"]
            ),
            "proxy_minus_live_portal_records": int(
                manifest["live_portal_comparison"]["proxy_minus_portal_records"]
            ),
        },
        "cohort_assignment": {
            "field": "publishingorgkey",
            "core_institutions": [
                {
                    "slug": item.slug,
                    "name": item.name,
                    "publisher_key": item.publisher_key,
                }
                for item in CORE_INSTITUTIONS
            ],
            "other_rule": "all publisher keys not matching the four core keys",
        },
        "recovery_lane_order": [
            "valid_coordinates",
            "unresolved_with_locality",
            "unresolved_structured_geography_only",
            "unresolved_no_location_evidence",
        ],
        "cohorts": cohort_entries,
        "derived": {
            "core_institution_records": core_total,
            "core_institution_share_percent": percent(core_total, source_records),
            "other_publisher_records": source_records - core_total,
            "other_publisher_share_percent": percent(
                source_records - core_total,
                source_records,
            ),
        },
        "limitations": [
            "Institution cohorts use publishingorgkey because institutionKey and collectionKey are absent.",
            "Media type does not identify label or accession-register imagery and no media URL is retained.",
            "The compact schema lacks verbatim coordinates and the richer NHM geography fields.",
            "Country or state/province is a coarse constraint, not a point-coordinate recovery.",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_REPORT_ROOT.with_suffix(".json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_REPORT_ROOT.with_suffix(".md"),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="trust the checksum recorded in the manifest",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing report artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    json_output = args.json_output.expanduser().resolve()
    markdown_output = args.markdown_output.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if input_path in {json_output, markdown_output}:
        raise ValueError("an output path resolves to the source Parquet")
    if json_output == markdown_output:
        raise ValueError("JSON and Markdown outputs must be distinct")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    for output in (json_output, markdown_output):
        if output.exists() and not args.force:
            raise FileExistsError(f"refusing to replace {output}; use --force")
        output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="placebot-dissco-profile-", dir="/tmp") as tmp:
        connection = connect_duckdb(args.threads, args.memory_limit, Path(tmp))
        try:
            report = build_report(
                connection,
                input_path,
                manifest_path,
                verify_checksum=not args.skip_checksum,
            )
        finally:
            connection.close()

    markdown = render_markdown(report)
    atomic_json(json_output, report)
    atomic_text(markdown_output, markdown)
    print(f"JSON report: {json_output}")
    print(f"Markdown report: {markdown_output}")
    print(f"Records: {report['cohorts']['all_snapshot']['metrics']['records']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
