#!/usr/bin/env python3
"""Recount PlaceBot target records from the completed, attested public extract.

Read-only source audit. No model calls, evidence builds or production writes.
The report separates raw specimens, eligible targets and repeated input keys.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import duckdb

import analyze_dissco_uk_snapshot as HISTORICAL
import gbif_locality_memory as INTEGRITY


def audit(source: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Audit output must be fresh")
    began = time.monotonic()
    verified = INTEGRITY.validate_source(source, "full", 16)
    paths = [str(source / "parts" / item["receipt"]["part"]) for item in verified["selected"]]
    normalize = lambda column: f"lower(regexp_replace(trim(coalesce({column}, '')), '\\s+', ' ', 'g'))"
    cohort_case = HISTORICAL.cohort_case_sql()
    query = f"""
    WITH classified AS (
      SELECT gbifid, {cohort_case} AS cohort, license,
             {HISTORICAL.VALID_COORDINATE_SQL} AS valid_coordinates,
             {HISTORICAL.HAS_LOCALITY_SQL} AS has_locality,
             {HISTORICAL.HAS_STRUCTURED_GEOGRAPHY_SQL} AS has_geography,
             coalesce(decimallatitude=0 AND decimallongitude=0, false) AS zero_zero,
             {normalize('locality')} AS locality_key,
             {normalize('countrycode')} AS country_key,
             {normalize('stateprovince')} AS state_key
      FROM read_parquet(?)
    )
    SELECT coalesce(cohort,'all_snapshot') AS cohort,
           count(*) AS records,
           count(*) FILTER (WHERE valid_coordinates) AS already_has_coordinates,
           count(*) FILTER (WHERE NOT valid_coordinates AND has_locality) AS placebot_candidates,
           count(*) FILTER (WHERE NOT valid_coordinates AND NOT has_locality AND has_geography) AS structured_only,
           count(*) FILTER (WHERE NOT valid_coordinates AND NOT has_locality AND NOT has_geography) AS no_location_evidence,
           count(*) FILTER (WHERE valid_coordinates AND has_locality) AS coordinate_and_locality,
           count(*) FILTER (WHERE zero_zero AND has_locality) AS zero_zero_with_locality,
           count(*) FILTER (WHERE (NOT valid_coordinates OR zero_zero) AND has_locality) AS candidates_if_zero_zero_missing,
           count(DISTINCT (locality_key,country_key)) FILTER (WHERE NOT valid_coordinates AND has_locality) AS distinct_locality_country,
           count(DISTINCT (locality_key,country_key,state_key)) FILTER (WHERE NOT valid_coordinates AND has_locality) AS distinct_locality_country_state,
           count(*) FILTER (WHERE NOT valid_coordinates AND has_locality AND country_key='') AS candidates_without_country,
           count(*) FILTER (WHERE NOT valid_coordinates AND has_locality AND state_key<>'') AS candidates_with_state,
           count(*) FILTER (WHERE NOT valid_coordinates AND has_locality AND license='CC_BY_NC_4_0') AS candidates_cc_by_nc,
           count(*) FILTER (WHERE NOT valid_coordinates AND has_locality AND license IN ('CC0_1_0','CC_BY_4_0')) AS candidates_cc0_or_cc_by
    FROM classified
    GROUP BY GROUPING SETS ((cohort),())
    ORDER BY cohort
    """
    print("Verified all source part hashes; counting eligibility and repeated input keys", flush=True)
    db = duckdb.connect()
    try:
        db.execute("SET threads=8")
        db.execute("SET memory_limit='12GB'")
        db.execute("SET preserve_insertion_order=false")
        cursor = db.execute(query, [paths])
        names = [item[0] for item in cursor.description]
        cohorts = {row[0]: dict(zip(names, row)) for row in cursor.fetchall()}
    finally:
        db.close()
    overall = cohorts["all_snapshot"]
    if overall["records"] != sum(item["receipt"]["records"] for item in verified["selected"]):
        raise ValueError("Read row count differs from complete source manifest")
    for values in cohorts.values():
        if sum(values[key] for key in ("already_has_coordinates", "placebot_candidates", "structured_only", "no_location_evidence")) != values["records"]:
            raise ValueError("Eligibility lanes do not account for every source record")
    if INTEGRITY.validate_source(source, "full", 16) != verified:
        raise ValueError("Source changed during audit")
    historical_path = Path(__file__).resolve().parents[1] / "reports/dissco_uk_georeferencing_opportunity_assessment_20260721.json"
    report = {
        "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source), "source_snapshot": verified["snapshot"],
        "source_identity_sha256": verified["identity_sha256"],
        "source_manifest_sha256": verified["complete_manifest_sha256"],
        "source_parts": len(paths), "cohorts": cohorts,
        "definition": "No finite in-range latitude/longitude pair, with nonblank locality text. Raw target eligibility before benchmark exclusions or quality routing.",
        "unique_key_qualification": "Case and whitespace normalization only. Locality/country/state keys retain more context than locality/country alone; neither proves semantic equivalence or final model-call count.",
        "publisher_qualification": "Institutions assigned by publishing organization; this is the DiSSCo UK-aligned GBIF proxy.",
        "historical_audit": {"path": str(historical_path), "sha256": INTEGRITY.sha(historical_path)},
        "code_sha256": {Path(p).name: INTEGRITY.sha(p) for p in (Path(__file__), Path(HISTORICAL.__file__), Path(INTEGRITY.__file__))},
        "duckdb_version": duckdb.__version__, "query": query,
        "seconds": time.monotonic() - began, "model_calls": 0,
    }
    INTEGRITY.save(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source.resolve(), args.output.resolve())
    print(json.dumps({k: v for k, v in result.items() if k not in {"query", "code_sha256"}}, indent=2))
