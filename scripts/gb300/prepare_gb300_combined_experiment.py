#!/usr/bin/env python3
"""Prepare a blind, deduplicated production/evaluation union. No model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import duckdb
import gbif_locality_memory as integrity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from placebot.cli import hpc

COLUMNS = {"locality": "locality", "country": "countrycode",
           "latitude": "decimallatitude", "longitude": "decimallongitude",
           "identifier": "gbifid", "context": ["stateprovince"]}


def prepare(source_root: Path, output: Path) -> dict:
    source_root, output = source_root.resolve(), output.resolve()
    if source_root == output or source_root in output.parents or output in source_root.parents:
        raise ValueError("Use a separate, fresh preparation directory")
    source = source_root / "source.parquet"
    attestation = integrity.read(source_root / "COMPLETE.json")
    source_sha = integrity.sha(source)
    if attestation.get("status") != "complete" or source_sha != attestation["compacted_source_sha256"]:
        raise ValueError("Source hash does not match completed preparation")
    if output.exists():
        raise ValueError("Use a fresh preparation directory; existing artifacts are preserved")
    output.mkdir(parents=True, mode=0o700)
    (output / "evaluation_only").mkdir(mode=0o700)
    e = hpc._selection_sql(COLUMNS, include_existing=True, treat_zero_zero_missing=False)
    # Keep production eligibility unchanged. Exclude 0,0 from evaluation;
    # singleton zeros (e.g. Greenwich longitude) are NOT excluded.
    zero = f"({e['valid_coordinates']} AND {e['latitude']} = 0 AND {e['longitude']} = 0)"
    selected = f"({e['has_locality']} AND NOT {zero})"
    with duckdb.connect(str(output / "prepare.duckdb")) as db:
        db.execute("SET threads=8")
        db.execute("SET memory_limit='16GB'")
        db.execute(f"SET temp_directory={hpc._sql_literal(output / 'duckdb-tmp')}")
        db.execute(f"CREATE VIEW source AS SELECT * FROM read_parquet({hpc._sql_literal(source)})")
        db.execute(f"""CREATE TABLE selected AS SELECT
            {e['candidate_id']} AS candidate_id,
            trim({e['locality']}) AS locality, trim({e['country']}) AS country,
            {e['context_json']} AS context_json,
            cast(gbifid AS VARCHAR) AS gbifid,
            CASE WHEN {e['valid_coordinates']} THEN 'evaluation' ELSE 'production' END AS cohort
            FROM source WHERE {selected}""")
        rows, ids = db.execute("SELECT count(*), count(DISTINCT gbifid) FROM selected").fetchone()
        if rows != ids:
            raise ValueError("Specimen mapping requires unique non-null GBIF IDs")
        db.execute("""CREATE TABLE candidates AS SELECT candidate_id,
            min(locality) AS locality, min(country) AS country,
            count(*)::BIGINT AS occurrence_count, min(gbifid) AS example_identifier,
            min(context_json) AS context_json
            FROM selected GROUP BY candidate_id ORDER BY candidate_id""")
        artifacts = {
            "candidates": ("candidates.parquet", "SELECT * FROM candidates"),
            "specimen_mapping": ("specimen_mapping.parquet", "SELECT candidate_id, gbifid, cohort FROM selected ORDER BY candidate_id, gbifid"),
            "candidate_cohorts": ("candidate_cohorts.parquet", "SELECT candidate_id, count(*) FILTER (WHERE cohort='production')::BIGINT AS production_records, count(*) FILTER (WHERE cohort='evaluation')::BIGINT AS evaluation_records FROM selected GROUP BY candidate_id ORDER BY candidate_id"),
            "withheld_references": ("evaluation_only/references.parquet", "SELECT s.candidate_id, source.* FROM selected s JOIN source ON s.gbifid=cast(source.gbifid AS VARCHAR) WHERE s.cohort='evaluation'"),
        }
        for _, (relative, query) in artifacts.items():
            db.execute(f"COPY ({query}) TO {hpc._sql_literal(output / relative)} (FORMAT PARQUET, COMPRESSION ZSTD)")
        cohort_counts = dict(db.execute("SELECT cohort, count(*) FROM selected GROUP BY cohort").fetchall())
        unique_counts = dict(db.execute("SELECT cohort, count(DISTINCT candidate_id) FROM selected GROUP BY cohort").fetchall())
        unique = db.execute("SELECT count(*) FROM candidates").fetchone()[0]
        counts = {"input_records": db.execute("SELECT count(*) FROM source").fetchone()[0],
                  "selected_records_before_deduplication": rows, "unique_candidates": unique,
                  "records_by_cohort": cohort_counts, "unique_inputs_by_cohort": unique_counts,
                  "shared_unique_inputs": sum(unique_counts.values()) - unique,
                  "excluded_zero_zero_with_locality": db.execute(f"SELECT count(*) FROM source WHERE {e['has_locality']} AND {zero}").fetchone()[0]}
    if integrity.sha(source) != source_sha:
        raise ValueError("Source changed during preparation")
    manifest = {"schema_version": hpc.SCHEMA_VERSION, "status": "complete",
        "created_at": hpc._utc_now(), "source": {"path": str(source), "sha256": source_sha,
        "parent_attestation_sha256": integrity.sha(source_root / "COMPLETE.json"),
        "raw_identity_sha256": attestation.get("source_identity_sha256")},
        "columns": COLUMNS, "counts": counts,
        "policy": {"include_existing": True, "treat_zero_zero_missing": False,
            "exclude_existing_zero_zero": True, "context_columns": ["stateprovince"],
            "candidate_key_recipe": "locality-country-context-json-v1",
            "inference_fields": ["locality", "country", "context_json.stateprovince"],
            "deduplication": "lowercase, trim and collapse whitespace over locality+country+explicit context",
            "no_reference_coordinates_or_memory_in_inference": True,
            "qualification": "GBIF UK-publisher specimen proxy; locality is not certified verbatimLocality. Existing coordinates are a reference baseline, not infallible ground truth."},
        "artifacts": {name: {"path": relative, "sha256": integrity.sha(output / relative),
                               "bytes": (output / relative).stat().st_size}
                      for name, (relative, _) in artifacts.items()},
        "code_sha256": {str(Path(__file__)): integrity.sha(Path(__file__)), str(Path(hpc.__file__)): integrity.sha(Path(hpc.__file__))}}
    integrity.save(output / "manifest.json", manifest)
    integrity.save(output / "COMPLETE.json", {"status": "complete", "manifest_sha256": integrity.sha(output / "manifest.json"), "counts": counts})
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_root, args.output)["counts"], indent=2))
