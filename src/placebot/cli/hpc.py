#!/usr/bin/env python3
"""Disk-backed PlaceBot workflow for multi-million-row HPC datasets.

The desktop prep/expand commands intentionally favour simplicity and keep rows
in memory.  This module is the large-data path: DuckDB scans CSV/TSV/Parquet,
exact locality-country targets are deduplicated on disk, inference results are
checkpointed in SQLite, and reconstitution is a streaming DuckDB join.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
import re
import signal
import sqlite3
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from importlib import metadata, resources
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import requests

from placebot.core.ai_processor import AIProcessor
from placebot.core.model_selector import load_model_profile

SCHEMA_VERSION = 1
OUTPUT_FIELDS = (
    "country",
    "state",
    "region",
    "sector",
    "exact_site",
    "latitude",
    "longitude",
    "coordinate_source",
    "coordinate_radius_meters",
    "elevation_meters",
    "elevation_original",
    "confidence",
    "notes",
    "collection_notes",
)
STRING_OUTPUT_FIELDS = {
    "country",
    "state",
    "region",
    "sector",
    "exact_site",
    "coordinate_source",
    "elevation_original",
    "confidence",
    "notes",
    "collection_notes",
}
COLUMN_ALIASES = {
    "locality": (
        "locality",
        "verbatimLocality",
        "Locality verbatim",
        "label_verbatim",
    ),
    "country": ("countrycode", "countryCode", "country", "Country"),
    "latitude": ("decimallatitude", "decimalLatitude", "Latitude", "latitude"),
    "longitude": (
        "decimallongitude",
        "decimalLongitude",
        "Longitude",
        "longitude",
    ),
    "identifier": (
        "gbifid",
        "gbifID",
        "occurrenceid",
        "occurrenceID",
        "catalognumber",
        "catalogNumber",
        "Barcode",
        "materialEntityID",
        "recordNumber",
    ),
}
RESULT_EXPORT_FIELDS = (
    "candidate_id",
    "locality",
    "input_country",
    "occurrence_count",
    "example_identifier",
    "completed_at",
    "api_success",
    "usable",
    *OUTPUT_FIELDS,
    "error",
    "latency_seconds",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "finish_reason",
    "stop_reason",
    "truncated",
    "response_content",
    "context_json",
)


class HpcWorkflowError(RuntimeError):
    """Raised when an HPC workflow guard fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise HpcWorkflowError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _duckdb_module():
    try:
        import duckdb
    except ImportError as exc:
        raise HpcWorkflowError(
            "DuckDB is required. Install the HPC extra with: "
            "python -m pip install -e '.[hpc]'"
        ) from exc
    return duckdb


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _detect_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffixes = [suffix.lower() for suffix in path.suffixes]
    logical = suffixes[-2] if suffixes and suffixes[-1] == ".gz" else suffixes[-1]
    if logical in {".parquet", ".pq"}:
        return "parquet"
    if logical in {".tsv", ".tab", ".txt"}:
        return "tsv"
    if logical == ".csv":
        return "csv"
    raise HpcWorkflowError(
        f"Cannot infer input format from {path}; pass --input-format"
    )


def _source_relation(path: Path, input_format: str, row_limit: Optional[int]) -> str:
    literal = _sql_literal(path)
    if input_format == "parquet":
        relation = f"read_parquet({literal})"
    else:
        delimiter = "\\t" if input_format == "tsv" else ","
        relation = (
            f"read_csv({literal}, header=true, delim={_sql_literal(delimiter)}, "
            "all_varchar=true, ignore_errors=false, sample_size=1048576)"
        )
    if row_limit is not None:
        relation = f"(SELECT * FROM {relation} LIMIT {int(row_limit)})"
    return relation


def _schema_columns(connection: Any, relation: str) -> list[str]:
    rows = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    return [str(row[0]) for row in rows]


def _resolve_column(
    columns: Sequence[str],
    concept: str,
    override: Optional[str],
    *,
    required: bool,
) -> Optional[str]:
    by_lower: dict[str, list[str]] = {}
    for column in columns:
        by_lower.setdefault(column.lower(), []).append(column)
    if override:
        if override in columns:
            return override
        matches = by_lower.get(override.lower(), [])
        if len(matches) == 1:
            return matches[0]
        raise HpcWorkflowError(f"Column not found or ambiguous: {override!r}")
    for alias in COLUMN_ALIASES[concept]:
        if alias in columns:
            return alias
        matches = by_lower.get(alias.lower(), [])
        if len(matches) == 1:
            return matches[0]
    if required:
        accepted = ", ".join(COLUMN_ALIASES[concept])
        raise HpcWorkflowError(
            f"No {concept} column found. Pass --{concept}-column; accepted aliases: "
            f"{accepted}"
        )
    return None


def _normalised_text_sql(expression: str) -> str:
    return f"lower(regexp_replace(trim(coalesce({expression}, '')), '\\s+', ' ', 'g'))"


def _resolve_context_columns(columns: Sequence[str], requested: Sequence[str]) -> list[str]:
    """Explicit geographic evidence only; never expose evaluation targets."""
    reserved = {
        "lat", "lon", "lng", "latitude", "longitude", "decimallatitude", "decimallongitude",
        "scientificname", "taxonid", "taxonkey", "acceptedscientificname",
        "kingdom", "phylum", "class", "order", "family", "genus", "species",
        "specificepithet", "infraspecificepithet", "taxonrank", "label",
    }
    forbidden_fragments = (
        "reference", "groundtruth", "gold", "expected", "benchmark", "distance", "error", "taxon",
    )
    resolved = set()
    for name in requested:
        column = _resolve_column(columns, "context", name, required=True)
        token = "".join(character for character in str(column).casefold() if character.isalnum())
        manual_coordinate = "manual" in token and (
            any(part in {
                "lat", "lon", "long", "lng", "radius", "uncertainty",
                "coordinate", "coordinates", "easting", "northing",
            } for part in re.split(r"[^a-z0-9]+", str(column).casefold()))
            or
            token.replace("manual", "").startswith((
                "lat", "lon", "lng", "radius", "uncertainty", "coordinate",
                "easting", "northing",
            ))
            or any(fragment in token for fragment in (
                "latitude", "longitude", "radius", "uncertainty",
                "coordinate", "easting", "northing",
            ))
        )
        if (
            token in reserved
            or manual_coordinate
            or any(fragment in token for fragment in forbidden_fragments)
            or token.startswith(("truelat", "truelon", "targetlat", "targetlon", "reflat", "reflon"))
        ):
            raise HpcWorkflowError(
                f"Context column {column!r} is reserved for coordinates, taxonomy or scoring; "
                "pass geographic source context only"
            )
        resolved.add(str(column))
    return sorted(resolved, key=lambda value: (value.casefold(), value))


def _selection_sql(
    columns: Mapping[str, Any],
    *,
    include_existing: bool,
    treat_zero_zero_missing: bool,
    table_alias: Optional[str] = None,
) -> dict[str, str]:
    def column_expression(name: str) -> str:
        quoted = _quote_identifier(name)
        return f"{_quote_identifier(table_alias)}.{quoted}" if table_alias else quoted

    locality = f"cast({column_expression(str(columns['locality']))} AS VARCHAR)"
    country = (
        f"cast({column_expression(str(columns['country']))} AS VARCHAR)"
        if columns.get("country")
        else "''"
    )
    identifier = (
        f"cast({column_expression(str(columns['identifier']))} AS VARCHAR)"
        if columns.get("identifier")
        else "''"
    )
    latitude = (
        f"try_cast({column_expression(str(columns['latitude']))} AS DOUBLE)"
        if columns.get("latitude")
        else "NULL::DOUBLE"
    )
    longitude = (
        f"try_cast({column_expression(str(columns['longitude']))} AS DOUBLE)"
        if columns.get("longitude")
        else "NULL::DOUBLE"
    )
    valid_coordinates = (
        f"({latitude} IS NOT NULL AND {longitude} IS NOT NULL "
        f"AND isfinite({latitude}) AND isfinite({longitude}) "
        f"AND {latitude} BETWEEN -90.0 AND 90.0 "
        f"AND {longitude} BETWEEN -180.0 AND 180.0)"
    )
    if treat_zero_zero_missing:
        valid_coordinates = (
            f"({valid_coordinates} AND NOT ({latitude} = 0.0 AND {longitude} = 0.0))"
        )
    has_locality = f"nullif(trim({locality}), '') IS NOT NULL"
    selected = (
        has_locality
        if include_existing
        else f"({has_locality} AND NOT {valid_coordinates})"
    )
    normalized_locality = _normalised_text_sql(locality)
    normalized_country = _normalised_text_sql(country)
    candidate_id = (
        f"sha256(concat_ws(chr(0), {normalized_locality}, {normalized_country}))"
    )
    context_columns = columns.get("context") or []
    context_json = "'{}'"
    if context_columns:
        raw_parts = []
        normalized_parts = []
        for name in context_columns:
            expression = f"cast({column_expression(name)} AS VARCHAR)"
            raw_parts.extend((_sql_literal(name), expression))
            normalized_parts.extend((_sql_literal(name), _normalised_text_sql(expression)))
        context_json = f"cast(json_object({', '.join(raw_parts)}) AS VARCHAR)"
        normalized_context = f"json_object({', '.join(normalized_parts)})"
        # JSON boundaries are unambiguous even if source text contains a NUL.
        candidate_id = (
            f"sha256(cast(json_array({normalized_locality}, {normalized_country}, "
            f"{normalized_context}) AS VARCHAR))"
        )
    return {
        "locality": locality,
        "country": country,
        "identifier": identifier,
        "latitude": latitude,
        "longitude": longitude,
        "valid_coordinates": valid_coordinates,
        "has_locality": has_locality,
        "selected": selected,
        "normalized_locality": normalized_locality,
        "normalized_country": normalized_country,
        "candidate_id": candidate_id,
        "context_json": context_json,
    }


def _configure_duckdb(connection: Any, args: argparse.Namespace, root: Path) -> None:
    temp_dir = root / "duckdb-tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory={_sql_literal(temp_dir)}")
    connection.execute(f"SET threads={int(args.duckdb_threads)}")
    if args.memory_limit:
        connection.execute(f"SET memory_limit={_sql_literal(args.memory_limit)}")


def prepare_command(args: argparse.Namespace) -> int:
    source = args.input.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise HpcWorkflowError(f"Input must be one regular file: {source}")
    if output_root.exists() and any(output_root.iterdir()):
        raise HpcWorkflowError(
            f"Output directory is not empty: {output_root}. Use a fresh path."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    input_format = _detect_format(source, args.input_format)
    relation = _source_relation(source, input_format, args.row_limit)
    duckdb = _duckdb_module()
    database = output_root / "prepare.duckdb"
    connection = duckdb.connect(str(database))
    try:
        _configure_duckdb(connection, args, output_root)
        schema = _schema_columns(connection, relation)
        columns = {
            "locality": _resolve_column(
                schema, "locality", args.locality_column, required=True
            ),
            "country": _resolve_column(
                schema, "country", args.country_column, required=False
            ),
            "latitude": _resolve_column(
                schema, "latitude", args.latitude_column, required=False
            ),
            "longitude": _resolve_column(
                schema, "longitude", args.longitude_column, required=False
            ),
            "identifier": _resolve_column(
                schema, "identifier", args.identifier_column, required=False
            ),
            "context": _resolve_context_columns(schema, args.context_column),
        }
        expressions = _selection_sql(
            columns,
            include_existing=args.include_existing,
            treat_zero_zero_missing=args.treat_zero_zero_missing,
        )
        connection.execute(f"""
            CREATE TABLE candidates AS
            SELECT
                candidate_id,
                min(locality) AS locality,
                min(country) AS country,
                count(*)::BIGINT AS occurrence_count,
                min(identifier) AS example_identifier,
                min(context_json) AS context_json
            FROM (
                SELECT
                    {expressions['candidate_id']} AS candidate_id,
                    trim({expressions['locality']}) AS locality,
                    trim({expressions['country']}) AS country,
                    trim({expressions['identifier']}) AS identifier,
                    {expressions['context_json']} AS context_json
                FROM {relation}
                WHERE {expressions['selected']}
            ) selected
            GROUP BY candidate_id
            ORDER BY candidate_id
            """)
        candidates_path = output_root / "candidates.parquet"
        connection.execute(
            f"COPY candidates TO {_sql_literal(candidates_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        counts = connection.execute(f"""
            SELECT
                count(*)::BIGINT AS total_records,
                count(*) FILTER (WHERE {expressions['has_locality']})::BIGINT,
                count(*) FILTER (WHERE {expressions['valid_coordinates']})::BIGINT,
                count(*) FILTER (WHERE {expressions['selected']})::BIGINT
            FROM {relation}
            """).fetchone()
        candidate_count = int(
            connection.execute("SELECT count(*) FROM candidates").fetchone()[0]
        )
    finally:
        connection.close()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": _utc_now(),
        "source": {
            "path": str(source),
            "format": input_format,
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "row_limit": args.row_limit,
        },
        "columns": columns,
        "policy": {
            "include_existing": args.include_existing,
            "treat_zero_zero_missing": args.treat_zero_zero_missing,
            "deduplication": (
                "lowercase, trim and collapse whitespace over locality+country"
                + ("+explicit context" if columns["context"] else "")
            ),
            "context_columns": columns["context"],
            "candidate_key_recipe": (
                "locality-country-context-json-v1" if columns["context"]
                else "locality-country-nul-v1"
            ),
        },
        "counts": {
            "input_records": int(counts[0]),
            "records_with_locality": int(counts[1]),
            "records_with_valid_coordinates": int(counts[2]),
            "selected_records_before_deduplication": int(counts[3]),
            "unique_candidates": candidate_count,
            "duplicates_removed": int(counts[3]) - candidate_count,
        },
        "artifacts": {
            "candidates": {
                "path": str(candidates_path),
                "sha256": _sha256(candidates_path),
            },
            "database": str(database),
        },
    }
    _atomic_json(output_root / "manifest.json", manifest)
    (output_root / "COMPLETE").write_text(f"{_utc_now()}\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Candidates: {candidates_path}")
    return 0


def _models_endpoint(chat_endpoint: str) -> str:
    if "/v1/" in chat_endpoint:
        return f"{chat_endpoint.split('/v1/', 1)[0]}/v1/models"
    return f"{chat_endpoint.rstrip('/')}/v1/models"


def _prepare_profile(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_model_profile(args.model_profile)
    if profile is None:
        raise HpcWorkflowError(f"Unknown model profile: {args.model_profile}")
    profile["quiet"] = getattr(args, "quiet", False)
    module = profile.get("module")
    if args.endpoint:
        endpoint = args.endpoint.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = (
                f"{endpoint}/chat/completions"
                if endpoint.endswith("/v1")
                else f"{endpoint}/v1/chat/completions"
            )
        profile["api_endpoint"] = endpoint
        if module is not None:
            setattr(module, "API_ENDPOINT", endpoint)
    if args.model_id:
        profile["model_id"] = args.model_id
        if module is not None:
            setattr(module, "MODEL_ID", args.model_id)
    if args.max_output_tokens is not None:
        profile["max_output_tokens"] = args.max_output_tokens
        if module is not None:
            setattr(module, "MAX_OUTPUT_TOKENS", args.max_output_tokens)
    if args.request_timeout is not None:
        profile["request_timeout"] = args.request_timeout
        if module is not None:
            setattr(module, "REQUEST_TIMEOUT", args.request_timeout)
    if args.max_retries is not None:
        profile["max_retries"] = args.max_retries
        if module is not None:
            setattr(module, "MAX_RETRIES", args.max_retries)
    return profile


def _probe_server(profile: Mapping[str, Any]) -> dict[str, Any]:
    endpoint = str(profile.get("api_endpoint") or "")
    if not endpoint:
        raise HpcWorkflowError("The model profile has no API endpoint")
    url = _models_endpoint(endpoint)
    module = profile.get("module")
    headers = module.get_headers(profile.get("api_key")) if module else {}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data", []) if isinstance(payload, dict) else []
    ids = [str(item.get("id")) for item in models if isinstance(item, dict)]
    wanted = str(profile.get("model_id") or "")
    if wanted not in ids:
        raise HpcWorkflowError(
            f"Configured model {wanted!r} is not served by {url}; available: {ids}"
        )
    selected = next(item for item in models if item.get("id") == wanted)
    return {"models_endpoint": url, "selected_model": selected}


def _stable_server_identity(server: Mapping[str, Any]) -> dict[str, Any]:
    """Keep serving identity, excluding vLLM's per-probe generated fields."""
    stable = dict(server)
    selected = stable.get("selected_model")
    if isinstance(selected, Mapping):
        stable["selected_model"] = {
            key: value
            for key, value in selected.items()
            if key not in {"created", "permission"}
        }
    return stable


def _stable_run_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    # Normalize old run manifests too, without rewriting their evidence.
    stable = dict(identity)
    if isinstance(stable.get("server"), Mapping):
        stable["server"] = _stable_server_identity(stable["server"])
    return stable


def _record_server_probe(run_root: Path, server: Mapping[str, Any]) -> None:
    """Retain the full live probe separately from immutable resume identity."""
    with (run_root / "server_probes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"at": _utc_now(), "server": server}, sort_keys=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _profile_identity(
    profile: Mapping[str, Any], reasoning_mode: Optional[str]
) -> dict[str, Any]:
    module = profile.get("module")
    source = Path(str(getattr(module, "__file__", ""))).resolve()
    getter = getattr(module, "get_reproducibility_metadata", None)
    settings: Mapping[str, Any] = {}
    if callable(getter):
        try:
            settings = getter(
                int(profile.get("max_output_tokens") or 0),
                reasoning_mode=reasoning_mode,
            )
        except TypeError:
            settings = getter(int(profile.get("max_output_tokens") or 0))
    return {
        "profile": profile.get("name"),
        "model_id": profile.get("model_id"),
        "endpoint": profile.get("api_endpoint"),
        "context_window": profile.get("max_tokens"),
        "max_output_tokens": profile.get("max_output_tokens"),
        "request_timeout": profile.get("request_timeout"),
        "max_retries": profile.get("max_retries"),
        "source_sha256": _sha256(source) if source.is_file() else None,
        "request_settings": dict(settings),
    }


def _implementation_identity() -> dict[str, Any]:
    """Pin new runs to the client implementation, not only its package label.

    A dirty/source installation can change without a version bump. Hash the
    package Python sources once per invocation, before any prediction requests.
    Existing manifests without this field retain their historical identity;
    never manufacture a retrospective attestation for them.
    """
    package = Path(__file__).resolve().parents[1]
    versions = {}
    for name in ("duckdb", "requests", "httpx", "pydantic", "transformers", "tiktoken"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "sqlite": sqlite3.sqlite_version,
        "packages": versions,
        "sources": {str(path.relative_to(package)): _sha256(path)
                    for path in sorted(package.rglob("*.py")) if path.is_file()},
    }


def _candidate_query(
    path: Path,
    shard_index: int,
    shard_count: int,
    offset: int,
    limit: Optional[int],
) -> str:
    relation = f"read_parquet({_sql_literal(path)})"
    connection = _duckdb_module().connect()
    try:
        columns = _schema_columns(connection, relation)
    finally:
        connection.close()
    context = "context_json" if "context_json" in columns else "'{}' AS context_json"
    where = ""
    if shard_count > 1:
        where = (
            "WHERE (cast('0x' || substr(candidate_id, 1, 16) AS UBIGINT) "
            f"% {shard_count}) = {shard_index}"
        )
    query = (
        "SELECT candidate_id, locality, country, occurrence_count, example_identifier, "
        f"{context} "
        f"FROM {relation} {where} ORDER BY candidate_id"
    )
    if limit is not None:
        query += f" LIMIT {limit}"
    if offset:
        query += f" OFFSET {offset}"
    return query


def _preparation_identity(path: Path, candidate_sha256: str) -> Optional[dict[str, Any]]:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    artifact = manifest.get("artifacts", {}).get("candidates", {})
    if not artifact:
        return None
    if manifest.get("status") != "complete" or artifact.get("sha256") != candidate_sha256:
        raise HpcWorkflowError("Candidate preparation manifest does not match the input")
    return {
        "manifest_sha256": _sha256(manifest_path),
        "source_sha256": manifest.get("source", {}).get("sha256"),
        "columns": manifest.get("columns", {}),
        "policy": manifest.get("policy", {}),
    }


def _init_results_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS results (
            candidate_id TEXT PRIMARY KEY,
            locality TEXT NOT NULL,
            input_country TEXT NOT NULL,
            occurrence_count INTEGER NOT NULL,
            example_identifier TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            api_success INTEGER NOT NULL,
            usable INTEGER NOT NULL,
            country TEXT,
            state TEXT,
            region TEXT,
            sector TEXT,
            exact_site TEXT,
            latitude REAL,
            longitude REAL,
            coordinate_source TEXT,
            coordinate_radius_meters REAL,
            elevation_meters REAL,
            elevation_original TEXT,
            confidence TEXT,
            notes TEXT,
            collection_notes TEXT,
            error TEXT,
            latency_seconds REAL NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            finish_reason TEXT,
            stop_reason TEXT,
            truncated INTEGER,
            reasoning_content TEXT,
            response_content TEXT,
            prediction_json TEXT NOT NULL
        )
        """)
    result_columns = {row[1] for row in connection.execute("PRAGMA table_info(results)")}
    if "context_json" not in result_columns:
        connection.execute(
            "ALTER TABLE results ADD COLUMN context_json TEXT NOT NULL DEFAULT '{}'"
        )
    connection.execute("""
        CREATE TABLE IF NOT EXISTS result_metrics (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            completed_candidates INTEGER NOT NULL,
            api_success_candidates INTEGER NOT NULL,
            usable_candidates INTEGER NOT NULL,
            represented_occurrences INTEGER NOT NULL,
            usable_represented_occurrences INTEGER NOT NULL,
            summed_request_seconds REAL NOT NULL
        )
        """)
    if connection.execute("SELECT 1 FROM result_metrics WHERE singleton = 1").fetchone() is None:
        # Existing checkpoints are upgraded with exactly one full-table scan.
        connection.execute("""
            INSERT INTO result_metrics
            SELECT 1, count(*), coalesce(sum(api_success), 0), coalesce(sum(usable), 0),
                   coalesce(sum(occurrence_count), 0),
                   coalesce(sum(CASE WHEN usable = 1 THEN occurrence_count ELSE 0 END), 0),
                   coalesce(sum(latency_seconds), 0)
            FROM results
            """)
    connection.execute("""
        CREATE TRIGGER IF NOT EXISTS result_metrics_insert AFTER INSERT ON results
        BEGIN
            UPDATE result_metrics SET
                completed_candidates = completed_candidates + 1,
                api_success_candidates = api_success_candidates + NEW.api_success,
                usable_candidates = usable_candidates + NEW.usable,
                represented_occurrences = represented_occurrences + NEW.occurrence_count,
                usable_represented_occurrences = usable_represented_occurrences +
                    CASE WHEN NEW.usable = 1 THEN NEW.occurrence_count ELSE 0 END,
                summed_request_seconds = summed_request_seconds + NEW.latency_seconds
            WHERE singleton = 1;
        END
        """)
    # Checkpoints are append-only. This also prevents counters silently drifting
    # if a future code path tries to amend a completed prediction.
    for operation in ("UPDATE", "DELETE"):
        connection.execute(f"""
            CREATE TRIGGER IF NOT EXISTS results_no_{operation.lower()}
            BEFORE {operation} ON results
            BEGIN
                SELECT RAISE(ABORT, 'Completed results are append-only');
            END
            """)
    # The first pass is an immutable attempt ledger. Failed cases are durable,
    # skipped on ordinary resume, and exposed explicitly for a separate retry
    # pass; they are never mistaken for successful predictions.
    connection.execute("CREATE INDEX IF NOT EXISTS results_failed ON results(candidate_id) WHERE usable = 0")
    connection.execute("CREATE VIEW IF NOT EXISTS deferred_failures AS SELECT * FROM results WHERE usable = 0")
    connection.commit()
    return connection


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _usable_prediction(prediction: Mapping[str, Any]) -> bool:
    if not prediction.get("success"):
        return False
    if any(field not in prediction for field in OUTPUT_FIELDS):
        return False
    if any(
        not isinstance(prediction.get(field), str) for field in STRING_OUTPUT_FIELDS
    ):
        return False
    if prediction.get("confidence") not in {"high", "medium", "low"}:
        return False
    lat = prediction.get("latitude")
    lon = prediction.get("longitude")
    if not _finite_number(lat) or not _finite_number(lon):
        return False
    if not -90 <= float(lat) <= 90 or not -180 <= float(lon) <= 180:
        return False
    for field in ("coordinate_radius_meters", "elevation_meters"):
        value = prediction.get(field)
        if value is not None and not _finite_number(value):
            return False
    radius = prediction.get("coordinate_radius_meters")
    if radius is not None and float(radius) < 0:
        return False
    if prediction.get("truncated") is True:
        return False
    finish_reason = prediction.get("finish_reason")
    if finish_reason not in (None, "", "stop"):
        return False
    return True


_thread_local = threading.local()


def _process_candidate(
    candidate: Sequence[Any],
    profile: Mapping[str, Any],
    reasoning_mode: Optional[str],
    evidence: str,
) -> dict[str, Any]:
    processor = getattr(_thread_local, "processor", None)
    if processor is None:
        processor = AIProcessor(dict(profile))
        _thread_local.processor = processor
    candidate_id, locality, country, occurrence_count, example_identifier = candidate[:5]
    context_json = str(candidate[5]) if len(candidate) > 5 else "{}"
    try:
        source_context = json.loads(context_json)
    except (ValueError, TypeError) as exc:
        raise HpcWorkflowError("Candidate source context is not valid JSON") from exc
    if not isinstance(source_context, dict):
        raise HpcWorkflowError("Candidate source context must be a JSON object")
    _resolve_context_columns(list(source_context), list(source_context))
    started = time.monotonic()
    kwargs: dict[str, Any] = {}
    if reasoning_mode:
        kwargs["reasoning_mode"] = reasoning_mode
    prompt_locality = str(locality)
    if source_context:
        prompt_locality += "\nAdditional source context (JSON):\n" + context_json
    prediction = processor.process_locality(prompt_locality, str(country or ""), **kwargs)
    latency = time.monotonic() - started
    usable = _usable_prediction(prediction)
    stored_prediction = dict(prediction)
    stored_prediction["source_context"] = source_context
    if evidence != "full":
        stored_prediction.pop("reasoning_content", None)
    if evidence == "minimal":
        stored_prediction.pop("response_content", None)
    return {
        "candidate_id": str(candidate_id),
        "locality": str(locality),
        "input_country": str(country or ""),
        "context_json": context_json,
        "occurrence_count": int(occurrence_count),
        "example_identifier": str(example_identifier or ""),
        "completed_at": _utc_now(),
        "api_success": bool(prediction.get("api_success")),
        "usable": usable,
        **{field: prediction.get(field) for field in OUTPUT_FIELDS},
        "error": str(prediction.get("error") or ""),
        "latency_seconds": latency,
        "input_tokens": prediction.get("input_tokens"),
        "output_tokens": prediction.get("output_tokens"),
        "total_tokens": prediction.get("total_tokens"),
        "finish_reason": prediction.get("finish_reason"),
        "stop_reason": prediction.get("stop_reason"),
        "truncated": prediction.get("truncated"),
        "reasoning_content": (
            prediction.get("reasoning_content") if evidence == "full" else None
        ),
        "response_content": (
            prediction.get("response_content") if evidence != "minimal" else None
        ),
        "prediction_json": json.dumps(
            stored_prediction, ensure_ascii=False, sort_keys=True
        ),
    }


def _diagnostic_text(value: Any) -> str:
    """Make failure diagnostics printable even when the failed input was not."""
    return str(value).encode("utf-8", errors="backslashreplace").decode("utf-8")


def _prepare_result_for_persistence(result: Mapping[str, Any]) -> dict[str, Any]:
    """Quarantine invalid Unicode without losing or silently repairing evidence.

    JSON can decode lone surrogates that SQLite cannot encode as UTF-8. Keep
    their original representation inside an escaped JSON *string*, so the
    surrounding evidence remains valid even for stricter readers like DuckDB.
    """
    invalid_fields: list[str] = []

    def safe_text(value: Any, path: str) -> Any:
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                invalid_fields.append(path)
                return "[quarantined invalid Unicode] " + json.dumps(value, ensure_ascii=True)
        elif isinstance(value, dict):
            return {
                safe_text(key, path + ".<key>"): safe_text(item, path + "[" + ascii(key) + "]")
                for key, item in value.items()
            }
        elif isinstance(value, (list, tuple)):
            return [safe_text(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    original = dict(result)
    normalized = safe_text(original, "result")
    prediction_json = original.get("prediction_json", "{}")
    prediction = json.loads(prediction_json)
    safe_prediction = safe_text(prediction, "prediction")
    safe_context = None
    if "context_json" in original:
        safe_context = safe_text(json.loads(original["context_json"]), "source_context")
    if not invalid_fields:
        return original
    # Candidate IDs are checkpoint keys: substituting one would break resume.
    if normalized.get("candidate_id") != original.get("candidate_id"):
        raise HpcWorkflowError("Candidate ID contains invalid Unicode; identity cannot be quarantined")
    error = "Invalid Unicode quarantined in " + ", ".join(sorted(set(invalid_fields)))
    normalized["usable"] = False
    normalized["error"] = error
    if safe_context is not None:
        normalized["context_json"] = json.dumps(safe_context, ensure_ascii=True, sort_keys=True)
    for field in OUTPUT_FIELDS:
        normalized[field] = "" if field in STRING_OUTPUT_FIELDS else None
    if not isinstance(safe_prediction, dict):
        safe_prediction = {"original_prediction": safe_prediction}
    safe_prediction.update({
        "success": False,
        "error": error,
        "persistence_quarantine": {
            "kind": "invalid_unicode",
            "invalid_fields": sorted(set(invalid_fields)),
            "original_result_json": json.dumps(original, ensure_ascii=True, sort_keys=True),
        },
    })
    normalized["prediction_json"] = json.dumps(safe_prediction, ensure_ascii=True, sort_keys=True)
    return normalized


def _insert_result(connection: sqlite3.Connection, result: Mapping[str, Any]) -> dict[str, Any]:
    result = _prepare_result_for_persistence(result)
    fields = (
        "candidate_id",
        "locality",
        "input_country",
        "occurrence_count",
        "example_identifier",
        "completed_at",
        "api_success",
        "usable",
        *OUTPUT_FIELDS,
        "error",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "finish_reason",
        "stop_reason",
        "truncated",
        "reasoning_content",
        "response_content",
        "prediction_json",
        "context_json",
    )
    placeholders = ", ".join("?" for _ in fields)
    # The result and its metrics-trigger update commit or roll back together.
    with connection:
        connection.execute(
            f"INSERT INTO results ({', '.join(fields)}) VALUES ({placeholders})",
            [result.get(field, "{}" if field == "context_json" else None) for field in fields],
        )
    return result


def _existing_ids(
    connection: sqlite3.Connection, candidates: Sequence[Sequence[Any]]
) -> set[str]:
    identifiers = [str(row[0]) for row in candidates]
    if not identifiers:
        return set()
    placeholders = ",".join("?" for _ in identifiers)
    return {
        str(row[0])
        for row in connection.execute(
            f"SELECT candidate_id FROM results WHERE candidate_id IN ({placeholders})",
            identifiers,
        )
    }


def _result_metrics(connection: sqlite3.Connection) -> dict[str, Any]:
    try:
        row = connection.execute("""
            SELECT completed_candidates, api_success_candidates, usable_candidates,
                   represented_occurrences, usable_represented_occurrences,
                   summed_request_seconds
            FROM result_metrics WHERE singleton = 1
            """).fetchone()
        if row is None:
            raise HpcWorkflowError("The durable result counter row is missing")
    except sqlite3.OperationalError as exc:
        if "no such table: result_metrics" not in str(exc):
            raise
        # Read-only status/export support for older checkpoints. A runner
        # upgrades these in _init_results_database before doing any inference.
        row = connection.execute("""
            SELECT count(*), coalesce(sum(api_success), 0), coalesce(sum(usable), 0),
                   coalesce(sum(occurrence_count), 0),
                   coalesce(sum(CASE WHEN usable = 1 THEN occurrence_count ELSE 0 END), 0),
                   coalesce(sum(latency_seconds), 0)
            FROM results
            """).fetchone()
    completed, api_success, usable, occurrences, usable_occurrences, latency = row
    return {
        "completed_candidates": int(completed),
        "api_success_candidates": int(api_success),
        "usable_candidates": int(usable),
        "represented_occurrences": int(occurrences),
        "usable_represented_occurrences": int(usable_occurrences),
        "summed_request_seconds": float(latency),
    }


def _write_run_progress(
    run_root: Path,
    identity: Mapping[str, Any],
    connection: sqlite3.Connection,
    total: int,
    started_monotonic: float,
    status: str,
    *,
    metrics_snapshot: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    metrics = dict(metrics_snapshot) if metrics_snapshot is not None else _result_metrics(connection)
    elapsed = max(time.monotonic() - started_monotonic, 0.000001)
    session_completed = max(
        0, metrics["completed_candidates"] - int(identity["resume_base_completed"])
    )
    metrics.update(
        {
            "selected_candidates": total,
            "remaining_candidates": max(0, total - metrics["completed_candidates"]),
            "session_elapsed_seconds": elapsed,
            "session_candidates_per_hour": session_completed / elapsed * 3600,
        }
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "updated_at": _utc_now(),
        "identity": {
            key: value
            for key, value in identity.items()
            if key != "resume_base_completed"
        },
        "metrics": metrics,
    }
    filename = "summary.json" if status == "complete" else "progress.json"
    _atomic_json(run_root / filename, payload)
    return payload


def run_command(args: argparse.Namespace) -> int:
    candidates_path = args.candidates.expanduser().resolve()
    run_root = args.output_dir.expanduser().resolve()
    if not candidates_path.is_file():
        raise HpcWorkflowError(f"Candidates file not found: {candidates_path}")
    run_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_root / ".run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise HpcWorkflowError(f"Another runner owns {run_root}") from exc

    try:
        results_path = run_root / "results.sqlite3"
        run_json = run_root / "run.json"
        if results_path.exists() and not run_json.exists():
            # Never adopt unknown predictions by manufacturing a new identity.
            # Check before migrations (and before any endpoint request), so
            # evidence from a foreign/orphaned checkpoint remains untouched.
            checkpoint = sqlite3.connect(results_path.as_uri() + "?mode=ro", uri=True)
            try:
                known_table = checkpoint.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='results'"
                ).fetchone()
                if not known_table or checkpoint.execute("SELECT 1 FROM results LIMIT 1").fetchone():
                    raise HpcWorkflowError(
                        "Checkpoint has no run.json identity; refusing to adopt unowned results"
                    )
            finally:
                checkpoint.close()
        failed_marker = run_root / "FAILED.json"
        if failed_marker.exists() and not args.resume_after_audit:
            raise HpcWorkflowError(
                f"Run is marked failed: {failed_marker}. Audit it, then use "
                "--resume-after-audit only if the endpoint and identity are safe."
            )
        profile = _prepare_profile(args)
        server = _probe_server(profile)
        prompt_path = Path(str(resources.files("placebot.data").joinpath("prompt.md")))
        duckdb = _duckdb_module()
        candidate_connection = duckdb.connect()
        query = _candidate_query(
            candidates_path,
            args.shard_index,
            args.shard_count,
            args.offset,
            args.limit,
        )
        total = int(
            candidate_connection.execute(
                f"SELECT count(*) FROM ({query}) selected"
            ).fetchone()[0]
        )
        if total == 0:
            raise HpcWorkflowError("The requested candidate selection is empty")
        if results_path.exists() and not args.resume and not args.resume_after_audit:
            raise HpcWorkflowError(
                f"Checkpoint already exists: {results_path}. Pass --resume."
            )
        results = _init_results_database(results_path)
        existing_metrics = _result_metrics(results)
        try:
            package_version = metadata.version("placebot")
        except metadata.PackageNotFoundError:
            package_version = "source-tree"
        candidate_sha256 = _sha256(candidates_path)
        preparation = _preparation_identity(candidates_path, candidate_sha256)
        identity = {
            "schema_version": SCHEMA_VERSION,
            "candidates": {
                "path": str(candidates_path),
                "sha256": candidate_sha256,
            },
            "selection": {
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "offset": args.offset,
                "limit": args.limit,
                "selected_candidates": total,
            },
            "model_profile": args.model_profile,
            "model": _profile_identity(profile, args.reasoning_mode),
            "server": _stable_server_identity(server),
            "reasoning_mode": args.reasoning_mode,
            "evidence": args.evidence,
            "prompt_sha256": _sha256(prompt_path),
            "placebot_version": package_version,
        }
        recorded = _read_json(run_json) if run_json.exists() else None
        if recorded is None or "implementation" in (recorded.get("identity") or {}):
            identity["implementation"] = _implementation_identity()
        # Preserve old context-free run identities while sealing the policy
        # for new runs and all context-bearing preparations.
        context_columns = (preparation or {}).get("columns", {}).get("context", [])
        if preparation and (
            recorded is None or context_columns
            or (
                isinstance(recorded.get("identity"), Mapping)
                and "preparation" in recorded["identity"]
            )
        ):
            identity["preparation"] = preparation
        if run_json.exists():
            if not isinstance(recorded.get("identity"), Mapping) or (
                _stable_run_identity(recorded["identity"]) != identity
            ):
                raise HpcWorkflowError(
                    "Cannot resume because candidate, model, prompt, server, implementation, or run settings changed"
                )
        else:
            _atomic_json(
                run_json,
                {"created_at": _utc_now(), "identity": identity},
            )
        _record_server_probe(run_root, server)

        stop_event = threading.Event()

        def request_stop(signum: int, _frame: Any) -> None:
            print(
                f"Received signal {signum}; stopping after durable in-flight checkpoints"
            )
            stop_event.set()

        previous_handlers = {
            signum: signal.signal(signum, request_stop)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        prior_markers = {
            marker.name: _read_json(marker)
            for marker in (run_root / "INTERRUPTED.json", failed_marker)
            if marker.exists()
        }
        if prior_markers:
            with (run_root / "resume_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "at": _utc_now(),
                    "resume_after_audit": args.resume_after_audit,
                    "prior_markers": prior_markers,
                }, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        for marker in (run_root / "INTERRUPTED.json", failed_marker):
            if marker.exists():
                marker.unlink()
        started = time.monotonic()
        progress_identity = {
            **identity,
            "resume_base_completed": existing_metrics["completed_candidates"],
        }
        consecutive_failures = 0
        fatal_reason: Optional[str] = None
        persistence_failures: list[dict[str, str]] = []
        last_known_metrics = existing_metrics
        cursor = candidate_connection.execute(query)
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures: dict[Future[dict[str, Any]], Sequence[Any]] = {}
                exhausted = False
                last_progress = time.monotonic()

                def refill() -> None:
                    nonlocal exhausted
                    while (
                        not exhausted
                        and not stop_event.is_set()
                        and len(futures) < args.workers
                    ):
                        chunk = cursor.fetchmany(args.workers - len(futures))
                        if not chunk:
                            exhausted = True
                            return
                        seen = _existing_ids(results, chunk)
                        for candidate in chunk:
                            if str(candidate[0]) not in seen:
                                future = executor.submit(
                                    _process_candidate, candidate, profile,
                                    args.reasoning_mode, args.evidence,
                                )
                                futures[future] = candidate

                refill()
                while futures:
                    completed, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    for future in completed:
                        candidate = futures.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            fatal_reason = fatal_reason or (
                                f"Worker failed for {_diagnostic_text(candidate[0])}: "
                                f"{type(exc).__name__}: {_diagnostic_text(exc)}"
                            )
                            stop_event.set()
                            continue
                        try:
                            result = _insert_result(results, result)
                            metrics = _result_metrics(results)
                            last_known_metrics = metrics
                        except Exception as exc:
                            # The transaction rolls back on insertion failure.
                            # Stop new requests, but still attempt to save the
                            # bounded in-flight set. Storage errors are terminal,
                            # never relabelled as unusable model predictions.
                            detail = {
                                "candidate_id": _diagnostic_text(candidate[0]),
                                "error": f"{type(exc).__name__}: {_diagnostic_text(exc)}",
                            }
                            persistence_failures.append(detail)
                            fatal_reason = fatal_reason or (
                                f"Persistence failed for {detail['candidate_id']}: {detail['error']}"
                            )
                            stop_event.set()
                            continue
                        if result["usable"]:
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                        if not args.quiet or not result["usable"]:
                            print(
                                f"[{metrics['completed_candidates']}/{total}] "
                                f"{result['candidate_id'][:12]} usable={result['usable']} "
                                f"{result['latency_seconds']:.2f}s",
                                flush=True,
                            )
                        if not fatal_reason and metrics["completed_candidates"] % args.checkpoint_every == 0:
                            _write_run_progress(
                                run_root,
                                progress_identity,
                                results,
                                total,
                                started,
                                "running",
                            )
                            last_progress = time.monotonic()
                        if consecutive_failures >= args.max_consecutive_failures:
                            fatal_reason = fatal_reason or f"{consecutive_failures} consecutive unusable model outputs"
                            stop_event.set()
                    if not fatal_reason and time.monotonic() - last_progress >= 30:
                        _write_run_progress(
                            run_root, progress_identity, results, total, started, "running"
                        )
                        last_progress = time.monotonic()
                    # Refill only after handling all completions already known
                    # to us, so a failure threshold cannot enqueue extra work.
                    # A stop drains the bounded in-flight set durably.
                    refill()
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

        checkpoint_metrics_verified = True
        try:
            metrics = _result_metrics(results)
        except (sqlite3.Error, HpcWorkflowError) as exc:
            metrics = last_known_metrics
            checkpoint_metrics_verified = False
            fatal_reason = fatal_reason or f"Checkpoint metrics failed: {type(exc).__name__}: {_diagnostic_text(exc)}"
        if fatal_reason:
            progress = _write_run_progress(
                run_root, progress_identity, results, total, started, "failed",
                metrics_snapshot=metrics,
            )
            _atomic_json(
                failed_marker,
                {
                    "at": _utc_now(),
                    "safe_to_resume": False,
                    "reason": fatal_reason,
                    "metrics": progress["metrics"],
                    "checkpoint_metrics_verified": checkpoint_metrics_verified,
                    "persistence_failures": persistence_failures,
                },
            )
            return 2
        if stop_event.is_set() and metrics["completed_candidates"] < total:
            progress = _write_run_progress(
                run_root, progress_identity, results, total, started, "interrupted"
            )
            _atomic_json(
                run_root / "INTERRUPTED.json",
                {
                    "at": _utc_now(),
                    "safe_to_resume": True,
                    "reason": "Scheduler/user signal after durable SQLite checkpoints",
                    "metrics": progress["metrics"],
                },
            )
            return 130
        summary = _write_run_progress(
            run_root, progress_identity, results, total, started, "complete"
        )
        complete = {
            "at": _utc_now(),
            "execution_complete": True,
            "quality_pass": summary["metrics"]["usable_candidates"] == total,
            "metrics": summary["metrics"],
        }
        _atomic_json(run_root / "COMPLETE.json", complete)
        qualification = (
            "QUALIFICATION_PASSED.json"
            if complete["quality_pass"]
            else "QUALIFICATION_FAILED.json"
        )
        _atomic_json(run_root / qualification, complete)
        return 0 if complete["quality_pass"] else 3
    finally:
        try:
            results.close()
        except (NameError, sqlite3.Error):
            pass
        try:
            candidate_connection.close()
        except NameError:
            pass
        lock_handle.close()


def export_command(args: argparse.Namespace) -> int:
    run_root = args.run_dir.expanduser().resolve()
    database = run_root / "results.sqlite3"
    output = (
        args.output.expanduser().resolve()
        if args.output
        else run_root / "results.tsv.gz"
    )
    if not database.is_file():
        raise HpcWorkflowError(f"Results database not found: {database}")
    if output == database or output in {
        run_root / name for name in (
                "run.json", "progress.json", "summary.json", "COMPLETE.json",
                "FAILED.json", "INTERRUPTED.json", "server_probes.jsonl",
                "resume_events.jsonl", "QUALIFICATION_PASSED.json",
                "QUALIFICATION_FAILED.json", "results.sqlite3-wal",
                "results.sqlite3-shm", ".run.lock", "continuation.json",
        )
    }:
        raise HpcWorkflowError("Export must not overwrite checkpoint or run evidence")
    if output.exists() and not args.replace:
        raise HpcWorkflowError(f"Output exists: {output}; pass --replace")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise HpcWorkflowError(f"Stale partial export exists: {temporary}; audit it first")
    connection = sqlite3.connect(database)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(results)")}
        projection = [
            "'{}' AS context_json" if field == "context_json" and field not in columns else field
            for field in RESULT_EXPORT_FIELDS
        ]
        cursor = connection.execute(
            f"SELECT {', '.join(projection)} FROM results ORDER BY candidate_id"
        )
        opener = gzip.open if output.suffix.lower() == ".gz" else open
        with opener(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(RESULT_EXPORT_FIELDS)
            exported_rows = 0
            for row in cursor:
                writer.writerow(["" if value is None else value for value in row])
                exported_rows += 1
    finally:
        connection.close()
    temporary.replace(output)
    run_manifest = run_root / "run.json"
    run_data = _read_json(run_manifest) if run_manifest.is_file() else {}
    _atomic_json(output.with_suffix(output.suffix + ".manifest.json"), {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "results_sha256": _sha256(output),
        "result_rows": exported_rows,
        "run_identity": run_data.get("identity"),
        **({"continuation": run_data["continuation"]} if "continuation" in run_data else {}),
    })
    print(f"Exported results sidecar: {output}")
    return 0


def merge_command(args: argparse.Namespace) -> int:
    shard_roots = [path.expanduser().resolve() for path in args.run_dirs]
    output_root = args.output_dir.expanduser().resolve()
    if len(shard_roots) < 2:
        raise HpcWorkflowError("Merge requires at least two shard run directories")
    if output_root.exists() and any(output_root.iterdir()):
        raise HpcWorkflowError(f"Merge output is not empty: {output_root}")

    recorded: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for root in shard_roots:
        run_path = root / "run.json"
        complete_path = root / "COMPLETE.json"
        database_path = root / "results.sqlite3"
        if (
            not run_path.is_file()
            or not complete_path.is_file()
            or not database_path.is_file()
        ):
            raise HpcWorkflowError(f"Shard is not execution-complete: {root}")
        run_data = _read_json(run_path)
        identity = run_data.get("identity")
        if not isinstance(identity, dict):
            raise HpcWorkflowError(f"Shard has invalid run identity: {root}")
        selection = identity.get("selection")
        if not isinstance(selection, dict):
            raise HpcWorkflowError(f"Shard has invalid selection identity: {root}")
        recorded.append((root, identity, selection))

    shard_counts = {int(item[2]["shard_count"]) for item in recorded}
    if len(shard_counts) != 1:
        raise HpcWorkflowError("Shard-count identity differs between inputs")
    shard_count = shard_counts.pop()
    indices = [int(item[2]["shard_index"]) for item in recorded]
    if len(set(indices)) != len(indices):
        raise HpcWorkflowError("Duplicate shard index in merge inputs")
    if set(indices) != set(range(shard_count)):
        raise HpcWorkflowError(
            f"Merge needs every shard index 0..{shard_count - 1}; got {sorted(indices)}"
        )

    def shared_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in _stable_run_identity(identity).items()
            if key not in {"selection"}
        }

    expected = shared_identity(recorded[0][1])
    if any(shared_identity(identity) != expected for _, identity, _ in recorded[1:]):
        raise HpcWorkflowError(
            "Model, prompt, candidate, server or evidence identity differs"
        )
    if any(
        selection.get("offset") != 0 or selection.get("limit") is not None
        for _, _, selection in recorded
    ):
        raise HpcWorkflowError("Cannot merge partial offset/limit shard selections")

    output_root.mkdir(parents=True, exist_ok=True)
    merged_database = output_root / "results.sqlite3"
    merged = _init_results_database(merged_database)
    try:
        for position, (root, _, _) in enumerate(
            sorted(recorded, key=lambda item: int(item[2]["shard_index"]))
        ):
            alias = f"shard_{position}"
            merged.execute(
                f"ATTACH DATABASE ? AS {alias}", (str(root / "results.sqlite3"),)
            )
            try:
                destination_columns = [row[1] for row in merged.execute("PRAGMA table_info(results)")]
                source_columns = {row[1] for row in merged.execute(f"PRAGMA {alias}.table_info(results)")}
                projection = [
                    "'{}'" if field == "context_json" and field not in source_columns else field
                    for field in destination_columns
                ]
                merged.execute(
                    f"INSERT INTO results ({', '.join(destination_columns)}) "
                    f"SELECT {', '.join(projection)} FROM {alias}.results"
                )
                merged.commit()
            finally:
                merged.execute(f"DETACH DATABASE {alias}")
        metrics = _result_metrics(merged)
    finally:
        merged.close()

    expected_total = sum(
        int(selection["selected_candidates"]) for _, _, selection in recorded
    )
    if metrics["completed_candidates"] != expected_total:
        raise HpcWorkflowError(
            "Merged checkpoint count does not equal the complete shard selections"
        )
    quality_pass = metrics["usable_candidates"] == expected_total
    merged_identity = {
        **expected,
        "selection": {
            "shard_index": "merged",
            "shard_count": shard_count,
            "offset": 0,
            "limit": None,
            "selected_candidates": expected_total,
        },
    }
    _atomic_json(
        output_root / "run.json",
        {
            "created_at": _utc_now(),
            "identity": merged_identity,
            "merged_from": [str(root) for root, _, _ in recorded],
        },
    )
    complete = {
        "at": _utc_now(),
        "execution_complete": True,
        "quality_pass": quality_pass,
        "metrics": {**metrics, "selected_candidates": expected_total},
    }
    _atomic_json(output_root / "summary.json", {"status": "complete", **complete})
    _atomic_json(output_root / "COMPLETE.json", complete)
    qualification = (
        "QUALIFICATION_PASSED.json" if quality_pass else "QUALIFICATION_FAILED.json"
    )
    _atomic_json(output_root / qualification, complete)
    print(f"Merged {shard_count} complete shards into {merged_database}")
    return 0 if quality_pass else 3


def reconstitute_command(args: argparse.Namespace) -> int:
    source = args.input.expanduser().resolve()
    results = args.results.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file() or not results.is_file():
        raise HpcWorkflowError("Input and results sidecar must both be regular files")
    sidecar_manifest_path = results.with_suffix(results.suffix + ".manifest.json")
    preparation: dict[str, Any] = {}
    if sidecar_manifest_path.is_file():
        sidecar_manifest = _read_json(sidecar_manifest_path)
        if sidecar_manifest.get("results_sha256") != _sha256(results):
            raise HpcWorkflowError("Results sidecar does not match its export manifest")
        run_identity = sidecar_manifest.get("run_identity") or {}
        preparation = run_identity.get("preparation") or {}
        if preparation.get("source_sha256") and preparation["source_sha256"] != _sha256(source):
            raise HpcWorkflowError("Source does not match the frozen preparation manifest")
        for option in ("include_existing", "treat_zero_zero_missing"):
            expected = preparation.get("policy", {}).get(option)
            if expected is not None and expected != getattr(args, option):
                raise HpcWorkflowError(f"Reconstitution policy differs from preparation: {option}")
    if output == source or output == results:
        raise HpcWorkflowError("Output must not overwrite an input")
    if output.exists():
        raise HpcWorkflowError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise HpcWorkflowError(
            f"Stale partial output exists: {temporary}; audit and remove it first"
        )
    input_format = _detect_format(source, args.input_format)
    relation = _source_relation(source, input_format, None)
    duckdb = _duckdb_module()
    connection = duckdb.connect()
    try:
        _configure_duckdb(connection, args, output.parent)
        schema = _schema_columns(connection, relation)
        columns = {
            "locality": _resolve_column(
                schema, "locality", args.locality_column, required=True
            ),
            "country": _resolve_column(
                schema, "country", args.country_column, required=False
            ),
            "latitude": _resolve_column(
                schema, "latitude", args.latitude_column, required=False
            ),
            "longitude": _resolve_column(
                schema, "longitude", args.longitude_column, required=False
            ),
            "identifier": None,
            "context": _resolve_context_columns(
                schema,
                args.context_column or preparation.get("columns", {}).get("context", []),
            ),
        }
        if preparation:
            for name in ("locality", "country", "latitude", "longitude", "context"):
                expected = preparation.get("columns", {}).get(
                    name, [] if name == "context" else None
                )
                if columns[name] != expected:
                    raise HpcWorkflowError(f"Reconstitution column differs from preparation: {name}")
        expressions = _selection_sql(
            columns,
            include_existing=args.include_existing,
            treat_zero_zero_missing=args.treat_zero_zero_missing,
            table_alias="source",
        )
        result_relation = (
            f"read_csv({_sql_literal(results)}, header=true, delim='\\t', "
            "all_varchar=true, ignore_errors=false)"
        )
        result_columns = _schema_columns(connection, result_relation)
        available_result_columns = {name.casefold() for name in result_columns}
        required_result_columns = {"candidate_id", "usable", "latitude", "longitude"}
        if not required_result_columns.issubset(available_result_columns):
            missing = sorted(required_result_columns - available_result_columns)
            raise HpcWorkflowError(f"Results sidecar is missing required fields: {missing}")
        latitude = "try_cast(latitude AS DOUBLE)"
        longitude = "try_cast(longitude AS DOUBLE)"
        valid_prediction_parts = [
            f"isfinite({latitude}) AND {latitude} BETWEEN -90 AND 90",
            f"isfinite({longitude}) AND {longitude} BETWEEN -180 AND 180",
        ]
        for field in ("coordinate_radius_meters", "elevation_meters"):
            if field in available_result_columns:
                number = f"try_cast({field} AS DOUBLE)"
                constraint = f"isfinite({number})"
                if field == "coordinate_radius_meters":
                    constraint += f" AND {number} >= 0"
                valid_prediction_parts.append(
                    f"(nullif(trim({field}), '') IS NULL OR ({constraint}))"
                )
        invalid_usable = connection.execute(f"""
            SELECT count(*) FROM {result_relation}
            WHERE try_cast(usable AS BOOLEAN) IS TRUE
              AND NOT coalesce(({' AND '.join(valid_prediction_parts)}), false)
            """).fetchone()[0]
        if invalid_usable:
            raise HpcWorkflowError(
                f"Results sidecar has {invalid_usable} usable predictions with invalid "
                "coordinates or optional numeric fields"
            )
        if not columns["context"] and "context_json" in result_columns:
            with_context = connection.execute(f"""
                SELECT count(*) FROM {result_relation}
                WHERE context_json IS NOT NULL AND trim(context_json) NOT IN ('', '{{}}')
                """).fetchone()[0]
            if with_context:
                raise HpcWorkflowError(
                    "Context-bearing results need their export manifest or --context-column options"
                )
        result_counts = connection.execute(f"""
            SELECT count(*), count(DISTINCT candidate_id),
                   count(*) FILTER (WHERE nullif(trim(candidate_id), '') IS NULL)
            FROM {result_relation}
            """).fetchone()
        if result_counts[0] != result_counts[1] or result_counts[2]:
            raise HpcWorkflowError(
                "Results sidecar must contain unique, nonblank candidate IDs"
            )
        source_count = int(connection.execute(
            f"SELECT count(*) FROM {relation}"
        ).fetchone()[0])
        candidate = (
            f"CASE WHEN {expressions['selected']} THEN {expressions['candidate_id']} "
            "ELSE NULL END"
        )
        joined = f"""
            SELECT
                source.*,
                try_cast(result.usable AS BOOLEAN) AS placebot_usable,
                result.country AS placebot_country,
                result.state AS placebot_state,
                result.region AS placebot_region,
                result.sector AS placebot_sector,
                result.exact_site AS placebot_exact_site,
                try_cast(result.latitude AS DOUBLE) AS placebot_latitude,
                try_cast(result.longitude AS DOUBLE) AS placebot_longitude,
                result.coordinate_source AS placebot_coordinate_source,
                try_cast(result.coordinate_radius_meters AS DOUBLE)
                    AS placebot_coordinate_radius_meters,
                try_cast(result.elevation_meters AS DOUBLE) AS placebot_elevation_meters,
                result.elevation_original AS placebot_elevation_original,
                result.confidence AS placebot_confidence,
                result.notes AS placebot_notes,
                result.collection_notes AS placebot_collection_notes,
                result.error AS placebot_error
            FROM {relation} source
            LEFT JOIN {result_relation} result
              ON result.candidate_id = {candidate}
        """
        suffix = output.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            copy_options = "FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000"
        elif suffix in {".tsv", ".tab"}:
            copy_options = "FORMAT CSV, HEADER true, DELIMITER '\\t'"
        elif suffix == ".csv":
            copy_options = "FORMAT CSV, HEADER true, DELIMITER ','"
        else:
            raise HpcWorkflowError("Output must end in .parquet, .csv, .tsv or .tab")
        connection.execute(
            f"COPY ({joined}) TO {_sql_literal(temporary)} ({copy_options})"
        )
        output_relation = _source_relation(
            temporary,
            (
                "parquet"
                if suffix in {".parquet", ".pq"}
                else ("tsv" if suffix in {".tsv", ".tab"} else "csv")
            ),
            None,
        )
        counts = connection.execute(
            f"SELECT count(*), count(*) FILTER (WHERE placebot_usable = true) "
            f"FROM {output_relation}"
        ).fetchone()
        if int(counts[0]) != source_count:
            raise HpcWorkflowError(
                f"Reconstitution changed row count: {source_count} to {counts[0]}"
            )
    finally:
        connection.close()
    temporary.replace(output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source": str(source),
        "results": str(results),
        "output": str(output),
        "input_records": source_count,
        "output_records": int(counts[0]),
        "records_with_usable_placebot_result": int(counts[1]),
        "policy": {
            "include_existing": args.include_existing,
            "treat_zero_zero_missing": args.treat_zero_zero_missing,
            "context_columns": columns["context"],
        },
    }
    _atomic_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def status_command(args: argparse.Namespace) -> int:
    run_root = args.run_dir.expanduser().resolve()
    for name in (
        "COMPLETE.json",
        "QUALIFICATION_PASSED.json",
        "QUALIFICATION_FAILED.json",
        "FAILED.json",
        "INTERRUPTED.json",
        "summary.json",
        "progress.json",
        "run.json",
    ):
        path = run_root / name
        if path.is_file():
            print(f"[{name}]")
            print(path.read_text(encoding="utf-8").rstrip())
            if name in {"COMPLETE.json", "FAILED.json", "INTERRUPTED.json"}:
                break
    database = run_root / "results.sqlite3"
    if database.is_file():
        connection = sqlite3.connect(database)
        try:
            print("[checkpoint]")
            print(json.dumps(_result_metrics(connection), indent=2))
        finally:
            connection.close()
    return 0


def _add_duckdb_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--duckdb-threads", type=int, default=max(1, os.cpu_count() or 1)
    )
    parser.add_argument(
        "--memory-limit",
        help="DuckDB memory limit such as 32GB; unset lets DuckDB choose",
    )


def _add_source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--context-column", action="append", default=[],
        help="Explicit source geographic context column; repeat for multiple fields",
    )
    parser.add_argument(
        "--input-format", choices=("auto", "parquet", "csv", "tsv"), default="auto"
    )
    parser.add_argument("--locality-column")
    parser.add_argument("--country-column")
    parser.add_argument("--latitude-column")
    parser.add_argument("--longitude-column")
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Process locality rows even when valid coordinates already exist",
    )
    parser.add_argument(
        "--treat-zero-zero-missing",
        action="store_true",
        help="Treat the syntactically valid 0,0 pair as unresolved",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="placebot-hpc",
        description="Disk-backed PlaceBot workflow for multi-million-row datasets",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build deduplicated candidates")
    prepare.add_argument("input", type=Path)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--identifier-column")
    prepare.add_argument("--row-limit", type=int, help="Bounded smoke-test scan")
    _add_source_options(prepare)
    _add_duckdb_options(prepare)
    prepare.set_defaults(func=prepare_command)

    run = subparsers.add_parser("run", help="Checkpointed model inference")
    run.add_argument("candidates", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--model-profile", default="nemotron_3_super_local")
    run.add_argument("--endpoint", help="Base URL or chat-completions URL")
    run.add_argument("--model-id")
    run.add_argument("--max-output-tokens", type=int)
    run.add_argument("--request-timeout", type=int)
    run.add_argument("--max-retries", type=int, choices=range(0, 11))
    run.add_argument("--reasoning-mode", choices=("fast", "full"))
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--checkpoint-every", type=int, default=1)
    run.add_argument("--quiet", action="store_true", help="Log errors, not every successful request; progress.json remains live")
    run.add_argument("--max-consecutive-failures", type=int, default=3)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--offset", type=int, default=0)
    run.add_argument("--limit", type=int)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--resume-after-audit", action="store_true")
    run.add_argument(
        "--evidence",
        choices=("minimal", "final", "full"),
        default="final",
        help="Retained response evidence; full reasoning can require enormous storage",
    )
    run.set_defaults(func=run_command)

    export = subparsers.add_parser("export", help="Export SQLite results sidecar")
    export.add_argument("run_dir", type=Path)
    export.add_argument("--output", type=Path)
    export.add_argument("--replace", action="store_true")
    export.set_defaults(func=export_command)

    merge = subparsers.add_parser("merge", help="Merge all complete shard checkpoints")
    merge.add_argument("run_dirs", type=Path, nargs="+")
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.set_defaults(func=merge_command)

    reconstitute = subparsers.add_parser(
        "reconstitute", help="Join PlaceBot results back to every source row"
    )
    reconstitute.add_argument("input", type=Path)
    reconstitute.add_argument("--results", type=Path, required=True)
    reconstitute.add_argument("--output", type=Path, required=True)
    _add_source_options(reconstitute)
    _add_duckdb_options(reconstitute)
    reconstitute.set_defaults(func=reconstitute_command)

    status = subparsers.add_parser("status", help="Show structured run state")
    status.add_argument("run_dir", type=Path)
    status.set_defaults(func=status_command)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in (
        "duckdb_threads",
        "workers",
        "checkpoint_every",
        "max_consecutive_failures",
        "shard_count",
    ):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "shard_index") and not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, shard-count)")
    if hasattr(args, "offset") and args.offset < 0:
        parser.error("--offset cannot be negative")
    for name in ("limit", "row_limit", "max_output_tokens", "request_timeout"):
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None and value <= 0:
                parser.error(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "resume", False) and getattr(args, "resume_after_audit", False):
        parser.error("Use either --resume or --resume-after-audit, not both")


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    try:
        return int(args.func(args))
    except (HpcWorkflowError, OSError, sqlite3.Error, requests.RequestException) as exc:
        print(f"placebot-hpc: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
