#!/usr/bin/env python3
"""Extract a DiSSCo UK-aligned occurrence snapshot from public GBIF Parquet.

The compact public GBIF snapshot does not include the interpreted
``institutionKey`` and ``collectionKey`` fields used by the live portal. This
tool therefore records and applies a reproducible proxy: a current GB-based
publishing organization, one of the portal's four specimen basis types, and a
non-empty institution or collection code. It never labels the result as an
exact citable GBIF occurrence download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_SNAPSHOT = "2026-07-01"
DEFAULT_OUTPUT_ROOT = Path("data/private/dissco-uk")
DEFAULT_MEMORY_LIMIT = "48GB"
GBIF_ORGANIZATIONS_URL = (
    "https://api.gbif.org/v1/organization?country=GB&limit=1000"
)
GBIF_SNAPSHOT_ROOT = (
    "s3://gbif-open-data-eu-central-1/occurrence/{snapshot}/"
    "occurrence.parquet/*"
)
NHM_DATASET_KEY = "7e380070-f762-11e1-a439-00145eb45e9a"
BASIS_TYPES = (
    "PRESERVED_SPECIMEN",
    "FOSSIL_SPECIMEN",
    "MATERIAL_SAMPLE",
    "LIVING_SPECIMEN",
)
LIVE_PORTAL_PREDICATE = (
    "AND(OR(isNotNull institutionKey,isNotNull collectionKey),"
    "basisOfRecord IN PRESERVED_SPECIMEN,FOSSIL_SPECIMEN,MATERIAL_SAMPLE,"
    "LIVING_SPECIMEN,publishingCountry=GB)"
)


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON without exposing a half-written state or manifest file."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Hash a local file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_organization_keys(payload: dict[str, Any]) -> list[str]:
    """Return sorted, active GB organization UUIDs from a registry response."""
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("GBIF organization response has no results list")
    keys = sorted(
        {
            str(item["key"])
            for item in results
            if isinstance(item, dict) and item.get("key") and not item.get("deleted")
        }
    )
    if not keys:
        raise ValueError("GBIF organization response contains no active keys")
    return keys


def proxy_predicate_sql() -> str:
    """Return the snapshot-compatible approximation of the portal predicate."""
    basis_sql = ", ".join(f"'{value}'" for value in BASIS_TYPES)
    return (
        "publishingorgkey IN (SELECT key FROM uk_organization_keys)\n"
        f"  AND basisofrecord IN ({basis_sql})\n"
        "  AND (nullif(trim(institutioncode), '') IS NOT NULL\n"
        "       OR nullif(trim(collectioncode), '') IS NOT NULL)"
    )


def build_select_sql(snapshot_path: str) -> str:
    """Build the filtered snapshot query."""
    escaped_path = snapshot_path.replace("'", "''")
    return (
        "SELECT *\n"
        f"FROM read_parquet('{escaped_path}')\n"
        f"WHERE {proxy_predicate_sql()}"
    )


def fetch_organizations(destination: Path) -> tuple[dict[str, Any], list[str]]:
    """Fetch and retain the GBIF registry response that defines the proxy."""
    request = urllib.request.Request(
        GBIF_ORGANIZATIONS_URL,
        headers={"User-Agent": "PlaceBot-DiSSCo-UK-snapshot/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    keys = parse_organization_keys(payload)
    atomic_json(destination, payload)
    return payload, keys


def connect_duckdb(threads: int, memory_limit: str, temp_directory: Path) -> Any:
    """Open DuckDB and configure anonymous access to the public GBIF bucket."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "duckdb is required; run this with the PlaceBot GBIF tools venv"
        ) from exc

    connection = duckdb.connect()
    connection.execute("LOAD httpfs")
    connection.execute("SET s3_region='eu-central-1'")
    connection.execute("SET s3_endpoint='s3.eu-central-1.amazonaws.com'")
    connection.execute("SET s3_url_style='path'")
    connection.execute("SET s3_access_key_id=''")
    connection.execute("SET s3_secret_access_key=''")
    connection.execute(f"SET threads={threads:d}")
    escaped_memory_limit = memory_limit.replace("'", "''")
    escaped_temp_directory = str(temp_directory).replace("'", "''")
    connection.execute(f"SET memory_limit='{escaped_memory_limit}'")
    connection.execute(f"SET temp_directory='{escaped_temp_directory}'")
    # Record order has no semantic meaning here. Disabling order preservation
    # lets a retry flush parallel scan batches instead of retaining them while
    # waiting for an earlier remote shard.
    connection.execute("SET preserve_insertion_order=false")
    return connection


def register_organization_keys(connection: Any, keys: Sequence[str]) -> None:
    """Load the exact registry key set into an in-memory DuckDB table."""
    connection.execute("CREATE TEMP TABLE uk_organization_keys(key VARCHAR PRIMARY KEY)")
    connection.executemany(
        "INSERT INTO uk_organization_keys VALUES (?)",
        [(key,) for key in keys],
    )


def write_state(path: Path, stage: str, **values: Any) -> None:
    """Write a durable status record for SSH/chat recovery."""
    state = {
        "stage": stage,
        "updated_at": utc_now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        **values,
    }
    atomic_json(path, state)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--memory-limit", default=DEFAULT_MEMORY_LIMIT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if not args.snapshot or "/" in args.snapshot or args.snapshot.startswith("."):
        raise ValueError("--snapshot must be a GBIF date directory such as 2026-07-01")

    output_dir = args.output_root.expanduser().resolve() / args.snapshot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "occurrence.parquet"
    partial_path = output_dir / "occurrence.parquet.partial"
    temp_directory = output_dir / "duckdb-tmp"
    organizations_path = output_dir / "gbif-gb-organizations.json"
    state_path = output_dir / "download-state.json"
    manifest_path = output_dir / "manifest.json"

    collisions = [path for path in (output_path, partial_path, manifest_path) if path.exists()]
    if collisions:
        joined = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"refusing to replace existing output: {joined}")

    started_at = utc_now()
    snapshot_path = GBIF_SNAPSHOT_ROOT.format(snapshot=args.snapshot)
    write_state(
        state_path,
        "fetching_organizations",
        snapshot=args.snapshot,
        snapshot_path=snapshot_path,
        output_path=str(output_path),
        partial_path=str(partial_path),
        started_at=started_at,
    )
    print(f"[{utc_now()}] fetching GBIF UK organization registry", flush=True)
    _, organization_keys = fetch_organizations(organizations_path)
    print(f"[{utc_now()}] active GB organization keys: {len(organization_keys):,}", flush=True)

    temp_directory.mkdir(exist_ok=True)
    connection = connect_duckdb(args.threads, args.memory_limit, temp_directory)
    try:
        register_organization_keys(connection, organization_keys)
        select_sql = build_select_sql(snapshot_path)
        escaped_partial = str(partial_path).replace("'", "''")
        copy_sql = (
            f"COPY ({select_sql}) TO '{escaped_partial}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        write_state(
            state_path,
            "exporting",
            snapshot=args.snapshot,
            snapshot_path=snapshot_path,
            output_path=str(output_path),
            partial_path=str(partial_path),
            started_at=started_at,
            organization_count=len(organization_keys),
            memory_limit=args.memory_limit,
            temp_directory=str(temp_directory),
        )
        print(f"[{utc_now()}] export started: {partial_path}", flush=True)
        copy_started = time.monotonic()
        connection.execute(copy_sql)
        copy_seconds = time.monotonic() - copy_started

        escaped_output = str(partial_path).replace("'", "''")
        verification = connection.execute(
            "SELECT count(*) AS records, "
            f"count(*) FILTER (WHERE datasetkey='{NHM_DATASET_KEY}') AS nhm_records "
            f"FROM read_parquet('{escaped_output}')"
        ).fetchone()
        records = int(verification[0])
        nhm_records = int(verification[1])
        if records < 1:
            raise RuntimeError("export verification found no records")
    finally:
        connection.close()

    output_sha256 = sha256_file(partial_path)
    output_bytes = partial_path.stat().st_size
    os.replace(partial_path, output_path)
    completed_at = utc_now()
    manifest = {
        "title": "DiSSCo UK-aligned GBIF occurrence snapshot extract",
        "status": "complete",
        "created_at": completed_at,
        "snapshot": args.snapshot,
        "snapshot_path": snapshot_path,
        "source": "GBIF public monthly occurrence Parquet snapshot",
        "output": {
            "path": str(output_path),
            "format": "Apache Parquet",
            "compression": "Zstandard",
            "bytes": output_bytes,
            "sha256": output_sha256,
            "records": records,
            "nhm_dataset_records": nhm_records,
        },
        "filter": {
            "live_portal_predicate": LIVE_PORTAL_PREDICATE,
            "snapshot_proxy_sql": proxy_predicate_sql(),
            "uk_organization_registry_url": GBIF_ORGANIZATIONS_URL,
            "uk_organization_registry_file": str(organizations_path),
            "active_organization_count": len(organization_keys),
        },
        "qualification": (
            "The compact snapshot lacks publishingCountry, institutionKey, and "
            "collectionKey. This extract uses current GB publishing-organization "
            "membership and non-empty institution/collection codes as proxies; "
            "it is not an exact GBIF citable download of the live portal predicate."
        ),
        "timing": {
            "started_at": started_at,
            "completed_at": completed_at,
            "export_seconds": round(copy_seconds, 3),
        },
        "tool": {
            "script": str(Path(__file__).resolve()),
            "python": sys.version,
            "hostname": socket.gethostname(),
            "duckdb_memory_limit": args.memory_limit,
            "duckdb_temp_directory": str(temp_directory),
        },
    }
    atomic_json(manifest_path, manifest)
    write_state(
        state_path,
        "complete",
        snapshot=args.snapshot,
        output_path=str(output_path),
        manifest_path=str(manifest_path),
        records=records,
        nhm_records=nhm_records,
        bytes=output_bytes,
        sha256=output_sha256,
        started_at=started_at,
        completed_at=completed_at,
    )
    print(
        f"[{completed_at}] complete: {records:,} records, "
        f"{output_bytes:,} bytes, sha256={output_sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
