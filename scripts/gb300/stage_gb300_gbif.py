#!/usr/bin/env python3
"""Resumable raw DiSSCo UK-aligned GBIF proxy extract, NOT a training corpus.

Retains every public snapshot column, including licence and provenance. Each
remote shard has a frozen inventory entry and a hashed local completion receipt.
Evaluation exclusions MUST be applied before this raw data becomes evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET

from download_dissco_uk_snapshot import (
    BASIS_TYPES, LIVE_PORTAL_PREDICATE, NHM_DATASET_KEY, atomic_json,
    build_select_sql, connect_duckdb, parse_organization_keys,
    proxy_predicate_sql, register_organization_keys, sha256_file, utc_now,
)

BUCKET = "https://gbif-open-data-eu-central-1.s3.eu-central-1.amazonaws.com/"
USER_AGENT = "PlaceBot-GB300-public-specimen-staging/1.0"
QUALIFICATION = (
    "New DiSSCo UK-aligned proxy from GBIF public monthly Parquet, not an exact "
    "DiSSCo portal export or DOI-bearing GBIF download. Uses current active GB "
    "publisher registry and institution/collection codes as proxies for the live "
    "portal predicate. RAW DATA: evaluation records have NOT been excluded. "
    "Do not use directly for retrieval, calibration, or training. Per-record "
    "licences remain applicable, including non-commercial restrictions."
)


def request(url, method="GET"):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read(), dict(response.headers.items())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def inventory(snapshot):
    prefix = f"occurrence/{snapshot}/occurrence.parquet/"
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    entries, continuation = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if continuation:
            params["continuation-token"] = continuation
        body, _ = request(BUCKET + "?" + urllib.parse.urlencode(params))
        root = ET.fromstring(body)
        for item in root.findall("s:Contents", ns):
            key = item.findtext("s:Key", namespaces=ns)
            size = int(item.findtext("s:Size", namespaces=ns))
            if not key.startswith(prefix) or not re.fullmatch(r"[0-9]+", key[len(prefix):]):
                raise ValueError(f"Unexpected source object: {key}")
            if size <= 0:
                raise ValueError("Empty source object")
            entries.append({"key": key, "bytes": size,
                            "etag": item.findtext("s:ETag", namespaces=ns),
                            "last_modified": item.findtext("s:LastModified", namespaces=ns)})
        if root.findtext("s:IsTruncated", namespaces=ns) == "false":
            break
        following = root.findtext("s:NextContinuationToken", namespaces=ns)
        if not following or following == continuation:
            raise ValueError("Invalid S3 pagination")
        continuation = following
    if not entries or len({e["key"] for e in entries}) != len(entries):
        raise ValueError("Empty/duplicate source inventory")
    return sorted(entries, key=lambda e: e["key"])


def organizations():
    pages, offset = [], 0
    while True:
        url = f"https://api.gbif.org/v1/organization?country=GB&limit=1000&offset={offset}"
        body, _ = request(url)
        page = json.loads(body)
        if not isinstance(page.get("results"), list) or int(page.get("offset", -1)) != offset:
            raise ValueError("Invalid organization pagination")
        pages.append({"url": url, "response": page})
        if page.get("endOfRecords") is True:
            break
        if not page["results"]:
            raise ValueError("Incomplete organization registry")
        offset += len(page["results"])
    keys = parse_organization_keys({"results": [r for p in pages for r in p["response"]["results"]]})
    return {"fetched_at": utc_now(), "pages": pages, "active_keys": keys}


def check_remote(entry):
    _, headers = request(BUCKET + entry["key"], method="HEAD")
    lower = {k.lower(): v for k, v in headers.items()}
    if lower.get("etag") != entry["etag"] or int(lower.get("content-length", -1)) != entry["bytes"]:
        raise ValueError(f"Source changed: {entry['key']}")


def validate_receipt(root, index, entry, identity_sha):
    receipt_path = root / "receipts" / f"{index:06d}.json"
    part = root / "parts" / f"{index:06d}.parquet"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt["identity_sha256"] != identity_sha or receipt["source"] != entry:
        raise ValueError("Part identity mismatch")
    if receipt["part"] != part.name or part.stat().st_size != receipt["bytes"]:
        raise ValueError("Missing/changed completed part")
    if sha256_file(part) != receipt["sha256"]:
        raise ValueError("Completed part checksum mismatch")
    if receipt["records"] < 0:
        raise ValueError("Invalid part row count")
    return receipt


def stage(args):
    import duckdb
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("parts", "receipts", "tmp"):
            (root / name).mkdir(exist_ok=True)
        state = {"started_at": utc_now(), "snapshot": args.snapshot, "qualification": QUALIFICATION}

        def progress(phase, **extra):
            state.update(extra)
            state.update(phase=phase, updated_at=utc_now(), pid=os.getpid())
            atomic_json(root / "progress.json", state)
            print(json.dumps(state, sort_keys=True), flush=True)

        try:
            progress("pinning_source_inventory")
            current_inventory = inventory(args.snapshot)
            registry_path = root / "organizations.json"
            if not registry_path.exists():
                atomic_json(registry_path, organizations())
            registry = json.loads(registry_path.read_text())
            citation_path = root / "citation.txt"
            citation_url = BUCKET + f"occurrence/{args.snapshot}/citation.txt"
            citation, _ = request(citation_url)
            if citation_path.exists() and citation_path.read_bytes() != citation:
                raise ValueError("Public snapshot citation changed")
            if not citation_path.exists():
                citation_path.write_bytes(citation)
            identity = {
                "schema": "placebot-gbif-raw-proxy-v1", "snapshot": args.snapshot,
                "bucket": BUCKET, "sources": current_inventory,
                "registry_sha256": sha256_file(registry_path),
                "citation_url": citation_url, "citation_sha256": sha256_file(citation_path),
                "active_organization_keys": registry["active_keys"],
                "predicate_sql": proxy_predicate_sql(), "live_portal_predicate": LIVE_PORTAL_PREDICATE,
                "qualification": QUALIFICATION, "duckdb_version": duckdb.__version__,
                "code_sha256": {p.name: sha256_file(p) for p in (
                    Path(__file__), Path(__file__).with_name("download_dissco_uk_snapshot.py"))},
            }
            identity_path = root / "identity.json"
            if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
                raise ValueError("Resume identity changed; use a fresh output root")
            atomic_json(identity_path, identity)
            identity_sha = sha256_file(identity_path)
            completed = {}
            for i, entry in enumerate(current_inventory):
                receipt = validate_receipt(root, i, entry, identity_sha)
                if receipt:
                    completed[i] = receipt
            if (root / "COMPLETE.json").exists():
                terminal = json.loads((root / "COMPLETE.json").read_text())
                if len(completed) != len(current_inventory) or terminal["manifest_sha256"] != sha256_file(root / "manifest.json"):
                    raise ValueError("Completed dataset attestation failed")
                progress("complete_verified", completed_parts=len(completed))
                return
            local = threading.local()

            def extract(index):
                entry = current_inventory[index]
                check_remote(entry)
                if shutil.disk_usage(root).free < 20 * 1024**3:
                    raise ValueError("Less than 20 GiB free; stopping safely")
                if not hasattr(local, "db"):
                    tmp = root / "tmp" / str(threading.get_ident())
                    tmp.mkdir(exist_ok=True)
                    local.db = connect_duckdb(2, "2GB", tmp)
                    register_organization_keys(local.db, registry["active_keys"])
                db = local.db
                part = root / "parts" / f"{index:06d}.parquet"
                # Orphaned outputs have no trusted receipt; retain them for audit.
                for stale in (part, part.with_suffix(".parquet.partial")):
                    if stale.exists():
                        stale.rename(stale.with_name(stale.name + ".orphan-" + uuid.uuid4().hex))
                partial = part.with_suffix(".parquet.partial")
                select = build_select_sql(BUCKET + entry["key"])
                escaped = str(partial).replace("'", "''")
                began = time.monotonic()
                db.execute(f"COPY ({select}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
                counts = db.execute(
                    f"SELECT count(*), count(*) FILTER (WHERE datasetkey='{NHM_DATASET_KEY}') "
                    f"FROM read_parquet('{escaped}')").fetchone()
                check_remote(entry)
                with partial.open("rb") as stream:
                    os.fsync(stream.fileno())
                partial.rename(part)
                receipt = {"source": entry, "identity_sha256": identity_sha,
                           "part": part.name, "bytes": part.stat().st_size,
                           "sha256": sha256_file(part), "records": counts[0], "nhm_records": counts[1],
                           "seconds": round(time.monotonic() - began, 3), "completed_at": utc_now()}
                atomic_json(root / "receipts" / f"{index:06d}.json", receipt)
                return receipt

            def report(phase):
                progress(phase, total_parts=len(current_inventory), completed_parts=len(completed),
                         records=sum(r["records"] for r in completed.values()),
                         nhm_records=sum(r["nhm_records"] for r in completed.values()),
                         output_bytes=sum(r["bytes"] for r in completed.values()),
                         source_inventory_bytes=sum(e["bytes"] for e in current_inventory))

            report("extracting")
            todo = iter(i for i in range(len(current_inventory)) if i not in completed)
            last_report = time.monotonic()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = {pool.submit(extract, i): i for i in [next(todo, None) for _ in range(args.workers)] if i is not None}
                while pending:
                    done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        completed[index] = future.result()
                        following = next(todo, None)
                        if following is not None:
                            pending[pool.submit(extract, following)] = following
                    if time.monotonic() - last_report >= 30 or not pending:
                        report("extracting")
                        last_report = time.monotonic()
            report("verifying_final_inventory")
            if inventory(args.snapshot) != current_inventory:
                raise ValueError("Remote snapshot inventory changed during extraction")
            total = sum(r["records"] for r in completed.values())
            if not total:
                raise ValueError("No specimen rows extracted")
            manifest = {"identity_sha256": identity_sha, "qualification": QUALIFICATION,
                        "evaluation_exclusions_applied": False, "completed_at": utc_now(),
                        "records": total, "nhm_records": sum(r["nhm_records"] for r in completed.values()),
                        "parts": [completed[i] for i in sorted(completed)],
                        "read_glob": str(root / "parts" / "*.parquet")}
            atomic_json(root / "manifest.json", manifest)
            atomic_json(root / "COMPLETE.json", {"status": "complete", "manifest_sha256": sha256_file(root / "manifest.json")})
            report("complete")
        except Exception as exc:
            progress("failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="2026-07-01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.snapshot) or not 1 <= args.workers <= 16:
        parser.error("Expected a date snapshot and 1–16 workers")
    stage(args)
