#!/usr/bin/env python3
"""Build a new, strictly excluded GBIF locality memory from attested raw shards.

Pilot mode is explicitly partial and unsuitable for accuracy claims. Full mode
requires a completed raw download. Neither mode promotes a production system.
No reference coordinates or taxonomy are stored as model-facing evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
import unicodedata

import build_benchmark_holdout_manifest as HOLDOUT
import nhm_locality_evidence as EVIDENCE

SCHEMA = "gbif-locality-memory-v1"
DB_FILE = "memory.sqlite"
NHM_DATASET = "7e380070-f762-11e1-a439-00145eb45e9a"
NHM_PUBLISHER = "19456090-b49a-11d8-abeb-b8a03c50a862"
ALLOWED_LICENSES = {"CC0_1_0", "CC_BY_4_0"}
FATAL_ISSUES = {
    "COORDINATE_INVALID", "COORDINATE_OUT_OF_RANGE", "COUNTRY_COORDINATE_MISMATCH",
    "PRESUMED_NEGATED_LATITUDE", "PRESUMED_NEGATED_LONGITUDE", "PRESUMED_SWAPPED_COORDINATE",
    "ZERO_COORDINATE", "COORDINATE_REPROJECTION_FAILED", "COORDINATE_REPROJECTION_SUSPICIOUS",
    "CONTINENT_COORDINATE_MISMATCH", "FOOTPRINT_WKT_MISMATCH", "INTERPRETATION_ERROR",
    "COUNTRY_INVALID", "COUNTRY_MISMATCH",
    "COORDINATE_UNCERTAINTY_METERS_INVALID", "COORDINATE_ACCURACY_INVALID",
    "COORDINATE_PRECISION_UNCERTAINTY_MISMATCH",
}
RAW_FIELDS = (
    "gbifid", "occurrenceid", "datasetkey", "catalognumber", "recordnumber", "publishingorgkey",
    "locality", "countrycode", "stateprovince", "decimallatitude", "decimallongitude",
    "coordinateuncertaintyinmeters", "license", "rightsholder", "basisofrecord",
    "institutioncode", "collectioncode", "issue",
)
RECORD_FIELDS = (
    "gbif_id", "place_id", "occurrence_id", "dataset_key", "catalog_number", "record_number",
    "publishing_org_key", "locality", "locality_normalized", "country_code", "state_province",
    "state_normalized", "latitude", "longitude", "uncertainty_m", "license", "rightsholder",
    "basis_of_record", "institution_code", "collection_code", "issues_json", "record_sha256",
)
QUALIFICATION = (
    "GBIF-derived DiSSCo UK-aligned proxy locality evidence, not NHM-native data or an exact DiSSCo export. "
    "Known benchmark identifiers and locality variants excluded before grouping/indexing. This does not "
    "prove all unknown transformed copies absent. Exact coordinates retained, no centroids. Support "
    "counts are specimens, not independent georeferences or confidence. No production approval."
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return EVIDENCE.sha256_file(Path(path))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def normalize_identifier(value):
    return "".join(c for c in unicodedata.normalize("NFKC", str(value or "")).casefold() if c.isalnum())


class Exclusions:
    def __init__(self, holdout_path, expected_sha256):
        path = Path(holdout_path)
        if not re.fullmatch("[0-9a-f]{64}", expected_sha256) or sha(path) != expected_sha256:
            raise ValueError("Holdout hash does not match explicit expected identity")
        payload = read(path)
        registry = EVIDENCE.HoldoutRegistry(path, policy="strict_locality")
        if payload["records_sha256"] != digest(payload["records"]):
            raise ValueError("Holdout aggregate records hash mismatch")
        if payload["benchmark_ids_sha256"] != digest(sorted(r["benchmark_id"] for r in payload["records"])):
            raise ValueError("Holdout benchmark ID hash mismatch")
        if payload["cohort_counts"] != dict(Counter(r["cohort"] for r in payload["records"])):
            raise ValueError("Holdout cohort counts mismatch")
        if payload["content_variant_count"] != sum(len(r["content_variants"]) for r in payload["records"]):
            raise ValueError("Holdout variant count mismatch")
        for info in payload["source_files"]:
            if sha(Path(info["path"])) != info["sha256"]:
                raise ValueError("Holdout input source changed")
        rebuilt = HOLDOUT.build_manifest(
            [HOLDOUT.SourceSpec(info["cohort"], Path(info["path"]), info["kind"]) for info in payload["source_files"]],
            expected_record_count=payload["record_count"], allow_content_variants=True, allow_multiple_references=True)
        if rebuilt != payload:
            raise ValueError("Holdout content does not reproduce from its declared sources")
        self.sha256 = expected_sha256
        self.registry = registry
        self.localities = set(registry.localities)
        identifiers = set(registry.barcode_identifiers) | set(registry.catalog_identifiers)
        self.identifiers = {normalize_identifier(v) for v in identifiers}
        self.nhm_numeric = {str(int(v)) for v in identifiers if v.isascii() and v.isdigit()}
        self.record_count = registry.record_count

    def match(self, row):
        reasons = set()
        if set(HOLDOUT.locality_holdout_keys(row.get("locality"))) & self.localities:
            reasons.add("holdout_locality")
        nhm_context = (
            row.get("datasetkey") == NHM_DATASET or row.get("publishingorgkey") == NHM_PUBLISHER
            or normalize_identifier(row.get("institutioncode")) in {"nhmuk", "bmnh"})
        # GBIF record IDs have their own namespace and are NOT barcode matches.
        for field in ("catalognumber", "recordnumber", "occurrenceid"):
            value = normalize_identifier(row.get(field))
            if value and value in self.identifiers:
                reasons.add("holdout_identifier")
            match = re.fullmatch(r"nhmuk([0-9]+)", value)
            numeric = match.group(1) if match else value if nhm_context and value.isascii() and value.isdigit() else ""
            if numeric and str(int(numeric)) in self.nhm_numeric:
                reasons.add("holdout_nhm_identifier")
        return reasons


def normalize_record(raw, exclusions):
    reasons = exclusions.match(raw)
    if reasons:
        return None, "+".join(sorted(reasons))
    locality = str(raw.get("locality") or "").strip()
    locality_normalized = EVIDENCE.normalize_text(locality)
    if not locality_normalized:
        return None, "missing_locality"
    if any(isinstance(raw.get(k), bool) for k in ("decimallatitude", "decimallongitude")):
        return None, "invalid_coordinates"
    try:
        latitude, longitude = float(raw["decimallatitude"]), float(raw["decimallongitude"])
    except (TypeError, ValueError, KeyError, OverflowError):
        return None, "invalid_coordinates"
    if not EVIDENCE.valid_coordinate_pair(latitude, longitude) or not all(map(math.isfinite, (latitude, longitude))):
        return None, "invalid_coordinates"
    issues = raw.get("issue")
    if issues is None:
        issues = []
    if not isinstance(issues, (list, tuple)) or any(not isinstance(v, str) for v in issues):
        return None, "malformed_issues"
    issues = sorted(set(issues))
    if set(issues) & FATAL_ISSUES:
        return None, "fatal_coordinate_issue"
    uncertainty = raw.get("coordinateuncertaintyinmeters")
    if isinstance(uncertainty, bool):
        return None, "invalid_uncertainty"
    if uncertainty is not None and str(uncertainty).strip() != "":
        try:
            uncertainty = float(uncertainty)
        except (ValueError, TypeError, OverflowError):
            return None, "invalid_uncertainty"
        if not math.isfinite(uncertainty) or uncertainty <= 0:
            return None, "invalid_uncertainty"
        if uncertainty > 100_000:
            return None, "uncertainty_over_100km"
    else:
        uncertainty = None
    license_value = str(raw.get("license") or "").strip()
    if license_value not in ALLOWED_LICENSES:
        return None, "license_not_allowlisted"
    gbif_id = str(raw.get("gbifid") or "").strip()
    dataset_key = str(raw.get("datasetkey") or "").strip()
    if not re.fullmatch("[0-9]+", gbif_id) or not dataset_key:
        return None, "missing_source_identity"
    country_code = str(raw.get("countrycode") or "").strip().upper()
    if country_code and not re.fullmatch("[A-Z]{2}", country_code):
        return None, "invalid_country_code"
    state = str(raw.get("stateprovince") or "").strip()
    record = {
        "gbif_id": gbif_id, "occurrence_id": str(raw.get("occurrenceid") or "").strip(),
        "dataset_key": dataset_key, "catalog_number": str(raw.get("catalognumber") or "").strip(),
        "record_number": str(raw.get("recordnumber") or "").strip(),
        "publishing_org_key": str(raw.get("publishingorgkey") or "").strip(),
        "locality": locality, "locality_normalized": locality_normalized, "country_code": country_code,
        "state_province": state, "state_normalized": EVIDENCE.normalize_text(state),
        "latitude": latitude + 0.0, "longitude": longitude + 0.0, "uncertainty_m": uncertainty,
        "license": license_value, "rightsholder": str(raw.get("rightsholder") or "").strip(),
        "basis_of_record": str(raw.get("basisofrecord") or "").strip(),
        "institution_code": str(raw.get("institutioncode") or "").strip(),
        "collection_code": str(raw.get("collectioncode") or "").strip(),
        "issues_json": json.dumps(issues, separators=(",", ":")),
    }
    record["place_id"] = "gbif-place:" + digest([record[k] for k in (
        "locality_normalized", "country_code", "state_normalized", "latitude", "longitude")])
    record["record_sha256"] = digest(record)
    return record, "accepted"


def regular_file(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Not a regular attested file: {path}")
    return path


def validate_source(source_root, mode, pilot_parts, selected_indices=None):
    root = Path(source_root).resolve()
    if mode not in {"pilot", "full"} or pilot_parts < 1:
        raise ValueError("Invalid source mode/part bound")
    raw_identity = read(regular_file(root / "identity.json"))
    identity_sha = sha(root / "identity.json")
    if raw_identity["schema"] != "placebot-gbif-raw-proxy-v1":
        raise ValueError("Wrong raw source schema")
    for name, key in (("organizations.json", "registry_sha256"), ("citation.txt", "citation_sha256")):
        if sha(regular_file(root / name)) != raw_identity[key]:
            raise ValueError("Raw source registry/citation changed")
    inventory = raw_identity["sources"]
    if not inventory or len({e["key"] for e in inventory}) != len(inventory):
        raise ValueError("Empty/duplicate source inventory")
    manifest = None
    if mode == "full":
        terminal = read(regular_file(root / "COMPLETE.json"))
        if terminal["status"] != "complete" or sha(regular_file(root / "manifest.json")) != terminal["manifest_sha256"]:
            raise ValueError("Raw download incomplete or manifest changed")
        manifest = read(root / "manifest.json")
        if manifest["identity_sha256"] != identity_sha or len(manifest["parts"]) != len(inventory):
            raise ValueError("Full source manifest identity/coverage mismatch")
        indices = list(range(len(inventory)))
        if selected_indices is not None and list(selected_indices) != indices:
            raise ValueError("Full source selection must cover entire inventory")
    elif selected_indices is None:
        indices = []
        # Completion receipts are the sole authority; ignore unreceipted parts.
        for path in sorted((root / "receipts").glob("*.json")):
            if not re.fullmatch(r"[0-9]{6}\.json", path.name):
                raise ValueError("Invalid source receipt filename")
            receipt = read(regular_file(path))
            if receipt["records"] > 0:
                indices.append(int(path.stem))
                if len(indices) >= pilot_parts:
                    break
    else:
        indices = list(selected_indices)
    if not indices or len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < len(inventory) for i in indices):
        raise ValueError("No valid unique completed source selection")
    selected = []
    for index in indices:
        receipt_path = regular_file(root / "receipts" / f"{index:06d}.json")
        receipt = read(receipt_path)
        expected_name = f"{index:06d}.parquet"
        if (receipt["identity_sha256"] != identity_sha or receipt["source"] != inventory[index]
                or receipt["part"] != expected_name or type(receipt["records"]) is not int or receipt["records"] < 0):
            raise ValueError("Source receipt membership/identity mismatch")
        part = regular_file(root / "parts" / expected_name)
        if part.stat().st_size != receipt["bytes"] or sha(part) != receipt["sha256"]:
            raise ValueError("Source part checksum/size mismatch")
        if manifest is not None and manifest["parts"][index] != receipt:
            raise ValueError("Source receipt differs from completed manifest")
        selected.append({"index": index, "receipt_sha256": sha(receipt_path), "receipt": receipt})
    if manifest is not None:
        if manifest["records"] != sum(s["receipt"]["records"] for s in selected):
            raise ValueError("Full source manifest row total mismatch")
        if manifest["nhm_records"] != sum(s["receipt"]["nhm_records"] for s in selected):
            raise ValueError("Full source NHM row total mismatch")
    return {"identity_sha256": identity_sha, "snapshot": raw_identity["snapshot"],
            "source_root": str(root), "indices": indices, "selected": selected,
            "complete_manifest_sha256": sha(root / "manifest.json") if manifest is not None else None}


def create_schema(db, identity_sha, identity):
    db.executescript("""
        BEGIN;
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
        CREATE TABLE records(
            gbif_id TEXT PRIMARY KEY, place_id TEXT NOT NULL, occurrence_id TEXT NOT NULL,
            dataset_key TEXT NOT NULL, catalog_number TEXT NOT NULL, record_number TEXT NOT NULL,
            publishing_org_key TEXT NOT NULL, locality TEXT NOT NULL, locality_normalized TEXT NOT NULL,
            country_code TEXT NOT NULL, state_province TEXT NOT NULL, state_normalized TEXT NOT NULL,
            latitude REAL NOT NULL, longitude REAL NOT NULL, uncertainty_m REAL,
            license TEXT NOT NULL, rightsholder TEXT NOT NULL, basis_of_record TEXT NOT NULL,
            institution_code TEXT NOT NULL, collection_code TEXT NOT NULL,
            issues_json TEXT NOT NULL, record_sha256 TEXT NOT NULL
        );
        CREATE TRIGGER no_conflicting_gbif_id BEFORE INSERT ON records
        WHEN EXISTS(SELECT 1 FROM records WHERE gbif_id=NEW.gbif_id AND record_sha256 != NEW.record_sha256)
        BEGIN SELECT RAISE(ABORT, 'Conflicting GBIF id'); END;
        CREATE TABLE completed_parts(part_index INTEGER PRIMARY KEY, receipt_sha256 TEXT NOT NULL,
                                     source_records INTEGER NOT NULL, added_records INTEGER NOT NULL,
                                     counters_json TEXT NOT NULL);
        CREATE TABLE raw_seen(gbif_id TEXT PRIMARY KEY, raw_sha256 TEXT NOT NULL);
        CREATE TRIGGER no_conflicting_raw_gbif_id BEFORE INSERT ON raw_seen
        WHEN EXISTS(SELECT 1 FROM raw_seen WHERE gbif_id=NEW.gbif_id AND raw_sha256 != NEW.raw_sha256)
        BEGIN SELECT RAISE(ABORT, 'Conflicting raw GBIF id'); END;
    """)
    metadata = {
        "schema": SCHEMA, "status": "building", "mode": identity["mode"], "identity_sha256": identity_sha,
        "holdout_sha256": identity["holdout_sha256"], "source_snapshot": identity["source"]["snapshot"],
        "source_inventory_identity_sha256": identity["source"]["identity_sha256"],
    }
    db.executemany("INSERT INTO metadata VALUES (?,?)", [(k, json.dumps(v)) for k, v in metadata.items()])
    db.commit()


def metadata(db):
    return {k: json.loads(v) for k, v in db.execute("SELECT key,value_json FROM metadata")}


def aggregate_places(db):
    # Transactional rebuild after all raw parts; resume cannot expose half an index.
    with db:
        db.executescript("""
            BEGIN;
            DROP TABLE IF EXISTS places_fts;
            DROP TABLE IF EXISTS places;
            CREATE INDEX IF NOT EXISTS records_place_idx ON records(place_id,gbif_id);
            CREATE TABLE places(
                place_id TEXT PRIMARY KEY, locality TEXT NOT NULL, locality_normalized TEXT NOT NULL,
                country_code TEXT NOT NULL, state_province TEXT NOT NULL, state_normalized TEXT NOT NULL,
                latitude REAL NOT NULL, longitude REAL NOT NULL, support_count INTEGER NOT NULL,
                distinct_occurrence_count INTEGER NOT NULL, uncertainty_m REAL, license_set_json TEXT NOT NULL,
                distinct_dataset_count INTEGER NOT NULL, uncertainty_missing_count INTEGER NOT NULL,
                uncertainty_min_m REAL, uncertainty_max_m REAL
            );
            INSERT INTO places
            SELECT place_id, min(locality), locality_normalized, country_code, min(state_province), state_normalized,
                   latitude, longitude, count(*),
                   count(DISTINCT json_array(dataset_key,
                         CASE WHEN occurrence_id='' THEN 'gbif_id' ELSE 'occurrence_id' END,
                         coalesce(nullif(occurrence_id,''),gbif_id))),
                   CASE WHEN count(uncertainty_m)=count(*) THEN max(uncertainty_m) ELSE NULL END,
                   json_group_array(DISTINCT license), count(DISTINCT dataset_key),
                   count(*)-count(uncertainty_m), min(uncertainty_m), max(uncertainty_m)
            FROM records GROUP BY place_id;
            CREATE INDEX places_locality_country_idx ON places(locality_normalized,country_code);
            CREATE INDEX places_country_locality_idx ON places(country_code,locality_normalized);
            CREATE VIRTUAL TABLE places_fts USING fts5(place_id UNINDEXED, locality,state_province,country_code,
                                                      tokenize='unicode61 remove_diacritics 2');
            INSERT INTO places_fts SELECT place_id, locality_normalized,state_normalized,country_code FROM places ORDER BY place_id;
            COMMIT;
        """)


def audit_exclusions(db, exclusions):
    checked = 0
    cursor = db.execute("SELECT " + ",".join(RECORD_FIELDS) + " FROM records ORDER BY gbif_id")
    while batch := cursor.fetchmany(5000):
        for values in batch:
            record = dict(zip(RECORD_FIELDS, values))
            raw = {
                "gbifid": record["gbif_id"], "occurrenceid": record["occurrence_id"], "datasetkey": record["dataset_key"],
                "catalognumber": record["catalog_number"], "recordnumber": record["record_number"],
                "publishingorgkey": record["publishing_org_key"], "locality": record["locality"],
                "countrycode": record["country_code"], "stateprovince": record["state_province"],
                "decimallatitude": record["latitude"], "decimallongitude": record["longitude"],
                "coordinateuncertaintyinmeters": record["uncertainty_m"], "license": record["license"],
                "rightsholder": record["rightsholder"], "basisofrecord": record["basis_of_record"],
                "institutioncode": record["institution_code"], "collectioncode": record["collection_code"],
                "issue": json.loads(record["issues_json"]),
            }
            normalized, reason = normalize_record(raw, exclusions)
            if normalized is None or normalized != record:
                raise ValueError("Output record failed independent post-ingestion normalization/exclusion check: " + reason)
            checked += 1
    if db.execute("SELECT count(*) FROM records r LEFT JOIN places p USING(place_id) WHERE p.place_id IS NULL").fetchone()[0]:
        raise ValueError("Evidence row missing from place index")
    if db.execute("SELECT count(*) FROM places p WHERE support_count != (SELECT count(*) FROM records r WHERE r.place_id=p.place_id)").fetchone()[0]:
        raise ValueError("Place support counts differ from source rows")
    if db.execute("SELECT count(*) FROM places").fetchone()[0] != db.execute("SELECT count(*) FROM places_fts").fetchone()[0]:
        raise ValueError("FTS membership count mismatch")
    return {"status": "passed", "checked_records": checked, "holdout_matches": 0,
            "holdout_sha256": exclusions.sha256,
            "qualification": "Independent scan of produced rows against the same declared known-key policy; not proof of unknown transformed-copy absence"}


def verify_output(output, expected_identity_sha=None):
    root = Path(output)
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = root / (DB_FILE + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            raise ValueError("Completed database has a mutable SQLite sidecar")
    terminal = read(regular_file(root / "COMPLETE.json"))
    manifest = read(regular_file(root / "manifest.json"))
    if terminal["status"] != "complete" or terminal["manifest_sha256"] != sha(root / "manifest.json"):
        raise ValueError("Output manifest completion mismatch")
    identity_sha = sha(regular_file(root / "identity.json"))
    identity = read(root / "identity.json")
    if manifest["identity_sha256"] != identity_sha or (expected_identity_sha and expected_identity_sha != identity_sha):
        raise ValueError("Output identity mismatch")
    if sha(regular_file(root / DB_FILE)) != manifest["database_sha256"]:
        raise ValueError("Output database changed")
    with closing(sqlite3.connect((root / DB_FILE).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        meta = metadata(db)
        if meta["identity_sha256"] != identity_sha or meta["status"] != "complete":
            raise ValueError("Output database metadata identity mismatch")
        if identity["schema"] != SCHEMA or identity["mode"] not in {"pilot", "full"}:
            raise ValueError("Unexpected output schema/mode")
        for key in ("schema", "mode", "holdout_sha256"):
            if meta[key] != identity[key] or manifest[key] != identity[key]:
                raise ValueError("Output schema/mode/holdout metadata mismatch")
        if (meta["source_snapshot"] != identity["source"]["snapshot"]
                or manifest["source_snapshot"] != meta["source_snapshot"]
                or meta["source_inventory_identity_sha256"] != identity["source"]["identity_sha256"]):
            raise ValueError("Output source metadata mismatch")
        if (manifest["records"] != db.execute("SELECT count(*) FROM records").fetchone()[0]
                or manifest["places"] != db.execute("SELECT count(*) FROM places").fetchone()[0]):
            raise ValueError("Output row counts mismatch")
    return manifest


def build(source_root, output, holdout, expected_holdout_sha256, *, mode="full", pilot_parts=16):
    import duckdb
    output, source_root, holdout = Path(output).resolve(), Path(source_root).resolve(), Path(holdout).resolve()
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError("Memory output must be separate from raw source tree")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (output / "identity.json").exists() and set(p.name for p in output.iterdir()) != {".lock"}:
            raise ValueError("Nonempty output without identity; refusing to guess recovery")
        existing = read(output / "identity.json") if (output / "identity.json").exists() else None
        exclusions = Exclusions(holdout, expected_holdout_sha256)
        source = validate_source(source_root, mode, pilot_parts, existing["source"]["indices"] if existing else None)
        identity = {
            "schema": SCHEMA, "mode": mode, "pilot_parts": pilot_parts if mode == "pilot" else None,
            "source": source, "holdout_sha256": exclusions.sha256, "holdout_path": str(holdout),
            "policy": "strict-locality-variants-and-nhm-aware-identifiers-v1", "grouping": "exact-coordinate-no-centroid",
            "allowed_licenses": sorted(ALLOWED_LICENSES), "fatal_issues": sorted(FATAL_ISSUES),
            "unknown_uncertainty": "retained_as_null", "maximum_known_uncertainty_m": 100_000,
            "selected_raw_fields": list(RAW_FIELDS), "qualification": QUALIFICATION,
            "duckdb_version": duckdb.__version__, "sqlite_version": sqlite3.sqlite_version,
            "source_hashes": {Path(p).name: sha(p) for p in (Path(__file__), Path(HOLDOUT.__file__), Path(EVIDENCE.__file__))},
        }
        if existing is not None and existing != identity:
            raise ValueError("Memory resume identity mismatch; use a fresh output")
        save(output / "identity.json", identity)
        identity_sha = sha(output / "identity.json")
        if (output / "COMPLETE.json").exists():
            return verify_output(output, identity_sha)
        began = time.monotonic()
        db = sqlite3.connect(output / DB_FILE)
        duck = duckdb.connect()
        duck.execute("SET threads=2")
        duck.execute("SET memory_limit='2GB'")
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA cache_size=-131072")
            if not db.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone()[0]:
                if db.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]:
                    raise ValueError("Unexpected partial database schema")
                create_schema(db, identity_sha, identity)
            if metadata(db)["identity_sha256"] != identity_sha:
                raise ValueError("Partial database identity differs from manifest")
            completed = {i: (s, n, added, json.loads(c)) for i, s, n, added, c in db.execute(
                "SELECT part_index,receipt_sha256,source_records,added_records,counters_json FROM completed_parts")}
            selection = {s["index"]: s for s in source["selected"]}
            if not set(completed).issubset(selection):
                raise ValueError("Unexpected completed part in partial memory")
            for index, (receipt_sha, count, added, counters) in completed.items():
                expected = selection[index]
                if expected["receipt_sha256"] != receipt_sha or expected["receipt"]["records"] != count:
                    raise ValueError("Resumed part differs from frozen receipt")
            total_counters = Counter()
            for _, _, _, counters in completed.values():
                total_counters.update(counters)

            def progress(phase):
                value = {"phase": phase, "mode": mode, "updated_at": now(), "completed_parts": len(completed),
                         "total_parts": len(selection), "source_records": sum(c[1] for c in completed.values()),
                         "added_records": sum(c[2] for c in completed.values()), "counters": dict(total_counters),
                         "seconds_this_invocation": time.monotonic() - began}
                save(output / "progress.json", value)
                print(json.dumps(value, sort_keys=True), flush=True)

            progress("ingesting")
            last_progress = time.monotonic()
            insert = "INSERT OR IGNORE INTO records (" + ",".join(RECORD_FIELDS) + ") VALUES (" + ",".join("?" for _ in RECORD_FIELDS) + ")"
            for item in source["selected"]:
                index, receipt = item["index"], item["receipt"]
                if index in completed:
                    continue
                path = source_root / "parts" / receipt["part"]
                if sha(path) != receipt["sha256"]:
                    raise ValueError("Source part changed immediately before ingestion")
                cursor = duck.execute("SELECT " + ",".join(RAW_FIELDS) + " FROM read_parquet(?)", [str(path)])
                count, added, counters = 0, 0, Counter()
                with db:
                    db.execute("BEGIN")
                    while batch := cursor.fetchmany(5000):
                        accepted, raw_ids = [], []
                        for values in batch:
                            raw = dict(zip(RAW_FIELDS, values))
                            count += 1
                            gbif_id = str(raw.get("gbifid") or "").strip()
                            if gbif_id:
                                raw_hash = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()).hexdigest()
                                raw_ids.append((gbif_id, raw_hash))
                            record, reason = normalize_record(raw, exclusions)
                            counters[reason] += 1
                            if record is not None:
                                accepted.append(tuple(record[k] for k in RECORD_FIELDS))
                        db.executemany("INSERT OR IGNORE INTO raw_seen VALUES (?,?)", raw_ids)
                        before = db.total_changes
                        db.executemany(insert, accepted)
                        added += db.total_changes - before
                    if count != receipt["records"]:
                        raise ValueError("Actual source row count differs from receipt")
                    if sha(path) != receipt["sha256"]:
                        raise ValueError("Source part changed during ingestion")
                    counters["duplicate_accepted_rows"] = counters["accepted"] - added
                    db.execute("INSERT INTO completed_parts VALUES (?,?,?,?,?)", (
                        index, item["receipt_sha256"], count, added, json.dumps(counters, sort_keys=True)))
                completed[index] = (item["receipt_sha256"], count, added, counters)
                total_counters.update(counters)
                if time.monotonic() - last_progress >= 15 or len(completed) == len(selection):
                    progress("ingesting")
                    last_progress = time.monotonic()
            if len(completed) != len(selection):
                raise ValueError("Incomplete memory ingestion")
            progress("grouping_and_indexing")
            aggregate_places(db)
            progress("auditing_exclusions")
            audit = audit_exclusions(db, exclusions)
            records = db.execute("SELECT count(*) FROM records").fetchone()[0]
            places = db.execute("SELECT count(*) FROM places").fetchone()[0]
            if records != sum(c[2] for c in completed.values()):
                raise ValueError("Completed part ledger differs from evidence record count")
            if sum(v for k, v in total_counters.items() if k != "duplicate_accepted_rows") != sum(c[1] for c in completed.values()):
                raise ValueError("Ingestion reason counts do not account for every source row")
            if not records or not places:
                raise ValueError("No eligible evidence: retain diagnostic output but do not mark complete")
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
            # Check every input again before issuing the completion attestation.
            if validate_source(source_root, mode, pilot_parts, source["indices"]) != source or sha(holdout) != exclusions.sha256:
                raise ValueError("Source or holdout changed before completion")
            with db:
                db.execute("UPDATE metadata SET value_json=? WHERE key='status'", [json.dumps("complete")])
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.execute("PRAGMA journal_mode=DELETE")
            manifest = {
                "schema": SCHEMA, "mode": mode, "production_approved": False, "qualification": QUALIFICATION,
                "identity_sha256": identity_sha, "holdout_sha256": exclusions.sha256,
                "source_snapshot": source["snapshot"], "records": records, "places": places,
                "source_parts": len(selection), "source_records": sum(c[1] for c in completed.values()),
                "counters": dict(total_counters), "exclusion_audit": audit, "completed_at": now(),
                "seconds_this_invocation": time.monotonic() - began,
            }
        except Exception as exc:
            db.rollback()
            save(output / "failure.json", {"at": now(), "error": f"{type(exc).__name__}: {exc}", "identity_sha256": identity_sha})
            if isinstance(exc, sqlite3.IntegrityError):
                raise ValueError(str(exc)) from exc
            raise
        finally:
            duck.close()
            db.close()
        manifest["database_sha256"] = sha(output / DB_FILE)
        save(output / "manifest.json", manifest)
        save(output / "COMPLETE.json", {"status": "complete", "manifest_sha256": sha(output / "manifest.json")})
        save(output / "progress.json", {"phase": "complete", "mode": mode, "records": records, "places": places, "updated_at": now()})
        return verify_output(output, identity_sha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--expected-holdout-sha256", required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), default="full")
    parser.add_argument("--pilot-parts", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.holdout, args.expected_holdout_sha256,
                           mode=args.mode, pilot_parts=args.pilot_parts), indent=2))
