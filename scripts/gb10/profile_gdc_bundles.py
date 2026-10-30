#!/usr/bin/env python3
"""Create a value-free aggregate profile of the TDWG GDC CSV bundles.

The output deliberately contains schema names and counts, never source values or
record identifiers, so it can be retained in the Thor orchestration workspace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_VERSION = 1
PLACEHOLDERS = {"-", "?", "n/a", "na", "none", "null", "unknown", "not known"}
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REPEATED_SPACE_RE = re.compile(r" {2,}")


def set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_schema(path: Path) -> dict[str, dict[str, str]]:
    tables: dict[str, dict[str, str]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            tables[row["table_name"]][row["column_name"]] = row["data_type"]
    return dict(tables)


def table_name_for(path: Path, tables: dict[str, dict[str, str]]) -> str:
    for table in sorted(tables, key=len, reverse=True):
        if path.stem.endswith(f"_{table}"):
            return table
    raise ValueError(f"Cannot map {path.name} to a schema table")


def parse_number(value: str, data_type: str) -> bool:
    try:
        if data_type in {"bigint", "integer"}:
            int(value)
        elif data_type == "double":
            return math.isfinite(float(value))
        else:
            return True
    except ValueError:
        return False
    return True


def as_float(value: str) -> float | None:
    if not value:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def profile_file(
    path: Path,
    table: str,
    types: dict[str, str],
    core_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = 0
    blank_rows = 0
    width_mismatches = 0
    duplicate_keys = 0
    missing_key_parts = 0
    orphan_irns = 0
    seen_keys: set[tuple[str, ...]] = set()
    column_metrics: dict[str, Counter[str]] = defaultdict(Counter)
    max_chars: Counter[str] = Counter()
    domain: Counter[str] = Counter()
    record_types: Counter[str] = Counter()
    bibliography_parent_refs: list[str] = []
    preferred_counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError(f"Duplicate header names in {path}")

        if table == "core":
            key_columns = ["irn"]
        else:
            key_columns = [name for name in ("irn", "pos_outer", "pos_inner") if name in header]

        for row in reader:
            rows += 1
            if len(row) != len(header):
                width_mismatches += 1
                continue
            if not any(row):
                blank_rows += 1
            record = dict(zip(header, row))

            key = tuple(record[name] for name in key_columns)
            if any(not value for value in key):
                missing_key_parts += 1
            elif key in seen_keys:
                duplicate_keys += 1
            else:
                seen_keys.add(key)

            irn = record.get("irn", "")
            if table == "core" and irn:
                core_ids.add(irn)
            elif table != "core" and irn and irn not in core_ids:
                orphan_irns += 1

            for name, value in record.items():
                metrics = column_metrics[name]
                if value == "":
                    metrics["empty"] += 1
                    continue
                metrics["nonempty"] += 1
                max_chars[name] = max(max_chars[name], len(value))
                if value != value.strip():
                    metrics["outer_whitespace"] += 1
                if "\n" in value or "\r" in value:
                    metrics["embedded_newline"] += 1
                if CONTROL_RE.search(value):
                    metrics["control_character"] += 1
                if "\ufffd" in value or any(marker in value for marker in ("Ã", "Â", "â€")):
                    metrics["encoding_marker"] += 1
                if REPEATED_SPACE_RE.search(value):
                    metrics["repeated_spaces"] += 1
                if value.strip().casefold() in PLACEHOLDERS:
                    metrics["placeholder_token"] += 1
                data_type = types.get(name)
                if data_type and not parse_number(value, data_type):
                    metrics["invalid_declared_type"] += 1

            bundle = path.parts[-3]
            if bundle == "ebibliography" and table == "core":
                record_type = record.get("bibrecordtype", "").strip()
                recognized = {
                    "Article", "Book", "Book Series", "Catalogue", "Chapter",
                    "Citation", "Electronic Resource", "Journal", "Manuscript",
                    "Other", "Thesis", "Web Site",
                }
                record_types[record_type if record_type in recognized else "<blank-or-other>"] += 1
                title_by_type = {
                    "Article": "arttitle", "Book": "bootitle", "Book Series": "bostitle",
                    "Catalogue": "cattitle", "Chapter": "chatitle",
                    "Electronic Resource": "eletitle", "Journal": "joutitle",
                    "Manuscript": "mantitle", "Other": "othtitle",
                    "Thesis": "thetitle", "Web Site": "webtitle",
                }
                title_field = title_by_type.get(record_type)
                if title_field and not record.get(title_field, ""):
                    domain["typed_records_missing_type_specific_title"] += 1
                for parent_name in (
                    "artparentref", "booparentref", "bosparentref", "catparentref",
                    "chaparentref", "citparentref", "eleparentref", "jouparentref",
                    "manparentref", "othparentref",
                ):
                    if record.get(parent_name, ""):
                        bibliography_parent_refs.append(record[parent_name])

            elif bundle == "elocations" and table == "core":
                levels = [record.get(f"loclevel{index}", "") for index in range(1, 11)]
                seen_gap = False
                hierarchy_gap = False
                for value in levels:
                    if not value:
                        seen_gap = True
                    elif seen_gap:
                        hierarchy_gap = True
                if hierarchy_gap:
                    domain["noncontiguous_location_hierarchy"] += 1
                if any(levels) and not record.get("loclevellowest", ""):
                    domain["hierarchy_present_but_lowest_level_missing"] += 1
                xyz = [record.get(name, "") for name in ("phylocationx", "phylocationy", "phylocationz")]
                if any(xyz) and not all(xyz):
                    domain["partial_physical_xyz"] += 1
                dimension_names = (
                    "locextimpheight", "locextimplength", "locextimpwidth",
                    "locextmetheight", "locextmetlength", "locextmetwidth",
                )
                if any((number := as_float(record.get(name, ""))) is not None and number < 0 for name in dimension_names):
                    domain["negative_external_dimension"] += 1

            elif bundle == "esites" and table == "core":
                continent = record.get("loccontinent", "")
                country = record.get("loccountry", "")
                province = record.get("locprovincestateterritory", "")
                district = record.get("locdistrictcountyshire", "")
                if country and not continent:
                    domain["country_without_continent"] += 1
                if (province or district) and not country:
                    domain["subcountry_without_country"] += 1
                geography = (
                    continent, country, province, district,
                    record.get("locislandgrouping", ""), record.get("locislandname", ""),
                    record.get("locpreciselocation", ""), record.get("locnhmlabellocality", ""),
                    record.get("lococean", ""), record.get("locseagulf", ""),
                )
                if not any(geography):
                    domain["no_primary_geographic_text"] += 1
                for suffix in ("ft", "mt"):
                    low = as_float(record.get(f"locelevationaslfrom{suffix}", ""))
                    high = as_float(record.get(f"locelevationaslto{suffix}", ""))
                    if low is not None and high is not None and low > high:
                        domain[f"elevation_from_greater_than_to_{suffix}"] += 1

            elif bundle == "esites" and table == "latlong":
                latitude = as_float(record.get("latcentroidlatitudedec", ""))
                longitude = as_float(record.get("latcentroidlongitudedec", ""))
                if (latitude is None) != (longitude is None):
                    domain["partial_decimal_centroid_pair"] += 1
                if latitude is not None and not -90 <= latitude <= 90:
                    domain["centroid_latitude_out_of_range"] += 1
                if longitude is not None and not -180 <= longitude <= 180:
                    domain["centroid_longitude_out_of_range"] += 1
                if (latitude is not None or longitude is not None) and not record.get("latdatum", ""):
                    domain["decimal_centroid_without_datum"] += 1
                if irn and record.get("latpreferred", "").strip().casefold() == "yes":
                    preferred_counts[irn] += 1

            elif bundle == "esites" and table == "latlongdetail":
                latitude = as_float(record.get("latlatitudedecimal", ""))
                longitude = as_float(record.get("latlongitudedecimal", ""))
                if (latitude is None) != (longitude is None):
                    domain["partial_decimal_point_pair"] += 1
                if latitude is not None and not -90 <= latitude <= 90:
                    domain["point_latitude_out_of_range"] += 1
                if longitude is not None and not -180 <= longitude <= 180:
                    domain["point_longitude_out_of_range"] += 1

            elif bundle == "esites" and table == "utmcoords":
                easting = record.get("mapeasting0", "")
                northing = record.get("mapnorthing0", "")
                if bool(easting) != bool(northing):
                    domain["partial_utm_pair"] += 1

    if path.parts[-3] == "ebibliography" and table == "core":
        domain["parent_references_not_in_core"] = sum(
            reference not in core_ids for reference in bibliography_parent_refs
        )
        domain["parent_references_checked"] = len(bibliography_parent_refs)
    if path.parts[-3] == "esites" and table == "latlong":
        domain["sites_with_multiple_preferred_determinations"] = sum(
            count > 1 for count in preferred_counts.values()
        )
        domain["sites_with_a_preferred_determination"] = len(preferred_counts)

    columns = {
        name: {
            "declared_type": types.get(name, "<missing>"),
            "max_chars": max_chars[name],
            **dict(sorted(column_metrics[name].items())),
        }
        for name in header
    }
    result = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "table": table,
        "column_count": len(header),
        "columns": columns,
        "rows": rows,
        "blank_rows": blank_rows,
        "row_width_mismatches": width_mismatches,
        "key_columns": key_columns,
        "duplicate_keys": duplicate_keys,
        "missing_key_parts": missing_key_parts,
        "orphan_irns": orphan_irns,
    }
    extras: dict[str, Any] = {"domain_findings": dict(sorted(domain.items()))}
    if record_types:
        extras["record_type_counts"] = dict(sorted(record_types.items()))
    return result, extras


def profile_bundle(bundle_root: Path) -> dict[str, Any]:
    gdc_dirs = sorted(path for path in bundle_root.glob("gdc_*") if path.is_dir())
    if len(gdc_dirs) != 1:
        raise ValueError(f"Expected one gdc_* directory under {bundle_root}")
    gdc_dir = gdc_dirs[0]
    schema_paths = sorted(gdc_dir.glob("*_schema.csv"))
    if len(schema_paths) != 1:
        raise ValueError(f"Expected one schema CSV under {gdc_dir}")
    schema_path = schema_paths[0]
    tables = load_schema(schema_path)
    data_paths = [path for path in sorted(gdc_dir.glob("*.csv")) if path != schema_path]
    data_paths.sort(key=lambda path: table_name_for(path, tables) != "core")

    core_ids: set[str] = set()
    profiles: list[dict[str, Any]] = []
    findings: dict[str, Any] = {}
    for path in data_paths:
        table = table_name_for(path, tables)
        profile, extras = profile_file(path, table, tables[table], core_ids)
        profiles.append(profile)
        findings[table] = extras

    raw_paths = sorted(bundle_root.glob("emu_*/*.texexport.txt"))
    if len(raw_paths) != 1:
        raise ValueError(f"Expected one raw texexport under {bundle_root}")
    raw_path = raw_paths[0]
    return {
        "bundle": bundle_root.name,
        "raw_export": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
            "profiled_as_records": False,
            "reason": "Retained as provenance; flattened GDC tables are the initial working layer.",
        },
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "tables": profiles,
        "domain": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    expected = [source_root / name for name in ("ebibliography", "elocations", "esites")]
    missing = [str(path) for path in expected if not path.is_dir()]
    if missing:
        parser.error(f"Missing bundle directories: {missing}")

    set_csv_field_limit()
    report = {
        "profile_version": PROFILE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "source_root": str(source_root),
        "contains_record_values": False,
        "bundles": [profile_bundle(path) for path in expected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
