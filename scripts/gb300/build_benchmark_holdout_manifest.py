#!/usr/bin/env python3
"""Build a deterministic, model-output-free PlaceBot benchmark holdout manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "placebot_benchmark_holdout"
DEFAULT_EXPECTED_RECORD_COUNT = 1617
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
LABEL_PREFIX_RE = re.compile(
    r"^\s*\d+(?:st|nd|rd|th)\s+label\s*:\s*",
    re.IGNORECASE,
)
COHORT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
ARCHIVE_KIND = "archived_comparison_json"
REFERENCE_KIND = "reference_tsv"
SOURCE_KIND_ORDER = {REFERENCE_KIND: 0, ARCHIVE_KIND: 1}
SOURCE_FIELDS = (
    "original_barcode",
    "original_locality",
    "original_country",
    "original_lat",
    "original_long",
)
REFERENCE_FIELDS = {
    "Barcode": "original_barcode",
    "label_verbatim": "original_locality",
    "Country": "original_country",
    "lat_manual": "original_lat",
    "long_manual": "original_long",
}


class HoldoutManifestError(RuntimeError):
    """Raised when an archived benchmark cannot produce a safe holdout."""


class SourceSpec(NamedTuple):
    cohort: str
    path: Path
    kind: str = ARCHIVE_KIND


def normalize_text(value: Any) -> str:
    """Return the conservative Unicode-preserving key used by PlaceBot memory."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return " ".join(TOKEN_RE.findall(text))


def locality_holdout_keys(value: Any) -> list[str]:
    """Return the full locality plus exact semicolon-separated label variants."""
    text = "" if value is None else str(value)
    raw_values = [text]
    raw_values.extend(part.strip() for part in text.split(";") if part.strip())
    raw_values.extend(
        LABEL_PREFIX_RE.sub("", part).strip()
        for part in text.split(";")
        if LABEL_PREFIX_RE.match(part)
    )
    keys: list[str] = []
    for raw_value in raw_values:
        key = normalize_text(raw_value)
        if key and key not in keys:
            keys.append(key)
    return keys


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value for stable hashing, independent of JSON formatting."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_text(
    record: dict[str, Any],
    field: str,
    *,
    allow_empty: bool,
    context: str,
) -> str:
    if field not in record:
        raise HoldoutManifestError(f"{context} is missing {field}")
    value = record[field]
    text = "" if value is None else str(value)
    if not allow_empty and not text.strip():
        raise HoldoutManifestError(f"{context} has an empty {field}")
    return text


def require_coordinate(
    record: dict[str, Any],
    field: str,
    *,
    minimum: float,
    maximum: float,
    context: str,
) -> float:
    if field not in record:
        raise HoldoutManifestError(f"{context} is missing {field}")
    raw = record[field]
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise HoldoutManifestError(f"{context} has an empty {field}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise HoldoutManifestError(
            f"{context} has invalid {field}: {raw!r}"
        ) from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise HoldoutManifestError(
            f"{context} has out-of-range {field}: {raw!r}"
        )
    rounded = round(value, 6)
    return 0.0 if rounded == 0 else rounded


def canonical_source_record(record: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise HoldoutManifestError(f"{context} is not a JSON object")
    missing = [field for field in SOURCE_FIELDS if field not in record]
    if missing:
        raise HoldoutManifestError(
            f"{context} is missing source fields: {', '.join(missing)}"
        )
    return {
        "benchmark_id": require_text(
            record,
            "original_barcode",
            allow_empty=False,
            context=context,
        ),
        "locality": require_text(
            record,
            "original_locality",
            allow_empty=False,
            context=context,
        ),
        "country": require_text(
            record,
            "original_country",
            allow_empty=True,
            context=context,
        ),
        "latitude": require_coordinate(
            record,
            "original_lat",
            minimum=-90.0,
            maximum=90.0,
            context=context,
        ),
        "longitude": require_coordinate(
            record,
            "original_long",
            minimum=-180.0,
            maximum=180.0,
            context=context,
        ),
    }


def content_fingerprint(record: dict[str, Any]) -> str:
    """Hash source-matchable content without relying on a benchmark-only ID."""
    payload = {
        "country_normalized": normalize_text(record["country"]),
        "latitude": round(float(record["latitude"]), 6),
        "locality_normalized": normalize_text(record["locality"]),
        "longitude": round(float(record["longitude"]), 6),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def load_verified_archive(spec: SourceSpec) -> list[dict[str, Any]]:
    try:
        with spec.path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutManifestError(f"cannot read {spec.path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HoldoutManifestError(f"{spec.path} root is not a JSON object")
    model_results = payload.get("model_results")
    if not isinstance(model_results, dict) or not model_results:
        raise HoldoutManifestError(
            f"{spec.path} has no non-empty model_results object"
        )

    canonical_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_name in sorted(model_results):
        model_result = model_results[model_name]
        if not isinstance(model_result, dict):
            raise HoldoutManifestError(
                f"{spec.path} model {model_name!r} is not a JSON object"
            )
        processed = model_result.get("processed_records")
        if not isinstance(processed, list) or not processed:
            raise HoldoutManifestError(
                f"{spec.path} model {model_name!r} has no processed_records"
            )
        canonical_by_model[model_name] = [
            canonical_source_record(
                record,
                context=(
                    f"{spec.path} model {model_name!r} "
                    f"processed_records[{index}]"
                ),
            )
            for index, record in enumerate(processed)
        ]

    reference_model = next(iter(canonical_by_model))
    reference = canonical_by_model[reference_model]
    for model_name, records in canonical_by_model.items():
        if len(records) != len(reference):
            raise HoldoutManifestError(
                f"{spec.path} model {model_name!r} has {len(records)} records; "
                f"reference model {reference_model!r} has {len(reference)}"
            )
        for index, (expected, observed) in enumerate(zip(reference, records)):
            if observed != expected:
                changed = [
                    key for key in expected if observed.get(key) != expected[key]
                ]
                raise HoldoutManifestError(
                    f"{spec.path} model {model_name!r} record {index} differs "
                    f"from reference model {reference_model!r}: "
                    f"{', '.join(changed)}"
                )
    return reference


def load_reference_tsv(spec: SourceSpec) -> list[dict[str, Any]]:
    try:
        with spec.path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = set(reader.fieldnames or [])
            missing = [field for field in REFERENCE_FIELDS if field not in fieldnames]
            if missing:
                raise HoldoutManifestError(
                    f"{spec.path} is missing TSV fields: {', '.join(missing)}"
                )
            records = [
                canonical_source_record(
                    {
                        target: row.get(source)
                        for source, target in REFERENCE_FIELDS.items()
                    },
                    context=f"{spec.path} TSV row {row_number}",
                )
                for row_number, row in enumerate(reader, start=2)
            ]
    except OSError as exc:
        raise HoldoutManifestError(f"cannot read {spec.path}: {exc}") from exc
    if not records:
        raise HoldoutManifestError(f"{spec.path} contains no reference records")
    return records


def load_source(spec: SourceSpec) -> list[dict[str, Any]]:
    if spec.kind == ARCHIVE_KIND:
        return load_verified_archive(spec)
    if spec.kind == REFERENCE_KIND:
        return load_reference_tsv(spec)
    raise HoldoutManifestError(f"unsupported source kind: {spec.kind!r}")


def validate_source_specs(
    source_specs: Sequence[SourceSpec], *, allow_multiple_references: bool = False
) -> list[SourceSpec]:
    if not source_specs:
        raise HoldoutManifestError("at least one --source is required")
    cohort_kinds: set[tuple[str, str]] = set()
    paths: set[Path] = set()
    validated: list[SourceSpec] = []
    for spec in source_specs:
        if not COHORT_RE.fullmatch(spec.cohort):
            raise HoldoutManifestError(
                f"invalid cohort label {spec.cohort!r}; use letters, numbers, "
                "dot, underscore or hyphen"
            )
        path = spec.path.expanduser().resolve()
        if spec.kind not in SOURCE_KIND_ORDER:
            raise HoldoutManifestError(f"unsupported source kind: {spec.kind!r}")
        cohort_kind = (spec.cohort, spec.kind)
        if cohort_kind in cohort_kinds and not (
            allow_multiple_references and spec.kind == REFERENCE_KIND
        ):
            raise HoldoutManifestError(
                f"duplicate {spec.kind} source for cohort {spec.cohort!r}"
            )
        if path in paths:
            raise HoldoutManifestError(f"duplicate source path: {path}")
        if not path.is_file():
            raise HoldoutManifestError(f"source is not a file: {path}")
        cohort_kinds.add(cohort_kind)
        paths.add(path)
        validated.append(SourceSpec(spec.cohort, path, spec.kind))
    return sorted(
        validated,
        key=lambda item: (
            item.cohort,
            SOURCE_KIND_ORDER[item.kind],
            str(item.path),
        ),
    )


def records_have_same_content(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    include_id: bool,
) -> bool:
    fields = ("locality", "country", "latitude", "longitude")
    if include_id:
        fields = ("benchmark_id", *fields)
    return all(left[field] == right[field] for field in fields)


def canonical_benchmark_id(
    raw_id: str,
    spec: SourceSpec,
) -> tuple[str, str]:
    """Reconcile the known eight/nine-digit Bombus TSV/JSON ID discrepancy."""
    if (
        spec.kind == REFERENCE_KIND
        and spec.cohort.casefold().startswith("bombus")
        and raw_id.isascii()
        and raw_id.isdigit()
        and len(raw_id) <= 9
    ):
        canonical = raw_id.zfill(9)
        method = "bombus_reference_numeric_zfill_9"
        return canonical, method
    return raw_id, "exact"


def build_manifest(
    source_specs: Sequence[SourceSpec],
    *,
    expected_record_count: int | None = DEFAULT_EXPECTED_RECORD_COUNT,
    allow_content_variants: bool = False,
    allow_multiple_references: bool = False,
) -> dict[str, Any]:
    specs = validate_source_specs(
        source_specs, allow_multiple_references=allow_multiple_references
    )
    records_by_id: dict[str, dict[str, Any]] = {}
    record_order: list[str] = []
    source_files: list[dict[str, Any]] = []

    for source_index, spec in enumerate(specs):
        source_records = load_source(spec)
        source_files.append(
            {
                "source_index": source_index,
                "cohort": spec.cohort,
                "kind": spec.kind,
                "path": str(spec.path),
                "sha256": sha256_file(spec.path),
                "record_count": len(source_records),
            }
        )
        source_seen_ids: set[str] = set()
        for record_index, source_record in enumerate(source_records):
            raw_id = source_record["benchmark_id"]
            benchmark_id, id_method = canonical_benchmark_id(raw_id, spec)
            if benchmark_id in source_seen_ids:
                raise HoldoutManifestError(
                    f"{spec.path} contains duplicate canonical benchmark_id "
                    f"{benchmark_id!r}"
                )
            source_seen_ids.add(benchmark_id)
            canonical_source = {**source_record, "benchmark_id": benchmark_id}
            membership = {
                "source_index": source_index,
                "record_index": record_index,
                "raw_benchmark_id": raw_id,
                "id_method": id_method,
            }
            existing = records_by_id.get(benchmark_id)
            if existing is not None:
                if existing["cohort"] != spec.cohort:
                    raise HoldoutManifestError(
                        f"duplicate benchmark_id {benchmark_id!r} occurs in "
                        f"cohorts {existing['cohort']!r} and {spec.cohort!r}"
                    )
                if not records_have_same_content(
                    existing,
                    canonical_source,
                    include_id=True,
                ):
                    if not allow_content_variants:
                        raise HoldoutManifestError(
                            f"benchmark_id {benchmark_id!r} content differs "
                            "between its source memberships"
                        )
                    variants = existing["content_variants"]
                    matching_variant = next(
                        (
                            variant
                            for variant in variants
                            if records_have_same_content(
                                variant,
                                canonical_source,
                                include_id=False,
                            )
                        ),
                        None,
                    )
                    if matching_variant is None:
                        matching_variant = {
                            "locality": canonical_source["locality"],
                            "locality_normalized": normalize_text(
                                canonical_source["locality"]
                            ),
                            "locality_holdout_keys": locality_holdout_keys(
                                canonical_source["locality"]
                            ),
                            "country": canonical_source["country"],
                            "country_normalized": normalize_text(
                                canonical_source["country"]
                            ),
                            "latitude": canonical_source["latitude"],
                            "longitude": canonical_source["longitude"],
                            "content_fingerprint": content_fingerprint(
                                canonical_source
                            ),
                            "source_memberships": [],
                        }
                        variants.append(matching_variant)
                    matching_variant["source_memberships"].append(membership)
                    continue
                existing["source_memberships"].append(membership)
                continue

            record = {
                "cohort": spec.cohort,
                **canonical_source,
                "locality_normalized": normalize_text(canonical_source["locality"]),
                "locality_holdout_keys": locality_holdout_keys(
                    canonical_source["locality"]
                ),
                "country_normalized": normalize_text(canonical_source["country"]),
                "content_fingerprint": content_fingerprint(canonical_source),
                "source_memberships": [membership],
                "content_variants": [],
            }
            records_by_id[benchmark_id] = record
            record_order.append(benchmark_id)

    records = [records_by_id[benchmark_id] for benchmark_id in record_order]

    if expected_record_count is not None and len(records) != expected_record_count:
        raise HoldoutManifestError(
            f"union contains {len(records)} records; "
            f"expected exactly {expected_record_count}"
        )

    cohort_counts: dict[str, int] = {}
    for record in records:
        cohort = record["cohort"]
        cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1
    cohort_counts = dict(sorted(cohort_counts.items()))
    benchmark_ids = sorted(records_by_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "source_files": source_files,
        "cohort_counts": cohort_counts,
        "record_count": len(records),
        "content_variant_count": sum(
            len(record["content_variants"]) for record in records
        ),
        "benchmark_ids_sha256": sha256_bytes(
            canonical_json_bytes(benchmark_ids)
        ),
        "records_sha256": sha256_bytes(canonical_json_bytes(records)),
        "records": records,
    }


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_fresh_atomic(path: Path, data: bytes) -> None:
    """Atomically publish data while refusing to replace an existing artifact."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise HoldoutManifestError(f"refusing to replace existing output: {path}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise HoldoutManifestError(
                f"refusing to replace existing output: {path}"
            ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_labeled_path(value: str, *, kind: str) -> SourceSpec:
    cohort, separator, raw_path = value.partition("=")
    if not separator or not cohort or not raw_path:
        raise argparse.ArgumentTypeError(
            "value must use COHORT=/path/to/source"
        )
    return SourceSpec(cohort=cohort, path=Path(raw_path), kind=kind)


def parse_source(value: str) -> SourceSpec:
    return parse_labeled_path(value, kind=ARCHIVE_KIND)


def parse_reference(value: str) -> SourceSpec:
    return parse_labeled_path(value, kind=REFERENCE_KIND)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        help=(
            "archived comparison as "
            "COHORT=/path/to/all_models_comparison_complete.json; repeatable"
        ),
    )
    parser.add_argument(
        "--reference",
        action="append",
        type=parse_reference,
        help=(
            "reference cohort as COHORT=/path/to/reference.tsv; repeatable; "
            "TSV fields are Barcode, label_verbatim, Country, lat_manual and "
            "long_manual"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-multiple-references", action="store_true",
        help=("explicitly union multiple reference TSVs in a cohort, retaining all "
              "source memberships; content differences still require --allow-content-variants"),
    )
    parser.add_argument(
        "--expected-record-count",
        type=int,
        default=DEFAULT_EXPECTED_RECORD_COUNT,
        help="fail unless the union has this many unique benchmark IDs",
    )
    parser.add_argument(
        "--allow-content-variants",
        action="store_true",
        help=(
            "preserve differing source versions under one benchmark ID; "
            "without this flag any difference fails closed"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_record_count <= 0:
        raise HoldoutManifestError("--expected-record-count must be positive")
    sources = [*(args.source or []), *(args.reference or [])]
    manifest = build_manifest(
        sources,
        expected_record_count=args.expected_record_count,
        allow_content_variants=args.allow_content_variants,
        allow_multiple_references=args.allow_multiple_references,
    )
    data = manifest_bytes(manifest)
    write_fresh_atomic(args.output, data)
    print(
        f"wrote {args.output}: {manifest['record_count']} records, "
        f"records_sha256={manifest['records_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
