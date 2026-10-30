"""Reproducible scoring helpers for PlaceBot reference benchmarks.

The paper-era benchmark files pair verbatim locality strings with manually
curated coordinates and uncertainty radii.  This module keeps the scoring
rules independent from any particular model runner so they can be tested
without making API calls.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REFERENCE_COLUMNS = (
    "Barcode",
    "label_verbatim",
    "Country",
    "lat_manual",
    "long_manual",
    "radius_km_manual",
)


def safe_float(value: Any) -> Optional[float]:
    """Return a finite float, or None for missing/invalid input."""
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def valid_coordinates(latitude: Any, longitude: Any) -> bool:
    """Return whether latitude/longitude form a valid decimal-degree pair."""
    lat = safe_float(latitude)
    lon = safe_float(longitude)
    return (
        lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180
    )


def haversine_distance_km(
    lat1: Any, lon1: Any, lat2: Any, lon2: Any
) -> Optional[float]:
    """Return great-circle distance in kilometres for two coordinate pairs."""
    if not valid_coordinates(lat1, lon1) or not valid_coordinates(lat2, lon2):
        return None
    lat1_f, lon1_f = math.radians(float(lat1)), math.radians(float(lon1))
    lat2_f, lon2_f = math.radians(float(lat2)), math.radians(float(lon2))
    dlat = lat2_f - lat1_f
    dlon = lon2_f - lon1_f
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_f) * math.cos(lat2_f) * math.sin(dlon / 2) ** 2
    )
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def normalise_model_label(value: str) -> str:
    """Convert a model name/ID into a stable TSV column suffix."""
    label = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not label:
        raise ValueError("Model label must contain at least one letter or number")
    return label


def result_fieldnames(reference_fields: Sequence[str], label: str) -> List[str]:
    """Return the deterministic column order for a scored benchmark run."""
    suffixes = [
        "benchmark_index",
        f"latitude_{label}",
        f"longitude_{label}",
        f"ai_radius_km_{label}",
        f"ai_confidence_{label}",
        f"country_processed_{label}",
        f"state_{label}",
        f"region_{label}",
        f"sector_{label}",
        f"exact_site_{label}",
        f"coordinate_source_{label}",
        f"elevation_meters_{label}",
        f"notes_{label}",
        f"collection_notes_{label}",
        f"distance_km_{label}",
        f"within_ai_radius_{label}",
        f"within_manual_radius_{label}",
        f"within_combined_uncertainty_{label}",
        f"api_success_{label}",
        f"success_{label}",
        f"error_{label}",
        f"latency_seconds_{label}",
        f"input_tokens_{label}",
        f"output_tokens_{label}",
        f"total_tokens_{label}",
        f"token_source_{label}",
        f"cached_tokens_{label}",
        f"reasoning_tokens_{label}",
        f"final_output_tokens_{label}",
        f"reasoning_token_source_{label}",
        f"response_token_ids_{label}",
        f"output_cap_tokens_{label}",
        f"finish_reason_{label}",
        f"stop_reason_{label}",
        f"truncated_{label}",
        f"reasoning_content_{label}",
        f"response_content_{label}",
    ]
    return list(reference_fields) + suffixes


def _yes_no(value: Optional[bool]) -> str:
    if value is None:
        return ""
    return "yes" if value else "no"


def score_prediction(
    reference: Mapping[str, Any],
    prediction: Mapping[str, Any],
    label: str,
    benchmark_index: int,
    latency_seconds: float,
) -> Dict[str, Any]:
    """Combine one reference row and PlaceBot prediction with accuracy fields."""
    result: Dict[str, Any] = dict(reference)
    lat = safe_float(prediction.get("latitude"))
    lon = safe_float(prediction.get("longitude"))
    ai_radius_m = safe_float(prediction.get("coordinate_radius_meters"))
    ai_radius_km = (
        ai_radius_m / 1000 if ai_radius_m is not None and ai_radius_m >= 0 else None
    )
    manual_radius_km = safe_float(reference.get("radius_km_manual"))

    distance = haversine_distance_km(
        reference.get("lat_manual"),
        reference.get("long_manual"),
        lat,
        lon,
    )
    within_ai = (
        distance <= ai_radius_km
        if distance is not None and ai_radius_km is not None
        else None
    )
    within_manual = (
        distance <= manual_radius_km
        if distance is not None and manual_radius_km is not None
        else None
    )
    within_combined = (
        distance <= ai_radius_km + manual_radius_km
        if distance is not None
        and ai_radius_km is not None
        and manual_radius_km is not None
        else None
    )

    # Transport/API success is distinct from a usable parsed prediction. Keep
    # backward compatibility with older callers that only provide ``success``.
    api_success_value = prediction.get("api_success")
    api_success = (
        bool(prediction.get("success"))
        if api_success_value is None
        else bool(api_success_value)
    )
    usable_prediction = api_success and valid_coordinates(lat, lon)
    error = str(prediction.get("error") or "")
    if api_success and not usable_prediction and not error:
        error = "Model response contained no valid coordinate pair"

    result.update(
        {
            "benchmark_index": benchmark_index,
            f"latitude_{label}": lat,
            f"longitude_{label}": lon,
            f"ai_radius_km_{label}": ai_radius_km,
            f"ai_confidence_{label}": prediction.get("confidence", ""),
            f"country_processed_{label}": prediction.get("country", ""),
            f"state_{label}": prediction.get("state", ""),
            f"region_{label}": prediction.get("region", ""),
            f"sector_{label}": prediction.get("sector", ""),
            f"exact_site_{label}": prediction.get("exact_site", ""),
            f"coordinate_source_{label}": prediction.get("coordinate_source", ""),
            f"elevation_meters_{label}": safe_float(prediction.get("elevation_meters")),
            f"notes_{label}": prediction.get("notes", ""),
            f"collection_notes_{label}": prediction.get("collection_notes", ""),
            f"distance_km_{label}": distance,
            f"within_ai_radius_{label}": _yes_no(within_ai),
            f"within_manual_radius_{label}": _yes_no(within_manual),
            f"within_combined_uncertainty_{label}": _yes_no(within_combined),
            f"api_success_{label}": _yes_no(api_success),
            f"success_{label}": _yes_no(usable_prediction),
            f"error_{label}": error,
            f"latency_seconds_{label}": round(latency_seconds, 6),
            f"input_tokens_{label}": int(
                safe_float(prediction.get("input_tokens")) or 0
            ),
            f"output_tokens_{label}": int(
                safe_float(prediction.get("output_tokens")) or 0
            ),
            f"total_tokens_{label}": int(
                safe_float(prediction.get("total_tokens")) or 0
            ),
            f"token_source_{label}": prediction.get("token_source", ""),
            f"cached_tokens_{label}": int(
                safe_float(prediction.get("cached_tokens")) or 0
            ),
            f"reasoning_tokens_{label}": (
                int(reasoning_tokens)
                if (reasoning_tokens := safe_float(prediction.get("reasoning_tokens")))
                is not None
                else ""
            ),
            f"final_output_tokens_{label}": (
                int(final_output_tokens)
                if (
                    final_output_tokens := safe_float(
                        prediction.get("final_output_tokens")
                    )
                )
                is not None
                else ""
            ),
            f"reasoning_token_source_{label}": prediction.get(
                "reasoning_token_source", ""
            ) or "",
            f"response_token_ids_{label}": (
                json.dumps(response_token_ids, separators=(",", ":"))
                if isinstance(
                    response_token_ids := prediction.get("response_token_ids"), list
                )
                else ""
            ),
            f"output_cap_tokens_{label}": (
                int(output_cap_tokens)
                if (
                    output_cap_tokens := safe_float(
                        prediction.get("output_cap_tokens")
                    )
                )
                is not None
                else ""
            ),
            f"finish_reason_{label}": prediction.get("finish_reason", ""),
            f"stop_reason_{label}": prediction.get("stop_reason", ""),
            f"truncated_{label}": _yes_no(prediction.get("truncated")),
            f"reasoning_content_{label}": prediction.get("reasoning_content", ""),
            f"response_content_{label}": prediction.get("response_content", ""),
        }
    )
    return result


def load_reference_rows(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    """Load and validate a CSV/TSV reference dataset."""
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = list(reader.fieldnames or [])
        missing = [name for name in REFERENCE_COLUMNS if name not in fields]
        if missing:
            raise ValueError(
                f"Reference dataset is missing columns: {', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Reference dataset has no records")
    return rows, fields


def load_tsv_rows(path: Path) -> List[Dict[str, str]]:
    """Load a benchmark result TSV, returning an empty list when absent."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def file_sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _rate_from_yes_no(values: Iterable[Any]) -> Optional[float]:
    observed = [str(value).lower() for value in values]
    if not observed:
        return None
    return sum(value == "yes" for value in observed) / len(observed)


def _distance_metrics(
    distances: Sequence[float], total_records: Optional[int] = None
) -> Dict[str, Optional[float]]:
    denominator = total_records if total_records is not None else len(distances)
    if not distances:
        return {
            "mean_distance_km": None,
            "median_distance_km": None,
            "p90_distance_km": None,
            "p95_distance_km": None,
            "within_1_km_rate": 0.0 if denominator else None,
            "within_5_km_rate": 0.0 if denominator else None,
            "within_10_km_rate": 0.0 if denominator else None,
            "within_50_km_rate": 0.0 if denominator else None,
            "within_100_km_rate": 0.0 if denominator else None,
        }
    return {
        "mean_distance_km": statistics.fmean(distances),
        "median_distance_km": statistics.median(distances),
        "p90_distance_km": _percentile(distances, 0.90),
        "p95_distance_km": _percentile(distances, 0.95),
        "within_1_km_rate": sum(value <= 1 for value in distances) / denominator,
        "within_5_km_rate": sum(value <= 5 for value in distances) / denominator,
        "within_10_km_rate": sum(value <= 10 for value in distances) / denominator,
        "within_50_km_rate": sum(value <= 50 for value in distances) / denominator,
        "within_100_km_rate": sum(value <= 100 for value in distances) / denominator,
    }


def summarise_scored_rows(
    rows: Sequence[Mapping[str, Any]],
    label: str,
    total_input_records: Optional[int] = None,
    elapsed_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarise quality, reliability, latency, and token usage for a run."""
    total_input = total_input_records if total_input_records is not None else len(rows)
    lat_field = f"latitude_{label}"
    lon_field = f"longitude_{label}"
    distance_field = f"distance_km_{label}"
    api_success_field = f"api_success_{label}"
    success_field = f"success_{label}"
    confidence_field = f"ai_confidence_{label}"
    source_field = f"coordinate_source_{label}"

    distances = [
        value
        for row in rows
        if (value := safe_float(row.get(distance_field))) is not None
    ]
    valid_count = sum(
        valid_coordinates(row.get(lat_field), row.get(lon_field)) for row in rows
    )
    api_success_count = sum(
        str(row.get(api_success_field, "")).lower() == "yes" for row in rows
    )
    usable_prediction_count = sum(
        str(row.get(success_field, "")).lower() == "yes" for row in rows
    )
    latencies = [
        value
        for row in rows
        if (value := safe_float(row.get(f"latency_seconds_{label}"))) is not None
    ]
    errors = Counter(
        str(row.get(f"error_{label}") or "unknown error")
        for row in rows
        if str(row.get(success_field, "")).lower() != "yes"
    )
    reasoning_token_values = [
        int(value)
        for row in rows
        if (
            value := safe_float(row.get(f"reasoning_tokens_{label}"))
        ) is not None
    ]
    final_output_token_values = [
        int(value)
        for row in rows
        if (
            value := safe_float(row.get(f"final_output_tokens_{label}"))
        ) is not None
    ]

    summary: Dict[str, Any] = {
        "input_records": total_input,
        "completed_records": len(rows),
        "completion_rate": len(rows) / total_input if total_input else 0,
        "api_success_records": api_success_count,
        "api_success_rate": api_success_count / len(rows) if rows else 0,
        "usable_prediction_records": usable_prediction_count,
        "usable_prediction_rate": (usable_prediction_count / len(rows) if rows else 0),
        "valid_coordinate_records": valid_count,
        "valid_coordinate_rate": valid_count / len(rows) if rows else 0,
        "scored_records": len(distances),
        "within_ai_radius_rate": _rate_from_yes_no(
            row.get(f"within_ai_radius_{label}") for row in rows
        ),
        "within_manual_radius_rate": _rate_from_yes_no(
            row.get(f"within_manual_radius_{label}") for row in rows
        ),
        "within_combined_uncertainty_rate": _rate_from_yes_no(
            row.get(f"within_combined_uncertainty_{label}") for row in rows
        ),
        "confidence_counts": dict(
            Counter(str(row.get(confidence_field) or "unknown").lower() for row in rows)
        ),
        "coordinate_source_counts": dict(
            Counter(str(row.get(source_field) or "unknown").lower() for row in rows)
        ),
        "error_counts": dict(errors.most_common()),
        "total_input_tokens": sum(
            int(safe_float(row.get(f"input_tokens_{label}")) or 0) for row in rows
        ),
        "total_output_tokens": sum(
            int(safe_float(row.get(f"output_tokens_{label}")) or 0) for row in rows
        ),
        "total_tokens": sum(
            int(safe_float(row.get(f"total_tokens_{label}")) or 0) for row in rows
        ),
        "cached_tokens": sum(
            int(safe_float(row.get(f"cached_tokens_{label}")) or 0) for row in rows
        ),
        "total_reasoning_tokens": (
            sum(reasoning_token_values) if reasoning_token_values else None
        ),
        "total_final_output_tokens": (
            sum(final_output_token_values) if final_output_token_values else None
        ),
        "reasoning_tokens_observed_records": len(reasoning_token_values),
        "reasoning_tokens_observed_rate": (
            len(reasoning_token_values) / len(rows) if rows else 0
        ),
        "final_output_tokens_observed_records": len(final_output_token_values),
        "final_output_tokens_observed_rate": (
            len(final_output_token_values) / len(rows) if rows else 0
        ),
        "reasoning_token_source_counts": dict(
            Counter(
                str(row.get(f"reasoning_token_source_{label}") or "unavailable")
                for row in rows
            )
        ),
        "finish_reason_counts": dict(
            Counter(str(row.get(f"finish_reason_{label}") or "unknown") for row in rows)
        ),
        "truncated_records": sum(
            str(row.get(f"truncated_{label}") or "").lower() == "yes"
            for row in rows
        ),
    }
    summary["truncated_rate"] = (
        summary["truncated_records"] / len(rows) if rows else 0
    )
    summary.update(_distance_metrics(distances, len(rows)))

    if latencies:
        summary.update(
            {
                "mean_latency_seconds": statistics.fmean(latencies),
                "median_latency_seconds": statistics.median(latencies),
                "p95_latency_seconds": _percentile(latencies, 0.95),
            }
        )
    else:
        summary.update(
            {
                "mean_latency_seconds": None,
                "median_latency_seconds": None,
                "p95_latency_seconds": None,
            }
        )
    if elapsed_seconds is not None:
        summary["elapsed_seconds"] = elapsed_seconds
        if elapsed_seconds > 0:
            attempted_records_per_hour = len(rows) * 3600 / elapsed_seconds
            summary.update(
                {
                    # Keep records_per_hour as a compatibility alias for the
                    # explicitly named attempted-record throughput.
                    "records_per_hour": attempted_records_per_hour,
                    "attempted_records_per_hour": attempted_records_per_hour,
                    "api_success_records_per_hour": (
                        api_success_count * 3600 / elapsed_seconds
                    ),
                    "usable_predictions_per_hour": (
                        usable_prediction_count * 3600 / elapsed_seconds
                    ),
                    "mean_seconds_per_record": (
                        elapsed_seconds / len(rows) if rows else None
                    ),
                    "input_tokens_per_second": (
                        summary["total_input_tokens"] / elapsed_seconds
                    ),
                    "output_tokens_per_second": (
                        summary["total_output_tokens"] / elapsed_seconds
                    ),
                    "total_tokens_per_second": (
                        summary["total_tokens"] / elapsed_seconds
                    ),
                }
            )
        else:
            summary.update(
                {
                    "records_per_hour": None,
                    "attempted_records_per_hour": None,
                    "api_success_records_per_hour": None,
                    "usable_predictions_per_hour": None,
                    "mean_seconds_per_record": None,
                    "input_tokens_per_second": None,
                    "output_tokens_per_second": None,
                    "total_tokens_per_second": None,
                }
            )

    calibration: Dict[str, Any] = {}
    for confidence in ("high", "medium", "low"):
        group = [
            row
            for row in rows
            if str(row.get(confidence_field, "")).lower() == confidence
        ]
        if group:
            group_distances = [
                value
                for row in group
                if (value := safe_float(row.get(distance_field))) is not None
            ]
            calibration[confidence] = {
                "records": len(group),
                "median_distance_km": (
                    statistics.median(group_distances) if group_distances else None
                ),
                "within_manual_radius_rate": _rate_from_yes_no(
                    row.get(f"within_manual_radius_{label}") for row in group
                ),
            }
    summary["confidence_calibration"] = calibration
    return summary


def _discover_archived_models(fieldnames: Sequence[str]) -> List[str]:
    prefix = "distance_km_"
    return sorted(name[len(prefix) :] for name in fieldnames if name.startswith(prefix))


def compare_with_archived_results(
    archive_path: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    candidate_label: str,
    candidate_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare a candidate against archived models on exactly the same Barcodes."""
    if not candidate_rows:
        return {"models": {}, "matched_records": 0}
    with archive_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        all_archive_rows = list(reader)

    candidate_counts = Counter(str(row.get("Barcode", "")) for row in candidate_rows)
    archive_counts = Counter(str(row.get("Barcode", "")) for row in all_archive_rows)
    candidate_by_id = {
        str(row.get("Barcode", "")): row
        for row in candidate_rows
        if candidate_counts[str(row.get("Barcode", ""))] == 1
    }
    archive_by_id = {
        str(row.get("Barcode", "")): row
        for row in all_archive_rows
        if archive_counts[str(row.get("Barcode", ""))] == 1
    }
    matched_ids = set(candidate_by_id) & set(archive_by_id)
    candidate_matched_rows = [
        row for row in candidate_rows if str(row.get("Barcode", "")) in matched_ids
    ]
    archive_rows = [
        archive_by_id[str(row.get("Barcode", ""))] for row in candidate_matched_rows
    ]

    mismatch_counts: Dict[str, int] = {}
    mismatch_samples: Dict[str, List[str]] = {}
    for field in REFERENCE_COLUMNS:
        mismatched_ids: List[str] = []
        for barcode in sorted(matched_ids):
            candidate_value = candidate_by_id[barcode].get(field)
            archive_value = archive_by_id[barcode].get(field)
            if field in {"lat_manual", "long_manual", "radius_km_manual"}:
                candidate_number = safe_float(candidate_value)
                archive_number = safe_float(archive_value)
                matches = (
                    candidate_number == archive_number
                    if candidate_number is not None and archive_number is not None
                    else candidate_number is None and archive_number is None
                )
            else:
                matches = str(candidate_value or "") == str(archive_value or "")
            if not matches:
                mismatched_ids.append(barcode)
        mismatch_counts[field] = len(mismatched_ids)
        mismatch_samples[field] = mismatched_ids[:10]

    prompt_input_parity = not (
        mismatch_counts["label_verbatim"] or mismatch_counts["Country"]
    )
    scoring_reference_parity = not any(
        mismatch_counts[field]
        for field in ("lat_manual", "long_manual", "radius_km_manual")
    )

    if len(candidate_matched_rows) == len(candidate_rows):
        matched_candidate_summary = dict(candidate_summary)
    else:
        matched_candidate_summary = summarise_scored_rows(
            candidate_matched_rows,
            candidate_label,
            total_input_records=len(candidate_matched_rows),
        )
    models: Dict[str, Dict[str, Any]] = {}
    for model in _discover_archived_models(fieldnames):
        distances = [
            value
            for row in archive_rows
            if (value := safe_float(row.get(f"distance_km_{model}"))) is not None
        ]
        model_summary: Dict[str, Any] = {
            "records": len(archive_rows),
            "scored_records": len(distances),
            "valid_coordinate_rate": (
                len(distances) / len(archive_rows) if archive_rows else 0
            ),
            "within_ai_radius_rate": _rate_from_yes_no(
                row.get(f"within_ai_radius_{model}") for row in archive_rows
            ),
            "within_manual_radius_rate": _rate_from_yes_no(
                row.get(f"within_manual_radius_{model}") for row in archive_rows
            ),
            "within_combined_uncertainty_rate": _rate_from_yes_no(
                row.get(f"within_combined_uncertainty_{model}") for row in archive_rows
            ),
        }
        model_summary.update(_distance_metrics(distances, len(archive_rows)))
        models[model] = model_summary

    models[candidate_label] = {
        key: matched_candidate_summary.get(key)
        for key in (
            "scored_records",
            "valid_coordinate_rate",
            "mean_distance_km",
            "median_distance_km",
            "p90_distance_km",
            "p95_distance_km",
            "within_1_km_rate",
            "within_5_km_rate",
            "within_10_km_rate",
            "within_50_km_rate",
            "within_100_km_rate",
            "within_ai_radius_rate",
            "within_manual_radius_rate",
            "within_combined_uncertainty_rate",
        )
    }

    def rank(metric: str, reverse: bool) -> Optional[int]:
        candidate_value = safe_float(models[candidate_label].get(metric))
        if candidate_value is None:
            return None
        values = [
            value
            for name, metrics in models.items()
            if name != candidate_label
            and (value := safe_float(metrics.get(metric))) is not None
        ]
        better = (
            sum(value > candidate_value for value in values)
            if reverse
            else sum(value < candidate_value for value in values)
        )
        return better + 1

    comparison: Dict[str, Any] = {
        "archive_path": str(archive_path),
        "matched_records": len(archive_rows),
        "cohort_audit": {
            "candidate_records": len(candidate_rows),
            "archive_records": len(all_archive_rows),
            "matched_unique_records": len(archive_rows),
            "candidate_only_records": len(set(candidate_counts) - set(archive_counts)),
            "archive_only_records": len(set(archive_counts) - set(candidate_counts)),
            "duplicate_candidate_barcodes": {
                barcode: count
                for barcode, count in candidate_counts.items()
                if count > 1
            },
            "duplicate_archive_barcodes": {
                barcode: count for barcode, count in archive_counts.items() if count > 1
            },
            "reference_field_mismatch_counts": mismatch_counts,
            "reference_field_mismatch_samples": mismatch_samples,
            "prompt_input_parity": prompt_input_parity,
            "scoring_reference_parity": scoring_reference_parity,
        },
        "models": models,
        "candidate_ranks": (
            {
                "median_distance_km": rank("median_distance_km", reverse=False),
                "within_manual_radius_rate": rank(
                    "within_manual_radius_rate", reverse=True
                ),
                "within_combined_uncertainty_rate": rank(
                    "within_combined_uncertainty_rate", reverse=True
                ),
            }
            if prompt_input_parity and scoring_reference_parity
            else None
        ),
    }
    if not prompt_input_parity or not scoring_reference_parity:
        comparison["comparison_warning"] = (
            "Archived ranks and deltas are suppressed because candidate and archive "
            "reference inputs are not identical."
        )
    local_baseline = models.get("qwen3_14b_local")
    if local_baseline and prompt_input_parity and scoring_reference_parity:
        candidate_median = safe_float(
            matched_candidate_summary.get("median_distance_km")
        )
        baseline_median = safe_float(local_baseline.get("median_distance_km"))
        candidate_within = safe_float(
            matched_candidate_summary.get("within_manual_radius_rate")
        )
        baseline_within = safe_float(local_baseline.get("within_manual_radius_rate"))
        comparison["qwen3_14b_local_delta"] = {
            "median_distance_km": (
                candidate_median - baseline_median
                if candidate_median is not None and baseline_median is not None
                else None
            ),
            "within_manual_radius_percentage_points": (
                (candidate_within - baseline_within) * 100
                if candidate_within is not None and baseline_within is not None
                else None
            ),
        }
    return comparison
