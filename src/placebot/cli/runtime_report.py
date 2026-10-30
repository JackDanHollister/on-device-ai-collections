#!/usr/bin/env python3
"""Combine two benchmark summaries into an accuracy-run time projection."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ATTEMPTED_TARGETS = (709, 908, 1617, 6_000_000)
USABLE_TARGETS = (6_000_000,)


class RuntimeReportError(ValueError):
    """Raised when benchmark summaries cannot be combined safely."""


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeReportError(f"Could not read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeReportError(f"{path} must contain a JSON object")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: Any, description: str) -> float:
    if isinstance(value, bool):
        raise RuntimeReportError(f"{description} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeReportError(f"{description} must be a finite number") from error
    if not math.isfinite(number):
        raise RuntimeReportError(f"{description} must be a finite number")
    return number


def _count(value: Any, description: str) -> int:
    number = _finite_number(value, description)
    if number < 0 or not number.is_integer():
        raise RuntimeReportError(f"{description} must be a non-negative integer")
    return int(number)


def _first_count(
    metrics: Mapping[str, Any], names: Sequence[str], description: str
) -> Tuple[int, str]:
    for name in names:
        if metrics.get(name) is not None:
            return _count(metrics[name], description), name
    raise RuntimeReportError(
        f"Missing {description}; expected one of: {', '.join(names)}"
    )


def _count_from_rate(
    metrics: Mapping[str, Any], rate_name: str, attempted: int, description: str
) -> Optional[int]:
    if metrics.get(rate_name) is None:
        return None
    rate = _finite_number(metrics[rate_name], rate_name)
    if not 0.0 <= rate <= 1.0:
        raise RuntimeReportError(f"{rate_name} must be between zero and one")
    inferred = int(round(rate * attempted))
    if attempted and not math.isclose(
        inferred / attempted, rate, rel_tol=1e-7, abs_tol=1e-7
    ):
        raise RuntimeReportError(
            f"{description} cannot be reconstructed exactly from {rate_name}"
        )
    return inferred


def _identity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    run = _mapping(payload.get("run"))
    model = _mapping(run.get("model"))
    profile = _mapping(model.get("profile_reproducibility"))
    prompt = _mapping(run.get("prompt"))
    server = _mapping(run.get("server"))
    selected_model = _mapping(server.get("selected_model"))

    model_id = model.get("model_id") or selected_model.get("id")
    if not model_id:
        raise RuntimeReportError("Missing run.model.model_id")

    request_settings = profile.get("request_settings")
    if request_settings is not None and not isinstance(request_settings, Mapping):
        raise RuntimeReportError(
            "run.model.profile_reproducibility.request_settings must be an object"
        )

    return {
        "model_id": str(model_id),
        "model_root": selected_model.get("root"),
        "model_profile": model.get("profile"),
        "max_output_tokens": model.get("max_output_tokens"),
        "prompt_sha256": prompt.get("sha256"),
        "request_settings_recorded": request_settings is not None,
        "request_settings": (
            dict(request_settings) if request_settings is not None else None
        ),
    }


def _load_summary(path: Path) -> Dict[str, Any]:
    payload = _read_json_object(path)
    metrics = _mapping(payload.get("metrics"))
    if not metrics:
        raise RuntimeReportError(f"{path} has no metrics object")

    try:
        attempted, attempted_field = _first_count(
            metrics,
            ("completed_records", "attempted_records"),
            "attempted record count",
        )
    except RuntimeReportError:
        if payload.get("status") != "complete":
            raise RuntimeReportError(
                f"{path} is not complete and does not record attempted records"
            )
        attempted, attempted_field = _first_count(
            metrics, ("input_records",), "attempted record count"
        )
    try:
        api_success, api_field = _first_count(
            metrics, ("api_success_records",), "API-success record count"
        )
    except RuntimeReportError:
        inferred = _count_from_rate(
            metrics, "api_success_rate", attempted, "API-success record count"
        )
        if inferred is None:
            raise
        api_success, api_field = inferred, "api_success_rate (reconstructed)"

    usable_field = ""
    for candidate in (
        "usable_prediction_records",
        "valid_coordinate_records",
        "scored_records",
    ):
        if metrics.get(candidate) is not None:
            usable = _count(metrics[candidate], "usable prediction count")
            usable_field = candidate
            break
    else:
        usable = _count_from_rate(
            metrics,
            "usable_prediction_rate",
            attempted,
            "usable prediction count",
        )
        if usable is None:
            usable = _count_from_rate(
                metrics,
                "valid_coordinate_rate",
                attempted,
                "usable prediction count",
            )
            usable_field = "valid_coordinate_rate (reconstructed)"
        else:
            usable_field = "usable_prediction_rate (reconstructed)"
        if usable is None:
            raise RuntimeReportError(
                "Missing usable prediction count; expected usable_prediction_records, "
                "valid_coordinate_records, scored_records, or a reconstructable rate"
            )

    elapsed = _finite_number(metrics.get("elapsed_seconds"), "elapsed_seconds")
    if elapsed <= 0:
        raise RuntimeReportError("elapsed_seconds must be greater than zero")
    if api_success > attempted:
        raise RuntimeReportError(
            "API-success record count cannot exceed attempted records"
        )
    if usable > api_success:
        raise RuntimeReportError(
            "usable prediction count cannot exceed API-success records"
        )

    run = _mapping(payload.get("run"))
    input_metadata = _mapping(run.get("input"))
    return {
        "path": str(path.resolve()),
        "status": payload.get("status"),
        "summary_schema_version": payload.get("schema_version"),
        "run_schema_version": run.get("schema_version"),
        "input": {
            "path": input_metadata.get("path"),
            "sha256": input_metadata.get("sha256"),
        },
        "identity": _identity(payload),
        "counts": {
            "attempted": attempted,
            "api_success": api_success,
            "usable": usable,
        },
        "elapsed_seconds": elapsed,
        "legacy_fallbacks": {
            "attempted": attempted_field,
            "api_success": api_field,
            "usable": usable_field,
        },
    }


def _validate_compatible(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not summaries:
        raise RuntimeReportError("At least one benchmark summary is required")
    expected = summaries[0]["identity"]
    incompatible: List[str] = []
    for summary in summaries[1:]:
        identity = summary["identity"]
        for field in (
            "model_id",
            "model_root",
            "model_profile",
            "max_output_tokens",
            "prompt_sha256",
            "request_settings_recorded",
            "request_settings",
        ):
            if identity.get(field) != expected.get(field):
                incompatible.append(field)
    if incompatible:
        fields = ", ".join(sorted(set(incompatible)))
        raise RuntimeReportError(
            f"Benchmark summaries use incompatible model/request settings: {fields}"
        )
    return dict(expected)


def _throughput(count: int, elapsed_seconds: float) -> float:
    return count * 3600.0 / elapsed_seconds


def _duration(target: int, records_per_hour: float) -> Dict[str, Optional[float]]:
    if records_per_hour <= 0:
        return {"seconds": None, "hours": None, "days": None}
    hours = target / records_per_hour
    return {"seconds": hours * 3600.0, "hours": hours, "days": hours / 24.0}


def build_runtime_report(summary_paths: Sequence[Path]) -> Dict[str, Any]:
    """Load compatible summaries and combine their observed wall-clock throughput."""
    if len(summary_paths) != 2:
        raise RuntimeReportError(
            "Exactly two summaries are required (Bombus and Odonata)"
        )
    summaries = [_load_summary(Path(path)) for path in summary_paths]
    identity = _validate_compatible(summaries)

    elapsed = sum(float(summary["elapsed_seconds"]) for summary in summaries)
    attempted = sum(int(summary["counts"]["attempted"]) for summary in summaries)
    api_success = sum(int(summary["counts"]["api_success"]) for summary in summaries)
    usable = sum(int(summary["counts"]["usable"]) for summary in summaries)
    attempted_per_hour = _throughput(attempted, elapsed)
    api_success_per_hour = _throughput(api_success, elapsed)
    usable_per_hour = _throughput(usable, elapsed)
    request_settings_verified = bool(identity["request_settings_recorded"])

    return {
        "schema_version": 1,
        "model": identity,
        "compatibility": {
            "request_settings_verified": request_settings_verified,
            "warning": (
                None
                if request_settings_verified
                else "Legacy summaries do not record full request settings; "
                "compatibility is based on the available model metadata."
            ),
        },
        "sources": summaries,
        "combined": {
            "elapsed_seconds": elapsed,
            "attempted_records": attempted,
            "api_success_records": api_success,
            "usable_prediction_records": usable,
            "api_success_rate": api_success / attempted if attempted else None,
            "usable_prediction_rate": usable / attempted if attempted else None,
            "throughput_records_per_hour": {
                "attempted": attempted_per_hour,
                "api_success": api_success_per_hour,
                "usable": usable_per_hour,
            },
        },
        "projected_durations": {
            "attempted_records": {
                str(target): _duration(target, attempted_per_hour)
                for target in ATTEMPTED_TARGETS
            },
            "usable_predictions": {
                str(target): _duration(target, usable_per_hour)
                for target in USABLE_TARGETS
            },
        },
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _human_duration(duration: Mapping[str, Optional[float]]) -> str:
    seconds = duration.get("seconds")
    if seconds is None:
        return "unavailable (zero observed throughput)"
    seconds_int = int(round(seconds))
    days, remainder = divmod(seconds_int, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds_int = divmod(remainder, 60)
    if days:
        return f"{days:,}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {seconds_int:02d}s"
    return f"{minutes}m {seconds_int:02d}s"


def _print_human(report: Mapping[str, Any]) -> None:
    combined = report["combined"]
    throughput = combined["throughput_records_per_hour"]
    print(f"Model: {report['model']['model_id']}")
    print(
        "Observed: "
        f"{combined['attempted_records']:,} attempted, "
        f"{combined['api_success_records']:,} API-success, "
        f"{combined['usable_prediction_records']:,} usable in "
        f"{combined['elapsed_seconds']:.1f}s"
    )
    print("Throughput (records/hour):")
    print(f"  attempted   {throughput['attempted']:.2f}")
    print(f"  API-success {throughput['api_success']:.2f}")
    print(f"  usable      {throughput['usable']:.2f}")
    warning = report["compatibility"]["warning"]
    if warning:
        print(f"Warning: {warning}")
    print("Projected duration:")
    attempted = report["projected_durations"]["attempted_records"]
    for target in ATTEMPTED_TARGETS:
        print(f"  {target:,} attempted: {_human_duration(attempted[str(target)])}")
    usable = report["projected_durations"]["usable_predictions"]
    for target in USABLE_TARGETS:
        print(f"  {target:,} usable:    {_human_duration(usable[str(target)])}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="placebot-runtime-report",
        description=(
            "Combine compatible Bombus and Odonata summary.json files into "
            "observed throughput and runtime projections."
        ),
    )
    parser.add_argument("bombus_summary", type=Path)
    parser.add_argument("odonata_summary", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optionally write the full machine-readable report to this path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON instead of a human summary",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = build_runtime_report([args.bombus_summary, args.odonata_summary])
        if args.output_json:
            _atomic_write_json(args.output_json, report)
    except RuntimeReportError as error:
        raise SystemExit(f"placebot-runtime-report: error: {error}") from error

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
        if args.output_json:
            print(f"JSON report: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
