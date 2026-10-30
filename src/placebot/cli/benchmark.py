#!/usr/bin/env python3
"""Non-interactive, resumable PlaceBot reference benchmark runner."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import requests

from placebot.core.ai_processor import AIProcessor
from placebot.core.benchmarking import (
    compare_with_archived_results,
    file_sha256,
    load_reference_rows,
    normalise_model_label,
    result_fieldnames,
    safe_float,
    score_prediction,
    summarise_scored_rows,
)
from placebot.core.model_selector import load_model_profile

DEFAULT_MODEL_PROFILE = "nemotron_3_super_local"


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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _git_state(repo_root: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _models_endpoint(chat_endpoint: str) -> str:
    marker = "/v1/"
    if marker in chat_endpoint:
        return f"{chat_endpoint.split(marker, 1)[0]}/v1/models"
    return f"{chat_endpoint.rstrip('/')}/v1/models"


def _probe_model_server(model_config: Mapping[str, Any]) -> Dict[str, Any]:
    endpoint = str(model_config.get("api_endpoint") or "")
    if not endpoint:
        raise RuntimeError("The selected model profile has no API endpoint")
    module = model_config.get("module")
    headers = module.get_headers(model_config.get("api_key")) if module else {}
    url = _models_endpoint(endpoint)
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = [
        str(item.get("id"))
        for item in models
        if isinstance(item, dict) and item.get("id")
    ]
    wanted = str(model_config.get("model_id") or "")
    if wanted and wanted not in model_ids:
        raise RuntimeError(
            f"Model {wanted!r} is not served by {url}; available: {', '.join(model_ids)}"
        )
    selected = next(
        (
            item
            for item in models
            if isinstance(item, dict) and item.get("id") == wanted
        ),
        None,
    )
    if isinstance(selected, Mapping):
        live_context = safe_float(selected.get("max_model_len"))
        expected_context = safe_float(model_config.get("max_tokens"))
        max_output = safe_float(model_config.get("max_output_tokens"))
        if live_context is not None:
            if (
                str(model_config.get("type") or "").lower() == "local"
                and expected_context is not None
                and expected_context > live_context
            ):
                raise RuntimeError(
                    f"Profile expects context {int(expected_context)}, but the "
                    f"live model advertises only {int(live_context)} at {url}. "
                    "Relaunch the server with the required max model length."
                )
            if max_output is not None and max_output > live_context:
                raise RuntimeError(
                    f"Requested output cap {int(max_output)} exceeds live model "
                    f"context {int(live_context)} at {url}"
                )
    return {
        "models_endpoint": url,
        "available_model_ids": model_ids,
        "selected_model": selected,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="placebot-benchmark",
        description="Run a resumable PlaceBot model against a manual reference TSV/CSV.",
    )
    parser.add_argument(
        "input", type=Path, help="Reference TSV/CSV with manual coordinates"
    )
    parser.add_argument("--model-profile", default=DEFAULT_MODEL_PROFILE)
    parser.add_argument(
        "--endpoint", help="Override the profile chat-completions endpoint"
    )
    parser.add_argument(
        "--model-id", help="Override the served OpenAI-compatible model ID"
    )
    parser.add_argument("--label", help="Stable suffix used for result columns")
    parser.add_argument(
        "--output-dir", type=Path, help="Run directory (required when resuming)"
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        help="Archived comparison TSV for the same dataset",
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Zero-based first reference row"
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum number of records to process"
    )
    parser.add_argument(
        "--max-output-tokens", type=int, help="Override profile output-token limit"
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only records not already checkpointed",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate files/profile/server without inference",
    )
    return parser


def _prepare_model(args: argparse.Namespace) -> Dict[str, Any]:
    profile = load_model_profile(args.model_profile)
    if profile is None:
        raise RuntimeError(f"Unknown or invalid model profile: {args.model_profile}")
    module = profile.get("module")
    if args.endpoint:
        profile["api_endpoint"] = args.endpoint
    if args.model_id:
        profile["model_id"] = args.model_id
        if module is not None:
            setattr(module, "MODEL_ID", args.model_id)
    if args.max_output_tokens is not None:
        if args.max_output_tokens <= 0:
            raise ValueError("--max-output-tokens must be positive")
        profile["max_output_tokens"] = args.max_output_tokens
        if module is not None:
            # Profile formatters may enforce their module-level safety cap.
            # Keep that cap aligned so an explicit CLI override is effective,
            # not merely recorded in run metadata.
            setattr(module, "MAX_OUTPUT_TOKENS", args.max_output_tokens)
    return profile


def _profile_reproducibility(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Describe the profile source and its effective non-secret request settings."""
    module = profile.get("module")
    if module is None:
        return {}

    result: Dict[str, Any] = {"module": getattr(module, "__name__", "")}
    source = getattr(module, "__file__", None)
    if source:
        source_path = Path(source).resolve()
        result["source_path"] = str(source_path)
        if source_path.is_file():
            result["source_sha256"] = file_sha256(source_path)

    metadata_getter = getattr(module, "get_reproducibility_metadata", None)
    if callable(metadata_getter):
        settings = metadata_getter(int(profile.get("max_output_tokens") or 0))
        if not isinstance(settings, Mapping):
            raise TypeError("get_reproducibility_metadata() must return a mapping")
        result["request_settings"] = dict(settings)
    return result


def _code_reproducibility(repo_root: Path) -> Dict[str, str]:
    """Hash the runner, request, parsing, selection, and scoring implementation."""
    relative_paths = (
        "placebot/cli/benchmark.py",
        "placebot/core/ai_processor.py",
        "placebot/core/benchmarking.py",
        "placebot/core/model_selector.py",
    )
    return {
        relative_path: file_sha256(repo_root / relative_path)
        for relative_path in relative_paths
    }


def _baseline_metadata(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None:
        return None
    return {"path": str(path), "sha256": file_sha256(path)}


def _server_identity(server: Mapping[str, Any]) -> Dict[str, Any]:
    selected = server.get("selected_model")
    selected = selected if isinstance(selected, Mapping) else {}
    return {
        "models_endpoint": server.get("models_endpoint"),
        "model_id": selected.get("id"),
        "root": selected.get("root"),
        "max_model_len": selected.get("max_model_len"),
    }


def _resume_identity(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only fields that must remain invariant while appending a run."""
    input_metadata = metadata.get("input")
    input_metadata = input_metadata if isinstance(input_metadata, Mapping) else {}
    prompt_metadata = metadata.get("prompt")
    prompt_metadata = prompt_metadata if isinstance(prompt_metadata, Mapping) else {}
    model_metadata = metadata.get("model")
    model_metadata = model_metadata if isinstance(model_metadata, Mapping) else {}
    profile_metadata = model_metadata.get("profile_reproducibility")
    profile_metadata = profile_metadata if isinstance(profile_metadata, Mapping) else {}
    baseline_metadata = metadata.get("baseline")
    baseline_metadata = (
        baseline_metadata if isinstance(baseline_metadata, Mapping) else {}
    )
    server_metadata = metadata.get("server")
    server_metadata = server_metadata if isinstance(server_metadata, Mapping) else {}
    return {
        "input": {
            "sha256": input_metadata.get("sha256"),
            "selected_records": input_metadata.get("selected_records"),
            "offset": input_metadata.get("offset"),
            "limit": input_metadata.get("limit"),
        },
        "prompt_sha256": prompt_metadata.get("sha256"),
        "baseline_sha256": baseline_metadata.get("sha256"),
        "model": {
            "profile": model_metadata.get("profile"),
            "label": model_metadata.get("label"),
            "model_id": model_metadata.get("model_id"),
            "endpoint": model_metadata.get("endpoint"),
            "max_output_tokens": model_metadata.get("max_output_tokens"),
            "context_window": model_metadata.get("context_window"),
            "max_retries": model_metadata.get("max_retries"),
            "request_timeout": model_metadata.get("request_timeout"),
            "request_settings": profile_metadata.get("request_settings"),
            "profile_module": profile_metadata.get("module"),
            "profile_source_sha256": profile_metadata.get("source_sha256"),
        },
        "server": _server_identity(server_metadata),
        "code_sha256": metadata.get("code_sha256"),
    }


def _validate_resume_metadata(
    recorded: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    recorded_identity = _resume_identity(recorded)
    current_identity = _resume_identity(current)
    if recorded_identity == current_identity:
        return

    differences = [
        key
        for key in recorded_identity
        if recorded_identity.get(key) != current_identity.get(key)
    ]
    raise RuntimeError(
        "Cannot resume because run identity changed in: "
        + ", ".join(differences)
        + ". Start a new output directory."
    )


def _run_metadata(
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    input_path: Path,
    prompt_path: Path,
    server: Mapping[str, Any],
    selected_records: int,
    label: str,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 2,
        "started_at": _utc_now(),
        "placebot_version": version("placebot"),
        "git": _git_state(repo_root),
        "code_sha256": _code_reproducibility(repo_root),
        "python": sys.version,
        "input": {
            "path": str(input_path),
            "sha256": file_sha256(input_path),
            "selected_records": selected_records,
            "offset": args.offset,
            "limit": args.limit,
        },
        "prompt": {"path": str(prompt_path), "sha256": file_sha256(prompt_path)},
        "baseline": _baseline_metadata(args.baseline_results),
        "model": {
            "profile": args.model_profile,
            "label": label,
            "name": profile.get("name"),
            "provider": profile.get("provider"),
            "model_id": profile.get("model_id"),
            "endpoint": profile.get("api_endpoint"),
            "max_output_tokens": profile.get("max_output_tokens"),
            "context_window": profile.get("max_tokens"),
            "max_retries": profile.get("max_retries"),
            "request_timeout": profile.get("request_timeout"),
            "api_key_configured": bool(profile.get("api_key")),
            "profile_reproducibility": _profile_reproducibility(profile),
        },
        "server": server,
        "runner": {
            "checkpoint_every": args.checkpoint_every,
            "max_consecutive_failures": args.max_consecutive_failures,
        },
    }


def _load_and_validate_resume_rows(
    path: Path,
    expected_fieldnames: Sequence[str],
    selected: Sequence[tuple[int, Mapping[str, Any]]],
    reference_fields: Sequence[str],
    label: str,
) -> List[Dict[str, str]]:
    """Reject corrupt or incompatible checkpoints before deciding what to skip."""
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        actual_fieldnames = list(reader.fieldnames or [])
        if actual_fieldnames != list(expected_fieldnames):
            raise RuntimeError(
                "Cannot resume because records.tsv columns do not match this run"
            )
        rows = list(reader)

    selected_by_index = {index: reference for index, reference in selected}
    seen_indices = set()
    for row_number, row in enumerate(rows, 2):
        truncated_fields = [
            field for field in expected_fieldnames if row.get(field) is None
        ]
        if truncated_fields:
            raise RuntimeError(
                f"Cannot resume: records.tsv row {row_number} is truncated; "
                f"missing {truncated_fields[0]!r}"
            )

        index_value = safe_float(row.get("benchmark_index"))
        if index_value is None or not index_value.is_integer():
            raise RuntimeError(
                f"Cannot resume: records.tsv row {row_number} has an invalid benchmark_index"
            )
        index = int(index_value)
        if index not in selected_by_index:
            raise RuntimeError(
                f"Cannot resume: records.tsv row {row_number} index {index} is outside this selection"
            )
        if index in seen_indices:
            raise RuntimeError(
                f"Cannot resume: records.tsv contains duplicate benchmark_index {index}"
            )
        seen_indices.add(index)

        reference = selected_by_index[index]
        for field in reference_fields:
            expected = reference.get(field)
            expected_text = "" if expected is None else str(expected)
            actual_text = "" if row.get(field) is None else str(row.get(field))
            if actual_text != expected_text:
                raise RuntimeError(
                    f"Cannot resume: records.tsv row {row_number} field {field!r} "
                    "does not match the reference input"
                )

        for field in (f"api_success_{label}", f"success_{label}"):
            success = str(row.get(field) or "").lower()
            if success not in {"yes", "no"}:
                raise RuntimeError(
                    f"Cannot resume: records.tsv row {row_number} has an invalid "
                    f"flag in {field!r}"
                )
        latency = safe_float(row.get(f"latency_seconds_{label}"))
        if latency is None or latency < 0:
            raise RuntimeError(
                f"Cannot resume: records.tsv row {row_number} has an invalid latency"
            )
    return rows


def _write_checkpoint(
    run_dir: Path,
    metadata: Mapping[str, Any],
    rows: List[Mapping[str, Any]],
    label: str,
    total: int,
    elapsed: float,
    status: str,
    baseline_path: Optional[Path] = None,
) -> Dict[str, Any]:
    metrics = summarise_scored_rows(
        rows, label, total_input_records=total, elapsed_seconds=elapsed
    )
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "updated_at": _utc_now(),
        "run": metadata,
        "metrics": metrics,
    }
    if baseline_path and rows:
        payload["archived_comparison"] = compare_with_archived_results(
            baseline_path, rows, label, metrics
        )
    _atomic_json(
        run_dir / ("summary.json" if status == "complete" else "progress.json"), payload
    )
    return payload


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    seconds_i = max(0, int(seconds))
    hours, remainder = divmod(seconds_i, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    if args.max_consecutive_failures <= 0:
        parser.error("--max-consecutive-failures must be positive")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input does not exist: {input_path}")
    if args.baseline_results:
        args.baseline_results = args.baseline_results.expanduser().resolve()
        if not args.baseline_results.exists():
            parser.error(f"Baseline results do not exist: {args.baseline_results}")

    try:
        references, reference_fields = load_reference_rows(input_path)
        indexed = list(enumerate(references))
        selected = indexed[args.offset :]
        if args.limit is not None:
            selected = selected[: args.limit]
        if not selected:
            raise ValueError("The requested offset/limit selected no records")

        profile = _prepare_model(args)
        label = normalise_model_label(
            args.label or str(profile.get("model_id") or profile.get("name"))
        )
        server = _probe_model_server(profile)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else (
                Path.home()
                / ".placebot"
                / "benchmarks"
                / f"{timestamp}_{input_path.stem}_{label}"
            )
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        records_path = run_dir / "records.tsv"
        metadata_path = run_dir / "run.json"
        progress_path = run_dir / "progress.json"
        summary_path = run_dir / "summary.json"

        lock_handle = (run_dir / ".benchmark.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_handle.close()
            raise RuntimeError(
                f"Another benchmark process is already using {run_dir}"
            ) from exc

        if records_path.exists() and not args.resume:
            raise FileExistsError(
                f"Result file already exists; use --resume: {records_path}"
            )

        prompt_path = Path(str(files("placebot.data").joinpath("prompt.md"))).resolve()
        fieldnames = result_fieldnames(reference_fields, label)

        if args.resume:
            metadata = _read_json(metadata_path)
            if not metadata:
                raise RuntimeError(
                    f"Cannot resume without valid metadata: {metadata_path}"
                )
            current_metadata = _run_metadata(
                args,
                profile,
                input_path,
                prompt_path,
                server,
                len(selected),
                label,
            )
            _validate_resume_metadata(metadata, current_metadata)
            existing_rows = _load_and_validate_resume_rows(
                records_path, fieldnames, selected, reference_fields, label
            )
            state_paths = [
                path for path in (progress_path, summary_path) if path.exists()
            ]
            latest_state = (
                _read_json(max(state_paths, key=lambda path: path.stat().st_mtime_ns))
                if state_paths
                else {}
            )
            previous_elapsed = safe_float(
                latest_state.get("metrics", {}).get("elapsed_seconds")
            ) or sum(
                safe_float(row.get(f"latency_seconds_{label}")) or 0.0
                for row in existing_rows
            )
        else:
            metadata = _run_metadata(
                args,
                profile,
                input_path,
                prompt_path,
                server,
                len(selected),
                label,
            )
            _atomic_json(metadata_path, metadata)
            existing_rows = []
            previous_elapsed = 0.0

        existing_indices = {int(row["benchmark_index"]) for row in existing_rows}
        remaining = [
            (index, row) for index, row in selected if index not in existing_indices
        ]

        print(f"Run directory: {run_dir}")
        print(f"Model: {profile.get('model_id')} via {profile.get('api_endpoint')}")
        print(
            f"Records: {len(selected)} total, {len(existing_rows)} checkpointed, {len(remaining)} remaining"
        )
        print(f"Prompt SHA-256: {file_sha256(prompt_path)}")
        if args.check_only:
            print("Validation passed; no inference requested.")
            return 0
        if not remaining:
            elapsed = previous_elapsed
            _write_checkpoint(
                run_dir,
                metadata,
                existing_rows,
                label,
                len(selected),
                elapsed,
                "complete",
                args.baseline_results,
            )
            print("Benchmark was already complete; summary refreshed.")
            return 0

        mode = "a" if records_path.exists() else "w"
        all_rows: List[Mapping[str, Any]] = list(existing_rows)
        processor = AIProcessor(profile)
        started = time.monotonic()
        consecutive_failures = 0
        status = "running"

        with records_path.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
            )
            if mode == "w":
                writer.writeheader()
                handle.flush()
                os.fsync(handle.fileno())

            try:
                for position, (benchmark_index, reference) in enumerate(remaining, 1):
                    request_started = time.monotonic()
                    prediction = processor.process_locality(
                        str(reference.get("label_verbatim") or ""),
                        str(reference.get("Country") or ""),
                    )
                    latency = time.monotonic() - request_started
                    scored = score_prediction(
                        reference, prediction, label, benchmark_index, latency
                    )
                    writer.writerow(scored)
                    handle.flush()
                    if position % args.checkpoint_every == 0:
                        os.fsync(handle.fileno())
                    all_rows.append(scored)

                    if str(scored.get(f"success_{label}") or "").lower() == "yes":
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

                    elapsed = previous_elapsed + (time.monotonic() - started)
                    completed = len(all_rows)
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(selected) - completed) / rate if rate > 0 else None
                    distance = safe_float(scored.get(f"distance_km_{label}"))
                    distance_text = (
                        f"{distance:.2f} km" if distance is not None else "unscored"
                    )
                    print(
                        f"[{completed}/{len(selected)}] {reference.get('Barcode')} | "
                        f"{distance_text} | {latency:.2f}s | ETA {_format_eta(eta)}",
                        flush=True,
                    )

                    if position % args.checkpoint_every == 0:
                        _write_checkpoint(
                            run_dir,
                            metadata,
                            all_rows,
                            label,
                            len(selected),
                            elapsed,
                            "running",
                        )
                    if consecutive_failures >= args.max_consecutive_failures:
                        status = "failed"
                        raise RuntimeError(
                            f"Aborting after {consecutive_failures} consecutive model failures; resume after fixing the server"
                        )
            except KeyboardInterrupt:
                status = "interrupted"
            except Exception:
                status = "failed"
                raise
            finally:
                handle.flush()
                os.fsync(handle.fileno())
                elapsed = previous_elapsed + (time.monotonic() - started)
                if status != "failed":
                    status = "complete" if len(all_rows) == len(selected) else status
                final_payload = _write_checkpoint(
                    run_dir,
                    metadata,
                    all_rows,
                    label,
                    len(selected),
                    elapsed,
                    status,
                    args.baseline_results if status == "complete" else None,
                )

        metrics = final_payload["metrics"]
        print(
            f"Status: {status}; valid coordinates {metrics['valid_coordinate_records']}/"
            f"{metrics['completed_records']}; median error {metrics['median_distance_km']} km"
        )
        print(
            f"Summary: {run_dir / ('summary.json' if status == 'complete' else 'progress.json')}"
        )
        return 0 if status == "complete" else 130
    except Exception as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
