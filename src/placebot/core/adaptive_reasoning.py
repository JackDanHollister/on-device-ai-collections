"""Durable two-pass adaptive-reasoning workflow primitives.

The engine is model-agnostic.  A caller supplies one request function and two
explicit request-setting dictionaries.  Every record is checkpointed in the
fast pass before routing is frozen; only then can the serial full pass begin.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = 1
ROUTER_RULE_ID = "nemotron_low_confidence_or_unusable_v1"
MERGE_RULE_ID = "nemotron_full_else_fast_fallback_v1"
REASONING_MODES = ("fast", "full")


class AdaptiveReasoningError(RuntimeError):
    """Raised when an adaptive run cannot safely continue."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveReasoningError(f"Cannot read {path}") from exc
    if not isinstance(value, dict):
        raise AdaptiveReasoningError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise AdaptiveReasoningError(
                        f"{path}:{line_number} is not a JSON object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveReasoningError(f"Cannot read {path}") from exc
    return rows


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise AdaptiveReasoningError(f"Refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_once(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise AdaptiveReasoningError(f"Refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            for value in values:
                handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite_coordinate(value: Any, minimum: float, maximum: float) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        coordinate = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(coordinate) and minimum <= coordinate <= maximum


def is_usable_attempt(attempt: Mapping[str, Any]) -> bool:
    """Return whether an attempt has a successful, valid coordinate pair."""
    return (
        attempt.get("success") is True
        and _finite_coordinate(attempt.get("latitude"), -90.0, 90.0)
        and _finite_coordinate(attempt.get("longitude"), -180.0, 180.0)
    )


def route_fast_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen, reference-free v1 routing rule."""
    reasons: list[str] = []
    if not is_usable_attempt(attempt):
        reasons.append("fast_unusable")
    if attempt.get("confidence") == "low":
        reasons.append("confidence_low")
    return {
        "route": bool(reasons),
        "route_reasons": reasons,
        "router_rule_id": ROUTER_RULE_ID,
    }


def merge_attempts(
    fast_attempt: Mapping[str, Any],
    route_decision: Mapping[str, Any],
    full_attempt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the frozen v1 selection and fallback rule."""
    routed = route_decision.get("route") is True
    fast_usable = is_usable_attempt(fast_attempt)
    full_usable = (
        full_attempt is not None and is_usable_attempt(full_attempt)
    )

    if not routed and fast_usable:
        return {"final_source": "fast", "resolved": True}
    if routed and full_usable:
        return {"final_source": "full", "resolved": True}
    if routed and fast_usable:
        return {"final_source": "fast_fallback", "resolved": True}
    return {"final_source": "unresolved", "resolved": False}


class AdaptiveReasoningRunner:
    """Checkpoint and resume a frozen fast-route-full-merge workflow."""

    def __init__(
        self,
        *,
        output_root: Path,
        records: Sequence[Mapping[str, Any]],
        run_identity: Mapping[str, Any],
        request_settings: Mapping[str, Mapping[str, Any]],
        request_fn: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
    ) -> None:
        self.output_root = Path(output_root)
        self.records = [dict(record) for record in records]
        self.run_identity = dict(run_identity)
        self.request_settings = {
            mode: dict(request_settings[mode]) for mode in REASONING_MODES
        }
        self.request_fn = request_fn
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if not self.records:
            raise AdaptiveReasoningError("At least one input record is required")
        seen: set[str] = set()
        for position, record in enumerate(self.records):
            record_id = record.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise AdaptiveReasoningError(
                    f"Input position {position} has no non-empty record_id"
                )
            if record_id in seen:
                raise AdaptiveReasoningError(f"Duplicate record_id: {record_id}")
            seen.add(record_id)
        for mode in REASONING_MODES:
            if self.request_settings[mode].get("reasoning_mode") != mode:
                raise AdaptiveReasoningError(
                    f"Request settings for {mode} do not identify that mode"
                )

    def _manifest_identity(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "adaptive_reasoning_run",
            "router_rule_id": ROUTER_RULE_ID,
            "merge_rule_id": MERGE_RULE_ID,
            "run_identity": self.run_identity,
            "request_settings": self.request_settings,
            "records": [
                {
                    "record_id": record["record_id"],
                    "cohort": record.get("cohort"),
                    "input_index": record.get("input_index", position),
                }
                for position, record in enumerate(self.records)
            ],
        }

    def _ensure_manifest(self) -> None:
        path = self.output_root / "run.json"
        identity = self._manifest_identity()
        digest = sha256_bytes(canonical_json_bytes(identity))
        if path.exists():
            existing = _read_json(path)
            if existing.get("manifest_identity_sha256") != digest:
                raise AdaptiveReasoningError(
                    "Existing run.json does not match this run identity"
                )
            return
        _write_json_once(
            path,
            {
                **identity,
                "manifest_identity_sha256": digest,
                "created_at": utc_now(),
            },
        )

    def _progress(self, stage: str, **counts: Any) -> None:
        _replace_json(
            self.output_root / "progress.json",
            {"stage": stage, "updated_at": utc_now(), **counts},
        )

    @staticmethod
    def _indexed_rows(
        path: Path,
        *,
        expected_mode: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(path):
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise AdaptiveReasoningError(f"Invalid record_id in {path}")
            if record_id in indexed:
                raise AdaptiveReasoningError(
                    f"Duplicate record_id {record_id} in {path}"
                )
            if expected_mode is not None and row.get("reasoning_mode") != expected_mode:
                raise AdaptiveReasoningError(
                    f"Unexpected reasoning mode for {record_id} in {path}"
                )
            indexed[record_id] = row
        return indexed

    def _run_pass(
        self,
        *,
        mode: str,
        selected_records: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        path = self.output_root / f"{mode}_results.jsonl"
        indexed = self._indexed_rows(path, expected_mode=mode)
        selected_ids = {record["record_id"] for record in selected_records}
        unexpected = set(indexed) - selected_ids
        if unexpected:
            raise AdaptiveReasoningError(
                f"Unexpected {mode} result ids: {sorted(unexpected)}"
            )

        if not selected_records and not path.exists():
            _write_jsonl_once(path, [])

        for position, record in enumerate(selected_records):
            record_id = record["record_id"]
            if record_id in indexed:
                continue
            attempt = dict(self.request_fn(dict(record), mode))
            raw_transport = attempt.pop("raw_transport", None)
            if raw_transport is not None:
                transport_name = sha256_bytes(record_id.encode("utf-8"))
                transport_path = (
                    self.output_root
                    / "raw_transports"
                    / mode
                    / f"{transport_name}.json"
                )
                if transport_path.exists():
                    if _read_json(transport_path) != raw_transport:
                        raise AdaptiveReasoningError(
                            f"Raw transport mismatch for {record_id}"
                        )
                else:
                    if not isinstance(raw_transport, Mapping):
                        raise AdaptiveReasoningError(
                            f"Raw transport for {record_id} is not an object"
                        )
                    _write_json_once(transport_path, raw_transport)
                attempt["raw_transport_artifact"] = {
                    "path": str(transport_path.relative_to(self.output_root)),
                    "sha256": sha256_file(transport_path),
                }
            row = {
                "record_id": record_id,
                "cohort": record.get("cohort"),
                "input_index": record.get("input_index", position),
                "reasoning_mode": mode,
                "request_settings": self.request_settings[mode],
                "attempt": attempt,
            }
            _append_jsonl(path, row)
            indexed[record_id] = row
            self._progress(
                f"{mode}_pass",
                completed=len(indexed),
                selected=len(selected_records),
            )
        return indexed

    def _freeze_routes(
        self, fast_rows: Mapping[str, Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        path = self.output_root / "route_decisions.jsonl"
        expected: list[dict[str, Any]] = []
        for position, record in enumerate(self.records):
            record_id = record["record_id"]
            decision = route_fast_attempt(fast_rows[record_id]["attempt"])
            expected.append(
                {
                    "record_id": record_id,
                    "cohort": record.get("cohort"),
                    "input_index": record.get("input_index", position),
                    **decision,
                }
            )

        if path.exists():
            observed = _read_jsonl(path)
            if observed != expected:
                raise AdaptiveReasoningError(
                    "Frozen route decisions do not match fast results"
                )
        else:
            _write_jsonl_once(path, expected)

        marker = self.output_root / "ROUTES_FROZEN.json"
        marker_value = {
            "router_rule_id": ROUTER_RULE_ID,
            "record_count": len(expected),
            "routed_count": sum(row["route"] for row in expected),
            "route_decisions_sha256": sha256_file(path),
        }
        if marker.exists():
            if _read_json(marker) != marker_value:
                raise AdaptiveReasoningError("ROUTES_FROZEN.json is inconsistent")
        else:
            _write_json_once(marker, marker_value)
        return expected, {row["record_id"]: row for row in expected}

    def _write_final_rows(
        self,
        *,
        fast_rows: Mapping[str, Mapping[str, Any]],
        route_rows: Mapping[str, Mapping[str, Any]],
        full_rows: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        expected: list[dict[str, Any]] = []
        for position, record in enumerate(self.records):
            record_id = record["record_id"]
            fast_attempt = fast_rows[record_id]["attempt"]
            full_row = full_rows.get(record_id)
            full_attempt = full_row["attempt"] if full_row else None
            selection = merge_attempts(
                fast_attempt,
                route_rows[record_id],
                full_attempt,
            )
            source = selection["final_source"]
            if source in {"fast", "fast_fallback"}:
                final_attempt: Mapping[str, Any] | None = fast_attempt
            elif source == "full":
                final_attempt = full_attempt
            else:
                final_attempt = None
            expected.append(
                {
                    "record_id": record_id,
                    "cohort": record.get("cohort"),
                    "input_index": record.get("input_index", position),
                    "route": route_rows[record_id]["route"],
                    "route_reasons": route_rows[record_id]["route_reasons"],
                    **selection,
                    "final_attempt": final_attempt,
                }
            )

        path = self.output_root / "final_results.jsonl"
        if path.exists():
            if _read_jsonl(path) != expected:
                raise AdaptiveReasoningError(
                    "Existing final results do not match frozen merge"
                )
        else:
            _write_jsonl_once(path, expected)
        return expected

    def _complete(
        self,
        route_rows: Sequence[Mapping[str, Any]],
        final_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        summary = {
            "record_count": len(final_rows),
            "routed_count": sum(row["route"] is True for row in route_rows),
            "final_source_counts": {
                source: sum(
                    row["final_source"] == source for row in final_rows
                )
                for source in (
                    "fast",
                    "full",
                    "fast_fallback",
                    "unresolved",
                )
            },
            "resolved_count": sum(row["resolved"] is True for row in final_rows),
            "router_rule_id": ROUTER_RULE_ID,
            "merge_rule_id": MERGE_RULE_ID,
        }
        summary_path = self.output_root / "summary.json"
        if summary_path.exists():
            if _read_json(summary_path) != summary:
                raise AdaptiveReasoningError("Existing summary.json is inconsistent")
        else:
            _write_json_once(summary_path, summary)

        artifact_names = [
            "run.json",
            "fast_results.jsonl",
            "route_decisions.jsonl",
            "ROUTES_FROZEN.json",
            "full_results.jsonl",
            "final_results.jsonl",
            "summary.json",
        ]
        transport_root = self.output_root / "raw_transports"
        if transport_root.is_dir():
            artifact_names.extend(
                str(path.relative_to(self.output_root))
                for path in sorted(transport_root.rglob("*.json"))
            )
        checksums = {
            name: sha256_file(self.output_root / name) for name in artifact_names
        }
        checksums_path = self.output_root / "checksums.json"
        if checksums_path.exists():
            if _read_json(checksums_path) != checksums:
                raise AdaptiveReasoningError("Existing checksums.json is inconsistent")
        else:
            _write_json_once(checksums_path, checksums)

        complete = {
            "status": "complete",
            "record_count": len(final_rows),
            "resolved_count": summary["resolved_count"],
            "checksums_sha256": sha256_file(checksums_path),
        }
        complete_path = self.output_root / "COMPLETE.json"
        if complete_path.exists():
            if _read_json(complete_path) != complete:
                raise AdaptiveReasoningError("COMPLETE.json is inconsistent")
        else:
            _write_json_once(complete_path, complete)
        self._progress("complete", record_count=len(final_rows))
        return summary

    def run(self) -> dict[str, Any]:
        """Run or safely resume all stages under an exclusive local lock."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_root / ".adaptive_reasoning.lock"
        with lock_path.open("a+b") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AdaptiveReasoningError(
                    f"Another process holds {lock_path}"
                ) from exc

            try:
                self._ensure_manifest()
                fast_rows = self._run_pass(
                    mode="fast", selected_records=self.records
                )
                route_list, route_rows = self._freeze_routes(fast_rows)
                routed_records = [
                    record
                    for record in self.records
                    if route_rows[record["record_id"]]["route"] is True
                ]
                full_rows = self._run_pass(
                    mode="full", selected_records=routed_records
                )
                final_rows = self._write_final_rows(
                    fast_rows=fast_rows,
                    route_rows=route_rows,
                    full_rows=full_rows,
                )
                return self._complete(route_list, final_rows)
            except BaseException:
                self._progress("interrupted_or_failed")
                raise
