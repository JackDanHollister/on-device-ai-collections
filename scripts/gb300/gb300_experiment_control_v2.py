#!/usr/bin/env python3
"""Version 2 operational supervision for one append-only PlaceBot experiment.

The workload has no row limit or timed shards. A window only controls when the
same checkpoint pauses. Runs never silently change code, data or model settings.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time

import requests


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.partial")
    with temporary.open("x") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def event(root, kind, **details):
    item = {"at": now(), "event": kind, **details}
    with (root / "events.jsonl").open("a") as out:
        out.write(json.dumps(item, sort_keys=True) + "\n")
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(item, sort_keys=True), flush=True)


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, timeout=60, **kwargs).stdout


def docker_contract(container):
    details = json.loads(command(["docker", "inspect", container]))[0]
    return {"id": details["Id"], "image": details["Image"],
            "path": details["Path"], "args": details["Args"],
            "mounts": sorted(({k: m.get(k) for k in ("Type", "Source", "Destination", "RW")}
                              for m in details["Mounts"]), key=lambda m: m["Destination"]),
            "environment": details["Config"].get("Env"),
            "host": {k: details["HostConfig"].get(k) for k in (
                "NetworkMode", "IpcMode", "DeviceRequests", "ShmSize")}}


def model_files(root):
    # Full hashes for parser, tokenizer, configuration and index. Weight shards
    # are pinned by local size/mtime plus the downloaded revision, not rehashed
    # on each restart (80 GB). This is a documented local integrity limitation.
    result = {}
    for path in sorted(Path(root).iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        result[path.name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path.suffix in (".py", ".json", ".jinja"):
            result[path.name]["sha256"] = sha(path)
    return result


def runtime_files(root):
    root = Path(root)
    paths = sorted((root / "placebot").rglob("*.py")) + sorted((root / "placebot/data").rglob("*.md"))
    paths += [root / "research/scripts/gb300_experiment_control.py"]
    return {str(p.relative_to(root)): sha(p) for p in paths}


def verify(config):
    verify_operations(config)
    if runtime_files(config["runtime"]) != config["runtime_files"]:
        raise RuntimeError("Frozen inference/supervisor source changed")
    if docker_contract(config["container"]) != config["docker_contract"]:
        raise RuntimeError("Docker model/image/arguments/mounts changed")
    if model_files(config["model_directory"]) != config["model_files"]:
        raise RuntimeError("Local model assets changed")
    prep = Path(config["preparation"])
    if sha(prep / "manifest.json") != config["preparation_manifest_sha256"]:
        raise RuntimeError("Preparation manifest changed")
    manifest = read(prep / "manifest.json")
    for artifact in manifest["artifacts"].values():
        if sha(prep / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"Preparation artifact changed: {artifact['path']}")


def verify_operations(config):
    """Pin optional operational additions independently of frozen inference."""
    manifest_path = Path(config["root"]) / "operations.json" if config.get("root") else None
    if manifest_path is None or not manifest_path.exists():
        return
    manifest = read(manifest_path)
    if sha(Path(config["root"]) / "experiment.json") != manifest["experiment_sha256"]:
        raise RuntimeError("Original experiment configuration changed")
    if manifest.get("inference_run_sha256") and sha(Path(config["root"]) / "run/run.json") != manifest["inference_run_sha256"]:
        raise RuntimeError("Original inference run identity changed")
    for name, expected in manifest["files"].items():
        path = Path(config["root"]) / name
        if sha(path) != (expected["sha256"] if isinstance(expected, dict) else expected):
            raise RuntimeError(f"Operational source changed: {name}")


def prepare_telemetry(root, control, timeout=45):
    """Require both observational baselines before this new window submits work."""
    operations = root / "operations.json"
    if not operations.exists():
        return
    manifest = read(operations)
    if not manifest.get("telemetry_unit"):
        return
    command(["systemctl", "--user", "start", manifest["telemetry_unit"]])
    ready_path = root / "telemetry" / control["window"] / "READY.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_path.exists():
            ready = read(ready_path)
            if (ready.get("window") == control["window"] and ready.get("baseline_complete")
                    and ready.get("gpu_ok") and ready.get("metrics_ok")
                    and datetime.fromisoformat(ready["at"].replace("Z", "+00:00")).timestamp() >= control["started_epoch"]
                    and ready.get("collector", {}).get("script_sha256") == sha(manifest["telemetry_script"])):
                event(root, "telemetry_baseline_ready", readiness=ready)
                return
        time.sleep(1)
    raise RuntimeError("GPU and model-server telemetry baselines were not ready; no inference started")


def acquire_start_lock(root, unit):
    lock = (root / ".window-start.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = command(["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"]).strip()
        if state not in ("inactive", "failed"):
            raise RuntimeError(f"Supervisor is {state}; wait for its drain/report to finish before a new window")
    except Exception:
        lock.close()
        raise
    return lock


def post_window_report(root, config, window, timeout=45):
    """Ancillary reporting failure never invalidates a safely paused checkpoint."""
    operations = root / "operations.json"
    if not operations.exists():
        return
    try:
        verify_operations(config)
        manifest = read(operations)
        if not manifest.get("report_script"):
            return
        if manifest.get("telemetry_unit"):
            summary_path = root / "telemetry" / window.name / "summary.json"
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if summary_path.exists():
                    summary = read(summary_path)
                    if summary.get("window") == window.name and summary.get("closed"):
                        break
                time.sleep(1)
            else:
                event(root, "telemetry_final_summary_unavailable", window=window.name)
        output = root / "reports" / f"{window.name}-{manifest.get('report_suffix', 'v2')}"
        log_path = window / f"full-report-{time.time_ns()}.log"
        event(root, "full_report_started", window=window.name, output=str(output), log=str(log_path))
        with log_path.open("x") as log:
            subprocess.run([config["runner_command"][0], manifest["report_script"],
                "--root", str(root), "--window", window.name, "--output", str(output)],
                check=True, text=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        if not (output / "COMPLETE.json").exists():
            raise RuntimeError("Full report did not publish a completion manifest")
        event(root, "full_report_completed", window=window.name, output=str(output),
              manifest_sha256=sha(output / "COMPLETE.json"))
    except Exception as exc:
        event(root, "full_report_failed", window=window.name,
              error=f"{type(exc).__name__}: {exc}", checkpoint_preserved=True)


def latest_rowid(database):
    if not database.exists():
        return 0
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='results'").fetchone():
            return 0
        return db.execute("SELECT coalesce(max(rowid),0) FROM results").fetchone()[0]


def failure_class(api_success, usable, error):
    """Only recognized request/transport errors qualify for automatic recovery."""
    if usable:
        return None
    message = str(error or "").lower()
    network = ("connection refused", "connection reset", "connection aborted",
        "read timed out", "connecttimeout", "connection timed out", "readtimeout",
        "service unavailable", "remote end closed connection", "remote disconnected",
        "502 server error", "503 server error", "504 server error")
    transient_status = re.search(r"\bapi error:\s*(502|503|504)\b", message)
    if not api_success and (transient_status or any(term in message for term in network)):
        return "transport"
    # Schema/truncation failures and unrecognized API errors keep the strict
    # quality guard; a healthy endpoint alone cannot justify accepting them.
    return "semantic_or_unknown"


def failure_summary(database, window_start_rowid, transport_handled_rowid):
    summary = {"recent_attempts": 0, "semantic_or_unknown": 0,
        "new_transport": 0, "last_rowid": window_start_rowid}
    if not database.exists():
        return summary
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='results'").fetchone():
            return summary
        rows = db.execute("SELECT rowid,api_success,usable,error FROM results ORDER BY rowid DESC LIMIT 500").fetchall()
    summary["recent_attempts"] = len(rows)
    for rowid, success, usable, error in rows:
        summary["last_rowid"] = max(summary["last_rowid"], rowid)
        kind = failure_class(success, usable, error)
        if kind == "semantic_or_unknown":
            summary[kind] += 1
        elif kind == "transport" and rowid > max(window_start_rowid, transport_handled_rowid):
            summary["new_transport"] += 1
    return summary


def begin_recovery(session, session_file, cause, rowid):
    """One persisted charge per outage episode, shared by client/model recovery."""
    if session.get("active_recovery"):
        return
    if session["recoveries"] >= 3:
        raise RuntimeError("Automatic recovery budget exhausted (3 per window)")
    session["recoveries"] += 1
    session["active_recovery"] = {"cause": cause, "began_at": now(), "rowid": rowid}
    save(session_file, session)


def acknowledge_transport(session, session_file, database):
    # Called only after a stopped/drained child. New errors from the replacement
    # runner must remain above this durable watermark and start a new episode.
    session["transport_handled_rowid"] = latest_rowid(database)
    save(session_file, session)


def quality_guard(summary):
    if summary["semantic_or_unknown"] >= 50:
        raise RuntimeError("At least 50 of the last 500 outputs have semantic or unknown failures; human audit required")


def healthy(config):
    try:
        response = requests.get(config["base_url"] + "/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def read_metrics(database):
    if not database.exists():
        return {"completed": 0, "usable": 0, "failed": 0, "represented_records": 0}
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=5) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='result_metrics'").fetchone():
            return {"completed": 0, "usable": 0, "failed": 0, "represented_records": 0}
        total, usable, represented = db.execute("SELECT completed_candidates, usable_candidates, represented_occurrences FROM result_metrics WHERE singleton=1").fetchone()
    return {"completed": total, "usable": usable, "failed": total - usable, "represented_records": represented}


def fast_audit_disposition(api_success, usable, error, item, rowid):
    """Missing telemetry on a rejected parse is unknown, never verified zero.

    A stray thinking delimiter can make the pinned splitter return None and
    leave an invalid final JSON response. Such an attempt is already excluded
    from usable outputs. Keep explicit reasoning violations fatal, and retain
    the existing quality-burst guard for repeated rejected responses.
    """
    if not isinstance(item, dict):
        raise RuntimeError(f"Invalid prediction evidence at result rowid={rowid}")
    tokens = item.get("reasoning_tokens")
    content = item.get("reasoning_content")
    if ((tokens is not None and (type(tokens) is not int or tokens != 0))
            or (content is not None and (not isinstance(content, str) or content.strip()))):
        raise RuntimeError(f"Fast-mode telemetry missing or nonzero at result rowid={rowid}")
    if not api_success:
        return "api_failure"
    if tokens == 0:
        return "verified_fast"
    parse_failure = isinstance(error, str) and (
        error.startswith("Response parsing failed: Unexpected vLLM response format:")
        or error.startswith("JSON parsing failed:")
        or error == "No valid JSON found in response")
    if (usable == 0 and item.get("success") is False and parse_failure
            and item.get("error") == error):
        return "deferred_unverified"
    raise RuntimeError(f"Fast-mode telemetry missing or nonzero at result rowid={rowid}")


def audit_fast(database, after_rowid, through_rowid=None, *, deferred_audit_path=None):
    if not database.exists():
        return after_rowid
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='results'").fetchone():
            return after_rowid
        boundary = " AND rowid <= ?" if through_rowid is not None else ""
        parameters = (after_rowid, through_rowid) if through_rowid is not None else (after_rowid,)
        for rowid, candidate_id, api_success, usable, error, prediction in db.execute(
                "SELECT rowid, candidate_id, api_success, usable, error, prediction_json FROM results WHERE rowid > ?"
                + boundary + " ORDER BY rowid LIMIT 10000", parameters):
            item = json.loads(prediction)
            disposition = fast_audit_disposition(api_success, usable, error, item, rowid)
            if disposition == "deferred_unverified" and deferred_audit_path is not None:
                record = {"at": now(), "rowid": rowid, "candidate_id": candidate_id,
                    "disposition": disposition, "usable": False,
                    "reasoning_tokens": None, "error": error,
                    "prediction_json_sha256": hashlib.sha256(prediction.encode("utf-8")).hexdigest(),
                    "interpretation": "Rejected parse with unavailable or ambiguous fast-mode telemetry; excluded from usable results, not counted as verified zero reasoning. Repeated audit entries refer to the same immutable attempt."}
                with Path(deferred_audit_path).open("a", encoding="utf-8") as out:
                    out.write(json.dumps(record, sort_keys=True) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
            after_rowid = rowid
    return after_rowid


def audit_checkpoint_prefix(database, session, session_file):
    """Validate inherited rows before launch so live checks start at new work."""
    boundary = session["window_start_rowid"]
    while session["fast_audited_rowid"] < boundary:
        previous = session["fast_audited_rowid"]
        audited = audit_fast(database, previous, through_rowid=boundary,
                             deferred_audit_path=session_file.parent / "deferred_fast_audit.jsonl")
        if audited <= previous:
            raise RuntimeError(
                f"Checkpoint prefix audit made no progress after rowid={previous}; "
                f"startup boundary rowid={boundary} was not verified")
        session["fast_audited_rowid"] = audited
        save(session_file, session)
    return session["fast_audited_rowid"]


def transient_failed_run(run):
    marker = read(run / "FAILED.json")
    match = re.fullmatch(r"(\d+) consecutive unusable model outputs", marker.get("reason", ""))
    if not match:
        return False
    with sqlite3.connect((run / "results.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("SELECT api_success,usable,error FROM results ORDER BY rowid DESC LIMIT ?", (int(match.group(1)),)).fetchall()
    # A recovered server does not license bypassing semantic/schema failures.
    return len(rows) >= 10 and len(rows) == int(match.group(1)) and all(failure_class(*row) == "transport" for row in rows)


def sqlite_audit(database):
    if database.exists():
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
            if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise RuntimeError("SQLite integrity check failed; checkpoint preserved for audit")


def checkpoint_snapshot(config, window_dir):
    """Only called after the child has drained; never a long live export read."""
    root = Path(config["root"])
    run = root / "run"
    database = run / "results.sqlite3"
    if not database.exists() or not (run / "run.json").exists():
        return
    snapshot = window_dir / "results.sqlite3"
    if snapshot.exists():
        manifest_path = window_dir / "checkpoint_manifest.json"
        if manifest_path.exists() and all(sha(window_dir / name) == item["sha256"] for name, item in read(manifest_path)["files"].items()):
            if read_metrics(snapshot) != read_metrics(database) or latest_rowid(snapshot) != latest_rowid(database):
                raise RuntimeError("Prior window snapshot no longer matches the live checkpoint; preserved for audit")
            return
        raise RuntimeError("Incomplete or changed prior snapshot; preserved for audit")
    sqlite_audit(database)
    temporary = snapshot.with_suffix(".partial")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as source, sqlite3.connect(temporary) as target:
        source.backup(target)
    with temporary.open("rb") as durable:
        os.fsync(durable.fileno())
    os.replace(temporary, snapshot)
    failures = window_dir / "deferred_failures.tsv"
    with sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True) as db, failures.open("x", newline="") as out:
        fields = ["candidate_id", "locality", "input_country", "occurrence_count", "example_identifier", "context_json", "error", "api_success", "truncated", "finish_reason"]
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(db.execute(f"SELECT {','.join(fields)} FROM deferred_failures ORDER BY candidate_id"))
    # Model-visible retry file has the same six columns as a normal preparation.
    import duckdb
    with duckdb.connect() as db:
        db.sql("SELECT candidate_id, locality, input_country AS country, cast(occurrence_count AS BIGINT) AS occurrence_count, example_identifier, context_json FROM read_csv(?, delim='\t', header=true, all_varchar=true)", params=[str(failures)]).write_parquet(str(window_dir / "retry_candidates.parquet"), compression="zstd")
    extra_files = []
    if config.get("runtime"):
        sys.path.insert(0, config["runtime"])
        from placebot.cli import hpc
        exported = window_dir / "results.tsv.gz"
        if hpc.main(["export", str(run), "--output", str(exported)]) != 0:
            raise RuntimeError("Paused result export failed")
        report = analyse_window(config, window_dir, exported)
        extra_files = [exported, exported.with_name(exported.name + ".manifest.json"), window_dir / "cohort_report.json", window_dir / "evaluation_errors.parquet"]
        event(root, "cohort_report", report=report)
    for path in (snapshot, failures, window_dir / "retry_candidates.parquet", *extra_files):
        with path.open("rb") as durable:
            os.fsync(durable.fileno())
    save(window_dir / "checkpoint_manifest.json", {
        "at": now(), "run_identity_sha256": sha(run / "run.json"),
        "config_sha256": sha(root / "experiment.json"), "metrics": read_metrics(snapshot),
        "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in (
            snapshot, failures, window_dir / "retry_candidates.parquet", *extra_files)},
        "retry_policy": "Separate later run; preserve this first-pass ledger and merge only explicitly audited successful retries by candidate_id.",
        "backup_scope": "Same machine/disk: protects against accidental checkpoint changes, not total storage loss."})
    event(root, "checkpoint_snapshot", path=str(snapshot), metrics=read_metrics(snapshot))


def analyse_window(config, window_dir, exported):
    """Post-inference scoring only; withheld references never enter model calls."""
    import duckdb
    prep = Path(config["preparation"])
    with duckdb.connect() as db:
        db.execute("SET threads=8")
        db.execute("SET memory_limit='16GB'")
        predictions = db.sql("SELECT * FROM read_csv(?, delim='\t', header=true, all_varchar=true)", params=[str(exported)])
        predictions.create_view("predictions")
        cohorts = db.sql("SELECT * FROM read_parquet(?)", params=[str(prep / "candidate_cohorts.parquet")])
        cohorts.create_view("cohorts")
        references = db.sql("SELECT * FROM read_parquet(?)", params=[str(prep / "evaluation_only/references.parquet")])
        references.create_view("refs")
        p_count, e_count, p_usable, e_usable = db.execute("""SELECT
            coalesce(sum(c.production_records),0), coalesce(sum(c.evaluation_records),0),
            coalesce(sum(c.production_records) FILTER (WHERE p.usable='1'),0),
            coalesce(sum(c.evaluation_records) FILTER (WHERE p.usable='1'),0)
            FROM predictions p JOIN cohorts c USING(candidate_id)""").fetchone()
        db.execute("""CREATE TEMP TABLE evaluation AS SELECT
            r.candidate_id, r.gbifid, r.publishingorgkey, r.datasetkey,
            try_cast(r.decimallatitude AS DOUBLE) AS reference_latitude,
            try_cast(r.decimallongitude AS DOUBLE) AS reference_longitude,
            try_cast(r.coordinateuncertaintyinmeters AS DOUBLE) AS reference_uncertainty_m,
            try_cast(p.latitude AS DOUBLE) AS prediction_latitude,
            try_cast(p.longitude AS DOUBLE) AS prediction_longitude,
            try_cast(p.coordinate_radius_meters AS DOUBLE) AS prediction_radius_m,
            12742.0 * asin(sqrt(least(1.0, greatest(0.0,
                pow(sin(radians(try_cast(p.latitude AS DOUBLE)-try_cast(r.decimallatitude AS DOUBLE))/2),2)
                + cos(radians(try_cast(r.decimallatitude AS DOUBLE))) * cos(radians(try_cast(p.latitude AS DOUBLE)))
                * pow(sin(radians(try_cast(p.longitude AS DOUBLE)-try_cast(r.decimallongitude AS DOUBLE))/2),2))))) AS error_km
            FROM predictions p JOIN refs r USING(candidate_id) WHERE p.usable='1'""")
        errors = window_dir / "evaluation_errors.parquet"
        db.sql("SELECT * FROM evaluation").write_parquet(str(errors), compression="zstd")
        columns = ["evaluated_specimens", "unique_inputs", "median_error_km", "mean_error_km", "p90_error_km", "within_1km", "within_10km", "within_50km", "within_100km", "references_with_positive_uncertainty", "within_reference_uncertainty", "predictions_with_radius", "within_prediction_radius"]
        summary = db.execute("""SELECT count(*), count(DISTINCT candidate_id), median(error_km), avg(error_km), quantile_cont(error_km,0.90),
            count(*) FILTER (WHERE error_km<=1), count(*) FILTER (WHERE error_km<=10),
            count(*) FILTER (WHERE error_km<=50), count(*) FILTER (WHERE error_km<=100),
            count(*) FILTER (WHERE isfinite(reference_uncertainty_m) AND reference_uncertainty_m>0),
            count(*) FILTER (WHERE isfinite(reference_uncertainty_m) AND reference_uncertainty_m>0 AND error_km*1000<=reference_uncertainty_m),
            count(*) FILTER (WHERE isfinite(prediction_radius_m) AND prediction_radius_m>=0),
            count(*) FILTER (WHERE isfinite(prediction_radius_m) AND prediction_radius_m>=0 AND error_km*1000<=prediction_radius_m)
            FROM evaluation""").fetchone()
    report = {"at": now(), "production": {"attempted_specimens": p_count, "usable_specimens": p_usable},
        "evaluation": {"attempted_specimens": e_count, "usable_specimens": e_usable, **dict(zip(columns, summary))},
        "interpretation": "Cumulative coordinate-agreement baseline, not verified ground-truth accuracy. Specimen-weighted metrics; repeated locality inputs are not independent observations. (0,0) source pairs excluded. Original source coordinates are unchanged. Scoring runs only after inference has stopped."}
    save(window_dir / "cohort_report.json", report)
    return report


def stop_reason(control, timestamp, free_bytes, stalled_seconds):
    if control["desired"] != "running":
        return "operator_pause"
    if control.get("deadline_epoch") is not None and timestamp >= control["deadline_epoch"]:
        return "window_deadline"
    if free_bytes < 50 * 1024**3:
        return "disk_space_below_50_GiB"
    if stalled_seconds >= 900:
        return "no_committed_progress_for_15_minutes"
    return None


def supervise(root):
    config = read(root / "experiment.json")
    lock = (root / ".supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    control = read(root / "control.json")
    if control["desired"] != "running":
        return
    window = root / "windows" / control["window"]
    window.mkdir(parents=True, exist_ok=True)
    session_file = window / "session.json"
    run = root / "run"
    database = run / "results.sqlite3"
    session = read(session_file) if session_file.exists() else {
        "recoveries": 0, "fast_audited_rowid": 0,
        "window_start_rowid": latest_rowid(database),
        "transport_handled_rowid": latest_rowid(database), "active_recovery": None}
    if not session_file.exists():
        save(session_file, session)
    # Preserve progress if this controller restarts inside the same window;
    # never reset its recovery budget or transport watermark on process restart.
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    child = None
    log_handle = None
    reason = "unknown"
    last_completed = read_metrics(database)["completed"]
    last_progress = time.monotonic()
    last_log = 0.0
    unhealthy_since = None
    audit_resume = False
    try:
        if "window_start_rowid" not in session:
            raise RuntimeError("Unversioned existing window session requires explicit audit")
        reason = stop_reason(control, time.time(), shutil.disk_usage(root).free, 0)
        if reason:
            return
        # On reboot the lingering user service can precede the Docker daemon.
        # Wait for that specific transient dependency, but honor the old deadline.
        for attempt in range(40):
            try:
                docker_contract(config["container"])
                break
            except subprocess.CalledProcessError as exc:
                message = (exc.stderr or "").lower()
                if "docker daemon" not in message:
                    raise
                reason = stop_reason(read(root / "control.json"), time.time(), shutil.disk_usage(root).free, 0)
                if reason or stopping.is_set():
                    reason = reason or "service_stop"
                    return
                if attempt == 39:
                    raise RuntimeError("Docker daemon unavailable for ten minutes") from exc
                stopping.wait(15)
        verify(config)
        sqlite_audit(database)
        audit_checkpoint_prefix(database, session, session_file)
        event(root, "supervisor_started", control=control, existing=read_metrics(database))
        if (run / "COMPLETE.json").exists():
            reason = "workload_complete"
            return
        if (run / "FAILED.json").exists():
            if not transient_failed_run(run):
                raise RuntimeError("Runner is marked FAILED; human audit required")
            begin_recovery(session, session_file, "existing_transport_failure", latest_rowid(database))
            acknowledge_transport(session, session_file, database)
            audit_resume = True
        while True:
            control = read(root / "control.json")
            metrics = read_metrics(database)
            if metrics["completed"] != last_completed:
                last_completed = metrics["completed"]
                last_progress = time.monotonic()
            free = shutil.disk_usage(root).free
            reason = stop_reason(control, time.time(), free, time.monotonic() - last_progress)
            if stopping.is_set():
                reason = "window_deadline" if control.get("deadline_epoch") is not None and time.time() >= control["deadline_epoch"] else "service_stop"
            if reason:
                break
            ok = healthy(config)
            if ok:
                unhealthy_since = None
            elif unhealthy_since is None:
                unhealthy_since = time.monotonic()
                event(root, "endpoint_unhealthy")
            if child is not None and child.poll() is not None:
                rc = child.returncode
                event(root, "runner_exited", returncode=rc, metrics=metrics)
                child = None
                if log_handle:
                    log_handle.close()
                    log_handle = None
                if rc in (0, 3) and (run / "COMPLETE.json").exists():
                    reason = "workload_complete" if rc == 0 else "first_pass_complete_with_deferred_failures"
                    break
                if (run / "FAILED.json").exists():
                    if not transient_failed_run(run):
                        raise RuntimeError("Runner is marked FAILED with non-network errors; human audit required")
                    audit_resume = True
                elif rc not in (-signal.SIGKILL, -signal.SIGTERM):
                    raise RuntimeError(f"Unexpected runner exit {rc}; human audit required")
                begin_recovery(session, session_file, "runner_exit", latest_rowid(database))
                acknowledge_transport(session, session_file, database)
                verify(config)
                sqlite_audit(database)
            failures = failure_summary(database, session["window_start_rowid"], session["transport_handled_rowid"])
            quality_guard(failures)
            transport_burst = failures["new_transport"] >= 50
            outage_pause = not ok and time.monotonic() - unhealthy_since >= 30
            if child is not None and (transport_burst or outage_pause):
                cause = "transport_failure_burst" if transport_burst else "endpoint_outage"
                begin_recovery(session, session_file, cause, failures["last_rowid"])
                child.send_signal(signal.SIGTERM)
                child.wait(timeout=630)
                event(root, "outage_drain", cause=cause, returncode=child.returncode)
                if (run / "COMPLETE.json").exists() and child.returncode in (0, 3):
                    reason = "workload_complete" if child.returncode == 0 else "first_pass_complete_with_deferred_failures"
                    break
                if (run / "FAILED.json").exists():
                    if not transient_failed_run(run):
                        raise RuntimeError("Outage coincided with non-network failures; human audit required")
                    audit_resume = True
                elif child.returncode != 130:
                    raise RuntimeError("Unexpected outage drain result")
                child = None
                if log_handle:
                    log_handle.close()
                    log_handle = None
                acknowledge_transport(session, session_file, database)
                failures = failure_summary(database, session["window_start_rowid"], session["transport_handled_rowid"])
                quality_guard(failures)
            if not ok:
                # Freeze submission during an outage. The runner drains errors
                # durably and we audit its disposition before resuming.
                running = command(["docker", "inspect", config["container"], "--format", "{{.State.Running}}"]).strip()
                if running != "true":
                    verify(config)
                    if (session.get("active_recovery") or {}).get("model_start_attempted"):
                        # A container dying again before health recovery is
                        # another failed attempt, not a free restart loop.
                        session["active_recovery"] = None
                    begin_recovery(session, session_file, "model_container_stopped", latest_rowid(database))
                    session["active_recovery"]["model_start_attempted"] = True
                    save(session_file, session)
                    command(["docker", "start", config["container"]])
                    event(root, "model_container_started")
                elif time.monotonic() - unhealthy_since > 600:
                    raise RuntimeError("Model is running but unhealthy for 10 minutes; human audit required")
            elif child is None:
                # Draining/verification may take minutes. Check the deadline
                # again so no replacement request is launched after it expires.
                reason = stop_reason(read(root / "control.json"), time.time(), shutil.disk_usage(root).free, time.monotonic() - last_progress)
                if reason or stopping.is_set():
                    reason = reason or "service_stop"
                    break
                verify(config)
                sqlite_audit(database)
                args = list(config["runner_command"])
                if (run / "run.json").exists():
                    args.append("--resume-after-audit" if audit_resume else "--resume")
                if audit_resume:
                    event(root, "network_failure_resume_audit", marker=read(run / "FAILED.json"), sqlite_check="ok", identity_check="unchanged")
                reason = stop_reason(read(root / "control.json"), time.time(), shutil.disk_usage(root).free, time.monotonic() - last_progress)
                if reason or stopping.is_set():
                    reason = reason or "service_stop"
                    break
                log_path = window / f"runner-{time.time_ns()}.log"
                log_handle = log_path.open("x")
                child = subprocess.Popen(args, cwd=config["runtime"], env={**os.environ, **config["environment"]}, stdout=log_handle, stderr=subprocess.STDOUT)
                event(root, "runner_started", pid=child.pid, log=str(log_path), resumed=(run / "run.json").exists())
                audit_resume = False
                session["active_recovery"] = None
                save(session_file, session)
            elif ok and session.get("active_recovery"):
                # A quick model restart can recover before a client drain was
                # needed. Charge it once; subsequent distinct outages get their
                # own bounded episode. Do not skip unhandled client errors.
                session["active_recovery"] = None
                save(session_file, session)
            session["fast_audited_rowid"] = audit_fast(database, session["fast_audited_rowid"],
                deferred_audit_path=window / "deferred_fast_audit.jsonl")
            save(session_file, session)
            status = {"at": now(), "window": control["window"], "desired": control["desired"],
                "deadline_utc": control.get("deadline_utc"), "model_healthy": ok,
                "runner_pid": child.pid if child else None, "metrics": metrics,
                "disk_free_bytes": free, "checkpoint_bytes": {suffix: Path(str(database) + suffix).stat().st_size if Path(str(database) + suffix).exists() else 0 for suffix in ("", "-wal", "-shm")},
                "recoveries": session["recoveries"], "fast_audited_rowid": session["fast_audited_rowid"],
                "window_start_rowid": session["window_start_rowid"],
                "transport_handled_rowid": session["transport_handled_rowid"], "recent_failures": failures}
            save(root / "monitor.json", status)
            if time.monotonic() - last_log >= 60:
                event(root, "progress", **status)
                last_log = time.monotonic()
            stopping.wait(15)
    except Exception as exc:
        reason = "audit_required"
        event(root, "safety_stop", error=f"{type(exc).__name__}: {exc}")
    finally:
        # A supervisor crash during drain/export must not erase a safety stop
        # or authorize further inference through systemd Restart=on-failure.
        control = read(root / "control.json")
        control.update(desired="paused", stopping_at=now(), stop_reason=reason)
        save(root / "control.json", control)
        if child is not None and child.poll() is None:
            event(root, "draining", reason=reason, pid=child.pid)
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=630)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=30)
                reason = "drain_timeout_audit_required"
            event(root, "runner_drained", reason=reason, returncode=child.returncode,
                  metrics=read_metrics(database))
        if log_handle:
            log_handle.close()
        control.update(stop_reason=reason)
        save(root / "control.json", control)
        try:
            while True:
                audited = audit_fast(database, session["fast_audited_rowid"],
                    deferred_audit_path=window / "deferred_fast_audit.jsonl")
                if audited == session["fast_audited_rowid"]:
                    break
                session["fast_audited_rowid"] = audited
        except Exception as exc:
            event(root, "final_audit_failed", error=f"{type(exc).__name__}: {exc}")
            reason = "final_audit_required"
        # Preserve and report the stopped ledger even when its content needs
        # audit. A semantic failure must not prevent making a durable backup.
        try:
            checkpoint_snapshot(config, window)
        except Exception as exc:
            event(root, "checkpoint_snapshot_failed", error=f"{type(exc).__name__}: {exc}")
            reason = "snapshot_or_final_audit_required"
        control = read(root / "control.json")
        control.update(desired="paused", stopped_at=now(), stop_reason=reason)
        save(root / "control.json", control)
        save(root / "monitor.json", {"at": now(), "desired": "paused", "window": control["window"],
            "stop_reason": reason, "runner_pid": None, "metrics": read_metrics(database),
            "fast_audited_rowid": session["fast_audited_rowid"]})
        save(session_file, session)
        event(root, "paused", reason=reason, metrics=read_metrics(database))
        post_window_report(root, config, window)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    subs = parser.add_subparsers(dest="action", required=True)
    initialize = subs.add_parser("init")
    initialize.add_argument("--preparation", type=Path, required=True)
    initialize.add_argument("--runtime", type=Path, required=True)
    initialize.add_argument("--python", required=True)
    initialize.add_argument("--container", default="placebot-super-fast-c128-bf16-20260905-v1")
    initialize.add_argument("--model-directory", default="/data/placebot-super-bench/model")
    initialize.add_argument("--unit", default="placebot-combined-experiment.service")
    start = subs.add_parser("start")
    group = start.add_mutually_exclusive_group(required=True)
    group.add_argument("--hours", type=float)
    group.add_argument("--continuous", action="store_true")
    subs.add_parser("pause")
    subs.add_parser("status")
    subs.add_parser("supervise")
    deadline_action = subs.add_parser("deadline")
    deadline_action.add_argument("--window", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "init":
        if (root / "experiment.json").exists() or (root / "run").exists():
            raise RuntimeError("Experiment already initialized; preserve it and use start")
        runtime = args.runtime.resolve()
        preparation = args.preparation.resolve()
        config = {"schema_version": 1, "created_at": now(), "root": str(root),
            "runtime": str(runtime), "preparation": str(preparation), "unit": args.unit,
            "container": args.container, "model_directory": args.model_directory,
            "base_url": "http://127.0.0.1:8000", "docker_contract": docker_contract(args.container),
            "model_files": model_files(args.model_directory), "runtime_files": runtime_files(runtime),
            "preparation_manifest_sha256": sha(preparation / "manifest.json"),
            "environment": {"PYTHONPATH": str(runtime), "PYTHONUNBUFFERED": "1",
                "PLACEBOT_NEMOTRON_CONTEXT_WINDOW": "16384", "PLACEBOT_NEMOTRON_SEED": "123",
                "PLACEBOT_NEMOTRON_TOKENIZER_REVISION": "4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6"},
            "runner_command": [args.python, "-m", "placebot.cli.hpc", "run", str(preparation / "candidates.parquet"),
                "--output-dir", str(root / "run"), "--model-profile", "nemotron_3_super_local",
                "--endpoint", "http://127.0.0.1:8000/v1/chat/completions", "--model-id", "placebot-nemotron-super-fast",
                "--reasoning-mode", "fast", "--max-output-tokens", "1024", "--request-timeout", "300",
                "--max-retries", "0", "--workers", "100", "--checkpoint-every", "100",
                "--max-consecutive-failures", "100", "--evidence", "final", "--quiet"],
            "counts": read(preparation / "manifest.json")["counts"],
            "protocol": "One shared, append-only first pass. No thinking, RAG, coordinate references or fixed-duration shards in inference. Separate deferred failure retry pass. Window boundaries do not change the scientific run identity."}
        verify(config)
        save(root / "experiment.json", config)
        save(root / "control.json", {"desired": "paused", "initialized_at": now()})
        print(json.dumps({"root": str(root), "config_sha256": sha(root / "experiment.json"), "counts": config["counts"]}, indent=2))
        return
    config = read(root / "experiment.json")
    if args.action == "supervise":
        supervise(root)
    elif args.action == "deadline":
        control = read(root / "control.json")
        if control.get("window") == args.window and control.get("deadline_epoch") is not None and time.time() >= control["deadline_epoch"]:
            control.update(desired="paused", deadline_requested_at=now())
            save(root / "control.json", control)
            command(["systemctl", "--user", "--no-block", "stop", config["unit"]])
    elif args.action == "status":
        print(json.dumps({"control": read(root / "control.json"), "monitor": read(root / "monitor.json") if (root / "monitor.json").exists() else None, "live_metrics": read_metrics(root / "run/results.sqlite3")}, indent=2))
    elif args.action == "pause":
        control = read(root / "control.json")
        control.update(desired="paused", pause_requested_at=now())
        save(root / "control.json", control)
        # The supervisor sees this within 15 seconds and drains safely. It is
        # intentionally not restarted by an external polling/session boundary.
        print("Pause requested; check status for the durable stopped_at marker.")
    elif args.action == "start":
        if args.hours is not None and (not 0 < args.hours <= 24 * 30):
            raise ValueError("Window hours must be positive and no greater than 30 days")
        # Serialize launch requests, including baseline collection, and do not
        # replace control.json while the previous service is still draining.
        start_lock = acquire_start_lock(root, config["unit"])
        if (root / "control.json").exists() and "audit_required" in read(root / "control.json").get("stop_reason", ""):
            raise RuntimeError("Previous safety stop requires audit before a new window")
        verify(config)
        if (root / "run/COMPLETE.json").exists():
            raise RuntimeError("First pass already complete; use the separate deferred-failure queue")
        epoch = time.time()
        deadline = None if args.continuous else epoch + args.hours * 3600
        control = {"window": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "desired": "running", "started_at": now(), "started_epoch": epoch,
            "deadline_epoch": deadline, "deadline_utc": datetime.fromtimestamp(deadline, timezone.utc).isoformat() if deadline else None}
        save(root / "control.json", control)
        event(root, "window_authorized", **control)
        try:
            prepare_telemetry(root, control)
            if deadline is not None:
                # Independent systemd deadline: still stops the service if the
                # Python monitor stalls. A stale timer cannot stop a later window.
                deadline_calendar = datetime.fromtimestamp(deadline + 1, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                command(["systemd-run", "--user", "--collect", "--unit", f"placebot-deadline-{control['window']}",
                    "--on-calendar", deadline_calendar, "--timer-property=AccuracySec=1s",
                    config["runner_command"][0], str(Path(__file__).resolve()),
                    "--root", str(root), "deadline", "--window", control["window"]])
            command(["systemctl", "--user", "start", config["unit"]])
        except Exception as exc:
            control.update(desired="paused", stopped_at=now(), stop_reason="launch_failed")
            save(root / "control.json", control)
            event(root, "launch_failed", window=control["window"], error=f"{type(exc).__name__}: {exc}")
            raise
        print(json.dumps(control, indent=2))
        start_lock.close()


if __name__ == "__main__":
    main()
