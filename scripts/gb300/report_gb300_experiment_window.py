#!/usr/bin/env python3
"""Rebuild rich PlaceBot reporting from immutable checkpoints; never calls a model.

SQLite is streamed into a disk-backed DuckDB staging database. Source snapshots,
the frozen inference runtime, and earlier reports are never modified. Each report
is a fresh, explicit directory; COMPLETE.json is written last.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import duckdb


TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "total_tokens", "cached_tokens",
    "cache_creation_input_tokens", "cache_read_input_tokens",
    "cached_content_token_count", "reasoning_tokens", "final_output_tokens",
)
EXTRA_NUMERIC = TOKEN_FIELDS[3:] + ("output_cap_tokens",)
EXTRA_TEXT = ("token_source", "reasoning_token_source")
THRESHOLDS = (1, 5, 10, 50, 100)
DEFAULT_MEMORY_LIMIT = "16GB"
DEFAULT_THREADS = 2
DEFAULT_PARQUET_ROW_GROUP_SIZE = 16384


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    with Path(path).open("x") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def sql_identifier(value):
    return '"' + value.replace('"', '""') + '"'


def query_one(db, sql, params=None):
    cursor = db.execute(sql, params or [])
    names = [item[0] for item in cursor.description]
    return dict(zip(names, cursor.fetchone()))


def epoch(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def read_window_events(path, window):
    """Return this and previous windows, never events from later windows."""
    grouped = {}
    current = None
    with Path(path).open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == "window_authorized":
                if current == window:
                    break
                current = event["window"]
                grouped.setdefault(current, [])
            if current:
                grouped[current].append(event)
    if window not in grouped:
        raise ValueError(f"No authorization event for window {window}")
    return grouped


def timing_for_window(events, last_completion):
    """Observed timestamps, with honest bounds where historical exit was unlogged.

    The old controller logged runner starts/exits on crashes, but only 'draining'
    before the final graceful exit. Its final completion is a lower bound on that
    exit; the checkpoint/report event is an upper bound. Never call these exact
    GPU-active seconds. Wall rates use authorization through last saved completion.
    """
    authorization = events[0]
    start = float(authorization.get("started_epoch", epoch(authorization["at"])))
    final = epoch(last_completion) if last_completion else None
    intervals = []
    running = None
    uncertainty = False
    upper = None
    for event in events:
        when = epoch(event["at"])
        name = event["event"]
        if name == "runner_started":
            if running is not None:
                # A controller reboot can leave no exit event. Do not silently
                # count the entire unobserved gap as active runtime.
                uncertainty = True
                intervals.append({"start_epoch": running, "end_epoch": None,
                                  "end_source": "unobserved_before_next_start"})
            running = when
        elif name in ("runner_exited", "outage_drain", "runner_drained") and running is not None:
            intervals.append({"start_epoch": running, "end_epoch": when,
                              "end_source": name})
            running = None
        elif name in ("cohort_report", "checkpoint_snapshot", "paused"):
            if upper is None:
                upper = when
    if running is not None:
        endpoint = max(running, final) if final is not None else None
        intervals.append({"start_epoch": running, "end_epoch": endpoint,
                          "end_source": "last_committed_completion_lower_bound"})
        uncertainty = True
    known = sum(item["end_epoch"] - item["start_epoch"] for item in intervals
                if item["end_epoch"] is not None)
    unknown_gap = any(item["end_epoch"] is None for item in intervals)
    upper_seconds = None if unknown_gap else known
    if running is not None:
        upper_seconds = (known + max(0, upper - max(running, final))
                         if upper is not None and final is not None and not unknown_gap else None)
    paused = next((event["at"] for event in reversed(events) if event["event"] == "paused"), None)
    return {
        "authorized_at": authorization.get("started_at", authorization["at"]),
        "last_committed_at": last_completion,
        "paused_at": paused,
        "scheduled_seconds": (authorization["deadline_epoch"] - start
                              if authorization.get("deadline_epoch") is not None else None),
        "authorization_to_last_completion_seconds": max(0, final - start) if final is not None else None,
        "authorization_to_paused_seconds": epoch(paused) - start if paused else None,
        "observed_client_seconds_lower_bound": known,
        "observed_client_seconds_upper_bound": upper_seconds,
        "client_runtime_has_unobserved_exit": uncertainty,
        "client_intervals": intervals,
        "interpretation": "Wall rates include startup/recovery through the last saved completion. Client spans include request waiting and CPU work; they are not GPU-active time. The bounds bracket the unlogged final graceful exit within observed event spans; earlier polled exit timestamps may lag the actual process exit. User pauses between windows are excluded from cumulative window-active wall rates.",
    }


def stage_requests(db, snapshot, output, events, continuation=None):
    staging = output / "requests_staging.tsv"
    # immutable=1 avoids even creating SQLite -shm sidecars. Only checkpoint
    # snapshots with no nonempty WAL may be read this way.
    wal = Path(str(snapshot) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Snapshot has a nonempty WAL; use a completed backup")
    with closing(sqlite3.connect(snapshot.as_uri() + "?mode=ro&immutable=1", uri=True)) as source:
        if source.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Checkpoint integrity check failed")
        schema = source.execute("PRAGMA table_info(results)").fetchall()
        cursor = source.execute("SELECT rowid AS committed_rowid, * FROM results ORDER BY rowid")
        names = [item[0] for item in cursor.description]
        with staging.open("x", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(names)
            while rows := cursor.fetchmany(4096):
                writer.writerows(tuple("\\N" if value is None else value for value in row) for row in rows)
    types = {"committed_rowid": "BIGINT"}
    types.update({row[1]: {"TEXT": "VARCHAR", "INTEGER": "BIGINT", "REAL": "DOUBLE"}[row[2]] for row in schema})
    declaration = "{" + ",".join(f"{sql_string(k)}:{sql_string(v)}" for k, v in types.items()) + "}"
    db.execute(f"CREATE TABLE raw_requests AS SELECT * FROM read_csv({sql_string(staging)}, delim='\t', header=true, nullstr='\\N', columns={declaration})")
    # Malformed JSON is an audit failure: richer fields must not silently vanish.
    if db.execute("SELECT count(*) FROM raw_requests WHERE NOT json_valid(prediction_json)").fetchone()[0]:
        raise ValueError("Malformed saved prediction JSON")
    extra = [f"try_cast(json_extract_string(prediction_json,'$.{key}') AS BIGINT) AS {key}" for key in EXTRA_NUMERIC]
    extra += [f"json_extract_string(prediction_json,'$.{key}') AS {key}" for key in EXTRA_TEXT]
    extra += ["cast(json_extract(prediction_json,'$.response_token_ids') AS VARCHAR) AS response_token_ids_json",
              "try_cast(json_array_length(json_extract(prediction_json,'$.response_token_ids')) AS BIGINT) AS response_token_ids_count"]
    db.execute("CREATE TABLE requests AS SELECT *, " + ",".join(extra) + ", input_tokens/nullif(latency_seconds,0) AS input_tokens_per_request_second, output_tokens/nullif(latency_seconds,0) AS output_tokens_per_request_second FROM raw_requests")
    db.execute("ALTER TABLE requests ADD COLUMN window_id VARCHAR")
    for window, rows in events.items():
        start = rows[0].get("started_at", rows[0]["at"])
        db.execute("UPDATE requests SET window_id=? WHERE try_cast(completed_at AS TIMESTAMPTZ)>=try_cast(? AS TIMESTAMPTZ)", [window, start])
    if continuation is not None:
        db.execute("ALTER TABLE requests ADD COLUMN source_runtime_origin VARCHAR")
        db.execute("UPDATE requests SET source_runtime_origin=CASE WHEN committed_rowid<=? THEN 'parent' ELSE 'current' END",
                   [continuation["copied_through_rowid"]])
    db.execute("DROP TABLE raw_requests")
    # Only this report's owned temporary artifact is removed.
    staging.unlink()


def request_summary(db, source, elapsed_seconds):
    report = query_one(db, f"""SELECT count(*) AS completed_requests,
        count(*) FILTER (WHERE api_success=1) AS api_success_requests,
        count(*) FILTER (WHERE usable=1) AS usable_requests,
        count(*) FILTER (WHERE usable<>1) AS deferred_requests,
        count(*) FILTER (WHERE truncated=1) AS truncated_requests,
        sum(occurrence_count) AS represented_specimens,
        min(completed_at) AS first_completed_at, max(completed_at) AS last_completed_at,
        count(*) FILTER (WHERE length(trim(coalesce(notes,'')))>0) AS requests_with_notes,
        count(*) FILTER (WHERE length(trim(coalesce(collection_notes,'')))>0) AS requests_with_collection_notes,
        count(*) FILTER (WHERE response_token_ids_count IS NOT NULL) AS requests_with_token_ids,
        count(*) FILTER (WHERE reasoning_tokens>0) AS requests_with_positive_reasoning_tokens,
        count(*) FILTER (WHERE api_success=1 AND reasoning_tokens IS NULL) AS api_success_requests_with_unverified_reasoning,
        count(*) FILTER (WHERE api_success=1 AND usable<>1 AND reasoning_tokens IS NULL) AS deferred_requests_with_unverified_reasoning,
        count(*) FILTER (WHERE usable=1 AND reasoning_tokens IS NULL) AS usable_requests_with_unverified_reasoning,
        count(*) FILTER (WHERE output_tokens IS NOT NULL AND final_output_tokens IS NOT NULL AND reasoning_tokens IS NOT NULL AND output_tokens<>final_output_tokens+reasoning_tokens) AS token_partition_mismatches,
        count(*) FILTER (WHERE input_tokens IS NOT NULL AND output_tokens IS NOT NULL AND total_tokens IS NOT NULL AND total_tokens<>input_tokens+output_tokens) AS token_total_mismatches
        FROM {source}""")
    report["latency_seconds"] = query_one(db, f"""SELECT count(latency_seconds) AS observed_requests,
        sum(latency_seconds) AS summed_request_seconds, min(latency_seconds) AS min,
        avg(latency_seconds) AS mean, median(latency_seconds) AS median,
        quantile_cont(latency_seconds,0.90) AS p90, quantile_cont(latency_seconds,0.95) AS p95,
        quantile_cont(latency_seconds,0.99) AS p99, max(latency_seconds) AS max FROM {source}""")
    report["tokens"] = {}
    for key in TOKEN_FIELDS:
        values = query_one(db, f"SELECT count({key}) AS observed_requests,sum({key}) AS total,avg({key}) AS mean_per_observed_request,median({key}) AS median_per_observed_request,quantile_cont({key},0.95) AS p95_per_observed_request FROM {source}")
        values["per_wall_second"] = values["total"] / elapsed_seconds if elapsed_seconds and values["total"] is not None else None
        report["tokens"][key] = values
    for column in ("confidence", "token_source", "reasoning_token_source", "finish_reason", "stop_reason", "output_cap_tokens"):
        report[column + "_counts"] = dict(db.execute(f"SELECT coalesce(cast({column} AS VARCHAR),'unavailable'),count(*) FROM {source} GROUP BY 1 ORDER BY 1").fetchall())
    report["rate_denominator_wall_seconds"] = elapsed_seconds
    for key in ("completed_requests", "api_success_requests", "usable_requests"):
        report[key + "_per_hour"] = report[key] * 3600 / elapsed_seconds if elapsed_seconds else None
    report["mean_seconds_per_completed_request"] = elapsed_seconds / report["completed_requests"] if elapsed_seconds and report["completed_requests"] else None
    report["mean_output_tokens_per_request_second"] = db.execute(f"SELECT avg(output_tokens_per_request_second) FROM {source}").fetchone()[0]
    report["token_interpretation"] = "Totals cover saved completions only, including failed/truncated saved attempts when telemetry exists. Interrupted unsaved requests may consume additional unknown tokens. Per-request output tokens/latency includes prefill and queue waiting; it is not decode speed or time-to-first-token. Cache counters may overlap and must not be added to token totals. Missing telemetry stays null."
    return report


def evaluation_tables(db, prep):
    db.execute(f"CREATE VIEW cohorts AS SELECT * FROM read_parquet({sql_string(prep / 'candidate_cohorts.parquet')})")
    db.execute(f"CREATE VIEW refs AS SELECT * FROM read_parquet({sql_string(prep / 'evaluation_only/references.parquet')})")
    db.execute("""CREATE TABLE evaluation AS SELECT p.candidate_id,p.window_id,r.gbifid,
        r.datasetkey,r.publishingorgkey,r.countrycode,p.confidence,
        p.api_success,p.usable,p.locality,p.context_json,
        r.decimallatitude AS reference_latitude,r.decimallongitude AS reference_longitude,
        r.coordinateuncertaintyinmeters AS reference_uncertainty_m,
        p.latitude AS prediction_latitude,p.longitude AS prediction_longitude,
        p.coordinate_radius_meters AS prediction_radius_m,
        CASE WHEN p.usable=1 AND isfinite(p.latitude) AND isfinite(p.longitude)
            AND abs(p.latitude)<=90 AND abs(p.longitude)<=180
            AND isfinite(r.decimallatitude) AND isfinite(r.decimallongitude)
            AND abs(r.decimallatitude)<=90 AND abs(r.decimallongitude)<=180
        THEN 12742.0*asin(sqrt(least(1.0,greatest(0.0,
            pow(sin(radians(p.latitude-r.decimallatitude)/2),2)
            +cos(radians(r.decimallatitude))*cos(radians(p.latitude))
            *pow(sin(radians(p.longitude-r.decimallongitude)/2),2)))))
        END AS error_km
        FROM requests p JOIN refs r USING(candidate_id)""")
    for name, condition in {
        **{f"within_{km}km": f"error_km<={km}" for km in THRESHOLDS},
        "within_reference_radius": "CASE WHEN isfinite(reference_uncertainty_m) AND reference_uncertainty_m>0 THEN error_km*1000<=reference_uncertainty_m END",
        "within_model_radius": "CASE WHEN isfinite(prediction_radius_m) AND prediction_radius_m>=0 THEN error_km*1000<=prediction_radius_m END",
        "within_combined_radius": "CASE WHEN isfinite(reference_uncertainty_m) AND reference_uncertainty_m>0 AND isfinite(prediction_radius_m) AND prediction_radius_m>=0 THEN error_km*1000<=reference_uncertainty_m+prediction_radius_m END",
    }.items():
        db.execute(f"ALTER TABLE evaluation ADD COLUMN {name} BOOLEAN")
        db.execute(f"UPDATE evaluation SET {name}={condition}")


AGREEMENT_FIELDS = tuple(f"within_{km}km" for km in THRESHOLDS) + ("within_reference_radius", "within_model_radius", "within_combined_radius")


def agreement_summary(db, source):
    result = query_one(db, f"""SELECT count(*) AS attempted_specimens,
        count(DISTINCT candidate_id) AS attempted_unique_inputs,
        count(error_km) AS evaluated_specimens,
        count(DISTINCT candidate_id) FILTER (WHERE error_km IS NOT NULL) AS evaluated_unique_inputs,
        avg(error_km) AS specimen_mean_error_km,median(error_km) AS specimen_median_error_km,
        quantile_cont(error_km,0.90) AS specimen_p90_error_km,
        quantile_cont(error_km,0.95) AS specimen_p95_error_km,
        count(*) FILTER (WHERE reference_uncertainty_m=0) AS reference_zero_radius_specimens,
        count(*) FILTER (WHERE reference_uncertainty_m<0 OR NOT isfinite(reference_uncertainty_m)) AS reference_invalid_radius_specimens,
        count(*) FILTER (WHERE reference_uncertainty_m IS NULL) AS reference_missing_radius_specimens
        FROM {source}""")
    input_query = f"(SELECT candidate_id,median(error_km) AS median_error_km,count(*) AS specimens FROM {source} WHERE error_km IS NOT NULL GROUP BY candidate_id)"
    result["equal_input_distance"] = query_one(db, f"SELECT count(*) AS inputs,avg(median_error_km) AS mean_of_input_medians_km,median(median_error_km) AS median_of_input_medians_km,quantile_cont(median_error_km,0.90) AS p90_of_input_medians_km,quantile_cont(median_error_km,0.95) AS p95_of_input_medians_km,max(specimens) AS largest_input_specimens FROM {input_query}")
    evaluated = result["evaluated_specimens"]
    largest = result["equal_input_distance"]["largest_input_specimens"]
    result["equal_input_distance"]["largest_input_specimen_fraction"] = largest / evaluated if evaluated else None
    result["agreement"] = {}
    for key in AGREEMENT_FIELDS:
        values = query_one(db, f"SELECT count({key}) AS eligible_specimens,count(*) FILTER (WHERE {key}) AS agreeing_specimens,avg(cast({key} AS DOUBLE)) AS specimen_rate FROM {source}")
        values.update(query_one(db, f"SELECT count(*) AS eligible_inputs,avg(input_rate) AS equal_input_mean_rate FROM (SELECT candidate_id,avg(cast({key} AS DOUBLE)) AS input_rate FROM {source} GROUP BY candidate_id HAVING count({key})>0)"))
        values["all_attempted_specimens_rate"] = values["agreeing_specimens"] / result["attempted_specimens"] if result["attempted_specimens"] else None
        result["agreement"][key] = values
    result["interpretation"] = "Agreement with existing coordinates, not verified accuracy. Each specimen contributes once to specimen metrics. Equal-input rates average each input's eligible-specimen agreement fraction; inputs lacking the required radii are excluded with denominators shown. The compatibility all_attempted_specimens_rate instead divides by every attempted reference specimen, including missing/invalid predictions or radii. Distance summaries give each input's median error one vote. Reference radii must be finite and positive (zero is excluded); model radii may be zero. Combined radius is the sum, not quadrature. Confidence is a model self-assessment, not a calibrated probability."
    return result


def group_summaries(db, source, column):
    # One grouped query per dimension avoids rescanning millions of specimens
    # for every country/publisher and every individual metric.
    specimen_agreement = ",".join(
        f"count({key}) AS {key}__eligible_specimens,"
        f"count(*) FILTER (WHERE {key}) AS {key}__agreeing_specimens,"
        f"avg(cast({key} AS DOUBLE)) AS {key}__specimen_rate"
        for key in AGREEMENT_FIELDS)
    input_agreement = ",".join(f"avg(cast({key} AS DOUBLE)) AS {key}_input_rate" for key in AGREEMENT_FIELDS)
    grouped_input_agreement = ",".join(
        f"count({key}_input_rate) AS {key}__eligible_inputs,"
        f"avg({key}_input_rate) AS {key}__equal_input_mean_rate" for key in AGREEMENT_FIELDS)
    cursor = db.execute(f"""WITH base AS (SELECT *,coalesce({column},'unavailable') AS group_key FROM {source}),
        specimens AS (SELECT group_key,count(*) AS attempted_specimens,
            count(DISTINCT candidate_id) AS attempted_unique_inputs,
            count(error_km) AS evaluated_specimens,
            count(DISTINCT candidate_id) FILTER (WHERE error_km IS NOT NULL) AS evaluated_unique_inputs,
            avg(error_km) AS specimen_mean_error_km,median(error_km) AS specimen_median_error_km,
            quantile_cont(error_km,0.90) AS specimen_p90_error_km,
            quantile_cont(error_km,0.95) AS specimen_p95_error_km,
            count(*) FILTER (WHERE reference_uncertainty_m=0) AS reference_zero_radius_specimens,
            count(*) FILTER (WHERE reference_uncertainty_m<0 OR NOT isfinite(reference_uncertainty_m)) AS reference_invalid_radius_specimens,
            count(*) FILTER (WHERE reference_uncertainty_m IS NULL) AS reference_missing_radius_specimens,
            {specimen_agreement} FROM base GROUP BY group_key),
        inputs AS (SELECT group_key,candidate_id,count(error_km) AS evaluated_specimens,
            median(error_km) AS input_median_error_km,{input_agreement}
            FROM base GROUP BY group_key,candidate_id),
        input_groups AS (SELECT group_key,count(input_median_error_km) AS inputs,
            avg(input_median_error_km) AS mean_of_input_medians_km,
            median(input_median_error_km) AS median_of_input_medians_km,
            quantile_cont(input_median_error_km,0.90) AS p90_of_input_medians_km,
            quantile_cont(input_median_error_km,0.95) AS p95_of_input_medians_km,
            max(evaluated_specimens) FILTER (WHERE evaluated_specimens>0) AS largest_input_specimens,
            {grouped_input_agreement} FROM inputs GROUP BY group_key)
        SELECT * FROM specimens JOIN input_groups USING(group_key) ORDER BY group_key""")
    names = [item[0] for item in cursor.description]
    result = {}
    for row in cursor.fetchall():
        values = dict(zip(names, row))
        key = values.pop("group_key")
        agreement = {name: {field.split("__", 1)[1]: values.pop(field)
                           for field in list(values) if field.startswith(name + "__")}
                     for name in AGREEMENT_FIELDS}
        distance = {field: values.pop(field) for field in (
            "inputs", "mean_of_input_medians_km", "median_of_input_medians_km",
            "p90_of_input_medians_km", "p95_of_input_medians_km", "largest_input_specimens")}
        distance["largest_input_specimen_fraction"] = distance["largest_input_specimens"] / values["evaluated_specimens"] if values["evaluated_specimens"] else None
        for item in agreement.values():
            item["all_attempted_specimens_rate"] = item["agreeing_specimens"] / values["attempted_specimens"] if values["attempted_specimens"] else None
        result[key] = {**values, "equal_input_distance": distance, "agreement": agreement}
    return result


def cohort_counts(db, source):
    return query_one(db, f"""SELECT count(*) AS attempted_unique_inputs,
        coalesce(sum(c.production_records),0) AS attempted_production_specimens,
        coalesce(sum(c.evaluation_records),0) AS attempted_evaluation_specimens,
        coalesce(sum(c.production_records) FILTER (WHERE p.usable=1),0) AS usable_production_specimens,
        coalesce(sum(c.evaluation_records) FILTER (WHERE p.usable=1),0) AS usable_evaluation_specimens,
        count(*) FILTER (WHERE c.production_records>0) AS production_unique_inputs,
        count(*) FILTER (WHERE c.evaluation_records>0) AS evaluation_unique_inputs,
        count(*) FILTER (WHERE c.production_records>0 AND c.evaluation_records>0) AS shared_unique_inputs
        FROM {source} p JOIN cohorts c USING(candidate_id)""")


def markdown_report(report):
    req = report["window"]["requests"]
    ev = report["cumulative"]["evaluation"]
    lines = ["# PlaceBot experiment report", "", f"Window: `{report['window_id']}`. Reported from an immutable checkpoint; no inference rerun.", "",
        f"This window saved **{req['completed_requests']:,} requests**, {req['usable_requests']:,} usable, {req['deferred_requests']:,} deferred. Rates include startup and recovery through the final saved completion.", "",
        "| Token field | Recorded total | Mean per observed request | Observed requests |", "| --- | ---: | ---: | ---: |"]
    for key, value in req["tokens"].items():
        mean = value["mean_per_observed_request"]
        lines.append(f"| {key} | {value['total']} | {mean:.3f} | {value['observed_requests']} |" if mean is not None else f"| {key} | unavailable | unavailable | 0 |")
    lines.extend(["", f"Latency: mean {req['latency_seconds']['mean']} s; median {req['latency_seconds']['median']} s; p95 {req['latency_seconds']['p95']} s. Notes: {req['requests_with_notes']:,}/{req['completed_requests']:,} requests.", "",
        f"Cumulative evaluation: {ev['evaluated_specimens']:,} specimens from {ev['evaluated_unique_inputs']:,} inputs. Specimen median error: {ev['specimen_median_error_km']} km; median of input medians: {ev['equal_input_distance']['median_of_input_medians_km']} km.", "",
        "| Agreement | Agreeing / eligible specimens | Specimen rate | Equal-input mean rate | Eligible inputs |", "| --- | ---: | ---: | ---: | ---: |"])
    for key, values in ev["agreement"].items():
        lines.append(f"| {key} | {values['agreeing_specimens']} / {values['eligible_specimens']} | {values['specimen_rate']} | {values['equal_input_mean_rate']} | {values['eligible_inputs']} |")
    hardware = report["window"].get("hardware_telemetry", {"available": False})
    hardware_note = ("No continuous GPU telemetry summary was recorded for this window, so historical power/energy, temperature and utilisation cannot be reconstructed from saved predictions."
                     if hardware.get("available") is False else "Recorded GPU/server telemetry and its measurement coverage are included under window.hardware_telemetry in report.json; measurement gaps and estimated energy remain labelled by the sampler.")
    if hardware.get("available") is not False and not hardware.get("closed"):
        hardware_note += " The collector had not finalized this window when the report was produced; its telemetry summary is incomplete."
    lines.extend(["", ev["interpretation"], "", req["token_interpretation"], "",
        f"Fast-mode telemetry: {req['deferred_requests_with_unverified_reasoning']:,} deferred API-successful responses have unavailable or ambiguous reasoning counts; {req['usable_requests_with_unverified_reasoning']:,} usable responses have unavailable counts. Missing counts remain unknown and are not treated as verified zero reasoning.", "",
        "Full JSON includes per-window and cumulative token/latency/coverage summaries and country, publisher and confidence breakdowns. requests.parquet retains one row per saved request, raw prediction JSON, response token IDs, notes, all recorded cache/token counters, status and timing. evaluation.parquet retains per-specimen comparisons and original reference coordinates/radii; evaluation_inputs.parquet retains per-input medians.", "",
        hardware_note + " Per-request time-to-first-token is unavailable from saved non-streaming completions. No model-written explanation or model confidence is treated as independent evidence of correctness.", ""])
    return "\n".join(lines)


def generate_report(root, window, output, preparation=None, *,
                    memory_limit=DEFAULT_MEMORY_LIMIT, threads=DEFAULT_THREADS,
                    parquet_row_group_size=DEFAULT_PARQUET_ROW_GROUP_SIZE):
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if (not isinstance(parquet_row_group_size, int) or isinstance(parquet_row_group_size, bool)
            or parquet_row_group_size < 2048):
        raise ValueError("parquet_row_group_size must be an integer of at least 2048")
    root, output = Path(root).resolve(), Path(output).resolve()
    config = read_json(root / "experiment.json")
    prep = Path(preparation or config["preparation"]).resolve()
    window_dir = root / "windows" / window
    snapshot = window_dir / "results.sqlite3"
    checkpoint = read_json(window_dir / "checkpoint_manifest.json")
    snapshot_hash = sha256(snapshot)
    if checkpoint["files"]["results.sqlite3"]["sha256"] != snapshot_hash:
        raise ValueError("Checkpoint snapshot hash does not match its manifest")
    prep_manifest = read_json(prep / "manifest.json")
    for key in ("candidate_cohorts", "withheld_references"):
        artifact = prep_manifest["artifacts"][key]
        if sha256(prep / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Preparation artifact hash mismatch: {key}")
    events = read_window_events(root / "events.jsonl", window)
    run_manifest = root / "run" / "run.json"
    continuation = read_json(run_manifest).get("continuation") if run_manifest.exists() else None
    if continuation is not None:
        boundary = continuation.get("copied_through_rowid")
        if not isinstance(boundary, int) or isinstance(boundary, bool) or boundary < 0:
            raise ValueError("Continuation copied_through_rowid must be a nonnegative integer")
    # Exclusive creation is the publication guard; existing reports never change.
    output.mkdir(parents=True, exist_ok=False)
    db = duckdb.connect(str(output / "report_staging.duckdb"))
    try:
        db.execute(f"SET threads={threads}")
        db.execute(f"SET memory_limit={sql_string(memory_limit)}")
        # Keep request order while bounding each Parquet writer's buffered rows.
        # The previous eight writers with default 122,880-row groups exhausted
        # the 4GB allowance when exporting the 868,100-request checkpoint.
        db.execute("SET preserve_insertion_order=true")
        db.execute(f"SET temp_directory={sql_string(output / 'duckdb_temp')}")
        stage_requests(db, snapshot, output, events, continuation)
        last_by_window = dict(db.execute("SELECT window_id,max(completed_at) FROM requests GROUP BY window_id").fetchall())
        if None in last_by_window:
            raise ValueError("Requests predate all recorded window authorizations")
        timing = {key: timing_for_window(rows, last_by_window.get(key)) for key, rows in events.items()}
        wall = timing[window]["authorization_to_last_completion_seconds"]
        total_wall = sum(item["authorization_to_last_completion_seconds"] or 0 for item in timing.values())
        db.execute(f"CREATE VIEW window_requests AS SELECT * FROM requests WHERE window_id={sql_string(window)}")
        evaluation_tables(db, prep)
        db.execute(f"CREATE VIEW window_evaluation AS SELECT * FROM evaluation WHERE window_id={sql_string(window)}")
        db.execute("CREATE TABLE evaluation_inputs AS SELECT candidate_id,window_id,count(*) AS attempted_specimens,count(error_km) AS evaluated_specimens,median(error_km) AS median_error_km,avg(error_km) AS mean_error_km," + ",".join(f"count({key}) AS {key}_eligible_specimens,avg(cast({key} AS DOUBLE)) AS {key}_rate" for key in AGREEMENT_FIELDS) + " FROM evaluation GROUP BY candidate_id,window_id")
        cumulative = {"requests": request_summary(db, "requests", total_wall),
                      "cohorts": cohort_counts(db, "requests"),
                      "evaluation": agreement_summary(db, "evaluation")}
        window_summary = {"requests": request_summary(db, "window_requests", wall),
                          "cohorts": cohort_counts(db, "window_requests"),
                          "evaluation": agreement_summary(db, "window_evaluation")}
        # Group counts are bounded by country/publisher/confidence, never specimen IDs.
        cumulative["evaluation_by"] = {label: group_summaries(db, "evaluation", column) for label, column in (("country", "countrycode"), ("publisher", "publishingorgkey"), ("confidence", "confidence"))}
        window_summary["evaluation_by"] = (cumulative["evaluation_by"] if len(events) == 1 else {label: group_summaries(db, "window_evaluation", column) for label, column in (("country", "countrycode"), ("publisher", "publishingorgkey"), ("confidence", "confidence"))})
        report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                  "window_id": window, "timing_by_window": timing,
                  "cumulative": cumulative, "window": window_summary,
                  "runtime": {"duckdb_version": duckdb.__version__,
                    "memory_limit": str(memory_limit),
                    "effective_memory_limit": db.execute("SELECT current_setting('memory_limit')").fetchone()[0],
                    "threads": threads, "parquet_row_group_size": parquet_row_group_size,
                    "preserve_insertion_order": True},
                  "unavailable_historical_telemetry": ["GPU power/energy time series", "GPU utilisation/memory/temperature time series", "per-request time-to-first-token", "tokens consumed by unsaved interrupted requests"],
                  "grouping_note": "Country is source countrycode. Publisher is source publishingorgkey. An input can occur in several publisher groups; group unique-input counts need not add up to the overall count. All reference records remain a baseline, not independently verified truth.",
                  "provenance": {"checkpoint_snapshot": str(snapshot), "checkpoint_sha256": snapshot_hash,
                    "checkpoint_manifest_sha256": sha256(window_dir / "checkpoint_manifest.json"),
                    "preparation_manifest_sha256": sha256(prep / "manifest.json"),
                    "preparation_source": prep_manifest.get("source"),
                    "run_identity_sha256": checkpoint.get("run_identity_sha256"),
                    "report_script_sha256": sha256(__file__),
                    "inference_unchanged": True, "new_model_requests": 0}}
        if run_manifest.exists():
            report["provenance"]["run_manifest_sha256"] = sha256(run_manifest)
        if continuation is not None:
            report["provenance"]["continuation"] = continuation
            report["provenance"]["source_runtime_origin_note"] = (
                "requests.parquet source_runtime_origin labels committed_rowid <= continuation.copied_through_rowid "
                "as parent; later rows are current. The continuation identifies the parent experiment and run manifest; "
                "run_manifest_sha256 identifies this experiment's run manifest.")
        telemetry = root / "telemetry" / window / "summary.json"
        report["window"]["hardware_telemetry"] = (read_json(telemetry) if telemetry.exists() else {
            "available": False, "reason": "No telemetry summary recorded for this window; retrospective GPU measurements are unavailable."})
        if telemetry.exists():
            report["provenance"]["telemetry_summary_sha256"] = sha256(telemetry)
            report["unavailable_historical_telemetry"] = [
                "per-request time-to-first-token", "tokens consumed by unsaved interrupted requests"]
        for name in ("requests", "evaluation", "evaluation_inputs"):
            db.execute(f"COPY {name} TO {sql_string(output / (name + '.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {parquet_row_group_size})")
        write_json(output / "report.json", report)
        with (output / "report.md").open("x") as handle:
            handle.write(markdown_report(report))
        write_json(output / "events_used.json", events)
    finally:
        db.close()
    # Staging is our reproducible temporary data, not a source or published result.
    (output / "report_staging.duckdb").unlink()
    artifacts = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output / "COMPLETE.json", {"status": "complete", "window": window, "files": artifacts,
               "source_checkpoint_sha256": snapshot_hash, "new_model_requests": 0})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--output-dir", "--output", dest="output", type=Path, required=True)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--memory-limit", default=DEFAULT_MEMORY_LIMIT,
                        help="DuckDB memory allowance (default: %(default)s)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help="DuckDB worker threads (default: %(default)s)")
    parser.add_argument("--parquet-row-group-size", type=int, default=DEFAULT_PARQUET_ROW_GROUP_SIZE,
                        help="Rows buffered per Parquet writer, minimum 2048 (default: %(default)s)")
    args = parser.parse_args()
    result = generate_report(args.root, args.window, args.output, args.preparation,
                             memory_limit=args.memory_limit, threads=args.threads,
                             parquet_row_group_size=args.parquet_row_group_size)
    print(json.dumps({"output": str(args.output), "window": result["window_id"],
                      "requests": result["window"]["requests"]["completed_requests"],
                      "new_model_requests": 0}))


if __name__ == "__main__":
    main()
