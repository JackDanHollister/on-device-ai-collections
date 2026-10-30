import json
from pathlib import Path
import sqlite3
import sys

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "gb300"))
import report_gb300_experiment_window as report
from placebot.cli import hpc


def fixture(tmp_path, second_window=False):
    root = tmp_path / "experiment"
    prep = tmp_path / "prep"
    prep.mkdir()
    (prep / "evaluation_only").mkdir()
    window = "two" if second_window else "one"
    snapshot_dir = root / "windows" / window
    snapshot_dir.mkdir(parents=True)
    snapshot = snapshot_dir / "results.sqlite3"
    with hpc._init_results_database(snapshot) as db:
        for key, done, success, tokens in [("a", 2, True, 100), ("b", 12, True, 200), ("c", 13, False, None)] + ([("d", 25, True, 300)] if second_window else []):
            prediction = {"reasoning_tokens": 0 if success else None,
                          "final_output_tokens": tokens // 10 if tokens else None,
                          "cached_tokens": 0 if success else None,
                          "token_source": "openai" if success else None,
                          "reasoning_token_source": "observed" if success else None,
                          "response_token_ids": [1, 2] if success else None,
                          "output_cap_tokens": 1024}
            hpc._insert_result(db, {"candidate_id": key, "locality": "A\tplace\nwith newline" if key == "a" else key,
                "input_country": "GB", "occurrence_count": 3 if key == "a" else 1,
                "example_identifier": key, "completed_at": f"2026-09-05T00:00:{done:02d}Z",
                "api_success": success, "usable": success, "latitude": 0 if success else None,
                "longitude": 0 if success else None, "coordinate_radius_meters": 5000 if key == "a" else 130000,
                "confidence": "high" if key == "a" else "low", "notes": "selected because" if key == "a" else "",
                "collection_notes": "", "latency_seconds": 2,
                "input_tokens": tokens, "output_tokens": tokens // 10 if tokens else None,
                "total_tokens": tokens * 11 // 10 if tokens else None,
                "truncated": False, "finish_reason": "stop" if success else "error",
                "prediction_json": json.dumps(prediction)})
    db.close()
    with duckdb.connect() as db:
        db.execute("""CREATE TABLE refs AS SELECT * FROM (VALUES
            ('a','a1','publisher1','dataset1','GB',0.0,0.0,1000.0),
            ('a','a2','publisher1','dataset1','GB',0.0,0.0,0.0),
            ('a','a3','publisher1','dataset1','GB',0.0,0.1,NULL),
            ('b','b1','publisher2','dataset2','GB',0.0,2.0,100000.0))
            t(candidate_id,gbifid,publishingorgkey,datasetkey,countrycode,decimallatitude,decimallongitude,coordinateuncertaintyinmeters)""")
        db.execute(f"COPY refs TO '{prep}/evaluation_only/references.parquet' (FORMAT PARQUET)")
        db.execute("CREATE TABLE cohorts AS SELECT * FROM (VALUES ('a',0,3),('b',0,1),('c',1,0),('d',1,0)) t(candidate_id,production_records,evaluation_records)")
        db.execute(f"COPY cohorts TO '{prep}/candidate_cohorts.parquet' (FORMAT PARQUET)")
    report.write_json(prep / "manifest.json", {"artifacts": {
        key: {"path": name, "sha256": report.sha256(prep / name)}
        for key, name in [("candidate_cohorts", "candidate_cohorts.parquet"), ("withheld_references", "evaluation_only/references.parquet")]}})
    report.write_json(root / "experiment.json", {"preparation": str(prep)})
    report.write_json(snapshot_dir / "checkpoint_manifest.json", {"files": {"results.sqlite3": {"sha256": report.sha256(snapshot)}}, "run_identity_sha256": "unchanged"})
    events = [
        {"event": "window_authorized", "at": "2026-09-05T00:00:00Z", "window": "one"},
        {"event": "runner_started", "at": "2026-09-05T00:00:01Z"},
        {"event": "runner_exited", "at": "2026-09-05T00:00:05Z", "returncode": -9},
        {"event": "runner_started", "at": "2026-09-05T00:00:10Z"},
        {"event": "draining", "at": "2026-09-05T00:00:12Z"},
        {"event": "checkpoint_snapshot", "at": "2026-09-05T00:00:15Z"},
        {"event": "paused", "at": "2026-09-05T00:00:16Z"},
    ]
    if second_window:
        events += [{"event": "window_authorized", "at": "2026-09-05T00:00:20Z", "window": "two"},
                   {"event": "runner_started", "at": "2026-09-05T00:00:21Z"},
                   {"event": "runner_drained", "at": "2026-09-05T00:00:26Z"},
                   {"event": "paused", "at": "2026-09-05T00:00:27Z"}]
    with (root / "events.jsonl").open("w") as handle:
        handle.writelines(json.dumps(event) + "\n" for event in events)
    return root, window, snapshot


def test_retrospective_tokens_weighting_radii_and_immutable_sources(tmp_path):
    root, window, snapshot = fixture(tmp_path)
    before = report.sha256(snapshot)
    output = tmp_path / "report"
    result = report.generate_report(root, window, output)
    assert result["runtime"]["memory_limit"] == "16GB"
    assert result["runtime"]["threads"] == 2
    assert result["runtime"]["parquet_row_group_size"] == 16384
    assert result["runtime"]["preserve_insertion_order"] is True
    assert result["runtime"]["duckdb_version"] == duckdb.__version__
    req = result["window"]["requests"]
    assert req["completed_requests"] == 3
    assert req["deferred_requests"] == 1
    assert req["tokens"]["total_tokens"]["total"] == 330
    assert req["tokens"]["total_tokens"]["observed_requests"] == 2
    assert req["tokens"]["total_tokens"]["per_wall_second"] == pytest.approx(330 / 13)
    assert req["latency_seconds"]["p95"] == 2
    assert req["requests_with_notes"] == 1
    assert req["tokens"]["reasoning_tokens"]["total"] == 0
    assert req["tokens"]["cache_read_input_tokens"]["total"] is None
    timing = result["timing_by_window"][window]
    assert timing["authorization_to_last_completion_seconds"] == 13
    assert timing["observed_client_seconds_lower_bound"] == 7
    assert timing["observed_client_seconds_upper_bound"] == 9
    ev = result["cumulative"]["evaluation"]
    assert ev["specimen_median_error_km"] == pytest.approx(5.559746, abs=0.00001)
    assert ev["equal_input_distance"]["median_of_input_medians_km"] == pytest.approx(111.194927, abs=0.00001)
    assert ev["agreement"]["within_10km"]["specimen_rate"] == 0.5
    assert ev["agreement"]["within_10km"]["equal_input_mean_rate"] == pytest.approx(1 / 3)
    assert ev["agreement"]["within_reference_radius"]["eligible_specimens"] == 2
    assert ev["agreement"]["within_reference_radius"]["specimen_rate"] == 0.5
    assert ev["agreement"]["within_reference_radius"]["all_attempted_specimens_rate"] == 0.25
    assert ev["agreement"]["within_5km"]["agreeing_specimens"] == 2
    assert ev["specimen_p95_error_km"] == pytest.approx(190.6983, abs=0.001)
    assert ev["agreement"]["within_combined_radius"]["specimen_rate"] == 1
    assert ev["agreement"]["within_model_radius"]["specimen_rate"] == 0.5
    assert ev["reference_zero_radius_specimens"] == 1
    assert result["window"]["hardware_telemetry"]["available"] is False
    assert result["cumulative"]["evaluation_by"]["publisher"]["publisher1"]["evaluated_specimens"] == 3
    with duckdb.connect() as db:
        row = db.execute("SELECT locality,response_token_ids_json,output_cap_tokens FROM read_parquet(?) WHERE candidate_id='a'", [str(output / "requests.parquet")]).fetchone()
        assert row == ("A\tplace\nwith newline", "[1,2]", 1024)
        columns = [item[0] for item in db.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(output / "requests.parquet")]).fetchall()]
        assert "source_runtime_origin" not in columns
    assert report.sha256(snapshot) == before
    assert not Path(str(snapshot) + "-shm").exists()
    assert (output / "COMPLETE.json").exists()
    saved_report = report.sha256(output / "report.json")
    with pytest.raises(FileExistsError):
        report.generate_report(root, window, output)
    assert report.sha256(output / "report.json") == saved_report


def test_later_window_separates_new_tokens_excludes_pause_and_preserves_cumulative(tmp_path):
    root, window, snapshot = fixture(tmp_path, second_window=True)
    telemetry = root / "telemetry" / window
    telemetry.mkdir(parents=True)
    report.write_json(telemetry / "summary.json", {"available": True, "estimated_gpu_energy_kwh": 0.1})
    result = report.generate_report(root, window, tmp_path / "report")
    assert result["window"]["requests"]["completed_requests"] == 1
    assert result["window"]["requests"]["tokens"]["total_tokens"]["total"] == 330
    assert result["window"]["requests"]["rate_denominator_wall_seconds"] == 5
    assert result["cumulative"]["requests"]["completed_requests"] == 4
    assert result["cumulative"]["requests"]["tokens"]["total_tokens"]["total"] == 660
    assert result["cumulative"]["requests"]["rate_denominator_wall_seconds"] == 18
    assert result["window"]["evaluation"]["evaluated_specimens"] == 0
    assert result["timing_by_window"]["two"]["client_runtime_has_unobserved_exit"] is False
    assert result["window"]["hardware_telemetry"]["estimated_gpu_energy_kwh"] == 0.1
    assert "Recorded GPU/server telemetry" in (tmp_path / "report" / "report.md").read_text()
    assert "GPU power/energy time series" not in result["unavailable_historical_telemetry"]


def test_rejects_snapshot_tampering_before_creating_report(tmp_path):
    root, window, snapshot = fixture(tmp_path)
    manifest = snapshot.parent / "checkpoint_manifest.json"
    data = json.loads(manifest.read_text())
    data["files"]["results.sqlite3"]["sha256"] = "wrong"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        report.generate_report(root, window, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_missing_runner_exit_is_never_counted_as_known_active_time():
    events = [{"event": "window_authorized", "at": "2026-09-05T00:00:00Z"},
              {"event": "runner_started", "at": "2026-09-05T00:00:01Z"},
              {"event": "runner_started", "at": "2026-09-05T00:00:10Z"}]
    result = report.timing_for_window(events, "2026-09-05T00:00:13Z")
    assert result["observed_client_seconds_lower_bound"] == 3
    assert result["observed_client_seconds_upper_bound"] is None
    assert result["client_runtime_has_unobserved_exit"] is True


def update_snapshot_hash(snapshot):
    manifest = snapshot.parent / "checkpoint_manifest.json"
    data = json.loads(manifest.read_text())
    data["files"]["results.sqlite3"]["sha256"] = report.sha256(snapshot)
    manifest.write_text(json.dumps(data))


def test_report_distinguishes_unverified_deferred_reasoning_from_verified_zero(tmp_path):
    root, window, snapshot = fixture(tmp_path)
    prediction = {"success": False, "api_success": True, "reasoning_tokens": None,
                  "final_output_tokens": None, "response_token_ids": [22, 13, 42]}
    with hpc._init_results_database(snapshot) as db:
        hpc._insert_result(db, {"candidate_id": "d", "locality": "Malformed result",
            "input_country": "GB", "occurrence_count": 1, "example_identifier": "d",
            "completed_at": "2026-09-05T00:00:14Z", "api_success": True,
            "usable": False, "latency_seconds": 1, "output_tokens": 3,
            "error": "Response parsing failed: Unexpected vLLM response format: invalid JSON",
            "prediction_json": json.dumps(prediction)})
    db.close()
    update_snapshot_hash(snapshot)
    output = tmp_path / "report"
    result = report.generate_report(root, window, output)
    req = result["cumulative"]["requests"]
    assert req["api_success_requests_with_unverified_reasoning"] == 1
    assert req["deferred_requests_with_unverified_reasoning"] == 1
    assert req["usable_requests_with_unverified_reasoning"] == 0
    assert req["tokens"]["reasoning_tokens"]["observed_requests"] == 2
    assert req["tokens"]["reasoning_tokens"]["total"] == 0
    with duckdb.connect() as db:
        assert db.execute("SELECT usable,reasoning_tokens,output_tokens FROM read_parquet(?) WHERE candidate_id='d'",
                          [str(output / "requests.parquet")]).fetchone() == (0, None, 3)
    assert "not treated as verified zero" in (output / "report.md").read_text()


def test_configurable_export_groups_keep_wide_requests_and_order(tmp_path):
    root, window, snapshot = fixture(tmp_path)
    source = sqlite3.connect(snapshot)
    try:
        cursor = source.execute("SELECT * FROM results WHERE candidate_id='a'")
        columns = [item[0] for item in cursor.description]
        template = dict(zip(columns, cursor.fetchone()))
        additional = []
        for index in range(4097):
            row = dict(template, candidate_id=f"wide-{index}", locality=f"wide-{index}:" + "place" * 1024)
            additional.append(tuple(row[column] for column in columns))
        source.executemany("INSERT INTO results VALUES (" + ",".join("?" for _ in columns) + ")", additional)
        source.commit()
    finally:
        source.close()
    update_snapshot_hash(snapshot)
    before = report.sha256(snapshot)
    output = tmp_path / "bounded-report"
    result = report.generate_report(root, window, output, memory_limit="256MB", threads=1,
                                    parquet_row_group_size=2048)
    assert result["runtime"]["memory_limit"] == "256MB"
    assert result["runtime"]["threads"] == 1
    assert result["runtime"]["parquet_row_group_size"] == 2048
    assert result["window"]["requests"]["completed_requests"] == 4100
    with duckdb.connect() as db:
        parquet = str(output / "requests.parquet")
        groups = db.execute("SELECT DISTINCT row_group_id,row_group_num_rows FROM parquet_metadata(?) ORDER BY 1", [parquet]).fetchall()
        assert groups == [(0, 2048), (1, 2048), (2, 4)]
        rowids = db.execute("SELECT committed_rowid FROM read_parquet(?)", [parquet]).fetchall()
        assert rowids == [(index,) for index in range(1, 4101)]
        locality = db.execute("SELECT locality FROM read_parquet(?) WHERE candidate_id='wide-4096'", [parquet]).fetchone()[0]
        assert locality == "wide-4096:" + "place" * 1024
    assert report.sha256(snapshot) == before
    assert (output / "COMPLETE.json").exists()


def test_quarantined_unicode_evidence_survives_report_as_escaped_string(tmp_path):
    root, window, snapshot = fixture(tmp_path)
    original = {"notes": "malformed \udcc4", "prediction_json": json.dumps({"notes": "malformed \udcc4"})}
    prediction = {"persistence_quarantine": {"kind": "invalid_unicode", "invalid_fields": ["notes"],
                   "original_result_json": json.dumps(original, ensure_ascii=True, sort_keys=True)},
                  "reasoning_tokens": 0, "final_output_tokens": 10,
                  "response_token_ids": [1, 2], "output_cap_tokens": 1024}
    saved_prediction = json.dumps(prediction, ensure_ascii=True)
    source = sqlite3.connect(snapshot)
    try:
        cursor = source.execute("SELECT * FROM results WHERE candidate_id='a'")
        columns = [item[0] for item in cursor.description]
        row = dict(zip(columns, cursor.fetchone()))
        row.update(candidate_id="quarantined", notes="[invalid_unicode] malformed \\udcc4",
                   prediction_json=saved_prediction, usable=0, api_success=0)
        source.execute("INSERT INTO results VALUES (" + ",".join("?" for _ in columns) + ")",
                       [row[column] for column in columns])
        source.commit()
    finally:
        source.close()
    update_snapshot_hash(snapshot)
    before = report.sha256(snapshot)
    output = tmp_path / "quarantine-report"
    result = report.generate_report(root, window, output)
    assert result["window"]["requests"]["deferred_requests"] == 2
    with duckdb.connect() as db:
        row = db.execute("SELECT prediction_json,response_token_ids_json,usable FROM read_parquet(?) WHERE candidate_id='quarantined'", [str(output / "requests.parquet")]).fetchone()
    assert row == (saved_prediction, "[1,2]", 0)
    recovered = json.loads(json.loads(row[0])["persistence_quarantine"]["original_result_json"])
    assert recovered == original
    assert report.sha256(snapshot) == before


def test_continuation_preserves_manifest_and_attributes_copied_requests(tmp_path):
    root, window, snapshot = fixture(tmp_path, second_window=True)
    continuation = {"parent_root": "/original/experiment", "parent_run_manifest_sha256": "parent-sha",
                    "parent_identity": {"runtime_sha256": "parent-runtime"}, "copied_through_rowid": 3,
                    "copied_completed_candidates": 3, "created_at": "2026-09-05T00:00:20Z"}
    run_manifest = root / "run" / "run.json"
    run_manifest.parent.mkdir()
    report.write_json(run_manifest, {"continuation": continuation})
    output = tmp_path / "continuation-report"
    result = report.generate_report(root, window, output)
    assert result["provenance"]["continuation"] == continuation
    assert result["provenance"]["run_manifest_sha256"] == report.sha256(run_manifest)
    with duckdb.connect() as db:
        rows = db.execute("SELECT candidate_id,source_runtime_origin FROM read_parquet(?) ORDER BY committed_rowid", [str(output / "requests.parquet")]).fetchall()
    assert rows == [("a", "parent"), ("b", "parent"), ("c", "parent"), ("d", "current")]
    assert result["cumulative"]["requests"]["completed_requests"] == 4


@pytest.mark.parametrize("settings", [{"threads": 0}, {"threads": True}, {"parquet_row_group_size": 2047}])
def test_invalid_export_settings_rejected_before_output_creation(tmp_path, settings):
    output = tmp_path / "report"
    with pytest.raises(ValueError):
        report.generate_report(tmp_path / "missing-root", "window", output, **settings)
    assert not output.exists()


def test_partial_output_is_preserved_and_requires_fresh_directory(tmp_path):
    root, window, snapshot = fixture(tmp_path)
    output = tmp_path / "partial-report"
    output.mkdir()
    partial = output / "requests.parquet"
    partial.write_bytes(b"unfinished export")
    with pytest.raises(FileExistsError):
        report.generate_report(root, window, output)
    assert partial.read_bytes() == b"unfinished export"
    assert not (output / "COMPLETE.json").exists()


@pytest.mark.parametrize("extra_args,expected", [
    ([], {"memory_limit": "16GB", "threads": 2, "parquet_row_group_size": 16384}),
    (["--memory-limit", "8GB", "--threads", "1", "--parquet-row-group-size", "4096"],
     {"memory_limit": "8GB", "threads": 1, "parquet_row_group_size": 4096}),
])
def test_cli_old_arguments_and_optional_export_settings(monkeypatch, capsys, extra_args, expected):
    captured = {}

    def generate(*args, **kwargs):
        captured.update(args=args, settings=kwargs)
        return {"window_id": "one", "window": {"requests": {"completed_requests": 3}}}

    monkeypatch.setattr(report, "generate_report", generate)
    monkeypatch.setattr(sys, "argv", ["report", "--root", "/experiment", "--window", "one", "--output", "/report"] + extra_args)
    report.main()
    assert captured["args"] == (Path("/experiment"), "one", Path("/report"), None)
    assert captured["settings"] == expected
    assert json.loads(capsys.readouterr().out)["new_model_requests"] == 0
