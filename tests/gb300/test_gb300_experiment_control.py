import json
import csv
from pathlib import Path
import sqlite3
import sys

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "gb300"))
import gb300_experiment_control as ctl
from placebot.cli import hpc


def add_result(db, name, *, usable=True, success=True, reasoning=0, error=None, **extra):
    hpc._insert_result(db, {"candidate_id": name, "locality": "Somewhere",
        "input_country": "GB", "occurrence_count": 2, "example_identifier": name,
        "completed_at": "2026-09-05T00:00:00Z", "api_success": success,
        "usable": usable, "latency_seconds": 1.0, "error": error,
        "prediction_json": json.dumps({"reasoning_tokens": reasoning}), **extra})


@pytest.mark.parametrize("failures", [0, 2])
def test_snapshot_and_retry_queue_preserve_every_attempt(tmp_path, failures):
    run = tmp_path / "run"
    run.mkdir()
    config = {"root": str(tmp_path)}
    ctl.save(tmp_path / "experiment.json", config)
    ctl.save(run / "run.json", {"identity": "frozen"})
    with hpc._init_results_database(run / "results.sqlite3") as db:
        add_result(db, "good")
        for i in range(failures):
            add_result(db, f"bad-{i}", usable=False, success=False, error="Connection refused")
        assert db.execute("SELECT count(*) FROM deferred_failures").fetchone()[0] == failures
        assert hpc._existing_ids(db, [("good",), ("bad-0",)]) == ({"good", "bad-0"} if failures else {"good"})
    window = tmp_path / "window"
    window.mkdir()
    ctl.checkpoint_snapshot(config, window)
    assert ctl.read_metrics(window / "results.sqlite3") == {"completed": 1 + failures, "usable": 1, "failed": failures, "represented_records": 2 * (1 + failures)}
    with duckdb.connect() as db:
        retry = db.sql("SELECT * FROM read_parquet(?)", params=[str(window / "retry_candidates.parquet")])
        assert retry.count("*").fetchone()[0] == failures
        assert retry.columns == ["candidate_id", "locality", "country", "occurrence_count", "example_identifier", "context_json"]
    before = ctl.sha(window / "results.sqlite3")
    ctl.checkpoint_snapshot(config, window)
    assert ctl.sha(window / "results.sqlite3") == before


def test_fast_audit_rejects_thinking_and_missing_telemetry(tmp_path):
    path = tmp_path / "results.sqlite3"
    with hpc._init_results_database(path) as db:
        add_result(db, "good")
        add_result(db, "failed", usable=False, success=False, reasoning=None)
        assert ctl.audit_fast(path, 0) == 2
        add_result(db, "thinking", reasoning=20)
        with pytest.raises(RuntimeError, match="nonzero"):
            ctl.audit_fast(path, 2)


def test_network_audit_does_not_accept_semantic_or_worker_failure(tmp_path):
    ctl.save(tmp_path / "FAILED.json", {"reason": "100 consecutive unusable model outputs"})
    with hpc._init_results_database(tmp_path / "results.sqlite3") as db:
        for i in range(100):
            add_result(db, str(i), usable=False, success=False, error="Connection refused")
        assert ctl.transient_failed_run(tmp_path)
        add_result(db, "semantic", usable=False, success=True, error="Malformed schema")
        assert not ctl.transient_failed_run(tmp_path)
    ctl.save(tmp_path / "FAILED.json", {"reason": "Worker failed: ValueError"})
    assert not ctl.transient_failed_run(tmp_path)


@pytest.mark.parametrize("control,timestamp,free,stalled,expected", [
    ({"desired": "running", "deadline_epoch": 200}, 199, 100 * 1024**3, 0, None),
    ({"desired": "running", "deadline_epoch": 200}, 200, 100 * 1024**3, 0, "window_deadline"),
    ({"desired": "paused", "deadline_epoch": 200}, 100, 100 * 1024**3, 0, "operator_pause"),
    ({"desired": "running", "deadline_epoch": None}, 1e12, 100 * 1024**3, 0, None),
    ({"desired": "running"}, 100, 49 * 1024**3, 0, "disk_space_below_50_GiB"),
    ({"desired": "running"}, 100, 100 * 1024**3, 901, "no_committed_progress_for_15_minutes"),
])
def test_window_is_not_a_workload_partition(control, timestamp, free, stalled, expected):
    assert ctl.stop_reason(control, timestamp, free, stalled) == expected


def test_reboot_after_expired_deadline_never_launches_model(tmp_path, monkeypatch):
    ctl.save(tmp_path / "experiment.json", {"root": str(tmp_path)})
    ctl.save(tmp_path / "control.json", {"desired": "running", "window": "expired", "deadline_epoch": 1})
    monkeypatch.setattr(ctl, "verify", lambda *_: None)
    monkeypatch.setattr(ctl, "healthy", lambda *_: pytest.fail("Expired window must not request model"))
    ctl.supervise(tmp_path)
    assert ctl.read(tmp_path / "control.json")["stop_reason"] == "window_deadline"
    assert ctl.read(tmp_path / "control.json")["desired"] == "paused"


def test_frozen_runtime_drift_fails_closed(tmp_path):
    (tmp_path / "placebot/data").mkdir(parents=True)
    (tmp_path / "research/scripts").mkdir(parents=True)
    (tmp_path / "placebot/a.py").write_text("original")
    (tmp_path / "research/scripts/gb300_experiment_control.py").write_text("original")
    config = {"runtime": str(tmp_path), "runtime_files": ctl.runtime_files(tmp_path)}
    (tmp_path / "placebot/a.py").write_text("changed")
    with pytest.raises(RuntimeError, match="source changed"):
        ctl.verify(config)


def test_sqlite_corruption_is_not_adopted(tmp_path):
    path = tmp_path / "results.sqlite3"
    path.write_bytes(b"This is not SQLite")
    with pytest.raises(sqlite3.DatabaseError):
        ctl.sqlite_audit(path)


def test_scoring_only_joins_usable_results_to_withheld_references(tmp_path):
    prep = tmp_path / "prep"
    (prep / "evaluation_only").mkdir(parents=True)
    exported = tmp_path / "results.tsv"
    with exported.open("w", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["candidate_id", "usable", "latitude", "longitude", "coordinate_radius_meters"])
        writer.writerows([("good", 1, 50, 0, 1000), ("far", 1, 51, 0, 1000), ("failed", 0, 0, 0, 1000)])
    with duckdb.connect() as db:
        db.sql("SELECT * FROM (VALUES ('good',2,1),('far',0,1),('failed',1,1)) t(candidate_id,production_records,evaluation_records)").write_parquet(str(prep / "candidate_cohorts.parquet"))
        db.sql("""SELECT * FROM (VALUES ('good','1','org','dataset',50.0,0.0,1000),
            ('far','2','org','dataset',50.0,0.0,2000),('failed','3','org','dataset',50.0,0.0,1000))
            t(candidate_id,gbifid,publishingorgkey,datasetkey,decimallatitude,decimallongitude,coordinateuncertaintyinmeters)""").write_parquet(str(prep / "evaluation_only/references.parquet"))
    report = ctl.analyse_window({"preparation": str(prep)}, tmp_path, exported)
    assert report["production"] == {"attempted_specimens": 3, "usable_specimens": 2}
    assert report["evaluation"]["evaluated_specimens"] == 2
    assert report["evaluation"]["within_1km"] == 1
    assert report["evaluation"]["within_100km"] == 1
    assert report["evaluation"]["within_reference_uncertainty"] == 1
    assert report["evaluation"]["median_error_km"] == pytest.approx(55.59746, rel=1e-5)


def test_full_paused_snapshot_export_and_scoring_pipeline(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    prep = tmp_path / "prep"
    (prep / "evaluation_only").mkdir(parents=True)
    with duckdb.connect() as db:
        db.sql("SELECT 'good' AS candidate_id, 1 AS production_records, 1 AS evaluation_records").write_parquet(str(prep / "candidate_cohorts.parquet"))
        db.sql("SELECT 'good' AS candidate_id, '1' AS gbifid, 'org' AS publishingorgkey, 'dataset' AS datasetkey, 50.0 AS decimallatitude, 0.0 AS decimallongitude, 1000 AS coordinateuncertaintyinmeters").write_parquet(str(prep / "evaluation_only/references.parquet"))
    config = {"root": str(tmp_path), "preparation": str(prep), "runtime": str(Path(hpc.__file__).resolve().parents[2])}
    ctl.save(tmp_path / "experiment.json", config)
    ctl.save(run / "run.json", {"identity": {"frozen": True}})
    with hpc._init_results_database(run / "results.sqlite3") as db:
        add_result(db, "good", latitude=50.0, longitude=0.0, coordinate_radius_meters=1000)
    window = tmp_path / "window"
    window.mkdir()
    ctl.checkpoint_snapshot(config, window)
    manifest = ctl.read(window / "checkpoint_manifest.json")
    assert len(manifest["files"]) == 7
    assert all(ctl.sha(window / name) == item["sha256"] for name,item in manifest["files"].items())
    assert ctl.read(window / "cohort_report.json")["evaluation"]["within_1km"] == 1
    ctl.checkpoint_snapshot(config, window)


def test_stale_deadline_cannot_stop_a_later_window(tmp_path, monkeypatch):
    ctl.save(tmp_path / "experiment.json", {"unit": "never-touch-a-real-unit"})
    ctl.save(tmp_path / "control.json", {"window": "new", "desired": "running", "deadline_epoch": 1})
    monkeypatch.setattr(sys, "argv", ["control", "--root", str(tmp_path), "deadline", "--window", "old"])
    monkeypatch.setattr(ctl, "command", lambda *_: pytest.fail("Old timer must not stop a new window"))
    ctl.main()
    assert ctl.read(tmp_path / "control.json")["desired"] == "running"


def test_matching_deadline_persists_pause_before_stopping_service(tmp_path, monkeypatch):
    ctl.save(tmp_path / "experiment.json", {"unit": "test-only"})
    ctl.save(tmp_path / "control.json", {"window": "current", "desired": "running", "deadline_epoch": 1})
    monkeypatch.setattr(sys, "argv", ["control", "--root", str(tmp_path), "deadline", "--window", "current"])
    calls = []
    def stop(args):
        assert ctl.read(tmp_path / "control.json")["desired"] == "paused"
        calls.append(args)
    monkeypatch.setattr(ctl, "command", stop)
    ctl.main()
    assert calls == [["systemctl", "--user", "--no-block", "stop", "test-only"]]
