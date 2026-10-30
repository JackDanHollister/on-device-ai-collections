"""Real SQLite ledgers and simulated processes; never launch inference/services."""
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "gb300"))
import gb300_experiment_control_v2 as ctl
from placebot.cli import hpc


def append_rows(root, count, *, error=None, success=False):
    database = root / "run/results.sqlite3"
    with hpc._init_results_database(database) as db:
        start = db.execute("SELECT count(*) FROM results").fetchone()[0]
        for i in range(count):
            hpc._insert_result(db, {"candidate_id": str(start + i),
                "locality": "Somewhere", "input_country": "GB",
                "occurrence_count": 1, "example_identifier": str(start + i),
                "completed_at": "2026-09-05T00:00:00Z", "api_success": not error or success,
                "usable": not error, "latency_seconds": 1.0, "error": error,
                "prediction_json": json.dumps({"reasoning_tokens": 0})})


class Simulator:
    def __init__(self, root, monkeypatch, on_launch=None, on_tick=None, deadline=1300):
        self.root, self.elapsed, self.children, self.commands = root, 0, [], []
        self.on_launch, self.on_tick, self.on_drain = on_launch, on_tick, None
        self.ok = True
        self.container_running = True
        (root / "run").mkdir()
        ctl.save(root / "run/run.json", {"identity": "frozen"})
        ctl.save(root / "experiment.json", {"root": str(root), "runtime": str(root),
            "container": "test-only", "runner_command": ["never-execute"], "environment": {}})
        ctl.save(root / "control.json", {"desired": "running", "window": "test",
            "deadline_epoch": deadline})
        monkeypatch.setattr(ctl.time, "time", lambda: 1000 + self.elapsed)
        monkeypatch.setattr(ctl.time, "monotonic", lambda: self.elapsed)
        monkeypatch.setattr(ctl.signal, "signal", lambda *_: None)
        monkeypatch.setattr(ctl, "verify", lambda *_: None)
        monkeypatch.setattr(ctl, "docker_contract", lambda *_: {})
        monkeypatch.setattr(ctl, "healthy", lambda *_: self.ok)
        monkeypatch.setattr(ctl, "checkpoint_snapshot", lambda *_: None)
        monkeypatch.setattr(ctl.shutil, "disk_usage", lambda *_: SimpleNamespace(free=100 * 1024**3))
        monkeypatch.setattr(ctl.subprocess, "Popen", self.launch)
        monkeypatch.setattr(ctl, "command", self.command)
        sim = self
        class Event:
            def __init__(self): self.stopped = False
            def is_set(self): return self.stopped
            def set(self): self.stopped = True
            def wait(self, seconds):
                sim.elapsed += seconds
                if sim.on_tick: sim.on_tick(sim)
                assert sim.elapsed <= 2000, "Simulation did not terminate"
        monkeypatch.setattr(ctl.threading, "Event", Event)

    def command(self, args):
        self.commands.append(args)
        if args[:2] == ["docker", "inspect"]:
            return "true" if self.container_running else "false"
        if args[:2] == ["docker", "start"]:
            return "test-only"
        pytest.fail(f"Unexpected external command {args}")

    def launch(self, args, **kwargs):
        sim = self
        class Child:
            def __init__(self):
                self.pid = 100 + len(sim.children)
                self.returncode = None
                self.args = args
                self.signals = []
            def poll(self): return self.returncode
            def send_signal(self, sig):
                self.signals.append(sig)
                self.returncode = 130
                if sim.on_drain: sim.on_drain(sim, self)
            def wait(self, timeout): return self.returncode
            def kill(self): self.returncode = -signal.SIGKILL
        child = Child()
        self.children.append(child)
        if "--resume-after-audit" in args:
            (self.root / "run/FAILED.json").unlink()
        if self.on_launch: self.on_launch(self, child)
        return child

    def pause(self):
        control = ctl.read(self.root / "control.json")
        ctl.save(self.root / "control.json", {**control, "desired": "paused"})

    def run(self):
        ctl.supervise(self.root)
        self.control = ctl.read(self.root / "control.json")
        self.session = ctl.read(self.root / "windows/test/session.json")
        return self


@pytest.mark.parametrize("failed_marker", [False, True])
def test_network_burst_recovers_once_and_old_rows_do_not_poison_resume(tmp_path, monkeypatch, failed_marker):
    def launch(sim, child):
        if len(sim.children) == 1:
            append_rows(sim.root, 100, error="Request failed: Connection refused")
            if failed_marker:
                ctl.save(sim.root / "run/FAILED.json", {"reason": "100 consecutive unusable model outputs"})
                child.returncode = 2
        else:
            append_rows(sim.root, 10)
    def tick(sim):
        if len(sim.children) == 2: sim.pause()
    sim = Simulator(tmp_path, monkeypatch, launch, tick)
    append_rows(tmp_path, 500)
    sim.run()
    assert len(sim.children) == 2
    assert sim.session["recoveries"] == 1
    assert sim.session["transport_handled_rowid"] == 600
    assert sim.control["stop_reason"] == "operator_pause"
    assert ctl.read_metrics(tmp_path / "run/results.sqlite3")["completed"] == 610
    assert ("--resume-after-audit" in sim.children[1].args) == failed_marker


def test_health_outage_drain_defers_network_rows_and_resumes(tmp_path, monkeypatch):
    def tick(sim):
        if sim.elapsed == 15: sim.ok = False
        if sim.elapsed >= 60: sim.ok = True
        if len(sim.children) == 2: sim.pause()
    sim = Simulator(tmp_path, monkeypatch, on_tick=tick)
    def drain(sim, child):
        if len(sim.children) == 1: append_rows(sim.root, 100, error="Read timed out")
    sim.on_drain = drain
    sim.run()
    assert len(sim.children) == 2
    assert sim.session["recoveries"] == 1
    assert sim.session["transport_handled_rowid"] == 100
    assert sim.control["stop_reason"] == "operator_pause"


@pytest.mark.parametrize("success,error", [(True, "Malformed schema"), (False, "401 unauthorized"), (True, "Truncated response")])
def test_semantic_and_unknown_failure_bursts_still_stop(tmp_path, monkeypatch, success, error):
    sim = Simulator(tmp_path, monkeypatch, on_launch=lambda sim, child: append_rows(sim.root, 50, error=error, success=success)).run()
    assert len(sim.children) == 1
    assert sim.session["recoveries"] == 0
    assert sim.control["stop_reason"] == "audit_required"


def test_repeated_transport_bursts_exhaust_three_episode_budget(tmp_path, monkeypatch):
    sim = Simulator(tmp_path, monkeypatch, on_launch=lambda sim, child: append_rows(sim.root, 50, error="Connection reset by peer")).run()
    assert len(sim.children) == 4  # Initial launch plus three recoveries.
    assert sim.session["recoveries"] == 3
    assert sim.session["transport_handled_rowid"] == 150
    assert sim.control["stop_reason"] == "audit_required"


def test_stale_failures_do_not_poison_new_window(tmp_path, monkeypatch):
    sim = Simulator(tmp_path, monkeypatch, on_tick=lambda sim: sim.pause())
    append_rows(tmp_path, 500, error="Connection refused")
    sim.run()
    assert sim.session["window_start_rowid"] == 500
    assert sim.control["stop_reason"] == "operator_pause"
    assert len(sim.children) == 1


def test_semantic_history_survives_window_boundary(tmp_path, monkeypatch):
    sim = Simulator(tmp_path, monkeypatch, on_launch=lambda sim, child: append_rows(sim.root, 1, error="Malformed schema", success=True))
    append_rows(tmp_path, 451)
    append_rows(tmp_path, 49, error="Malformed schema", success=True)
    sim.run()
    assert sim.control["stop_reason"] == "audit_required"
    assert sim.session["recoveries"] == 0


@pytest.mark.parametrize("status,expected", [(502, "transport"), (503, "transport"), (504, "transport"), (500, "semantic_or_unknown"), (401, "semantic_or_unknown"), (429, "semantic_or_unknown")])
def test_saved_api_error_status_is_explicitly_classified(status, expected):
    assert ctl.failure_class(False, False, f"API error: {status}") == expected


def test_deadline_during_drain_prevents_replacement_launch(tmp_path, monkeypatch):
    sim = Simulator(tmp_path, monkeypatch, on_launch=lambda sim, child: append_rows(sim.root, 50, error="Connection refused"), deadline=1030)
    sim.on_drain = lambda sim, child: setattr(sim, "elapsed", sim.elapsed + 30)
    sim.run()
    assert len(sim.children) == 1
    assert sim.control["stop_reason"] == "window_deadline"


def test_already_expired_window_never_launches(tmp_path, monkeypatch):
    sim = Simulator(tmp_path, monkeypatch, deadline=999).run()
    assert sim.children == []
    assert sim.control["stop_reason"] == "window_deadline"


def test_model_crashing_before_ready_has_bounded_restart_attempts(tmp_path, monkeypatch):
    sim = Simulator(tmp_path, monkeypatch)
    sim.ok = sim.container_running = False
    sim.run()
    starts = [cmd for cmd in sim.commands if cmd[:2] == ["docker", "start"]]
    assert len(starts) == 3
    assert sim.session["recoveries"] == 3
    assert sim.control["stop_reason"] == "audit_required"


def test_unknown_failed_marker_is_not_bypassed(tmp_path, monkeypatch):
    sim = Simulator(tmp_path, monkeypatch)
    append_rows(tmp_path, 100, error="Connection refused")
    ctl.save(tmp_path / "run/FAILED.json", {"reason": "Worker failed: ValueError"})
    sim.run()
    assert not sim.children
    assert sim.control["stop_reason"] == "audit_required"


def test_killed_client_cannot_override_unknown_failed_marker(tmp_path, monkeypatch):
    def launch(sim, child):
        append_rows(sim.root, 100, error="401 unauthorized")
        ctl.save(sim.root / "run/FAILED.json", {"reason": "100 consecutive unusable model outputs"})
        child.returncode = -signal.SIGKILL
    sim = Simulator(tmp_path, monkeypatch, on_launch=launch).run()
    assert len(sim.children) == 1
    assert sim.session["recoveries"] == 0
    assert sim.control["stop_reason"] == "audit_required"


def test_operational_hashes_are_independent_of_inference_config(tmp_path):
    ctl.save(tmp_path / "experiment.json", {"frozen": True})
    script = tmp_path / "controller.py"
    script.write_text("original")
    ctl.save(tmp_path / "operations.json", {"experiment_sha256": ctl.sha(tmp_path / "experiment.json"), "files": {str(script): ctl.sha(script)}})
    ctl.verify_operations({"root": str(tmp_path)})
    script.write_text("changed")
    with pytest.raises(RuntimeError, match="Operational source changed"):
        ctl.verify_operations({"root": str(tmp_path)})


def test_same_recovery_episode_and_watermark_survive_supervisor_restart(tmp_path, monkeypatch):
    sim = Simulator(tmp_path, monkeypatch, on_tick=lambda sim: sim.pause())
    append_rows(tmp_path, 100, error="Connection refused")
    ctl.save(tmp_path / "run/FAILED.json", {"reason": "100 consecutive unusable model outputs"})
    ctl.save(tmp_path / "windows/test/session.json", {"recoveries": 2,
        "fast_audited_rowid": 0, "window_start_rowid": 0, "transport_handled_rowid": 100,
        "active_recovery": {"cause": "transport_failure_burst", "rowid": 100}})
    sim.run()
    assert sim.session["recoveries"] == 2
    assert sim.control["stop_reason"] == "operator_pause"
    assert "--resume-after-audit" in sim.children[0].args


def test_initial_session_is_durable_before_first_child_launch(tmp_path, monkeypatch):
    observed = []
    def launch(sim, child):
        session_path = sim.root / "windows/test/session.json"
        observed.append(ctl.read(session_path) if session_path.exists() else None)
        append_rows(sim.root, 3)
    sim = Simulator(tmp_path, monkeypatch, on_launch=launch, on_tick=lambda sim: sim.pause())
    append_rows(tmp_path, 7)
    sim.run()
    assert observed and observed[0] is not None
    assert observed[0]["window_start_rowid"] == 7
    assert observed[0]["transport_handled_rowid"] == 7
    assert observed[0]["recoveries"] == 0
    assert sim.session["window_start_rowid"] == 7


@pytest.mark.parametrize("reason", ["operator_pause", "window_deadline", "audit_required"])
def test_stop_intent_is_durable_before_drain_and_snapshot(tmp_path, monkeypatch, reason):
    observed = []
    def record(phase):
        control = ctl.read(tmp_path / "control.json")
        observed.append((phase, control.get("desired"), control.get("stop_reason")))
    on_launch = None
    if reason == "audit_required":
        on_launch = lambda sim, child: append_rows(sim.root, 50, error="Malformed schema", success=True)
    sim = Simulator(tmp_path, monkeypatch, on_launch=on_launch,
        on_tick=(lambda sim: sim.pause()) if reason == "operator_pause" else None,
        deadline=1010 if reason == "window_deadline" else 1300)
    sim.on_drain = lambda sim, child: record("drain")
    monkeypatch.setattr(ctl, "checkpoint_snapshot", lambda *_: record("snapshot"))
    sim.run()
    assert observed == [("drain", "paused", reason), ("snapshot", "paused", reason)]
    assert sim.control["stop_reason"] == reason


def test_existing_snapshot_is_rejected_if_live_ledger_advanced(tmp_path):
    (tmp_path / "run").mkdir()
    config = {"root": str(tmp_path)}
    ctl.save(tmp_path / "experiment.json", config)
    ctl.save(tmp_path / "run/run.json", {"identity": "frozen"})
    window = tmp_path / "window"
    window.mkdir()
    append_rows(tmp_path, 1)
    ctl.checkpoint_snapshot(config, window)
    snapshot_hash = ctl.sha(window / "results.sqlite3")
    manifest_hash = ctl.sha(window / "checkpoint_manifest.json")
    append_rows(tmp_path, 2)
    with pytest.raises(RuntimeError):
        ctl.checkpoint_snapshot(config, window)
    assert ctl.sha(window / "results.sqlite3") == snapshot_hash
    assert ctl.sha(window / "checkpoint_manifest.json") == manifest_hash
    assert ctl.read_metrics(tmp_path / "run/results.sqlite3")["completed"] == 3
