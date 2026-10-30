#!/usr/bin/env python3
"""Observe GPU/server activity for active experiment windows without model calls.

Telemetry is independent of inference and its checkpoint. JSONL chunks retain
every raw Prometheus scrape; summaries subtract the first successful baseline.
Energy is sampled GPU-board power only, with explicit missing-sample coverage.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from urllib.request import urlopen
import uuid


VERSION = 1
CHUNK_BYTES = 32 * 1024 * 1024
MAX_METRICS_BYTES = 2 * 1024 * 1024
GPU_FIELDS = ("index", "uuid", "name", "utilization.gpu", "memory.used",
              "memory.total", "temperature.gpu", "power.draw", "power.limit")
GPU_KEYS = ("index", "uuid", "name", "utilization_percent", "memory_used_mib",
            "memory_total_mib", "temperature_c", "power_w", "power_limit_w")
SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)(?:\s+.*)?$")


def utc(epoch=None):
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    partial = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.partial")
    with partial.open("x") as out:
        json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(partial, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def collector_identity():
    with Path(__file__).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"version": VERSION, "script_sha256": digest}


def parse_prometheus(raw):
    """Keep exact label strings and all cumulative histogram components."""
    types, counters, gauges = {}, {}, {}
    for line in raw.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split()[:4]
            types[name] = kind
    for line in raw.splitlines():
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        if not (name.startswith("vllm:") or name == "process_start_time_seconds"):
            continue
        try:
            numeric = float(value)
        except ValueError:
            continue
        if not math.isfinite(numeric):
            continue
        key = name + (labels or "")
        histogram = any(name.endswith(suffix) and types.get(name[:-len(suffix)]) == "histogram"
                        for suffix in ("_bucket", "_sum", "_count"))
        if types.get(name) == "counter" or histogram:
            counters[key] = numeric
        else:
            gauges[key] = numeric
    if not any(key.startswith("vllm:") for key in (*counters, *gauges)):
        raise ValueError("Metrics response contained no vLLM samples")
    return {"counters": counters, "gauges": gauges,
            "process_start_time_seconds": gauges.get("process_start_time_seconds")}


def parse_nvidia_smi(raw):
    gpus = []
    for row in csv.reader(io.StringIO(raw), skipinitialspace=True):
        if not row:
            continue
        if len(row) != len(GPU_KEYS):
            raise ValueError(f"Expected {len(GPU_KEYS)} GPU columns; received {len(row)}")
        item = dict(zip(GPU_KEYS, (value.strip() for value in row)))
        for key in GPU_KEYS:
            if key in ("uuid", "name"):
                continue
            try:
                number = float(item[key])
                item[key] = number if math.isfinite(number) and number >= 0 else None
            except ValueError:
                item[key] = None
        if not item["uuid"]:
            raise ValueError("GPU UUID is missing")
        gpus.append(item)
    if not gpus:
        raise ValueError("nvidia-smi returned no GPUs")
    return gpus


def capture(*, metrics_url, collector_id, wall=time.time, monotonic=time.monotonic):
    epoch = wall()
    result = {"kind": "sample", "at": utc(epoch), "epoch": epoch,
              "monotonic": monotonic(), "collector_id": collector_id,
              "gpus": None, "metrics": None, "errors": {}}
    try:
        raw = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS),
                              "--format=csv,noheader,nounits"], check=True, text=True,
                             capture_output=True, timeout=10).stdout
        result["gpus"] = parse_nvidia_smi(raw)
        result["gpu_sample_monotonic"] = monotonic()
    except Exception as exc:
        result["errors"]["gpu"] = f"{type(exc).__name__}: {exc}"
    try:
        with urlopen(metrics_url, timeout=8) as response:
            raw = response.read(MAX_METRICS_BYTES + 1)
        if len(raw) > MAX_METRICS_BYTES:
            raise ValueError("Metrics response exceeded 2 MiB sampling limit")
        raw = raw.decode("utf-8")
        result["metrics_raw"] = raw
        result["metrics"] = parse_prometheus(raw)
    except Exception as exc:
        result["errors"]["metrics"] = f"{type(exc).__name__}: {exc}"
    result["collection_seconds"] = monotonic() - result["monotonic"]
    return result


class Aggregate:
    def __init__(self, control, interval_seconds):
        self.window = control["window"]
        self.started_epoch = control.get("started_epoch")
        self.interval = interval_seconds
        self.samples = self.gpu_successes = self.metrics_successes = 0
        self.first_epoch = self.last_epoch = None
        self.errors = {"gpu": 0, "metrics": 0}
        self.gpus = {}
        self.previous_power = {}
        self.counters = {}
        self.previous_counters = None
        self.previous_process = None
        self.previous_collector = None
        self.collector_sessions = {}
        self.last_gauges = {}
        self.resets = []
        self.server_segments = []
        self.counter_series_baselined = 0
        self.metrics_baseline_at = None
        self.gpu_baseline_at = None

    def consume(self, sample):
        self.samples += 1
        epoch = sample["epoch"]
        self.first_epoch = epoch if self.first_epoch is None else self.first_epoch
        self.last_epoch = epoch
        collector = sample["collector_id"]
        session = self.collector_sessions.setdefault(collector, {
            "collector_id": collector, "collector": sample.get("collector"),
            "first_sample_at": sample["at"], "samples": 0})
        session.update(last_sample_at=sample["at"], samples=session["samples"] + 1)
        if collector != self.previous_collector:
            self.previous_power.clear()
        self.previous_collector = collector
        for kind in self.errors:
            self.errors[kind] += int(kind in sample.get("errors", {}))
        if sample["gpus"] is None:
            self.previous_power.clear()
        else:
            self.gpu_successes += 1
            self.gpu_baseline_at = self.gpu_baseline_at or sample["at"]
            present = set()
            for gpu in sample["gpus"]:
                key = gpu["uuid"]
                present.add(key)
                stats = self.gpus.setdefault(key, {"name": gpu["name"], "samples": 0,
                    "energy_joules": 0.0, "integrated_seconds": 0.0,
                    "omitted_intervals": 0, "measurements": {}})
                stats["samples"] += 1
                for field in GPU_KEYS:
                    value = gpu.get(field)
                    if field in ("index", "uuid", "name") or value is None:
                        continue
                    values = stats["measurements"].setdefault(field, {"min": value,
                        "max": value, "sum": 0.0, "count": 0})
                    values.update(min=min(values["min"], value), max=max(values["max"], value),
                                  sum=values["sum"] + value, count=values["count"] + 1)
                previous = self.previous_power.pop(key, None)
                power = gpu.get("power_w")
                current_time = sample.get("gpu_sample_monotonic", sample["monotonic"])
                if power is not None:
                    if previous:
                        elapsed = current_time - previous[0]
                        if 0 < elapsed <= self.interval * 1.5 + 5:
                            stats["energy_joules"] += (previous[1] + power) * 0.5 * elapsed
                            stats["integrated_seconds"] += elapsed
                        else:
                            stats["omitted_intervals"] += 1
                    self.previous_power[key] = (current_time, power)
            self.previous_power = {key: value for key, value in self.previous_power.items() if key in present}
        metrics = sample.get("metrics")
        if metrics is None:
            return
        self.metrics_successes += 1
        self.metrics_baseline_at = self.metrics_baseline_at or sample["at"]
        counters = metrics["counters"]
        process = metrics.get("process_start_time_seconds")
        previous = self.previous_counters
        reset = previous is not None and (
            (process is not None and self.previous_process is not None and process != self.previous_process)
            or any(value < previous[key] for key, value in counters.items() if key in previous))
        if reset:
            self.resets.append({"at": sample["at"], "previous_process_start": self.previous_process,
                                "process_start": process,
                                "reason": "process changed or cumulative metric decreased"})
        if previous is None or reset:
            self.server_segments.append({"segment": len(self.server_segments),
                "started_at": sample["at"], "process_start_time_seconds": process,
                "counter_delta_policy": "count_since_reset_zero" if reset else "subtract_first_observation"})
        for key, value in counters.items():
            if previous is None or key not in previous:
                self.counter_series_baselined += 1
                delta = value if reset else 0.0
            else:
                delta = value if reset else value - previous[key]
            self.counters[key] = self.counters.get(key, 0.0) + delta
        self.previous_counters = counters
        self.previous_process = process
        self.last_gauges = metrics["gauges"]

    def summary(self, *, closed=False, reason=None, identity=None):
        elapsed = max(0, (self.last_epoch or 0) - (self.first_epoch or 0))
        observed_window = max(0, (self.last_epoch or 0) - (self.started_epoch or self.first_epoch or 0))
        gpus = {}
        for key, stats in self.gpus.items():
            item = {k: v for k, v in stats.items() if k != "measurements"}
            item.update(energy_wh=stats["energy_joules"] / 3600,
                        observed_window_coverage_fraction=(stats["integrated_seconds"] / observed_window
                            if observed_window else None),
                        unintegrated_observed_window_seconds=max(0, observed_window - stats["integrated_seconds"]),
                        measurements={field: {"min": values["min"], "max": values["max"],
                            "sample_mean": values["sum"] / values["count"], "samples": values["count"]}
                            for field, values in stats["measurements"].items()})
            gpus[key] = item
        return {"schema_version": VERSION, "collector": identity, "window": self.window,
                "closed": closed, "close_reason": reason,
                "window_started_epoch": self.started_epoch,
                "collection_started_at": utc(self.first_epoch) if self.first_epoch is not None else None,
                "last_sample_at": utc(self.last_epoch) if self.last_epoch is not None else None,
                "sample_span_seconds": elapsed, "samples": self.samples,
                "gpu_successful_samples": self.gpu_successes,
                "metrics_successful_samples": self.metrics_successes, "errors": self.errors,
                "gpu_baseline_at": self.gpu_baseline_at, "metrics_baseline_at": self.metrics_baseline_at,
                "gpus": gpus, "cumulative_metric_deltas": self.counters,
                "last_gauges": self.last_gauges, "server_resets": self.resets,
                "server_segments": self.server_segments,
                "collector_sessions": list(self.collector_sessions.values()),
                "counter_series_baselined": self.counter_series_baselined,
                "limitations": [
                    "GPU-board energy only; excludes CPU, host RAM, cooling, and other system power.",
                    "Power uses trapezoidal integration of consecutive valid samples; gaps, long intervals, and collector restarts are not bridged.",
                    "GPU activity and server counters include every process/request using this GPU/server during observation; they are not exclusive attribution.",
                    "Counter deltas begin at the first successful scrape; newly appearing series are conservatively baselined.",
                    "A detected server reset starts a segment at zero; unobserved work immediately before a reset cannot be recovered.",
                    "Telemetry collected now cannot reconstruct an earlier completed window.",
                    "Server histograms contain aggregate observations, not per-request time-to-first-token or stage timings."]}


class SampleLog:
    """Append-only chunks; an incomplete crash tail stays intact in its old chunk."""
    def __init__(self, directory, chunk_bytes=CHUNK_BYTES):
        self.directory = Path(directory)
        self.chunk_bytes = chunk_bytes
        self.directory.mkdir(parents=True, exist_ok=True)
        chunks = sorted(self.directory.glob("samples-*.jsonl"))
        self.index = int(chunks[-1].stem.split("-")[-1]) + 1 if chunks else 0
        self.path = self.directory / f"samples-{self.index:06d}.jsonl"

    def replay(self):
        for path in sorted(self.directory.glob("samples-*.jsonl")):
            with path.open("rb") as source:
                for line in source:
                    if not line.endswith(b"\n"):
                        break  # A crash may leave the final append incomplete.
                    yield json.loads(line)

    def append(self, sample):
        payload = (json.dumps(sample, sort_keys=True, allow_nan=False) + "\n").encode()
        if self.path.exists() and self.path.stat().st_size + len(payload) > self.chunk_bytes:
            self.index += 1
            self.path = self.directory / f"samples-{self.index:06d}.jsonl"
        with self.path.open("ab") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())


class WindowRecorder:
    def __init__(self, root, control, interval, identity):
        self.directory = root / "telemetry" / control["window"]
        self.log = SampleLog(self.directory)
        self.aggregate = Aggregate(control, interval)
        self.identity = identity
        for sample in self.log.replay():
            self.aggregate.consume(sample)

    def append(self, sample):
        sample = {**sample, "collector": self.identity}
        self.log.append(sample)
        self.aggregate.consume(sample)
        save(self.directory / "summary.json", self.aggregate.summary(identity=self.identity))
        ready = self.directory / "READY.json"
        if not ready.exists() or not read(ready).get("baseline_complete"):
            # Both measurements must succeed in this sample for an initial
            # launch gate; first successful baselines remain in the summary.
            save(ready, {"window": self.aggregate.window, "at": sample["at"],
                "collector": self.identity, "collector_id": sample["collector_id"],
                "gpu_ok": sample["gpus"] is not None, "metrics_ok": sample["metrics"] is not None,
                "baseline_complete": sample["gpus"] is not None and sample["metrics"] is not None,
                "errors": sample.get("errors", {})})

    def close(self, reason):
        save(self.directory / "summary.json", self.aggregate.summary(
            closed=True, reason=reason, identity=self.identity))


def valid_window(value):
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", value) is not None


def window_phase(control, monitor, epoch, current_window=None):
    """Never open an expired/paused old window or trust another window's status."""
    window = control.get("window")
    if not valid_window(window):
        return "inactive"
    deadline = control.get("deadline_epoch")
    expired = deadline is not None and epoch >= deadline
    if current_window != window:
        return "active" if control.get("desired") == "running" and not expired else "inactive"
    if control.get("stopped_at"):
        return "closed"
    if monitor.get("window") == window and monitor.get("desired") == "paused":
        return "closed"
    if expired and epoch > deadline + 900:
        return "drain_observation_timeout"
    return "active" if control.get("desired") == "running" and not expired else "draining"


def acquire_lock(root):
    telemetry = root / "telemetry"
    telemetry.mkdir(parents=True, exist_ok=True)
    handle = (telemetry / ".monitor.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("Another telemetry collector holds this experiment lock")
    return handle


def run(args):
    root = args.root.resolve(strict=True)
    identity = collector_identity()
    collector_id = str(uuid.uuid4())
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    lock = acquire_lock(root)
    recorder = None
    last_sample = -math.inf
    try:
        while not stopping.is_set():
            try:
                control = read(root / "control.json") if (root / "control.json").exists() else {}
                monitor = read(root / "monitor.json") if (root / "monitor.json").exists() else {}
                window = control.get("window")
                if recorder and window != recorder.aggregate.window:
                    recorder.close("control_window_changed")
                    recorder = None
                phase = window_phase(control, monitor, time.time(),
                                     recorder.aggregate.window if recorder else None)
                if recorder is None and phase == "active":
                    summary_path = root / "telemetry" / window / "summary.json"
                    if not summary_path.exists() or not read(summary_path).get("closed"):
                        recorder = WindowRecorder(root, control, args.interval_seconds, identity)
                        last_sample = -math.inf
                final = phase in ("closed", "drain_observation_timeout")
                interval = args.interval_seconds
                if recorder and not (recorder.aggregate.gpu_successes and recorder.aggregate.metrics_successes):
                    interval = min(5, interval)
                if recorder and (final or time.monotonic() - last_sample >= interval):
                    sample = capture(metrics_url=args.metrics_url, collector_id=collector_id)
                    recorder.append(sample)
                    last_sample = sample["monotonic"]
                    if sample["errors"]:
                        print(json.dumps({"at": sample["at"], "window": window,
                                          "telemetry_errors": sample["errors"]}), flush=True)
                    if final:
                        recorder.close(control.get("stop_reason", phase))
                        recorder = None
                if args.once:
                    return 0
            except Exception as exc:
                # Independent observation must never control inference. A
                # repeated filesystem failure is visible in the user journal.
                print(json.dumps({"at": utc(), "telemetry_error": f"{type(exc).__name__}: {exc}"}), flush=True)
                if args.once:
                    return 1
            stopping.wait(min(2, args.interval_seconds))
    finally:
        # Service restarts preserve an open window for replay/resume. The next
        # collector UUID prevents energy integration over this unobserved gap.
        if recorder:
            save(recorder.directory / "summary.json", recorder.aggregate.summary(identity=identity))
        lock.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30)
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--once", action="store_true", help="Sample an active window once; do nothing for a completed window")
    args = parser.parse_args()
    if not math.isfinite(args.interval_seconds) or args.interval_seconds < 1:
        parser.error("--interval-seconds must be finite and at least 1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
