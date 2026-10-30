#!/usr/bin/env python3
"""Run a new local-model benchmark; preserve measured time and held-out answers."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import signal
import threading
import time
from typing import Callable

TASKS = ('country', 'bibliography', 'locality')
TARGET_KEYS = {
    'country': {'country', 'country_name', 'country_code'},
    'bibliography': {'title', 'arttitle', 'article_title'},
    'locality': {'latitude', 'longitude', 'decimalLatitude', 'decimalLongitude', 'coordinates'},
}


def load_records(path: Path, task: str) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    identifiers: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'].strip():
            raise ValueError('Each record needs a string id and an object context.')
        if row['id'] in identifiers or not isinstance(row.get('context'), dict):
            raise ValueError('Record ids must be unique and contexts must be objects.')
        identifiers.add(row['id'])
        if row.get('reference') is not None:
            validate_value(task, row['reference'])
    return rows, hashlib.sha256(raw).hexdigest()


def plan_records(rows: list[dict], *, seed: int, max_records: int | None = None) -> list[dict]:
    """Select nine genuinely blank targets for each masked known-value check."""
    if max_records is not None and (max_records < 10 or max_records % 10):
        raise ValueError('max_records must be a positive multiple of ten.')
    blanks = [r for r in rows if r.get('reference') is None]
    known = [r for r in rows if r.get('reference') is not None]
    rng = random.Random(seed)
    rng.shuffle(blanks)
    rng.shuffle(known)
    if not blanks or not known:
        raise ValueError('Supply at least nine blank targets and one known-value check.')
    if max_records is not None:
        blanks = blanks[:max_records // 10 * 9]
    blocks = math.ceil(len(blanks) / 9)
    if len(known) < blocks:
        raise ValueError('Supply one known-value check for each nine blank targets; no blank records were discarded.')
    planned = []
    for i in range(blocks):
        planned.extend(dict(r, cohort='blank') for r in blanks[i * 9:(i + 1) * 9])
        planned.append(dict(known[i], cohort='held_out'))
    return planned


def packet(task: str, row: dict) -> dict:
    # Remove only the target's answer fields. Candidate titles remain evidence.
    normalized = lambda name: re.sub(r'[^a-z0-9]', '', name.casefold())
    target_names = {normalized(k) for k in TARGET_KEYS[task]}
    target_names |= {'loccountry'} if task == 'country' else set()
    context = {k: v for k, v in row['context'].items() if normalized(k) not in target_names}
    if task == 'bibliography' and isinstance(context.get('target'), dict):
        context['target'] = {k: v for k, v in context['target'].items() if normalized(k) not in target_names}
    return {'task': task, 'context': context}


def validate_value(task: str, value):
    if value is None:
        return None
    if task != 'locality':
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Expected a non-empty string or null.')
        return value
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError('Expected [latitude, longitude] or null.')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError('Coordinates must be finite numbers.')
    if not (-90 <= value[0] <= 90 and -180 <= value[1] <= 180):
        raise ValueError('Coordinates are outside geographic bounds.')
    return value


def distance_km(a: list, b: list) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1, h)))


def run(task: str, planned: list[dict], output: Path, *, hours: float,
        infer: Callable, model: str, input_sha256: str, seed: int,
        clock: Callable = time.monotonic, execution_mode: str = 'local_model',
        configuration: dict | None = None) -> dict:
    if task not in TASKS or not math.isfinite(hours) or hours <= 0:
        raise ValueError('Select a supported task and a positive finite duration.')
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    started = clock()
    started_epoch = time.time()
    deadline = started + hours * 3600
    counts = dict(attempted=0, valid=0, failed=0, abstained=0, blank_answered=0,
                  blank_attempted=0, held_out_attempted=0, held_out_answered=0,
                  held_out_exact=0, input_tokens=0, output_tokens=0, usage_missing=0)
    distances = []
    reason = 'dataset_exhausted'
    consecutive_errors = 0
    planned_fraction = sum(r['cohort'] == 'held_out' for r in planned) / len(planned) if planned else None
    def summary():
        elapsed = max(0, clock() - started)
        result = {'schema': 'collections-benchmark-v1', 'task': task, 'model': model,
            'execution_mode': execution_mode, 'input_sha256': input_sha256, 'seed': seed,
            'configuration': configuration or {}, 'record_limit': None,
            'started_epoch': started_epoch, 'deadline_epoch': started_epoch + hours * 3600,
            'configured_max_hours': hours, 'elapsed_seconds': elapsed,
            'stop_reason': reason, 'planned_records': len(planned),
            'planned_held_out_fraction': planned_fraction,
            'observed_held_out_fraction': counts['held_out_attempted'] / counts['attempted'] if counts['attempted'] else None,
            'counts': dict(counts),
            'projected_records_per_24h': counts['attempted'] / elapsed * 86400 if elapsed else None,
            'held_out_exact_fraction': counts['held_out_exact'] / counts['held_out_attempted'] if counts['held_out_attempted'] and task != 'locality' else None,
            'held_out_median_distance_km': statistics.median(distances) if distances else None,
            'distance_count': len(distances)}
        if execution_mode == 'offline_demonstration':
            for name in ['projected_records_per_24h', 'held_out_exact_fraction', 'held_out_median_distance_km']:
                result[name] = None
        temporary = output / '.summary.json.partial'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(output / 'summary.json')
        return result
    class TimeBudgetReached(Exception):
        pass
    def stop_at_deadline(signum, frame):
        raise TimeBudgetReached()
    # GB10/Linux: also interrupt a blocked HTTP read when the wall-time budget ends.
    use_alarm = (clock is time.monotonic and hasattr(signal, 'setitimer')
                 and threading.current_thread() is threading.main_thread())
    previous_handler = None
    previous_timer = None
    if use_alarm:
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        use_alarm = previous_timer == (0.0, 0.0)
    if use_alarm:
        previous_handler = signal.signal(signal.SIGALRM, stop_at_deadline)
    reason = 'running'
    summary()
    private = output / 'records.jsonl'
    try:
        with private.open('x', encoding='utf8') as stream:
            private.chmod(0o600)
            for row in planned:
                remaining = deadline - clock()
                if remaining <= 0:
                    reason = 'time_limit'
                    break
                counts['attempted'] += 1
                counts[row['cohort'] + '_attempted'] += 1
                receipt = {'id': row['id'], 'cohort': row['cohort']}
                usage_known = False
                try:
                    if use_alarm:
                        signal.setitimer(signal.ITIMER_REAL, remaining)
                    response = infer(task, packet(task, row), remaining)
                    usage = response.get('usage', {})
                    if isinstance(usage.get('prompt_tokens'), int) and isinstance(usage.get('completion_tokens'), int):
                        counts['input_tokens'] += usage['prompt_tokens']
                        counts['output_tokens'] += usage['completion_tokens']
                        usage_known = True
                    value = validate_value(task, response['value'])
                    if task == 'bibliography' and value is not None:
                        candidates = row['context'].get('candidates', [])
                        titles = [c.get('title') if isinstance(c, dict) else c for c in candidates]
                        if value not in titles:
                            raise ValueError('Suggested title was not copied from a supplied candidate.')
                    counts['valid'] += 1
                    counts['abstained'] += int(value is None)
                    counts['blank_answered'] += int(row['cohort'] == 'blank' and value is not None)
                    consecutive_errors = 0
                    receipt.update(value=value, status='abstained' if value is None else 'candidate')
                    if 'details' in response:
                        receipt['details'] = response['details']
                    if row['cohort'] == 'held_out':
                        receipt['reference'] = row['reference']
                        if value is not None:
                            counts['held_out_answered'] += 1
                            if task == 'locality':
                                distances.append(distance_km(value, row['reference']))
                            else:
                                counts['held_out_exact'] += int(value == row['reference'])
                except (Exception, KeyboardInterrupt) as exc:
                    counts['failed'] += 1
                    consecutive_errors += 1
                    # Exception messages can contain a server's copy of museum text.
                    receipt.update(status='failed', error_type=type(exc).__name__)
                    if isinstance(exc, KeyboardInterrupt):
                        reason = 'interrupted'
                    elif isinstance(exc, TimeBudgetReached):
                        reason = 'time_limit'
                    elif consecutive_errors >= 5:
                        reason = 'consecutive_errors'
                finally:
                    if use_alarm:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                if not usage_known:
                    counts['usage_missing'] += 1
                stream.write(json.dumps(receipt, ensure_ascii=False) + '\n')
                stream.flush()
                summary()
                if reason in {'interrupted', 'consecutive_errors', 'time_limit'}:
                    break
            else:
                reason = 'dataset_exhausted'
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
    return summary()


if __name__ == '__main__':
    from .gb10 import main
    main()
