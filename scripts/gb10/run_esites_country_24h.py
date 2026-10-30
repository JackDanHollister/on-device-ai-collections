#!/usr/bin/env python3
"""GB10-resident, checkpointed thinking-on country extraction and review."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import tempfile
import threading
import time
import uuid

import requests

import local_span_pilot as p
import run_local_span_pilots as base
import run_esites_review_comparison as reviewer

RUN = '20260913_esites_country_24h_v2'
CONTRACT = 'esites-country-24h-thinking-v2'
UNIT = 'tdwg-esites-country-24h-v2.service'
PARENT_RUN = '20260913_esites_country_24h_v1'
SECONDS = 86400
MAX_ATTEMPTS = 2  # Transport interruptions only; never retry a model validation failure.
MIN_FREE_BYTES = 1024 ** 3
PROMPT_LIMIT = 131072
DEPENDENCIES = {
    'local_span_pilot.py': '64699260c3381bb0f63a5af605cc7f867aa8e6db278b6dd6e161923e0deb3e16',
    'run_local_span_pilots.py': 'a65a790491eef2e4a8b4935423ac23ed93cf3aae3507071376892538a2c5eac7',
    'run_esites_review_comparison.py': '67c815b66d318e73632f51ea754d58da12f2489929955737ee07a008ac865859',
}
CONFIG = {'contract': CONTRACT, 'run': RUN, 'seconds': SECONDS, 'thinking': True,
    'review_control_gate': 'positive_supported_negative_withheld',
    'masked_per_block': 1, 'actual_per_block': 9, 'max_transport_attempts': MAX_ATTEMPTS,
    'invalid_output_retries': 0, 'consecutive_failed_record_pause': 5,
    'prompt_limit': PROMPT_LIMIT, 'output_limit': base.OUTPUT_TOKENS,
    'model': p.MODEL, 'revision': p.REVISION, 'source_edits': 0}
TERMINALS = ('COMPLETE.json', 'BUDGET_COMPLETE.json', 'FAILED.json')
FIELDS = ('locpreciselocation', 'locnhmlabellocality')
CONTEXT = ('loccontinent', 'locprovincestateterritory', 'locdistrictcountyshire',
           'locislandname', 'locislandgrouping', 'lococean', 'locseagulf')


class Pause(Exception):
    pass


class Deadline(Exception):
    pass


class Interrupted(Exception):
    pass


def utc(epoch=None):
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return p.strict_json(path.read_bytes())


def atomic(path, data):
    """Unique temporary names cannot obstruct recovery after an interrupted rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.partial', dir=path.parent)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(path, value):
    atomic(path, p.encoded(value))


def publish(path, value):
    data = p.encoded(value)
    if path.exists():
        if path.read_bytes() != data:
            raise p.PilotError('IMMUTABLE_ARTIFACT_CHANGED')
    else:
        atomic(path, data)


def roots():
    return (p.resident(p.ROOT / 'work' / (RUN + '_prepared')),
            p.resident(p.ROOT / 'outputs' / RUN),
            p.resident(p.ROOT / 'runtime' / RUN))


def group(packet):
    return sha(p.encoded([b['text'].casefold() for b in packet['evidence']]))


def locality_key(packet):
    texts = {b['source_field']: b['text'] for b in packet['evidence']}
    return sha(p.encoded([texts.get(field, '') for field in FIELDS]))


def interleave(actual, masked):
    """The ratio is by source record, independent of how many review calls it needs."""
    for index, control in enumerate(masked):
        block = actual[index * 9:(index + 1) * 9]
        if not block:
            return
        yield control
        yield from block


def prepare():
    os.umask(0o077)
    prepared, output, runtime = roots()
    if Path(__file__).resolve().parent != runtime or prepared.exists() or output.exists():
        raise p.PilotError('FRESH_FIXED_PREPARATION_REQUIRED')
    for name, expected in DEPENDENCIES.items():
        if p.digest(runtime / name) != expected:
            raise p.PilotError('DEPENDENCY_CHANGED')
    if (p.ROOT / 'work' / (PARENT_RUN + '_prepared') / 'SEALED.json').is_file():
        return prepare_from_parent(prepared, runtime)
    start = time.monotonic()
    source_hashes = p.fingerprints()
    old_cases_path = p.ROOT / 'work/20260912_local_span_pilots_v2_prepared/cases.json'
    old_terminal = p.ROOT / 'outputs/20260912_local_span_pilots_v2/COMPLETE.json'
    if p.digest(old_terminal) != reviewer.TERMINAL_HASH:
        raise p.PilotError('BASELINE_IDENTITY_CHANGED')
    old_hash = p.digest(old_cases_path)
    old = [c for c in read(old_cases_path)['cases'] if c['packet']['dataset'] == 'esites'
           and c['cohort'] != 'synthetic']
    excluded_ids = {c['record_id'] for c in old}
    excluded_groups = {group(c['packet']) for c in old}
    source = p.ROOT / 'source' / p.SOURCES['esites'][0]
    vocabulary = Counter(row['loccountry'].strip() for row in p.rows(source) if row['loccountry'].strip())
    terms = [v for v, n in vocabulary.items() if n >= 5 and len(v) >= 3
             and not re.search(r'unknown|unspecified|not known|\?', v, re.I)]
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(map(re.escape, sorted(terms, key=len, reverse=True))) + r')(?!\w)', re.I)
    prepared.mkdir()
    catalog = sqlite3.connect(prepared / 'catalog.sqlite3')
    catalog.executescript('''
        CREATE TABLE candidates (record_id TEXT PRIMARY KEY, cohort TEXT NOT NULL,
            ordering TEXT NOT NULL, evidence_group TEXT NOT NULL, packet_key TEXT NOT NULL,
            payload TEXT NOT NULL);
        CREATE TABLE peers (packet_key TEXT, country TEXT, PRIMARY KEY(packet_key,country));
    ''')
    census = Counter(country_vocabulary_forms=len(terms))
    for row in p.rows(source):
        census['source_records'] += 1
        packet = p.packet('esites', 'loccountry', [(f, row[f]) for f in FIELDS],
                          {f: row[f] for f in CONTEXT if row[f]})
        if not packet['evidence']:
            continue
        if len(p.encoded(packet)) > p.PACKET_BYTES:
            census['oversize'] += 1
            continue
        key = locality_key(packet)
        value = row['loccountry']
        if value.strip():
            catalog.execute('INSERT OR IGNORE INTO peers VALUES (?,?)', (key, value))
        evidence_group = group(packet)
        if row['irn'] in excluded_ids or evidence_group in excluded_groups:
            census['prior_exposed_record_or_group_excluded'] += 1
            continue
        if not value and any(pattern.search(b['text']) for b in packet['evidence']):
            cohort, expected = 'actual_blank', None
        elif len(value.strip()) >= 3 and any(b['text'].count(value) == 1 for b in packet['evidence']):
            cohort, expected = 'masked_existing_value', {'decision': 'COPY_SPAN', 'quote': value}
        else:
            continue
        case = {'record_id': row['irn'], 'packet': packet, 'cohort': cohort,
                'expected': expected, 'evidence_group': evidence_group}
        catalog.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)', (row['irn'], cohort,
            sha((RUN + '\0' + row['irn']).encode()), evidence_group, key, p.encoded(case).decode()))
        census[cohort + '_eligible'] += 1
    if census['source_records'] != p.SOURCES['esites'][2]:
        raise p.PilotError('SOURCE_COUNT_CHANGED')
    catalog.executescript('''
        CREATE INDEX group_cohort ON candidates(evidence_group,cohort);
        CREATE INDEX ordered_cohort ON candidates(cohort,ordering);
    ''')
    actual = catalog.execute('''SELECT payload,packet_key FROM candidates WHERE cohort='actual_blank'
        ORDER BY ROW_NUMBER() OVER(PARTITION BY evidence_group ORDER BY ordering), ordering''').fetchall()
    needed = math.ceil(len(actual) / 9)
    masked = catalog.execute('''SELECT payload,packet_key FROM
        (SELECT payload,packet_key,ordering,
            ROW_NUMBER() OVER(PARTITION BY evidence_group ORDER BY ordering) AS occurrence
         FROM candidates m WHERE cohort='masked_existing_value' AND NOT EXISTS
            (SELECT 1 FROM candidates a WHERE a.evidence_group=m.evidence_group AND a.cohort='actual_blank'))
        WHERE occurrence=1 ORDER BY ordering LIMIT ?''', (needed,)).fetchall()
    if not masked or not actual:
        raise p.PilotError('NO_DISJOINT_MIXED_QUEUE')
    if len(masked) < needed:
        census['actual_not_queued_due_to_control_availability'] = len(actual) - len(masked) * 9
        actual = actual[:len(masked) * 9]
    queue = sqlite3.connect(prepared / 'queue.sqlite3')
    queue.execute('CREATE TABLE queue (seq INTEGER PRIMARY KEY, kind TEXT, cohort TEXT, payload TEXT)')
    seq = 0
    for case in p.controls('esites'):
        seq += 1
        queue.execute('INSERT INTO queue VALUES (?,?,?,?)', (seq, 'extraction_control', 'synthetic', p.encoded(case).decode()))
    for task in reviewer.review_controls():
        seq += 1
        queue.execute('INSERT INTO queue VALUES (?,?,?,?)', (seq, 'review_control', 'synthetic', p.encoded(task).decode()))
    for raw_case, key in interleave(actual, masked):
        seq += 1
        case = p.strict_json(raw_case)
        # No peer reference can reveal a masked record's held-out country.
        case['peer_countries'] = ([r[0] for r in catalog.execute('SELECT country FROM peers WHERE packet_key=? ORDER BY country', (key,))]
                                  if case['cohort'] == 'actual_blank' else [])
        case['case_id'] = f'country_{seq:07d}'
        queue.execute('INSERT INTO queue VALUES (?,?,?,?)', (seq, 'record', case['cohort'], p.encoded(case).decode()))
    queue.commit()
    counts = dict(queue.execute('SELECT cohort,COUNT(*) FROM queue GROUP BY cohort').fetchall())
    queue.close()
    catalog.commit()
    catalog.close()
    if p.fingerprints() != source_hashes or p.digest(old_cases_path) != old_hash:
        raise p.PilotError('PREPARATION_INPUT_CHANGED')
    manifest = {'contract': CONTRACT, 'config': CONFIG, 'source_hashes': source_hashes,
        'baseline_cases_sha256': old_hash, 'baseline_terminal_sha256': reviewer.TERMINAL_HASH,
        'code_hashes': {name: p.digest(runtime / name) for name in (*DEPENDENCIES, Path(__file__).name)},
        'file_hashes': {name: p.digest(prepared / name) for name in ('queue.sqlite3', 'catalog.sqlite3')},
        'queue_entries': seq, 'cohort_counts': counts, 'census': dict(census),
        'masked_actual_evidence_groups_disjoint': True, 'masked_peers_empty': True,
        'ordering': 'one masked then up to nine actual; distinct actual evidence groups first',
        'test_fraction_of_source_records': counts['masked_existing_value'] / (seq - 8),
        'actual_semantic_accuracy': None, 'representative_sample': False,
        'contains_record_values': False, 'preparation_seconds': time.monotonic() - start}
    save(prepared / 'manifest.json', manifest)
    save(prepared / 'SEALED.json', {'contract': CONTRACT, 'manifest_sha256': p.digest(prepared / 'manifest.json')})
    print(json.dumps(manifest, sort_keys=True))


def prepare_from_parent(prepared, runtime):
    """Reuse the unexposed queue after the documented synthetic-gate correction."""
    with base.client_lock():
        parent = p.ROOT / 'work' / (PARENT_RUN + '_prepared')
        source_output = p.ROOT / 'outputs' / PARENT_RUN
        previous = read(parent / 'manifest.json')
        if read(parent / 'SEALED.json')['manifest_sha256'] != p.digest(parent / 'manifest.json'):
            raise p.PilotError('PARENT_MANIFEST_CHANGED')
        if previous['contract'] != 'esites-country-24h-thinking-v1':
            raise p.PilotError('PARENT_CONTRACT_CHANGED')
        state = read(source_output / 'progress.json')
        if state['status'] != 'paused' or state['actual_started'] or state['masked_started']:
            raise p.PilotError('PARENT_NOT_UNEXPOSED_PAUSED')
        window = read(source_output / 'run.json')
        if window['deadline_epoch'] != window['started_epoch'] + SECONDS:
            raise p.PilotError('PARENT_DEADLINE_INVALID')
        if p.fingerprints() != previous['source_hashes']:
            raise p.PilotError('PARENT_SOURCE_CHANGED')
        for name, expected in previous['file_hashes'].items():
            if p.digest(parent / name) != expected:
                raise p.PilotError('PARENT_QUEUE_CHANGED')
        prepared.mkdir()
        for name in previous['file_hashes']:
            shutil.copyfile(parent / name, prepared / name)
        manifest = {**previous, 'contract': CONTRACT, 'config': CONFIG,
            'code_hashes': {name: p.digest(runtime / name) for name in (*DEPENDENCIES, Path(__file__).name)},
            'window_started_epoch': window['started_epoch'], 'window_deadline_epoch': window['deadline_epoch'],
            'parent_run': PARENT_RUN, 'parent_queue_unchanged': True,
            'parent_attempt_cost': {'calls': state['model_calls'], 'tokens': state['known_total_tokens'],
                                    'model_seconds': state['known_model_seconds'], 'actual_records': 0},
            'gate_change': 'Review negatives may be AMBIGUOUS or NOT_SUPPORTED; both withhold. Exact label scoring retained.'}
        save(prepared / 'manifest.json', manifest)
        save(prepared / 'SEALED.json', {'contract': CONTRACT, 'manifest_sha256': p.digest(prepared / 'manifest.json')})
        print(json.dumps(manifest, sort_keys=True))


def verify_inputs():
    prepared, output, runtime = roots()
    if Path(__file__).resolve().parent != runtime:
        raise p.PilotError('FIXED_RUNTIME_REQUIRED')
    manifest = read(prepared / 'manifest.json')
    if read(prepared / 'SEALED.json') != {'contract': CONTRACT, 'manifest_sha256': p.digest(prepared / 'manifest.json')}:
        raise p.PilotError('MANIFEST_SEAL_CHANGED')
    if manifest['config'] != CONFIG:
        raise p.PilotError('CONFIG_CHANGED')
    for root, hashes in ((prepared, manifest['file_hashes']), (runtime, manifest['code_hashes'])):
        for name, expected in hashes.items():
            if p.digest(p.resident(root / name)) != expected:
                raise p.PilotError('FROZEN_INPUT_CHANGED')
    if p.fingerprints() != manifest['source_hashes']:
        raise p.PilotError('SOURCE_CHANGED')
    return manifest


def open_queue():
    prepared, _, _ = roots()
    connection = sqlite3.connect((prepared / 'queue.sqlite3').as_uri() + '?mode=ro&immutable=1', uri=True)
    connection.row_factory = sqlite3.Row
    return connection


class Ledger:
    def __init__(self, output):
        self.output = output
        self.db = sqlite3.connect(output / 'checkpoint.sqlite3')
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS records (seq INTEGER PRIMARY KEY, kind TEXT, cohort TEXT,
                started REAL, finished REAL, valid INTEGER, status TEXT, exact INTEGER, control_passed INTEGER,
                result TEXT, result_sha256 TEXT);
            CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, seq INTEGER, phase TEXT,
                attempt INTEGER, request_sha TEXT, dispatched REAL, status TEXT, result TEXT,
                prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
                model_seconds REAL, result_sha256 TEXT, UNIQUE(seq,phase,attempt));
        ''')
        if self.db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise p.PilotError('CHECKPOINT_CORRUPT')

    def begin_record(self, seq, kind, cohort):
        self.db.execute('INSERT OR IGNORE INTO records(seq,kind,cohort,started) VALUES (?,?,?,?)',
                        (seq, kind, cohort, time.time()))
        self.db.commit()

    def calls(self, seq, phase):
        return self.db.execute('SELECT * FROM calls WHERE seq=? AND phase=? ORDER BY attempt', (seq, phase)).fetchall()

    def dispatch(self, seq, phase, body):
        attempt = len(self.calls(seq, phase)) + 1
        stamp = time.time()
        cursor = self.db.execute('INSERT INTO calls(seq,phase,attempt,request_sha,dispatched,status) VALUES (?,?,?,?,?,?)',
            (seq, phase, attempt, sha(p.encoded(body)), stamp, 'DISPATCHED'))
        self.db.commit()
        return self.db.execute('SELECT * FROM calls WHERE id=?', (cursor.lastrowid,)).fetchone()

    def call_result(self, call, result):
        used = result['usage'] or {}
        encoded = p.encoded(result).decode()
        self.db.execute('''UPDATE calls SET status=?,result=?,prompt_tokens=?,completion_tokens=?,
            total_tokens=?,model_seconds=?,result_sha256=? WHERE id=? AND result IS NULL''',
            (result['status'], encoded, used.get('prompt_tokens'), used.get('completion_tokens'),
             used.get('total_tokens'), result.get('model_seconds'), sha(encoded.encode()), call['id']))
        self.db.commit()

    def finish(self, seq, result):
        text = p.encoded(result).decode()
        self.db.execute('''UPDATE records SET finished=?,valid=?,status=?,exact=?,control_passed=?,result=?,result_sha256=?
            WHERE seq=? AND result IS NULL''', (time.time(), int(result['valid']), result['status'],
            result['exact_reference_match'], result.get('control_passed'), text, sha(text.encode()), seq))
        self.db.commit()
        self.materialize(seq)

    def materialize(self, seq=None):
        query = 'SELECT seq,result,result_sha256 FROM records WHERE result IS NOT NULL'
        args = ()
        if seq is not None:
            query += ' AND seq=?'
            args = (seq,)
        for row in self.db.execute(query, args):
            if sha(row['result'].encode()) != row['result_sha256']:
                raise p.PilotError('RESULT_CHECKPOINT_CHANGED')
            publish(self.output / 'records' / f'{row["seq"]:07d}.json', p.strict_json(row['result']))

    def completed(self, seq):
        row = self.db.execute('SELECT result FROM records WHERE seq=?', (seq,)).fetchone()
        return row is not None and row['result'] is not None

    def recent_failed(self):
        rows = self.db.execute('SELECT valid FROM records WHERE kind="record" AND result IS NOT NULL ORDER BY seq DESC LIMIT 5').fetchall()
        return len(rows) == 5 and all(row['valid'] == 0 for row in rows)

    def totals(self):
        cohort = {}
        for row in self.db.execute('''SELECT cohort,COUNT(*) AS completed, SUM(valid) AS valid,
            SUM(CASE WHEN exact=1 THEN 1 ELSE 0 END) AS exact,
            SUM(CASE WHEN control_passed=1 THEN 1 ELSE 0 END) AS controls_passed FROM records
            WHERE result IS NOT NULL GROUP BY cohort'''):
            cohort[row['cohort']] = dict(row)
        stages = {}
        for row in self.db.execute('''SELECT phase,COUNT(*) AS attempted,
            SUM(CASE WHEN result IS NOT NULL THEN 1 ELSE 0 END) AS finished,
            COUNT(total_tokens) AS usage_reported_calls, SUM(prompt_tokens) AS prompt_tokens,
            SUM(completion_tokens) AS completion_tokens,SUM(total_tokens) AS total_tokens,
            SUM(model_seconds) AS model_seconds FROM calls GROUP BY phase'''):
            stages[row['phase']] = dict(row)
        outcomes = {row['status']: row['n'] for row in self.db.execute('''SELECT status,COUNT(*) AS n FROM records
            WHERE kind="record" AND cohort="actual_blank" AND result IS NOT NULL GROUP BY status''')}
        statuses = {row['status']: row['n'] for row in self.db.execute('SELECT status,COUNT(*) AS n FROM calls GROUP BY status')}
        complete = sum(v['completed'] for v in cohort.values())
        started = dict(self.db.execute('SELECT cohort,COUNT(*) FROM records GROUP BY cohort').fetchall())
        actual = cohort.get('actual_blank', {}).get('completed', 0)
        masked = cohort.get('masked_existing_value', {}).get('completed', 0)
        return {'completed_entries': complete, 'actual_completed': actual, 'masked_completed': masked,
            'actual_started': started.get('actual_blank', 0), 'masked_started': started.get('masked_existing_value', 0),
            'test_fraction_completed': masked / (actual + masked) if actual + masked else None,
            'cohorts': cohort, 'actual_outcomes': outcomes, 'stages': stages, 'call_statuses': statuses,
            'model_calls': sum(v['attempted'] for v in stages.values()),
            'unknown_usage_calls': sum(v['attempted'] - v['usage_reported_calls'] for v in stages.values()),
            'known_total_tokens': sum(v['total_tokens'] or 0 for v in stages.values()),
            'known_model_seconds': sum(v['model_seconds'] or 0 for v in stages.values())}


def extraction_body(case):
    return base.request(case, p.PROMPTS['esites'], True)


def review_control_passed(result, expected):
    if result['status'] != 'VALID':
        return False
    label = result['answer']['assessment']
    return label == 'SUPPORTED' if expected == 'SUPPORTED' else label in {'AMBIGUOUS', 'NOT_SUPPORTED'}


def decode_transport(transport, phase, case):
    raw_bytes = base64.b64decode(transport['response_base64']) if transport.get('response_base64') is not None else b''
    if sha(raw_bytes) != transport['response_sha256']:
        raise p.PilotError('RESPONSE_RECEIPT_CHANGED')
    raw = None
    try:
        raw = p.strict_json(raw_bytes)
    except p.PilotError:
        pass
    result = {'status': 'INVALID', 'error': None, 'answer': None, 'proposal': None,
        'exact_reference_match': None, 'usage': base.usage(raw),
        'usage_raw': raw.get('usage') if isinstance(raw, dict) else None,
        'model_seconds': transport['model_seconds'], 'started_at': transport['started_at'],
        'tokenization_seconds': transport.get('tokenization_seconds'),
        'tokenized_prompt_count': transport.get('tokenized_prompt_count'),
        'finished_at': transport['finished_at'], 'http_status': transport['http_status'],
        'request_sha256': transport['request_sha256'], 'response_sha256': transport['response_sha256'],
        'response_id': raw.get('id') if isinstance(raw, dict) else None,
        'finish_reason': None, 'reasoning_content': None, 'reasoning_sha256': None,
        'reasoning_chars': None, 'reasoning_tokens': None,
        'reasoning_token_source': None, 'independent_semantic_truth': False}
    choices = raw.get('choices') if isinstance(raw, dict) else None
    if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
        result['finish_reason'] = choices[0].get('finish_reason')
        message = choices[0].get('message')
        if isinstance(message, dict):
            reasoning = message.get('reasoning')
            if not isinstance(reasoning, str):
                reasoning = message.get('reasoning_content')
            if isinstance(reasoning, str):
                result.update(reasoning_content=reasoning, reasoning_chars=len(reasoning),
                              reasoning_sha256=sha(reasoning.encode()))
    if isinstance(result['usage_raw'], dict):
        details = result['usage_raw'].get('completion_tokens_details')
        if isinstance(details, dict) and type(details.get('reasoning_tokens')) is int:
            result['reasoning_tokens'] = details['reasoning_tokens']
            result['reasoning_token_source'] = 'provider_usage_completion_tokens_details'
    status = transport['http_status']
    if status is None or status == 429 or (isinstance(status, int) and status >= 500):
        result.update(status='TRANSPORT_ERROR', error=transport.get('transport_error') or 'HTTP_TRANSIENT')
        return result
    if status != 200:
        result['error'] = 'HTTP_PERMANENT'
        return result
    try:
        if phase == 'review':
            answer = reviewer.review_result(raw)
            expected = case.get('expected_review')
            score = None if expected is None else answer['assessment'] == expected
            proposal = None
        else:
            answer, proposal, score = base.evaluate(case, raw)
        result.update(status='VALID', answer=answer, proposal=proposal, exact_reference_match=score)
    except (p.PilotError, TypeError, KeyError) as exc:
        result['error'] = str(exc) if isinstance(exc, p.PilotError) else 'FINAL_CONTENT_INVALID'
    return result


class Runner:
    def __init__(self, output, manifest):
        self.output, self.manifest = output, manifest
        self.state = {'phase': 'initializing'}
        self.stop_heartbeat = threading.Event()
        self.heartbeat_error = False
        self.session_id = uuid.uuid4().hex
        self.session = requests.Session()
        self.session.trust_env = False
        if any((output / name).exists() for name in TERMINALS):
            raise p.PilotError('TERMINAL_ROOT_CANNOT_RESUME')
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'run.json'
        manifest_hash = sha(p.encoded(manifest))
        if path.exists():
            self.run_info = read(path)
            if self.run_info['config'] != CONFIG or self.run_info['manifest_sha256'] != manifest_hash:
                raise p.PilotError('RESUME_CONFIG_CHANGED')
            if self.run_info['deadline_epoch'] != self.run_info['started_epoch'] + SECONDS:
                raise p.PilotError('DEADLINE_CHANGED')
        else:
            if any(output.iterdir()):
                raise p.PilotError('UNRECOGNIZED_OUTPUT_ROOT')
            epoch = manifest.get('window_started_epoch', time.time())
            if manifest.get('window_deadline_epoch', epoch + SECONDS) != epoch + SECONDS:
                raise p.PilotError('WINDOW_DEADLINE_CHANGED')
            self.run_info = {'config': CONFIG, 'manifest_sha256': manifest_hash,
                'started_epoch': epoch, 'started_at': utc(epoch),
                'deadline_epoch': epoch + SECONDS, 'deadline_at': utc(epoch + SECONDS),
                'reference_kind': 'masked_existing_database_value', 'contains_record_values': False}
            publish(path, self.run_info)
        self.deadline = self.run_info['deadline_epoch']
        self.ledger = Ledger(output)
        self.ledger.materialize()
        publish(output / 'sessions' / (self.session_id + '-start.json'),
            {'started_at': utc(), 'pid': os.getpid(), 'deadline_at': self.run_info['deadline_at']})

    def check_deadline(self):
        if time.time() >= self.deadline:
            raise Deadline()
        if self.heartbeat_error:
            raise Pause('HEARTBEAT_WRITE_FAILED')

    def checkpoint(self, status='running', **extra):
        self.state.update(extra)
        self.state.update(self.ledger.totals())
        elapsed = max(0.0, time.time() - self.run_info['started_epoch'])
        self.state.update(contract=CONTRACT, run=RUN, status=status, updated_at=utc(),
            started_at=self.run_info['started_at'], deadline_at=self.run_info['deadline_at'],
            deadline_epoch=self.deadline, wall_elapsed_seconds=elapsed,
            remaining_seconds=max(0.0, self.deadline - time.time()), pid=os.getpid(),
            session_id=self.session_id, queue_entries=self.manifest['queue_entries'],
            queue_remaining=self.manifest['queue_entries'] - self.state['completed_entries'],
            source_edits=0, thinking=True, contains_record_values=False,
            actual_semantic_accuracy=None, independent_semantic_truth=False,
            safe_to_resume=status in ('running', 'paused'))
        self.state['source_records_per_hour'] = ((self.state['actual_completed'] + self.state['masked_completed'])
                                                * 3600 / elapsed if elapsed > 0 else None)
        save(self.output / 'progress.json', self.state)

    def heartbeat(self):
        while not self.stop_heartbeat.is_set():
            try:
                save(self.output / 'heartbeat.json', {'contract': CONTRACT, 'updated_at': utc(),
                    'pid': os.getpid(), 'session_id': self.session_id, 'phase': self.state.get('phase'),
                    'deadline_at': self.run_info['deadline_at'], 'contains_record_values': False})
            except OSError:
                self.heartbeat_error = True
            self.stop_heartbeat.wait(15)

    def wait_service(self):
        while True:
            self.check_deadline()
            if shutil.disk_usage(self.output).free < MIN_FREE_BYTES:
                raise Pause('LOW_DISK_SPACE')
            reason = 'endpoint_unavailable'
            try:
                response = self.session.get(base.BASE + '/v1/models', timeout=(5, 10), allow_redirects=False)
                response.raise_for_status()
                models = p.strict_json(response.content).get('data', [])
                matches = [m for m in models if m.get('id') == p.MODEL and p.REVISION in m.get('root', '')]
                if len(matches) != 1:
                    raise p.PilotError('MODEL_IDENTITY_CHANGED')
                context = matches[0].get('max_model_len')
                if type(context) is not int or context < PROMPT_LIMIT + base.OUTPUT_TOKENS + base.MARGIN:
                    raise p.PilotError('MODEL_CONTEXT_CHANGED')
                metrics = self.session.get(base.BASE + '/metrics', timeout=(5, 10), allow_redirects=False)
                metrics.raise_for_status()
                if base.idle(metrics.text):
                    identity = {'model': p.MODEL, 'revision': p.REVISION, 'context_limit': context}
                    publish(self.output / 'service_identity.json', identity)
                    return identity
                reason = 'waiting_for_queue_drain'
            except requests.RequestException:
                pass
            self.checkpoint(phase='waiting_service', service_wait_reason=reason)
            time.sleep(min(30, max(0.0, self.deadline - time.time())))

    def attempt_dir(self, seq, phase, attempt):
        return self.output / 'attempts' / f'{seq:07d}' / phase / f'{attempt:02d}'

    def finish_call(self, call, result):
        directory = self.attempt_dir(call['seq'], call['phase'], call['attempt'])
        publish(directory / 'assessment.json', result)
        self.ledger.call_result(call, result)
        return result

    def recover_call(self, call, phase, case):
        directory = self.attempt_dir(call['seq'], phase, call['attempt'])
        transport = directory / 'transport.json'
        assessment = directory / 'assessment.json'
        if transport.exists():
            receipt = read(transport)
            if receipt['request_sha256'] != call['request_sha']:
                raise p.PilotError('CALL_RECEIPT_CHANGED')
            result = decode_transport(receipt, phase, case)
        else:
            # Never claim zero tokens or zero elapsed time for an unobserved result.
            result = {'status': 'UNKNOWN_INTERRUPTED', 'error': 'DISPATCH_WITHOUT_RESPONSE',
                'request_sha256': call['request_sha'], 'usage': None, 'model_seconds': None,
                'answer': None, 'proposal': None, 'exact_reference_match': None,
                'started_at': utc(call['dispatched']), 'finished_at': None,
                'reasoning_content': None, 'finish_reason': None}
        return self.finish_call(call, result)

    def call(self, seq, phase, body, case):
        request_hash = sha(p.encoded(body))
        calls = self.ledger.calls(seq, phase)
        for call in calls:
            if call['request_sha'] != request_hash:
                raise p.PilotError('RESUMED_REQUEST_CHANGED')
            if call['result'] is None:
                result = self.recover_call(call, phase, case)
            else:
                if sha(call['result'].encode()) != call['result_sha256']:
                    raise p.PilotError('CALL_CHECKPOINT_CHANGED')
                result = p.strict_json(call['result'])
            if result['status'] not in ('TRANSPORT_ERROR', 'UNKNOWN_INTERRUPTED'):
                return result
        if len(calls) >= MAX_ATTEMPTS:
            return result
        while len(calls) < MAX_ATTEMPTS:
            self.check_deadline()
            identity = self.wait_service()
            self.checkpoint(phase='tokenizing', active_sequence=seq, active_stage=phase)
            tick = time.monotonic()
            try:
                tokenized = self.session.post(base.BASE + '/tokenize', json={k: body[k] for k in
                    ('model', 'messages', 'chat_template_kwargs')}, timeout=(5, 60), allow_redirects=False)
                tokenized.raise_for_status()
            except requests.RequestException:
                self.checkpoint(phase='waiting_tokenizer')
                time.sleep(min(30, max(0.0, self.deadline - time.time())))
                continue
            tokenization_seconds = time.monotonic() - tick
            counts = p.strict_json(tokenized.content)
            prompt_count = counts.get('count')
            if type(prompt_count) is not int or prompt_count <= 0 or counts.get('max_model_len') != identity['context_limit']:
                raise p.PilotError('TOKENIZER_INVALID')
            if prompt_count > PROMPT_LIMIT or prompt_count + base.OUTPUT_TOKENS + base.MARGIN > identity['context_limit']:
                return {'status': 'INVALID', 'error': 'PROMPT_BUDGET_EXCEEDED', 'usage': None,
                    'model_seconds': 0.0, 'answer': None, 'proposal': None, 'exact_reference_match': None,
                    'reasoning_content': None, 'request_sha256': request_hash}
            self.check_deadline()
            directory = self.attempt_dir(seq, phase, len(calls) + 1)
            publish(directory / 'request.json', body)
            dispatched = self.ledger.dispatch(seq, phase, body)
            publish(directory / 'dispatch.json', dict(dispatched))
            self.checkpoint(phase='model_call', active_sequence=seq, active_stage=phase)
            tick = time.monotonic()
            transport_error, status, raw_bytes = None, None, b''
            try:
                response = self.session.post(base.BASE + '/v1/chat/completions', json=body,
                    timeout=(5, max(1, min(base.CALL_TIMEOUT, self.deadline - time.time()))), allow_redirects=False)
                raw_bytes, status = response.content, response.status_code
            except requests.RequestException as exc:
                transport_error = type(exc).__name__
            transport = {'request_sha256': request_hash, 'http_status': status,
                'started_at': utc(dispatched['dispatched']), 'finished_at': utc(),
                'model_seconds': time.monotonic() - tick, 'tokenization_seconds': tokenization_seconds,
                'tokenized_prompt_count': prompt_count, 'response_sha256': sha(raw_bytes),
                'response_base64': base64.b64encode(raw_bytes).decode(), 'transport_error': transport_error}
            publish(directory / 'transport.json', transport)
            result = self.finish_call(dispatched, decode_transport(transport, phase, case))
            self.checkpoint(phase='response_saved')
            if result['status'] not in ('TRANSPORT_ERROR', 'UNKNOWN_INTERRUPTED'):
                return result
            calls = self.ledger.calls(seq, phase)
            if len(calls) < MAX_ATTEMPTS:
                self.checkpoint(phase='transport_backoff')
                time.sleep(min(30, max(0.0, self.deadline - time.time())))
        return result

    def process(self, entry):
        seq, kind, cohort = entry['seq'], entry['kind'], entry['cohort']
        case = p.strict_json(entry['payload'])
        self.ledger.begin_record(seq, kind, cohort)
        self.checkpoint(phase='record_start', active_sequence=seq, active_cohort=cohort)
        extracted, reviewed, proposal, control_passed = None, None, None, None
        if kind == 'review_control':
            reviewed = self.call(seq, 'review', case['body'], {'expected_review': case['expected']})
            valid = reviewed['status'] == 'VALID'
            exact = reviewed['exact_reference_match']
            control_passed = review_control_passed(reviewed, case['expected'])
            status = 'CONTROL_PASS' if control_passed else 'CONTROL_FAIL'
        else:
            extracted = self.call(seq, 'extraction', extraction_body(case), case)
            valid = extracted['status'] == 'VALID'
            exact = extracted['exact_reference_match']
            proposal = extracted.get('proposal')
            status = ('INVALID_EXTRACTION' if not valid else extracted['answer']['decision'])
            if kind == 'extraction_control':
                control_passed = exact is True
                status = 'CONTROL_PASS' if exact is True else 'CONTROL_FAIL'
            elif proposal:
                peers = case['peer_countries'] if cohort == 'actual_blank' else []
                reviewed = self.call(seq, 'review', reviewer.review_body(case['packet'], proposal, peers), {})
                valid = reviewed['status'] == 'VALID'
                status = 'PROPOSAL_' + reviewed['answer']['assessment'] if valid else 'PROPOSAL_REVIEW_INVALID'
        call_rows = self.ledger.db.execute('SELECT * FROM calls WHERE seq=? ORDER BY id', (seq,)).fetchall()
        known_usage = {key: sum(row[key] or 0 for row in call_rows) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        unknown = sum(row['total_tokens'] is None for row in call_rows)
        started = self.ledger.db.execute('SELECT started FROM records WHERE seq=?', (seq,)).fetchone()[0]
        result = {'schema_version': CONTRACT, 'sequence': seq, 'kind': kind, 'cohort': cohort,
            'record_id': case.get('record_id'), 'dataset': 'esites', 'target_field': 'loccountry',
            'status': status, 'valid': valid, 'thinking': True, 'model': p.MODEL, 'revision': p.REVISION,
            'source_sha256': self.manifest['source_hashes']['esites'],
            'case_sha256': sha(entry['payload'].encode()), 'extraction': extracted, 'review': reviewed,
            'proposal': proposal, 'source_updated': False, 'independent_semantic_truth': False,
            'expected_for_scoring_only': case.get('expected') if cohort == 'masked_existing_value' else None,
            'exact_reference_match': exact, 'control_passed': control_passed,
            'supported_exact_reference_match': (exact is True and status == 'PROPOSAL_SUPPORTED') if cohort == 'masked_existing_value' else None,
            'notes': {'extraction_reasoning': extracted.get('reasoning_content') if extracted else None,
                'review_explanation': reviewed.get('answer', {}).get('explanation') if reviewed and reviewed.get('answer') else None,
                'explanation_source': 'Nemotron thinking and second-pass explanation; never fabricated by harness'},
            'started_at': utc(started), 'finished_at': utc(), 'elapsed_seconds_including_interruptions': time.time() - started,
            'known_model_seconds': sum(row['model_seconds'] or 0 for row in call_rows),
            'model_time_complete': all(row['model_seconds'] is not None for row in call_rows),
            'known_usage': known_usage, 'usage_complete': unknown == 0, 'unknown_usage_calls': unknown,
            'attempt_count': len(call_rows), 'attempt_paths': [str(self.attempt_dir(seq, row['phase'], row['attempt']).relative_to(self.output)) for row in call_rows],
            'contains_record_values': True}
        self.ledger.finish(seq, result)
        self.checkpoint(phase='record_saved')
        if kind != 'record' and not control_passed:
            raise Pause('SYNTHETIC_CONTROL_FAILED')
        if self.ledger.recent_failed():
            raise Pause('FIVE_CONSECUTIVE_INVALID_RECORDS')

    def export_results(self):
        # Durable per-record JSON is always current; aggregate JSONL is rebuilt on pause/end.
        path = self.output / 'results.jsonl'
        fd, temporary = tempfile.mkstemp(prefix='.results.', suffix='.partial', dir=self.output)
        with os.fdopen(fd, 'w') as stream:
            for row in self.ledger.db.execute('SELECT result FROM records WHERE result IS NOT NULL ORDER BY seq'):
                stream.write(row['result'])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def run(self, queue, verify=verify_inputs):
        thread = threading.Thread(target=self.heartbeat, daemon=True)
        thread.start()
        status, reason, marker = 'complete', 'QUEUE_EXHAUSTED', 'COMPLETE.json'
        try:
            self.check_deadline()
            for entry in queue.execute('SELECT * FROM queue ORDER BY seq'):
                self.check_deadline()
                if self.ledger.completed(entry['seq']):
                    continue
                if entry['kind'] == 'record':
                    controls = self.ledger.db.execute('SELECT COUNT(*) FROM records WHERE kind!="record" AND control_passed=1 AND result IS NOT NULL').fetchone()[0]
                    if controls != 8:
                        raise Pause('SYNTHETIC_GATE_CLOSED')
                self.process(entry)
                if entry['seq'] % 100 == 0:
                    verify()
        except Deadline:
            status, reason, marker = 'budget_complete', '24_HOUR_WINDOW_FINISHED', 'BUDGET_COMPLETE.json'
        except (Interrupted, Pause) as exc:
            status, reason, marker = 'paused', str(exc) or 'USER_STOPPED', 'PAUSED.json'
        except p.PilotError as exc:
            status, reason, marker = 'failed', str(exc), 'FAILED.json'
        except Exception as exc:
            # Unexpected failures preserve checkpoints for systemd restart; do not reset the clock.
            self.checkpoint(status='paused', phase='unexpected_error', error_category=type(exc).__name__)
            save(self.output / 'sessions' / (self.session_id + '-error.json'), self.state)
            raise
        finally:
            self.stop_heartbeat.set()
            thread.join(timeout=2)
        try:
            verify()
            self.state.update(source_immutable=True, frozen_inputs_immutable=True)
        except Exception:
            status, reason, marker = 'failed', 'FINAL_INTEGRITY_CHECK_FAILED', 'FAILED.json'
            self.state.update(source_immutable=None, frozen_inputs_immutable=None)
        self.export_results()
        self.checkpoint(status=status, phase='terminal' if status != 'paused' else 'paused',
                        reason=reason, active_sequence=None, active_stage=None, active_cohort=None)
        publish(self.output / 'sessions' / (self.session_id + '-end.json'), self.state)
        save(self.output / marker, self.state)
        print(json.dumps(self.state, sort_keys=True))
        return 0


def run():
    os.umask(0o077)
    _, output, _ = roots()
    if any((output / name).exists() for name in TERMINALS):
        print(json.dumps({'status': 'terminal_no_restart', 'run': RUN, 'contains_record_values': False}))
        return 0
    with base.client_lock():
        manifest = verify_inputs()
        runner = Runner(output, manifest)
        queue = open_queue()
        try:
            return runner.run(queue)
        finally:
            queue.close()
            runner.ledger.db.close()
            runner.session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'status'))
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(Interrupted('USER_STOPPED')))
    signal.signal(signal.SIGINT, lambda *args: (_ for _ in ()).throw(Interrupted('USER_STOPPED')))
    if args.action == 'prepare':
        prepare()
        return 0
    if args.action == 'status':
        _, output, _ = roots()
        print(json.dumps(read(output / 'progress.json'), sort_keys=True))
        return 0
    return run()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Record-bearing exceptions/tracebacks must not leak into orchestration logs.
        print(json.dumps({'startup_error': type(exc).__name__, 'contains_record_values': False}))
        raise SystemExit(1)
