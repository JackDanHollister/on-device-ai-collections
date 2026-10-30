#!/usr/bin/env python3
"""Run the frozen two-dataset pilot on the resident Nemotron service."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import time

import requests

import local_span_pilot as p

BASE = 'http://127.0.0.1:8000'
OUTPUT_TOKENS = 16384
MARGIN = 2048
CALL_TIMEOUT = 900
MAX_SECONDS = 21600


def now():
    return datetime.now(timezone.utc).isoformat()


def idle(raw):
    seen = set()
    for line in raw.splitlines():
        match = re.fullmatch(r'vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+(\S+)', line)
        if match:
            try:
                value = float(match[2])
            except ValueError:
                return False
            if not math.isfinite(value) or value != 0:
                return False
            seen.add(match[1])
    return seen == {'running', 'waiting'}


def service(session):
    response = session.get(BASE + '/v1/models', timeout=(5, 10), allow_redirects=False)
    response.raise_for_status()
    models = p.strict_json(response.content).get('data', [])
    matches = [m for m in models if m.get('id') == p.MODEL and p.REVISION in m.get('root', '')]
    metrics = session.get(BASE + '/metrics', timeout=(5, 10), allow_redirects=False)
    metrics.raise_for_status()
    if len(matches) != 1 or not idle(metrics.text):
        raise p.PilotError('MODEL_IDENTITY_OR_IDLE_INVALID')
    context = matches[0].get('max_model_len')
    if type(context) is not int or context <= OUTPUT_TOKENS + MARGIN:
        raise p.PilotError('MODEL_CONTEXT_INVALID')
    return {'model': p.MODEL, 'revision': p.REVISION, 'context_limit': context,
        'queue_idle_at_preflight': True, 'verified_at': now()}


@contextmanager
def client_lock():
    path = p.resident(p.ROOT / 'work/.nemotron-client.lock')
    with path.open('r+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise p.PilotError('CLIENT_LOCK_HELD') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load():
    prepared = p.resident(p.ROOT / 'work' / (p.RUN + '_prepared'))
    manifest = p.strict_json((prepared / 'manifest.json').read_bytes())
    seal = p.strict_json((prepared / 'COMPLETE.json').read_bytes())
    if seal != {'status': 'complete', 'manifest_sha256': p.digest(prepared / 'manifest.json'),
                'contains_record_values': False} or manifest['contract'] != p.VERSION:
        raise p.PilotError('PREPARATION_SEAL_INVALID')
    runtime = p.resident(Path(__file__).resolve().parent, p.ROOT / 'runtime')
    if runtime != p.ROOT / 'runtime' / p.RUN:
        raise p.PilotError('RUNTIME_PATH_INVALID')
    hashes = {prepared / name: value for name, value in manifest['file_hashes'].items()}
    hashes.update({runtime / name: value for name, value in manifest['code_hashes'].items()})
    hashes[prepared / 'manifest.json'] = p.digest(prepared / 'manifest.json')
    hashes[prepared / 'COMPLETE.json'] = p.digest(prepared / 'COMPLETE.json')
    check_hashes(hashes)
    cases = p.strict_json((prepared / 'cases.json').read_bytes())['cases']
    prompts = p.strict_json((prepared / 'prompts.json').read_bytes())
    if (manifest['case_count'] != 248 or len(cases) != 248 or len({c['case_id'] for c in cases}) != 248
            or prompts != p.PROMPTS or manifest['source_hashes'] != p.fingerprints()):
        raise p.PilotError('COHORT_INVALID')
    identities = set()
    for bundle in p.SOURCES:
        selected = [c for c in cases if c['packet']['dataset'] == bundle]
        if [c['cohort'] for c in selected] != ['synthetic'] * 4 + ['masked_existing_value'] * 20 + ['actual_blank'] * 100:
            raise p.PilotError('COHORT_ORDER_INVALID')
        for case in selected:
            if case['packet']['current_value'] != '' or ((case['expected'] is None) != (case['cohort'] == 'actual_blank')):
                raise p.PilotError('REFERENCE_OR_TARGET_INVALID')
            if case['cohort'] != 'synthetic':
                identity = (bundle, case['record_id'])
                if identity in identities:
                    raise p.PilotError('DUPLICATE_RECORD')
                identities.add(identity)
    return cases, prompts, manifest, hashes


def check_hashes(hashes):
    if any(p.digest(p.resident(path)) != expected for path, expected in hashes.items()):
        raise p.PilotError('FROZEN_INPUT_CHANGED')


def request(case, prompt, thinking):
    return {'model': p.MODEL, 'messages': [{'role': 'system', 'content': prompt},
        {'role': 'user', 'content': p.encoded({'packet': case['packet'],
            'output_schema': p.schema(case['packet'])}).decode()}],
        'max_tokens': OUTPUT_TOKENS, 'temperature': 1.0 if thinking else 0.0,
        'top_p': 0.95, 'seed': 123, 'stream': False,
        'chat_template_kwargs': {'enable_thinking': thinking, 'low_effort': False},
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'local_span', 'strict': True, 'schema': p.schema(case['packet'])}}}


def evaluate(case, response):
    if not isinstance(response, dict) or response.get('model') != p.MODEL:
        raise p.PilotError('RESPONSE_MODEL_INVALID')
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise p.PilotError('RESPONSE_CHOICES_INVALID')
    choice = choices[0]
    if choice.get('finish_reason') != 'stop':
        raise p.PilotError('NON_NATURAL_STOP')
    message = choice.get('message')
    if (not isinstance(message, dict) or message.get('role') != 'assistant'
            or message.get('tool_calls') or not isinstance(message.get('content'), str)):
        raise p.PilotError('FINAL_CONTENT_INVALID')
    answer = p.strict_json(message['content'])
    proposal = p.validate(case['packet'], answer)
    expected = case['expected']
    score = None if expected is None else answer['decision'] == expected['decision'] and answer['quote'] == expected['quote']
    return answer, proposal, score


def usage(response):
    values = response.get('usage') if isinstance(response, dict) else None
    keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    if (not isinstance(values, dict) or any(type(values.get(k)) is not int or values[k] < 0 for k in keys)
            or values['total_tokens'] != values['prompt_tokens'] + values['completion_tokens']):
        return None
    return {k: values[k] for k in keys}


def run(args):
    if type(args.prompt_token_limit) is not int or args.prompt_token_limit <= 0:
        raise p.PilotError('PROMPT_LIMIT_INVALID')
    os.umask(0o077)
    root = p.resident(p.ROOT / 'outputs' / p.RUN, p.ROOT / 'outputs')
    if root.exists():
        raise p.PilotError('FRESH_OUTPUT_REQUIRED')
    root.mkdir()
    start = time.monotonic()
    state = {'contract': p.VERSION, 'status': 'running', 'phase': 'preflight',
        'started_at': now(), 'case_count': 248, 'completed': 0, 'attempted': 0,
        'api_success': 0, 'valid_outputs': 0, 'invalid_outputs': 0, 'in_flight': False,
        'usage_reported_calls': 0, 'known_usage': dict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        'model_seconds': 0.0, 'tokenization_seconds': 0.0, 'errors': {},
        'max_prompt_tokens_seen': 0, 'prompt_token_limit': args.prompt_token_limit,
        'output_token_limit': OUTPUT_TOKENS, 'thinking': not args.no_thinking,
        'source_edits': 0, 'source_immutable': None, 'frozen_inputs_immutable': None,
        'safe_to_resume': False, 'production_qualified': False, 'contains_record_values': False,
        'datasets': {b: {'synthetic_completed': 0, 'synthetic_passed': 0,
            'masked_completed': 0, 'masked_exact': 0, 'baseline_masked_exact': 0,
            'actual_completed': 0, 'actual_proposals': 0, 'actual_unresolved': 0,
            'actual_review': 0, 'actual_invalid': 0, 'baseline_actual_proposals': 0,
            'actual_semantic_accuracy': None, 'model_seconds': 0.0} for b in p.SOURCES}}

    def checkpoint():
        state.update(updated_at=now(), elapsed_seconds=round(time.monotonic() - start, 3),
            usage_complete_for_attempted_calls=state['attempted'] == state['usage_reported_calls'])
        state['estimated_remaining_model_seconds'] = (
            round(state['model_seconds'] / state['completed'] * (248 - state['completed']))
            if state['completed'] >= 5 else None)
        p.write_json(root / 'progress.json', state)

    session = requests.Session()
    session.trust_env = False
    failed = None
    hashes = {}
    checkpoint()
    try:
        with client_lock():
            cases, prompts, manifest, hashes = load()
            model = service(session)
            p.write_json(root / 'run.json', {'contract': p.VERSION, 'model': model,
                'manifest': manifest, 'prompt_token_limit': args.prompt_token_limit,
                'output_token_limit': OUTPUT_TOKENS, 'thinking': not args.no_thinking,
                'call_timeout_seconds': CALL_TIMEOUT, 'maximum_seconds': MAX_SECONDS,
                'one_request_per_case': True, 'automatic_retries': 0})
            consecutive_invalid = 0
            for case in cases:
                if time.monotonic() - start >= MAX_SECONDS:
                    raise p.PilotError('WALL_BUDGET_EXHAUSTED')
                bundle, cohort = case['packet']['dataset'], case['cohort']
                ds = state['datasets'][bundle]
                if cohort != 'synthetic' and ds['synthetic_passed'] != 4:
                    raise p.PilotError('SYNTHETIC_GATE_FAILED')
                case_root = root / 'cases' / case['case_id']
                case_root.mkdir(parents=True)
                state.update(phase='tokenizing', active_case=case['case_id'], active_dataset=bundle, active_cohort=cohort)
                body = request(case, prompts[bundle], not args.no_thinking)
                p.write_json(case_root / 'request.json', body)
                p.write_json(case_root / 'case.json', case)
                checkpoint()
                tick = time.monotonic()
                tokenized = session.post(BASE + '/tokenize', json={k: body[k] for k in
                    ('model', 'messages', 'chat_template_kwargs')}, timeout=(5, 60), allow_redirects=False)
                tokenized.raise_for_status()
                counts = p.strict_json(tokenized.content)
                prompt_tokens = counts.get('count')
                if type(prompt_tokens) is not int or prompt_tokens <= 0 or counts.get('max_model_len') != model['context_limit']:
                    raise p.PilotError('TOKENIZER_INVALID')
                state['tokenization_seconds'] += time.monotonic() - tick
                state['max_prompt_tokens_seen'] = max(prompt_tokens, state['max_prompt_tokens_seen'])
                allowed = prompt_tokens <= args.prompt_token_limit and prompt_tokens + OUTPUT_TOKENS + MARGIN <= model['context_limit']
                p.write_json(case_root / 'budget.json', {'prompt_tokens': prompt_tokens,
                    'prompt_token_limit': args.prompt_token_limit, 'output_reserved': OUTPUT_TOKENS,
                    'margin_reserved': MARGIN, 'model_context_limit': model['context_limit'], 'allowed': allowed})
                if not allowed:
                    raise p.PilotError('PROMPT_BUDGET_EXCEEDED')
                state.update(phase='model_call', in_flight=True)
                state['attempted'] += 1
                checkpoint()
                tick = time.monotonic()
                response = session.post(BASE + '/v1/chat/completions', json=body,
                    timeout=(5, CALL_TIMEOUT), allow_redirects=False)
                seconds = time.monotonic() - tick
                p.write(case_root / 'response.json', response.content)
                state['in_flight'] = False
                state['model_seconds'] += seconds
                ds['model_seconds'] += seconds
                answer, proposal, score, used, error = None, None, None, None, None
                if response.status_code == 200:
                    state['api_success'] += 1
                try:
                    raw = p.strict_json(response.content)
                    used = usage(raw)
                    if response.status_code != 200:
                        raise p.PilotError('INFERENCE_HTTP_FAILED')
                    answer, proposal, score = evaluate(case, raw)
                except p.PilotError as exc:
                    error = str(exc)
                if used is not None:
                    state['usage_reported_calls'] += 1
                    for k in state['known_usage']:
                        state['known_usage'][k] += used[k]
                state['completed'] += 1
                assessment = {'case_id': case['case_id'], 'dataset': bundle, 'cohort': cohort,
                    'valid': error is None, 'error': error, 'decision': answer['decision'] if answer else None,
                    'exact_reference_match': score, 'semantic_accuracy': None,
                    'usage': used, 'model_seconds': seconds, 'contains_record_values': False}
                p.write_json(case_root / 'assessment.json', assessment)
                if error:
                    consecutive_invalid += 1
                    state['invalid_outputs'] += 1
                    state['errors'][error] = state['errors'].get(error, 0) + 1
                else:
                    consecutive_invalid = 0
                    state['valid_outputs'] += 1
                    p.write_json(case_root / 'answer.json', answer)
                if cohort == 'synthetic':
                    ds['synthetic_completed'] += 1
                    ds['synthetic_passed'] += int(score is True)
                elif cohort == 'masked_existing_value':
                    ds['masked_completed'] += 1
                    ds['masked_exact'] += int(score is True)
                    baseline = case['baseline']
                    ds['baseline_masked_exact'] += int(baseline is not None and baseline['quote'] == case['expected']['quote'])
                else:
                    ds['actual_completed'] += 1
                    ds['baseline_actual_proposals'] += int(case['baseline'] is not None)
                    if error:
                        ds['actual_invalid'] += 1
                    elif proposal:
                        ds['actual_proposals'] += 1
                        proposal.update(case_id=case['case_id'], record_id=case['record_id'],
                            dataset=bundle, source_file_sha256=manifest['source_hashes'][bundle],
                            status='UNVERIFIED_SOURCE_BACKED_PROPOSAL')
                        p.write_json(case_root / 'proposal.json', proposal)
                        with (root / (bundle + '-proposals.jsonl')).open('ab') as stream:
                            stream.write(p.encoded(proposal))
                            stream.flush()
                            os.fsync(stream.fileno())
                    elif answer['decision'] == 'UNRESOLVED':
                        ds['actual_unresolved'] += 1
                    else:
                        ds['actual_review'] += 1
                checkpoint()
                if response.status_code != 200:
                    raise p.PilotError('INFERENCE_HTTP_FAILED')
                if consecutive_invalid >= 3:
                    raise p.PilotError('CONSECUTIVE_INVALID_LIMIT')
            state['source_immutable'] = p.fingerprints() == manifest['source_hashes']
            check_hashes(hashes)
            state['frozen_inputs_immutable'] = True
    except BaseException as exc:
        failed = str(exc) if isinstance(exc, p.PilotError) else type(exc).__name__
        # Keep unexpected exception details and record-bearing tracebacks off stdout.
        state['error_category'] = failed
        try:
            state['source_immutable'] = p.fingerprints() == {b: spec[1] for b, spec in p.SOURCES.items()}
            check_hashes(hashes)
            state['frozen_inputs_immutable'] = bool(hashes)
        except BaseException:
            pass
    state.update(status='failed' if failed else 'complete', phase='terminal', active_case=None)
    checkpoint()
    p.write_json(root / ('FAILED.json' if failed else 'COMPLETE.json'), state)
    print(json.dumps(state, sort_keys=True))
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-token-limit', type=int, default=131072)
    parser.add_argument('--no-thinking', action='store_true')
    def interrupted(signum, frame):
        raise p.PilotError('PROCESS_INTERRUPTED')
    signal.signal(signal.SIGTERM, interrupted)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
