#!/usr/bin/env python3
"""Resident bibliography audit ending at the fixed 07:00 BST deadline."""
from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import signal
import threading
import time
import uuid

from jsonschema import Draft202012Validator
import requests

import ebib_peer_overnight as c
import local_span_pilot as p
import run_local_span_pilots as base
import run_esites_country_24h as engine

CONFIG = dict(contract=c.CONTRACT, run=c.RUN, seconds=c.SECONDS, model=p.MODEL, revision=p.REVISION,
    absolute_deadline_epoch=c.DEADLINE_EPOCH, inference_deadline_epoch=c.INFERENCE_DEADLINE_EPOCH,
    title_comparison_policy='case-space-unicode-plus-one-terminal-full-stop-v1',
    masked_disagreement_policy='retain-emitted-value-and-quarantine-individual-case',
    thinking=True, prompt_limit=65536, output_tokens=8192, max_transport_attempts=2,
    invalid_output_retries=0, consecutive_failed_pause=5, masked_per_nine_actual=1, source_edits=0)
TERMINALS = ('COMPLETE.json', 'BUDGET_COMPLETE.json', 'FAILED.json', 'STOPPED.json', 'STOP_REQUESTED.json')


def decode(transport, case):
    raw_bytes = base64.b64decode(transport['response_base64'])
    if engine.sha(raw_bytes) != transport['response_sha256']: raise p.PilotError('RESPONSE_RECEIPT_CHANGED')
    try: raw = p.strict_json(raw_bytes)
    except p.PilotError: raw = None
    result = dict(status='INVALID', error=None, answer=None, usage=base.usage(raw),
        model_seconds=transport['model_seconds'], http_status=transport['http_status'],
        request_sha256=transport['request_sha256'], response_sha256=transport['response_sha256'],
        started_at=transport['started_at'], finished_at=transport['finished_at'],
        reasoning_content=None, reasoning_tokens=None, finish_reason=None)
    status = transport['http_status']
    if status is None or status == 429 or status >= 500:
        result.update(status='TRANSPORT_ERROR', error=transport.get('transport_error') or 'TRANSIENT_HTTP')
        return result
    if status != 200:
        result['error'] = 'PERMANENT_HTTP'; return result
    try:
        if not isinstance(raw, dict) or raw.get('model') != p.MODEL: raise p.PilotError('MODEL_RESPONSE_INVALID')
        choices = raw.get('choices')
        if not isinstance(choices, list) or len(choices) != 1: raise p.PilotError('CHOICES_INVALID')
        choice = choices[0]; result['finish_reason'] = choice.get('finish_reason')
        message = choice.get('message', {})
        result['reasoning_content'] = message.get('reasoning') or message.get('reasoning_content')
        if result['finish_reason'] != 'stop': raise p.PilotError('NON_NATURAL_STOP')
        if message.get('role') != 'assistant' or message.get('tool_calls'): raise p.PilotError('MESSAGE_INVALID')
        answer = p.strict_json(message.get('content'))
        if not Draft202012Validator(c.SCHEMA).is_valid(answer) or not answer['explanation'].strip():
            raise p.PilotError('OUTPUT_SCHEMA_INVALID')
        c.selection(case['packet'], answer)
        result.update(status='VALID', answer=answer)
        details = raw.get('usage', {}).get('completion_tokens_details') or {}
        if type(details.get('reasoning_tokens')) is int: result['reasoning_tokens'] = details['reasoning_tokens']
    except (p.PilotError, TypeError, KeyError, AttributeError) as exc:
        result['error'] = str(exc) if isinstance(exc, p.PilotError) else 'FINAL_CONTENT_INVALID'
    return result


def verify():
    manifest, queue = c.load(); queue.close(); return manifest


class Runner(engine.Runner):
    """Reuse the exercised ledger, service-idle gate and atomic receipt primitives."""
    def __init__(self, output, manifest):
        self.output, self.manifest = output, manifest
        self.state = {'phase': 'initializing'}
        self.stop_heartbeat = threading.Event(); self.heartbeat_error = False
        self.session_id = uuid.uuid4().hex
        self.session = requests.Session(); self.session.trust_env = False
        if any((output / name).exists() for name in TERMINALS): raise p.PilotError('TERMINAL_ROOT_CANNOT_RESUME')
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'run.json'; digest = engine.sha(p.encoded(manifest))
        if path.exists():
            self.run_info = engine.read(path)
            if self.run_info['config'] != CONFIG or self.run_info['manifest_sha256'] != digest:
                raise p.PilotError('RESUME_IDENTITY_CHANGED')
            if self.run_info['deadline_epoch'] != c.INFERENCE_DEADLINE_EPOCH:
                raise p.PilotError('DEADLINE_CHANGED')
        else:
            if any(output.iterdir()): raise p.PilotError('UNRECOGNIZED_OUTPUT_ROOT')
            stamp = time.time()
            self.run_info = dict(config=CONFIG, manifest_sha256=digest, started_epoch=stamp,
                deadline_epoch=c.INFERENCE_DEADLINE_EPOCH, started_at=engine.utc(stamp),
                deadline_at=engine.utc(c.INFERENCE_DEADLINE_EPOCH), user_deadline_at=c.DEADLINE_AT,
                contains_record_values=False)
            engine.publish(path, self.run_info)
        self.deadline = self.run_info['deadline_epoch']
        self.ledger = engine.Ledger(output); self.ledger.materialize()
        engine.publish(output / 'sessions' / (self.session_id + '-start.json'),
            dict(started_at=engine.utc(), pid=os.getpid(), deadline_at=self.run_info['deadline_at']))

    def checkpoint(self, status='running', **extra):
        self.state.update(extra); self.state.update(self.ledger.totals())
        elapsed = max(0, time.time() - self.run_info['started_epoch'])
        self.state.update(contract=c.CONTRACT, run=c.RUN, status=status, updated_at=engine.utc(),
            started_at=self.run_info['started_at'], deadline_at=self.run_info['deadline_at'],
            deadline_epoch=self.deadline, wall_elapsed_seconds=elapsed, remaining_seconds=max(0, self.deadline - time.time()),
            pid=os.getpid(), session_id=self.session_id, queue_entries=self.manifest['queue_entries'],
            queue_remaining=self.manifest['queue_entries'] - self.state['completed_entries'],
            source_records_scanned=self.manifest['source_records_scanned'],
            source_edits=0, thinking=True, contains_record_values=False, independent_semantic_truth=False,
            safe_to_resume=status == 'running', proposal_accuracy=None, user_deadline_at=c.DEADLINE_AT)
        outcomes = self.state['actual_outcomes']
        self.state['actual_title_proposals'] = outcomes.get('TITLE_AND_LINK_PROPOSAL', 0)
        self.state['actual_link_only_proposals'] = outcomes.get('LINK_ONLY_PROPOSAL', 0)
        self.state['actual_review_title_candidates'] = outcomes.get('TITLE_REVIEW_CANDIDATE', 0)
        self.state['previous_link_rediscoveries'] = self.ledger.db.execute('''
            SELECT COUNT(*) FROM records WHERE cohort='actual_blank' AND result IS NOT NULL
            AND json_extract(result,'$.previous_link_rediscovered')=1''').fetchone()[0]
        self.state['masked_title_comparison_counts'] = dict(self.ledger.db.execute('''
            SELECT json_extract(result,'$.title_comparison'),COUNT(*) FROM records
            WHERE cohort='masked_existing_value' AND result IS NOT NULL
            GROUP BY json_extract(result,'$.title_comparison')''').fetchall())
        self.state['masked_quarantined_disagreements'] = self.ledger.db.execute('''
            SELECT COUNT(*) FROM records WHERE cohort='masked_existing_value'
            AND status='MASKED_TITLE_DISAGREEMENT' AND result IS NOT NULL''').fetchone()[0]
        completed = self.state['actual_completed'] + self.state['masked_completed']
        self.state['records_per_hour'] = completed * 3600 / elapsed if elapsed else None
        engine.save(self.output / 'progress.json', self.state)

    def heartbeat(self):
        while not self.stop_heartbeat.is_set():
            try:
                engine.save(self.output / 'heartbeat.json', dict(contract=c.CONTRACT, updated_at=engine.utc(),
                    pid=os.getpid(), session_id=self.session_id, phase=self.state.get('phase'),
                    deadline_at=self.run_info['deadline_at'], contains_record_values=False))
            except OSError: self.heartbeat_error = True
            self.stop_heartbeat.wait(15)

    def recover_call(self, call, phase, case):
        directory = self.attempt_dir(call['seq'], phase, call['attempt'])
        if (directory / 'transport.json').exists():
            receipt = engine.read(directory / 'transport.json')
            if receipt['request_sha256'] != call['request_sha']: raise p.PilotError('CALL_RECEIPT_CHANGED')
            result = decode(receipt, case)
        else:
            result = dict(status='UNKNOWN_INTERRUPTED', error='DISPATCH_WITHOUT_RESPONSE',
                request_sha256=call['request_sha'], usage=None, model_seconds=None, answer=None,
                reasoning_content=None, started_at=engine.utc(call['dispatched']), finished_at=None)
        return self.finish_call(call, result)

    def call(self, seq, phase, body, case):
        request_hash = engine.sha(p.encoded(body)); calls = self.ledger.calls(seq, phase)
        result = None
        for call in calls:
            if call['request_sha'] != request_hash: raise p.PilotError('RESUMED_REQUEST_CHANGED')
            if call['result'] is None: result = self.recover_call(call, phase, case)
            else:
                if engine.sha(call['result'].encode()) != call['result_sha256']: raise p.PilotError('CALL_CHECKPOINT_CHANGED')
                result = p.strict_json(call['result'])
            if result['status'] not in ('TRANSPORT_ERROR', 'UNKNOWN_INTERRUPTED'): return result
        while len(calls) < CONFIG['max_transport_attempts']:
            self.check_deadline(); identity = self.wait_service()
            self.checkpoint(phase='tokenizing', active_sequence=seq, active_stage=phase)
            tick = time.monotonic()
            try:
                response = self.session.post(base.BASE + '/tokenize', json={k: body[k] for k in
                    ('model', 'messages', 'chat_template_kwargs')}, timeout=(5, 60), allow_redirects=False)
                response.raise_for_status()
            except requests.RequestException:
                self.checkpoint(phase='waiting_tokenizer'); time.sleep(min(15, max(0, self.deadline - time.time()))); continue
            tokenization_seconds = time.monotonic() - tick
            tokenized = p.strict_json(response.content); count = tokenized.get('count')
            if type(count) is not int or count <= 0 or tokenized.get('max_model_len') != identity['context_limit']:
                raise p.PilotError('TOKENIZER_INVALID')
            if count > CONFIG['prompt_limit'] or count + CONFIG['output_tokens'] + base.MARGIN > identity['context_limit']:
                return dict(status='INVALID', error='PROMPT_BUDGET_EXCEEDED', answer=None,
                    usage=None, model_seconds=0.0, reasoning_content=None, request_sha256=request_hash)
            self.check_deadline()
            directory = self.attempt_dir(seq, phase, len(calls) + 1)
            engine.publish(directory / 'request.json', body)
            dispatched = self.ledger.dispatch(seq, phase, body)
            engine.publish(directory / 'dispatch.json', dict(dispatched))
            self.checkpoint(phase='model_call', active_sequence=seq, active_stage=phase)
            tick = time.monotonic(); status, raw_bytes, error = None, b'', None
            try:
                response = self.session.post(base.BASE + '/v1/chat/completions', json=body,
                    timeout=(5, max(1, min(600, self.deadline - time.time()))), allow_redirects=False)
                status, raw_bytes = response.status_code, response.content
            except requests.RequestException as exc: error = type(exc).__name__
            transport = dict(request_sha256=request_hash, http_status=status, started_at=engine.utc(dispatched['dispatched']),
                finished_at=engine.utc(), model_seconds=time.monotonic() - tick, tokenization_seconds=tokenization_seconds,
                tokenized_prompt_count=count, response_sha256=engine.sha(raw_bytes),
                response_base64=base64.b64encode(raw_bytes).decode(), transport_error=error)
            engine.publish(directory / 'transport.json', transport)
            result = self.finish_call(dispatched, decode(transport, case))
            self.checkpoint(phase='response_saved')
            if result['status'] not in ('TRANSPORT_ERROR', 'UNKNOWN_INTERRUPTED'): return result
            calls = self.ledger.calls(seq, phase)
            if len(calls) < CONFIG['max_transport_attempts']:
                self.checkpoint(phase='transport_backoff'); time.sleep(min(15, max(0, self.deadline - time.time())))
        return result

    def process(self, entry):
        seq, kind, cohort = entry['seq'], entry['kind'], entry['cohort']
        case = p.strict_json(entry['payload'])
        self.ledger.begin_record(seq, kind, cohort)
        self.checkpoint(phase='record_start', active_sequence=seq, active_cohort=cohort)
        first = self.call(seq, 'selection', c.request(case), case)
        valid = first['status'] == 'VALID'; answer = first['answer']
        candidate, checks = c.selection(case['packet'], answer) if valid else (None, {})
        qualified = candidate is not None and checks.get('minimum_evidence_met') is True
        review, accepted, control, exact, normal_exact = None, False, None, None, None
        status = 'INVALID_SELECTION' if not valid else answer['decision']
        if kind == 'control':
            expected = case['expected_decision']
            expected_decisions = ('SAME_PUBLICATION', 'POSSIBLE_MATCH') if expected == 'POSSIBLE_OR_SAME' else (expected,)
            control = bool(valid and ((qualified and answer['candidate_id'] == 'c1'
                and answer['decision'] in expected_decisions) if expected != 'WITHHOLD'
                else answer['decision'] in ('REVIEW', 'NO_MATCH')))
            status = 'CONTROL_PASS' if control else 'CONTROL_FAIL'
        elif qualified:
            review = self.call(seq, 'review', c.request(case, True), case)
            valid = review['status'] == 'VALID'
            accepted = bool(valid and review['answer']['decision'] in ('SAME_PUBLICATION', 'POSSIBLE_MATCH')
                            and review['answer']['candidate_id'] == answer['candidate_id'])
            transferable = bool(checks.get('title_transfer_eligible') and candidate['title'].strip()
                                and review['answer'] and review['answer']['decision'] == 'SAME_PUBLICATION')
            review_title = bool(candidate['title'].strip() and checks.get('review_candidate_eligible') and not transferable)
            status = ('TITLE_AND_LINK_PROPOSAL' if transferable else 'TITLE_REVIEW_CANDIDATE' if review_title
                      else 'LINK_ONLY_PROPOSAL') if accepted else 'REVIEW_DISAGREEMENT' if valid else 'INVALID_REVIEW'
        elif candidate is not None:
            status = 'REVIEW_EVIDENCE_GAP'
        title_proposal, link_proposal, title_review_candidate, quarantined_title_candidate = None, None, None, None
        if accepted:
            link_proposal = dict(target_record_id=case['record_id'], peer_record_id=case['candidate_record_ids'][answer['candidate_id']],
                relation='same_publication_proposal' if status == 'TITLE_AND_LINK_PROPOSAL' else 'possible_same_publication',
                evidence_checks=checks, independently_verified=False,
                source_merged=False, target_is_masked=cohort == 'masked_existing_value')
            if status in ('TITLE_AND_LINK_PROPOSAL', 'TITLE_REVIEW_CANDIDATE'):
                copied = dict(record_id=case['record_id'], target_field=case['target_field'],
                    old_value=case.get('original_title', ''), new_value=candidate['title'],
                    source_record_id=link_proposal['peer_record_id'], source_field=case['target_field'],
                    source_file_sha256=c.SOURCE_HASHES[str(c.CORE)], source_edited=False,
                    independently_verified=False, evaluation_only=cohort == 'masked_existing_value',
                    review_required=True, evidence_strength=checks['evidence_strength'],
                    missing_identity_anchors=checks['missing_identity_anchors'])
                if status == 'TITLE_AND_LINK_PROPOSAL': title_proposal = copied
                else: title_review_candidate = dict(copied, eligible_for_title_transfer=False)
        comparison = None
        if cohort == 'masked_existing_value':
            proposed = title_proposal or title_review_candidate
            predicted = proposed['new_value'] if proposed else None
            exact = predicted == case['expected_title'] if predicted is not None else False
            normal_exact = c.norm(predicted) == c.norm(case['expected_title']) if predicted is not None else False
            comparison = c.title_comparison(predicted, case['expected_title'])
            if comparison == 'SUBSTANTIVE_DIFFERENCE':
                quarantined_title_candidate = dict(proposed, emitted_status=status, reason='STORED_REFERENCE_DISAGREEMENT')
                status = 'MASKED_TITLE_DISAGREEMENT'
                title_proposal, title_review_candidate, link_proposal = None, None, None
        call_rows = self.ledger.db.execute('SELECT * FROM calls WHERE seq=? ORDER BY id', (seq,)).fetchall()
        usage = {k: sum(row[k] or 0 for row in call_rows) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        unknown = sum(row['total_tokens'] is None for row in call_rows)
        started = self.ledger.db.execute('SELECT started FROM records WHERE seq=?', (seq,)).fetchone()[0]
        result = dict(schema_version=c.CONTRACT, sequence=seq, kind=kind, cohort=cohort, case_id=case['case_id'],
            record_id=case['record_id'], status=status, valid=valid, control_passed=control,
            exact_reference_match=exact, normalized_reference_match=normal_exact,
            title_comparison=comparison, rule_baseline=c.rule_baseline(case['packet']),
            expected_for_scoring_only=case.get('expected_title'),
            reference_title_in_candidates=(any(c.title_key(peer['title']) == c.title_key(case['expected_title']) for peer in case['packet']['candidates']) if cohort == 'masked_existing_value' else None),
            title_proposal=title_proposal, link_proposal=link_proposal,
            title_review_candidate=title_review_candidate, quarantined_title_candidate=quarantined_title_candidate,
            previous_link_rediscovered=bool(link_proposal and answer['candidate_id'] in case.get('previously_proposed_candidate_ids', [])),
            selection=first, review=review, evidence_checks=checks,
            case_sha256=engine.sha(entry['payload'].encode()), source_hashes=c.SOURCE_HASHES,
            candidate_record_ids=case.get('candidate_record_ids'), thinking=True, model=p.MODEL, revision=p.REVISION,
            notes=dict(selection_explanation=answer['explanation'] if answer else None,
                review_explanation=review['answer']['explanation'] if review and review['answer'] else None,
                explanation_source='Nemotron; no harness-authored explanation'),
            started_at=engine.utc(started), finished_at=engine.utc(), elapsed_seconds=time.time() - started,
            known_model_seconds=sum(row['model_seconds'] or 0 for row in call_rows), known_usage=usage,
            unknown_usage_calls=unknown, token_usage_complete=unknown == 0,
            attempt_count=len(call_rows), source_edits=0, independently_verified=False, contains_record_values=True)
        self.ledger.finish(seq, result); self.checkpoint(phase='record_saved')
        if kind == 'control' and not control: raise engine.Pause('SYNTHETIC_CONTROL_FAILED')
        if self.ledger.recent_failed(): raise engine.Pause('FIVE_CONSECUTIVE_INVALID_RECORDS')

    def export_results(self):
        super().export_results()
        exports = {'title_proposals.jsonl': 'title_proposal',
                   'title_candidates_review_only.jsonl': 'title_review_candidate',
                   'possible_duplicate_links.jsonl': 'link_proposal'}
        rows = [p.strict_json(row[0]) for row in self.ledger.db.execute(
            "SELECT result FROM records WHERE cohort='actual_blank' AND result IS NOT NULL ORDER BY seq")]
        for name, field in exports.items():
            engine.atomic(self.output / name, b''.join(p.encoded(row) for row in rows if row.get(field)))

    def run(self, queue, verify_inputs=verify):
        thread = threading.Thread(target=self.heartbeat, daemon=True); thread.start()
        status, reason, marker = 'complete', 'QUEUE_EXHAUSTED', 'COMPLETE.json'
        try:
            self.check_deadline()
            for entry in queue.execute('SELECT * FROM queue ORDER BY seq'):
                self.check_deadline()
                if self.ledger.completed(entry['seq']): continue
                if entry['kind'] == 'record':
                    passed = self.ledger.db.execute('SELECT COUNT(*) FROM records WHERE kind="control" AND control_passed=1 AND result IS NOT NULL').fetchone()[0]
                    if passed != c.SYNTHETIC_COUNT: raise engine.Pause('SYNTHETIC_GATE_CLOSED')
                self.process(entry)
                if entry['seq'] % 100 == 0: verify_inputs()
        except engine.Deadline:
            status, reason, marker = 'budget_complete', 'ABSOLUTE_OVERNIGHT_DEADLINE', 'BUDGET_COMPLETE.json'
        except engine.Interrupted as exc:
            if time.time() >= c.INFERENCE_DEADLINE_EPOCH:
                status, reason, marker = 'budget_complete', 'ABSOLUTE_OVERNIGHT_DEADLINE', 'BUDGET_COMPLETE.json'
            else:
                status, reason, marker = 'stopped', str(exc) or 'USER_STOPPED', 'STOPPED.json'
        except engine.Pause as exc:
            status, reason, marker = 'failed', str(exc), 'FAILED.json'
        except p.PilotError as exc:
            status, reason, marker = 'failed', str(exc), 'FAILED.json'
        except Exception as exc:
            self.checkpoint(status='paused', phase='unexpected_error', error_category=type(exc).__name__)
            engine.save(self.output / 'sessions' / (self.session_id + '-error.json'), self.state)
            raise
        finally:
            self.stop_heartbeat.set(); thread.join(timeout=2)
        try:
            verify_inputs(); self.state.update(source_immutable=True, frozen_inputs_immutable=True)
        except Exception:
            status, reason, marker = 'failed', 'FINAL_INTEGRITY_CHECK_FAILED', 'FAILED.json'
            self.state.update(source_immutable=None, frozen_inputs_immutable=None)
        self.export_results()
        self.checkpoint(status=status, phase='terminal' if status != 'paused' else 'paused', reason=reason,
            active_sequence=None, active_stage=None, active_cohort=None)
        engine.publish(self.output / 'sessions' / (self.session_id + '-end.json'), self.state)
        engine.save(self.output / marker, self.state)
        print(p.encoded(self.state).decode()); return 0


def run():
    os.umask(0o077)
    output = p.resident(p.ROOT / 'outputs' / c.RUN)
    if any((output / marker).exists() for marker in TERMINALS):
        print(p.encoded(dict(status='terminal_no_restart', run=c.RUN, contains_record_values=False)).decode()); return 0
    with base.client_lock():
        manifest, queue = c.load()
        runner = Runner(output, manifest)
        try: return runner.run(queue)
        finally:
            queue.close(); runner.ledger.db.close(); runner.session.close()
            drain_until_idle(output)


def drain_until_idle(output):
    """Retain the resident lock during drain; the Thor guard outlives any hard stop."""
    with requests.Session() as session:
        session.trust_env = False
        while time.time() < c.DEADLINE_EPOCH - 5:
            try:
                response = session.get(base.BASE + '/metrics', timeout=(3, 5), allow_redirects=False)
                response.raise_for_status()
                if base.idle(response.text):
                    engine.save(output / 'drain.json', dict(idle=True, checked_at=engine.utc(), contains_record_values=False))
                    return
            except requests.RequestException:
                pass
            time.sleep(3)
    engine.save(output / 'drain.json', dict(idle=False, global_guard_required=True,
        checked_at=engine.utc(), contains_record_values=False))


def request_stop():
    output = p.resident(p.ROOT / 'outputs' / c.RUN)
    if not any((output / name).exists() for name in TERMINALS):
        engine.save(output / 'STOP_REQUESTED.json', dict(run=c.RUN, requested_at=engine.utc(),
            reason='ABSOLUTE_OVERNIGHT_DEADLINE' if time.time() >= c.INFERENCE_DEADLINE_EPOCH else 'USER_STOPPED',
            safe_to_resume=False, contains_record_values=False))


def interrupted(*args):
    request_stop()
    raise engine.Interrupted('USER_STOPPED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'status', 'stop-intent'))
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.action == 'prepare': c.prepare()
        elif args.action == 'stop-intent': request_stop()
        elif args.action == 'status': print(p.encoded(engine.read(p.resident(p.ROOT / 'outputs' / c.RUN / 'progress.json'))).decode())
        else: raise SystemExit(run())
    except Exception as exc:
        print(p.encoded(dict(startup_error=type(exc).__name__, contains_record_values=False)).decode()); raise SystemExit(1)
