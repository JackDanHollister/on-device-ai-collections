import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'gb10'))
import local_span_pilot as p
import run_local_span_pilots as base
import run_esites_review_comparison as reviewer
import run_esites_country_24h as run
import hold_esites_country_24h_v2_lock as guard


class SyntheticServer:
    def __init__(self):
        self.posted = []
        self.fail_actual = False
        self.fail_control = False
        self.transport_failures = 0
        self.ambiguous_review_label = 'AMBIGUOUS'

    @staticmethod
    def response(value, status=200, text=''):
        def raise_for_status():
            if status >= 400:
                raise run.requests.HTTPError('Synthetic HTTP error')
        return SimpleNamespace(status_code=status, content=p.encoded(value), text=text,
                               raise_for_status=raise_for_status)

    def get(self, url, **kwargs):
        if url.endswith('/metrics'):
            return self.response({}, text='vllm:num_requests_running{model_name="synthetic"} 0\nvllm:num_requests_waiting{model_name="synthetic"} 0\n')
        return self.response({'data': [{'id': p.MODEL, 'root': 'synthetic/' + p.REVISION, 'max_model_len': 1000000}]})

    def post(self, url, **kwargs):
        if url.endswith('/tokenize'):
            return self.response({'count': 250, 'max_model_len': 1000000})
        body = copy.deepcopy(kwargs['json'])
        self.posted.append(body)
        if self.transport_failures:
            self.transport_failures -= 1
            return self.response({'error': 'synthetic service failure'}, status=503)
        payload = p.strict_json(body['messages'][1]['content'])
        text = payload['packet']['evidence'][0]['text']
        if 'proposed_country' in payload:
            label = 'SUPPORTED'
            if 'postal address' in text or 'old attribution' in text:
                label = 'NOT_SUPPORTED'
            elif 'either France or Spain' in text:
                label = self.ambiguous_review_label
            answer = {'assessment': label, 'explanation': 'Synthetic source-based explanation.'}
        else:
            decision, quote, block = 'COPY_SPAN', 'France', 'b0'
            if 'INSTRUCTION' in text or 'country was not recorded' in text:
                decision, quote, block = 'UNRESOLVED', '', ''
            elif 'either France or Spain' in text:
                decision, quote, block = 'REVIEW', '', ''
            if self.fail_control and text.startswith('Country: France\nLocality:'):
                decision, quote, block = 'UNRESOLVED', '', ''
            if self.fail_actual and 'Synthetic record' in text:
                decision, quote, block = 'UNRESOLVED', '', 'b0'
            answer = {'decision': decision, 'quote': quote, 'block_id': block}
        return self.response({'model': p.MODEL, 'id': 'synthetic-response',
            'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant',
                'content': p.encoded(answer).decode(), 'reasoning_content': 'Synthetic model reasoning.'}}],
            'usage': {'prompt_tokens': 250, 'completion_tokens': 40, 'total_tokens': 290}})

    def close(self):
        pass


class LongrunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'output'
        self.server = SyntheticServer()
        self.queue = sqlite3.connect(':memory:')
        self.queue.row_factory = sqlite3.Row
        self.queue.execute('CREATE TABLE queue(seq INTEGER PRIMARY KEY,kind TEXT,cohort TEXT,payload TEXT)')
        seq = 0
        for case in p.controls('esites'):
            seq += 1
            self.insert(seq, 'extraction_control', 'synthetic', case)
        for task in reviewer.review_controls():
            seq += 1
            self.insert(seq, 'review_control', 'synthetic', task)
        for i in range(10):
            case = copy.deepcopy(p.controls('esites')[0])
            case.update(record_id=f'synthetic-{i}', cohort='masked_existing_value' if i == 0 else 'actual_blank',
                        expected={'decision': 'COPY_SPAN', 'quote': 'France'} if i == 0 else None,
                        peer_countries=[])
            case['packet']['evidence'][0]['text'] = f'Country: France\nSynthetic record {i}.'
            seq += 1
            self.insert(seq, 'record', case['cohort'], case)
        self.queue.commit()
        self.manifest = {'queue_entries': seq, 'source_hashes': {'esites': p.SOURCES['esites'][1]}}
        self.addCleanup(self.queue.close)
        self.session_patch = patch.object(run.requests, 'Session', return_value=self.server)
        self.session_patch.start()
        self.addCleanup(self.session_patch.stop)
        self.print_patch = patch('builtins.print')
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    def insert(self, seq, kind, cohort, payload):
        self.queue.execute('INSERT INTO queue VALUES (?,?,?,?)', (seq, kind, cohort, p.encoded(payload).decode()))

    def runner(self):
        runner = run.Runner(self.output, self.manifest)
        self.addCleanup(runner.ledger.db.close)
        return runner

    def case(self, seq=9):
        return p.strict_json(self.queue.execute('SELECT payload FROM queue WHERE seq=?', (seq,)).fetchone()[0])

    def test_schedule_one_masked_nine_actual_in_every_full_block(self):
        actual, masked = list(range(90)), ['test' + str(i) for i in range(10)]
        sequence = list(run.interleave(actual, masked))
        self.assertEqual(len(sequence), 100)
        self.assertEqual(sequence[::10], masked)
        self.assertEqual([x for x in sequence if isinstance(x, int)], actual)
        self.assertEqual(list(run.interleave([1, 2], ['a', 'b'])), ['a', 1, 2])

    def test_reference_and_record_identity_never_enter_either_inference_body(self):
        case = self.case()
        case['expected']['quote'] = 'HIDDEN-REFERENCE-SENTINEL'
        body = run.extraction_body(case)
        self.assertNotIn('HIDDEN-REFERENCE-SENTINEL', p.encoded(body).decode())
        self.assertNotIn('synthetic-0', p.encoded(body).decode())
        self.assertEqual(body, base.request(case, p.PROMPTS['esites'], True))
        proposal = p.validate(case['packet'], {'decision': 'COPY_SPAN', 'block_id': 'b0', 'quote': 'France'})
        review = reviewer.review_body(case['packet'], proposal, [])
        self.assertNotIn('HIDDEN-REFERENCE-SENTINEL', p.encoded(review).decode())
        self.assertEqual(p.strict_json(review['messages'][1]['content'])['peer_countries'], [])

    def test_full_synthetic_workflow_records_notes_tokens_times_and_ten_percent_tests(self):
        runner = self.runner()
        self.assertEqual(runner.run(self.queue, verify=lambda: None), 0)
        summary = run.read(self.output / 'COMPLETE.json')
        self.assertEqual(summary['actual_completed'], 9)
        self.assertEqual(summary['masked_completed'], 1)
        self.assertEqual(summary['test_fraction_completed'], .1)
        self.assertEqual(summary['model_calls'], 28)
        self.assertEqual(summary['known_total_tokens'], 28 * 290)
        self.assertEqual(summary['actual_outcomes'], {'PROPOSAL_SUPPORTED': 9})
        self.assertEqual(summary['cohorts']['masked_existing_value']['exact'], 1)
        result = run.read(self.output / 'records/0000009.json')
        self.assertEqual(result['known_usage']['total_tokens'], 580)
        self.assertEqual(result['attempt_count'], 2)
        self.assertTrue(result['supported_exact_reference_match'])
        self.assertEqual(result['notes']['extraction_reasoning'], 'Synthetic model reasoning.')
        self.assertEqual(result['notes']['review_explanation'], 'Synthetic source-based explanation.')
        self.assertIsNone(result['extraction']['reasoning_tokens'])
        self.assertTrue(result['usage_complete'])
        self.assertFalse(result['source_updated'])
        self.assertFalse(result['independent_semantic_truth'])
        self.assertGreaterEqual(result['elapsed_seconds_including_interruptions'], result['known_model_seconds'])
        self.assertEqual(len((self.output / 'results.jsonl').read_text().splitlines()), 18)

    def test_pause_between_extraction_and_review_resumes_without_repeating_extraction(self):
        first = self.runner()
        original_call = first.call
        def interrupt(seq, phase, body, case):
            if seq == 9 and phase == 'review':
                raise run.Interrupted('synthetic disconnect')
            return original_call(seq, phase, body, case)
        with patch.object(first, 'call', side_effect=interrupt):
            first.run(self.queue, verify=lambda: None)
        self.assertEqual(len(self.server.posted), 9)
        started, deadline = first.run_info['started_epoch'], first.deadline
        second = self.runner()
        second.run(self.queue, verify=lambda: None)
        self.assertEqual(len(self.server.posted), 28)
        self.assertEqual(second.run_info['started_epoch'], started)
        self.assertEqual(second.deadline, deadline)
        self.assertEqual(len(second.ledger.calls(9, 'extraction')), 1)

    def test_saved_transport_survives_crash_before_assessment_and_avoids_new_call(self):
        runner = self.runner()
        case = self.case()
        runner.ledger.begin_record(9, 'record', case['cohort'])
        body = run.extraction_body(case)
        with patch.object(runner, 'finish_call', side_effect=run.Interrupted('after response')):
            with self.assertRaises(run.Interrupted):
                runner.call(9, 'extraction', body, case)
        self.assertEqual(len(self.server.posted), 1)
        second = self.runner()
        result = second.call(9, 'extraction', body, case)
        self.assertEqual(result['status'], 'VALID')
        self.assertEqual(len(self.server.posted), 1)
        self.assertEqual(second.ledger.totals()['known_total_tokens'], 290)

    def test_unknown_interrupted_call_is_preserved_and_never_counted_as_zero_usage(self):
        runner = self.runner()
        case, body = self.case(), run.extraction_body(self.case())
        runner.ledger.begin_record(9, 'record', case['cohort'])
        runner.ledger.dispatch(9, 'extraction', body)
        result = runner.call(9, 'extraction', body, case)
        self.assertEqual(result['status'], 'VALID')
        rows = runner.ledger.calls(9, 'extraction')
        self.assertEqual([r['status'] for r in rows], ['UNKNOWN_INTERRUPTED', 'VALID'])
        self.assertIsNone(rows[0]['total_tokens'])
        self.assertIsNone(rows[0]['model_seconds'])
        self.assertEqual(runner.ledger.totals()['unknown_usage_calls'], 1)
        self.assertEqual(len(self.server.posted), 1)

    def test_transport_retry_is_bounded_and_all_attempts_are_saved(self):
        runner = self.runner()
        self.server.transport_failures = 10
        case = self.case()
        runner.ledger.begin_record(9, 'record', case['cohort'])
        with patch.object(run.time, 'sleep'):
            result = runner.call(9, 'extraction', run.extraction_body(case), case)
        self.assertEqual(result['status'], 'TRANSPORT_ERROR')
        self.assertEqual(len(self.server.posted), 2)
        self.assertEqual(len(runner.ledger.calls(9, 'extraction')), 2)
        runner.call(9, 'extraction', run.extraction_body(case), case)
        self.assertEqual(len(self.server.posted), 2)

    def test_invalid_model_output_is_preserved_without_retry_and_five_failures_pause(self):
        runner = self.runner()
        self.server.fail_actual = True
        runner.run(self.queue, verify=lambda: None)
        summary = run.read(self.output / 'PAUSED.json')
        self.assertEqual(summary['reason'], 'FIVE_CONSECUTIVE_INVALID_RECORDS')
        self.assertEqual(summary['completed_entries'], 13)
        self.assertEqual(len(self.server.posted), 13)
        first = run.read(self.output / 'records/0000009.json')
        self.assertEqual(first['extraction']['error'], 'ABSTENTION_WITH_VALUE')
        self.assertEqual(first['attempt_count'], 1)
        self.assertEqual(first['known_usage']['total_tokens'], 290)

    def test_control_failure_closes_gate_before_real_work_including_on_resume(self):
        runner = self.runner()
        self.server.fail_control = True
        runner.run(self.queue, verify=lambda: None)
        self.assertEqual(len(self.server.posted), 1)
        self.assertEqual(run.read(self.output / 'PAUSED.json')['reason'], 'SYNTHETIC_CONTROL_FAILED')
        second = self.runner()
        second.run(self.queue, verify=lambda: None)
        self.assertEqual(run.read(self.output / 'PAUSED.json')['actual_started'], 0)
        self.assertEqual(run.read(self.output / 'PAUSED.json')['reason'], 'SYNTHETIC_GATE_CLOSED')

    def test_deadline_survives_restart_and_expiry_prevents_any_new_request(self):
        first = self.runner()
        old_deadline = first.deadline
        with patch.object(run.time, 'time', return_value=old_deadline + 1):
            second = self.runner()
            self.assertEqual(second.deadline, old_deadline)
            second.run(self.queue, verify=lambda: None)
        self.assertEqual(self.server.posted, [])
        self.assertEqual(run.read(self.output / 'BUDGET_COMPLETE.json')['model_calls'], 0)
        with self.assertRaisesRegex(p.PilotError, 'TERMINAL_ROOT'):
            self.runner()

    def test_new_corrected_version_keeps_the_previously_authorized_window(self):
        original_start = run.time.time() - 3600
        self.manifest.update(window_started_epoch=original_start, window_deadline_epoch=original_start + run.SECONDS)
        runner = self.runner()
        self.assertEqual(runner.run_info['started_epoch'], original_start)
        self.assertEqual(runner.deadline, original_start + run.SECONDS)
        self.assertLess(runner.deadline - run.time.time(), run.SECONDS - 3599)

    def test_integrity_failure_prevents_complete_marker(self):
        runner = self.runner()
        def corrupted():
            raise p.PilotError('synthetic source drift')
        runner.run(self.queue, verify=corrupted)
        self.assertTrue((self.output / 'FAILED.json').exists())
        self.assertFalse((self.output / 'COMPLETE.json').exists())
        self.assertFalse(run.read(self.output / 'FAILED.json')['safe_to_resume'])

    def test_checkpoint_materializes_missing_json_and_rejects_modified_result(self):
        runner = self.runner()
        runner.process(self.queue.execute('SELECT * FROM queue WHERE seq=1').fetchone())
        path = self.output / 'records/0000001.json'
        original = path.read_bytes()
        path.unlink()
        runner.ledger.materialize()
        self.assertEqual(path.read_bytes(), original)
        run.save(path, {'corrupted': True})
        with self.assertRaisesRegex(p.PilotError, 'IMMUTABLE_ARTIFACT_CHANGED'):
            runner.ledger.materialize()

    def test_request_drift_is_rejected_on_resume(self):
        runner = self.runner()
        case = self.case()
        runner.ledger.begin_record(9, 'record', case['cohort'])
        body = run.extraction_body(case)
        runner.call(9, 'extraction', body, case)
        body['temperature'] = 0
        with self.assertRaisesRegex(p.PilotError, 'RESUMED_REQUEST_CHANGED'):
            runner.call(9, 'extraction', body, case)

    def test_guard_probe_is_fixed_and_compiles(self):
        compile(guard.PROBE, 'probe', 'exec')
        self.assertIn(run.RUN, guard.PROBE)
        self.assertIn(run.UNIT, guard.PROBE)
        self.assertIn('stopped_and_drained', guard.PROBE)

    def test_review_gate_accepts_both_withholding_labels_but_never_unsafe_support(self):
        for expected in ('AMBIGUOUS', 'NOT_SUPPORTED'):
            for label in ('AMBIGUOUS', 'NOT_SUPPORTED'):
                self.assertTrue(run.review_control_passed({'status':'VALID','answer':{'assessment':label}}, expected))
            self.assertFalse(run.review_control_passed({'status':'VALID','answer':{'assessment':'SUPPORTED'}}, expected))
        self.assertFalse(run.review_control_passed({'status':'VALID','answer':{'assessment':'AMBIGUOUS'}}, 'SUPPORTED'))
        self.assertFalse(run.review_control_passed({'status':'INVALID','answer':None}, 'AMBIGUOUS'))

    def test_withholding_label_difference_keeps_exact_score_false_and_allows_real_work(self):
        runner = self.runner()
        self.server.ambiguous_review_label = 'NOT_SUPPORTED'
        runner.run(self.queue, verify=lambda: None)
        result = run.read(self.output / 'records/0000007.json')
        self.assertFalse(result['exact_reference_match'])
        self.assertTrue(result['control_passed'])
        self.assertEqual(result['review']['answer']['assessment'], 'NOT_SUPPORTED')
        summary = run.read(self.output / 'COMPLETE.json')
        self.assertEqual(summary['cohorts']['synthetic']['exact'], 7)
        self.assertEqual(summary['cohorts']['synthetic']['controls_passed'], 8)
        self.assertEqual(summary['actual_completed'], 9)


if __name__ == '__main__':
    unittest.main()
