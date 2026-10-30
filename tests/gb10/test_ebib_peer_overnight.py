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
import ebib_peer_overnight as c
import run_ebib_peer_overnight as run
import hold_ebib_peer_overnight_lock as guard


class Server:
    def __init__(self):
        self.requests = []
        self.transient = 0
        self.fail_control = False
        self.force_possible = False

    @staticmethod
    def response(value, status=200, text=''):
        def raise_for_status():
            if status >= 400: raise run.requests.HTTPError('Synthetic')
        return SimpleNamespace(status_code=status, content=c.p.encoded(value), text=text, raise_for_status=raise_for_status)

    def get(self, url, **kwargs):
        if url.endswith('/metrics'):
            return self.response({}, text='vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n')
        return self.response({'data': [{'id': c.p.MODEL, 'root': 'synthetic/' + c.p.REVISION, 'max_model_len': 1000000}]})

    def post(self, url, json, **kwargs):
        if url.endswith('/tokenize'): return self.response({'count': 500, 'max_model_len': 1000000})
        self.requests.append(copy.deepcopy(json))
        if self.transient:
            self.transient -= 1
            return self.response({'error': 'Synthetic temporary error'}, status=503)
        packet = c.p.strict_json(json['messages'][1]['content'])['packet']
        _, checks = c.selection(packet, dict(decision='SAME_PUBLICATION', candidate_id='c1'))
        accept = checks['minimum_evidence_met'] and not (self.fail_control and len(self.requests) == 1)
        decision = 'SAME_PUBLICATION' if checks.get('strict_identity_evidence_met') and not self.force_possible else 'POSSIBLE_MATCH'
        answer = dict(decision=decision, candidate_id='c1', explanation='Synthetic evidence comparison.') if accept else dict(
            decision='REVIEW', candidate_id='', explanation='Synthetic insufficient or conflicting evidence.')
        return self.response({'model': c.p.MODEL, 'choices': [{'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': c.p.encoded(answer).decode(), 'reasoning_content': 'Synthetic thinking trace.'}}],
            'usage': dict(prompt_tokens=100, completion_tokens=20, total_tokens=120)})

    def close(self): pass


class OvernightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / 'run'
        self.server = Server()
        self.queue = sqlite3.connect(':memory:'); self.queue.row_factory = sqlite3.Row
        self.addCleanup(self.queue.close)
        self.queue.execute('CREATE TABLE queue(seq INTEGER PRIMARY KEY,kind TEXT,cohort TEXT,payload TEXT)')
        self.cases = c.controls()
        self.first = len(self.cases) + 1
        for cohort, variation in [('masked_existing_value', 'punctuation'), ('actual_blank', 'strong'),
                                  ('actual_blank', 'missing_pages'), ('actual_blank', 'both_blank'),
                                  ('masked_existing_value', 'disagreement'), ('actual_blank', 'strong')]:
            case = copy.deepcopy(c.controls()[0]); number = len(self.cases) + 1
            case.update(cohort=cohort, case_id='case_' + str(number), record_id='synthetic_' + str(number),
                target_field='arttitle', expected_title='Shells of an imaginary valley' if cohort.startswith('masked') else None,
                original_title='', candidate_record_ids={'c1': 'synthetic_peer_' + str(number)})
            if variation == 'punctuation': case['packet']['candidates'][0]['title'] += '.'
            if variation == 'missing_pages': case['packet']['target']['pages'] = ''
            if variation == 'both_blank': case['packet']['candidates'][0]['title'] = ''
            if variation == 'disagreement': case['expected_title'] = 'A different publication title'
            self.cases.append(case)
        for seq, case in enumerate(self.cases, 1):
            self.queue.execute('INSERT INTO queue VALUES(?,?,?,?)', (seq, 'control' if case['cohort'] == 'synthetic' else 'record', case['cohort'], c.p.encoded(case).decode()))
        self.queue.commit()
        self.manifest = dict(queue_entries=len(self.cases), source_records_scanned=505953)
        self.clock = c.INFERENCE_DEADLINE_EPOCH - 3600
        for obj, attr, value in [(run.requests, 'Session', lambda: self.server),
                                  (run.time, 'sleep', lambda _: None), (run.time, 'time', lambda: self.clock)]:
            context = patch.object(obj, attr, value); context.start(); self.addCleanup(context.stop)
        quiet = patch('builtins.print'); quiet.start(); self.addCleanup(quiet.stop)

    def runner(self):
        runner = run.Runner(self.output, self.manifest)
        self.addCleanup(runner.ledger.db.close)
        return runner

    def result(self, seq):
        return run.engine.read(self.output / 'records' / f'{seq:07d}.json')

    def test_controls_and_known_conflicts(self):
        for case in c.controls():
            _, checks = c.selection(case['packet'], dict(decision='SAME_PUBLICATION', candidate_id='c1'))
            self.assertEqual(checks['minimum_evidence_met'], case['expected_same'])
        packet = copy.deepcopy(c.controls()[0]['packet'])
        packet['target']['pages'] = ''
        for field, value in [('publication_date', '1905'), ('volume', '4'), ('issue', '2'), ('edition', '2')]:
            altered = copy.deepcopy(packet)
            if field in ('issue', 'edition'): altered['target'][field] = '1'
            altered['candidates'][0][field] = value
            self.assertFalse(c.evidence(altered['target'], altered['candidates'][0])['minimum_evidence_met'])

    def test_title_comparison_is_narrow_and_never_changes_copy(self):
        title = 'Shells of an imaginary valley'
        self.assertEqual(c.title_comparison(title + '.', title), 'TERMINAL_FULL_STOP_ONLY')
        for changed in (title + '..', title + ' Part II', title.replace('imaginary', 'different'), title.replace(' ', '-', 1)):
            self.assertEqual(c.title_comparison(changed, title), 'SUBSTANTIVE_DIFFERENCE')
        self.assertNotEqual(c.title_key('A. B.'), c.title_key('A. B'))
        self.assertEqual(c.title_comparison(title.upper(), title), 'CASE_SPACE_UNICODE_ONLY')

    def test_competing_title_punctuation_agrees_but_different_words_do_not(self):
        packet = copy.deepcopy(c.controls()[0]['packet'])
        packet['candidates'].append(dict(copy.deepcopy(packet['candidates'][0]), candidate_id='c2', title=packet['candidates'][0]['title'] + '.'))
        self.assertTrue(c.selection(packet, dict(decision='SAME_PUBLICATION', candidate_id='c1'))[1]['minimum_evidence_met'])
        packet['candidates'][1]['title'] += ' Part II'
        self.assertFalse(c.selection(packet, dict(decision='SAME_PUBLICATION', candidate_id='c1'))[1]['minimum_evidence_met'])

    def test_complete_retains_all_tiers_and_quarantines_disagreement(self):
        runner = self.runner(); runner.run(self.queue, lambda: self.manifest)
        state = run.engine.read(self.output / 'COMPLETE.json')
        self.assertEqual(state['actual_title_proposals'], 2)
        self.assertEqual(state['actual_review_title_candidates'], 1)
        self.assertEqual(state['actual_link_only_proposals'], 1)
        self.assertEqual(state['masked_quarantined_disagreements'], 1)
        punct = self.result(self.first)
        self.assertFalse(punct['exact_reference_match']); self.assertFalse(punct['normalized_reference_match'])
        self.assertEqual(punct['title_comparison'], 'TERMINAL_FULL_STOP_ONLY')
        self.assertTrue(punct['reference_title_in_candidates'])
        self.assertTrue(punct['title_proposal']['new_value'].endswith('.'))
        weak = self.result(self.first + 2)
        self.assertIsNone(weak['title_proposal']); self.assertIsNotNone(weak['title_review_candidate'])
        bad = self.result(self.first + 4)
        self.assertEqual(bad['status'], 'MASKED_TITLE_DISAGREEMENT')
        self.assertIsNotNone(bad['quarantined_title_candidate']); self.assertIsNone(bad['title_proposal'])
        self.assertIsNone(bad['link_proposal']); self.assertTrue(bad['valid'])
        self.assertIsNotNone(self.result(self.first + 5)['title_proposal'])
        self.assertEqual(state['known_total_tokens'], len(self.server.requests) * 120)
        self.assertFalse(state['safe_to_resume']); self.assertEqual(state['source_edits'], 0)
        self.assertEqual(len((self.output / 'title_candidates_review_only.jsonl').read_text().splitlines()), 1)

    def test_possible_model_decision_never_becomes_strong_fill(self):
        self.server.force_possible = True
        runner = self.runner()
        runner.process(self.queue.execute('SELECT * FROM queue WHERE seq=?', (self.first + 1,)).fetchone())
        result = self.result(self.first + 1)
        self.assertIsNone(result['title_proposal']); self.assertIsNotNone(result['title_review_candidate'])

    def test_prompts_do_not_contain_reference_or_rule_answer(self):
        case = copy.deepcopy(self.cases[self.first - 1]); case['expected_title'] = 'SECRET_HELD_REFERENCE'
        for review in (False, True):
            body = c.request(case, review); encoded = json.dumps(body)
            self.assertNotIn('SECRET_HELD_REFERENCE', encoded)
            self.assertNotIn('rule_baseline', encoded); self.assertNotIn('expected_title', encoded)
            self.assertTrue(body['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(c.p.strict_json(body['messages'][1]['content'])['packet']['target']['title'], '')

    def test_resume_keeps_absolute_deadline_and_saved_calls(self):
        runner = self.runner(); runner.process(self.queue.execute('SELECT * FROM queue WHERE seq=1').fetchone())
        self.assertEqual(runner.deadline, c.INFERENCE_DEADLINE_EPOCH)
        runner.ledger.db.close(); self.clock += 120
        resumed = self.runner(); self.assertEqual(resumed.deadline, c.INFERENCE_DEADLINE_EPOCH)
        resumed.run(self.queue, lambda: self.manifest)
        self.assertEqual(len(self.server.requests), len(self.cases) + 6)

    def test_manual_stop_is_terminal(self):
        runner = self.runner()
        with patch.object(runner, 'check_deadline', side_effect=run.engine.Interrupted('USER_STOPPED')):
            runner.run(self.queue, lambda: self.manifest)
        self.assertTrue((self.output / 'STOPPED.json').exists())
        with self.assertRaisesRegex(c.p.PilotError, 'TERMINAL_ROOT_CANNOT_RESUME'): self.runner()
        self.assertEqual(self.server.requests, [])

    def test_expired_deadline_never_dispatches(self):
        self.clock = c.DEADLINE_EPOCH + 100
        self.runner().run(self.queue, lambda: self.manifest)
        self.assertEqual(self.server.requests, [])
        self.assertEqual(run.engine.read(self.output / 'BUDGET_COMPLETE.json')['reason'], 'ABSOLUTE_OVERNIGHT_DEADLINE')

    def test_crash_after_transport_receipt_recovers_without_duplicate_request(self):
        runner = self.runner(); real_finish = runner.finish_call
        with patch.object(runner, 'finish_call', side_effect=RuntimeError('Synthetic crash after transport')):
            with self.assertRaises(RuntimeError): runner.process(self.queue.execute('SELECT * FROM queue WHERE seq=1').fetchone())
        self.assertEqual(len(self.server.requests), 1)
        self.assertTrue((self.output / 'attempts/0000001/selection/01/transport.json').exists())
        runner.ledger.db.close(); resumed = self.runner()
        resumed.process(self.queue.execute('SELECT * FROM queue WHERE seq=1').fetchone())
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(self.result(1)['known_usage']['total_tokens'], 120)

    def test_transport_retry_is_bounded_and_unknown_usage_visible(self):
        self.server.transient = 1
        self.runner().run(self.queue, lambda: self.manifest)
        state = run.engine.read(self.output / 'COMPLETE.json')
        self.assertEqual(state['unknown_usage_calls'], 1)
        self.assertEqual(state['known_total_tokens'], (len(self.server.requests) - 1) * 120)

    def test_synthetic_failure_stops_before_real_data(self):
        self.server.fail_control = True
        self.runner().run(self.queue, lambda: self.manifest)
        state = run.engine.read(self.output / 'FAILED.json')
        self.assertEqual(state['actual_completed'], 0)
        self.assertEqual(state['reason'], 'SYNTHETIC_CONTROL_FAILED')

    def test_guard_waits_for_both_terminal_and_drain(self):
        states = [dict(stopped_and_drained=False, terminal_marker='BUDGET_COMPLETE.json'),
                  dict(stopped_and_drained=True, terminal_marker=None),
                  dict(stopped_and_drained=True, terminal_marker='BUDGET_COMPLETE.json')]
        with patch.object(guard, 'remote', side_effect=[SimpleNamespace(stdout=json.dumps(s)) for s in states]) as probe:
            guard.monitor_until_drained()
            self.assertEqual(probe.call_count, 3)

    def test_guard_retains_ownership_across_all_startup_ssh_failures(self):
        running = SimpleNamespace(stdout=json.dumps(dict(terminal_marker=None, stopped_and_drained=False,
            deadline_timer=dict(ActiveState='active'))))
        done = SimpleNamespace(stdout=json.dumps(dict(terminal_marker='COMPLETE.json', stopped_and_drained=True)))
        def timeout(): return guard.subprocess.TimeoutExpired('synthetic-ssh', 50)
        sequence = [timeout(), running, timeout(), running, SimpleNamespace(), timeout(),
                    running, SimpleNamespace(), running, timeout(), done]
        with patch.object(guard, 'remote', side_effect=sequence) as remote:
            guard.launch_and_monitor()
            self.assertEqual(remote.call_count, len(sequence))

    def test_stop_intent_blocks_start_even_before_runner_initialization(self):
        root = Path(self.tmp.name) / 'resident'
        with patch.object(run.p, 'ROOT', root), patch.object(run.p, 'resident', side_effect=lambda path, *args: path):
            run.request_stop()
        output = root / 'outputs' / c.RUN
        self.assertTrue((output / 'STOP_REQUESTED.json').exists())
        with self.assertRaisesRegex(c.p.PilotError, 'TERMINAL_ROOT_CANNOT_RESUME'):
            run.Runner(output, self.manifest)

    def test_stop_helper_persists_intent_before_service_stop(self):
        with patch.object(sys, 'argv', ['guard', 'stop']), patch.object(guard, 'remote') as remote:
            guard.main()
        self.assertIn('stop-intent', remote.call_args_list[0].args[0])
        self.assertEqual(remote.call_args_list[1].args[0], 'systemctl --user stop ' + guard.CLIENT)

    def test_previous_link_is_explicitly_counted_as_rediscovery(self):
        seq = self.first + 3
        case = copy.deepcopy(self.cases[seq - 1]); case['previously_proposed_candidate_ids'] = ['c1']
        self.queue.execute('UPDATE queue SET payload=? WHERE seq=?', (c.p.encoded(case).decode(), seq))
        self.queue.commit(); self.runner().run(self.queue, lambda: self.manifest)
        self.assertTrue(self.result(seq)['previous_link_rediscovered'])
        self.assertEqual(run.engine.read(self.output / 'COMPLETE.json')['previous_link_rediscoveries'], 1)

    def test_absolute_timer_and_guard_identity(self):
        scripts = Path(__file__).resolve().parents[2] / 'archive' / 'gb10-services'
        timer = (scripts / 'tdwg-ebib-peer-overnight-v1-stop.timer').read_text()
        self.assertIn('OnCalendar=2026-09-15 05:59:00 UTC', timer)
        self.assertIn('Persistent=true', timer); self.assertIn('AccuracySec=1s', timer)
        compile(guard.PROBE, 'probe', 'exec')
        self.assertIn(c.RUN, guard.PROBE); self.assertIn(c.CONTRACT, guard.PROBE)
        self.assertIn('STOPPED.json', guard.PROBE)


if __name__ == '__main__': unittest.main()
