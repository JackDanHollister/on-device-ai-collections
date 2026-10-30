import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'gb10'))
import local_span_pilot as pilot
import run_local_span_pilots as runner


class SpanTests(unittest.TestCase):
    def case(self):
        return pilot.controls('esites')[0]

    def answer(self, quote='France', block='b0'):
        return dict(decision='COPY_SPAN', quote=quote, block_id=block)

    def response(self, answer, finish='stop'):
        return {'model': pilot.MODEL, 'choices': [{'finish_reason': finish,
            'message': {'role': 'assistant', 'content': pilot.encoded(answer).decode()}}]}

    def test_copy_has_exact_codepoint_provenance(self):
        packet = pilot.packet('esites', 'loccountry', [('locpreciselocation', '🦋 Country: France')], {})
        proposal = pilot.validate(packet, self.answer())
        self.assertEqual(proposal['start_codepoint'], 11)
        self.assertEqual(proposal['end_codepoint'], 17)
        self.assertEqual(proposal['new_value'], 'France')

    def test_repeated_invented_wrong_block_and_empty_quotes_rejected(self):
        packet = self.case()['packet']
        for answer in (self.answer('Spain'), self.answer(''), self.answer(block='b99')):
            with self.subTest(answer=answer), self.assertRaises(pilot.PilotError):
                pilot.validate(packet, answer)
        packet['evidence'][0]['text'] = 'France or France'
        with self.assertRaisesRegex(pilot.PilotError, 'UNIQUE_LITERAL'):
            pilot.validate(packet, self.answer())

    def test_overlapping_occurrences_are_ambiguous(self):
        packet = self.case()['packet']
        packet['evidence'][0]['text'] = 'aaaaa'
        with self.assertRaises(pilot.PilotError):
            pilot.validate(packet, self.answer('aaa'))

    def test_target_must_be_blank(self):
        packet = self.case()['packet']
        packet['current_value'] = 'France'
        with self.assertRaisesRegex(pilot.PilotError, 'POPULATED_TARGET'):
            pilot.validate(packet, self.answer())

    def test_abstention_cannot_smuggle_value(self):
        answer = self.answer()
        answer['decision'] = 'UNRESOLVED'
        with self.assertRaises(pilot.PilotError):
            pilot.validate(self.case()['packet'], answer)

    def test_required_fields_and_extra_keys(self):
        for answer in ({}, {**self.answer(), 'new_country': 'France'}):
            with self.assertRaisesRegex(pilot.PilotError, 'OUTPUT_SCHEMA'):
                pilot.validate(self.case()['packet'], answer)

    def test_reference_and_baseline_never_enter_model_request(self):
        case = self.case()
        case.update(record_id='private-marker', baseline='baseline-marker', expected={'quote': 'reference-marker'})
        request = runner.request(case, 'instructions', True)
        payload = request['messages'][1]['content']
        for text in ('private-marker', 'baseline-marker', 'reference-marker', 'expected', 'cohort'):
            self.assertNotIn(text, payload)
        self.assertEqual(request['chat_template_kwargs']['enable_thinking'], True)
        self.assertEqual(pilot.strict_json(payload)['output_schema'], request['response_format']['json_schema']['schema'])
        self.assertEqual(set(pilot.schema(case['packet'])['required']), {'decision', 'block_id', 'quote'})

    def test_response_must_finish_naturally_and_contain_valid_final(self):
        for response in (self.response(self.answer(), 'length'), self.response({}),
                         {**self.response(self.answer()), 'model': 'different-model'}):
            with self.assertRaises(pilot.PilotError):
                runner.evaluate(self.case(), response)
        _, proposal, score = runner.evaluate(self.case(), self.response(self.answer()))
        self.assertTrue(score)
        self.assertEqual(proposal['new_value'], 'France')

    def test_wrong_reference_is_recorded_and_actual_accuracy_stays_unknown(self):
        case = self.case()
        case['expected']['quote'] = 'Different reference'
        self.assertFalse(runner.evaluate(case, self.response(self.answer()))[2])
        case['expected'] = None
        self.assertIsNone(runner.evaluate(case, self.response(self.answer()))[2])

    def test_baseline_cannot_cross_empty_field_newline(self):
        packet = self.case()['packet']
        packet['evidence'][0]['text'] = 'Country: \nProvince: Somewhere'
        self.assertIsNone(pilot.baseline(packet))
        packet['evidence'][0]['text'] = 'Country: France\nProvince: Somewhere'
        self.assertEqual(pilot.baseline(packet)['quote'], 'France')

    def test_baseline_rejects_competing_values(self):
        packet = self.case()['packet']
        packet['evidence'][0]['text'] = 'Country: France\nCountry: Spain'
        self.assertIsNone(pilot.baseline(packet))

    def test_selections_are_stable_and_evidence_groups_disjoint(self):
        cases = []
        for n in range(8):
            case = self.case()
            case.update(record_id=str(n), stratum=str(n % 2))
            case['packet']['evidence'][0]['text'] = 'Synthetic evidence ' + str(n // 2)
            cases.append(case)
        groups = set()
        first = pilot.select(cases, 2, groups)
        second = pilot.select(cases, 2, groups)
        self.assertEqual(len(groups), 4)
        self.assertEqual(first, pilot.select(list(reversed(cases)), 2, set()))
        self.assertFalse({c['record_id'] for c in first} & {c['record_id'] for c in second})

    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in ('{"decision":1,"decision":2}', '{"x":NaN}'):
            with self.assertRaises(pilot.PilotError):
                pilot.strict_json(raw)

    def test_usage_and_service_gauges_fail_closed(self):
        self.assertIsNone(runner.usage({'usage': {'prompt_tokens': True, 'completion_tokens': 1, 'total_tokens': 2}}))
        self.assertIsNone(runner.usage({'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 3}}))
        idle = 'vllm:num_requests_running{engine="0"} 0.0\nvllm:num_requests_waiting{engine="0"} 0.0'
        self.assertTrue(runner.idle(idle))
        self.assertFalse(runner.idle(idle.replace('0.0', '1.0', 1)))
        self.assertFalse(runner.idle(''))


class RunnerAccountingTests(unittest.TestCase):
    def cohort(self):
        result = []
        for bundle in pilot.SOURCES:
            examples = pilot.controls(bundle)
            cases = copy.deepcopy(examples)
            for n in range(120):
                case = copy.deepcopy(examples[0])
                case['cohort'] = 'masked_existing_value' if n < 20 else 'actual_blank'
                case['record_id'] = str(n)
                if n >= 20:
                    case['expected'] = None
                cases.append(case)
            for n, case in enumerate(cases):
                case['case_id'] = f'{bundle}_{n:03d}'
                case['baseline'] = None
            result.extend(cases)
        return result

    def simulate(self, bad_first=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / 'outputs').mkdir()
        (root / 'work').mkdir()
        (root / 'work/.nemotron-client.lock').touch()
        cases = self.cohort()
        index = 0

        def post(url, **kwargs):
            nonlocal index
            if url.endswith('/tokenize'):
                content = {'count': 500, 'max_model_len': 1_000_000}
            else:
                case = cases[index]
                reference = case['expected'] or pilot.controls(case['packet']['dataset'])[0]['expected']
                decision = reference['decision']
                answer = {'decision': decision, 'quote': reference['quote'],
                    'block_id': 'b0' if decision == 'COPY_SPAN' else ''}
                if bad_first and index == 0:
                    answer = {}
                index += 1
                content = {'model': pilot.MODEL, 'usage': dict(prompt_tokens=500, completion_tokens=20, total_tokens=520),
                    'choices': [{'finish_reason': 'stop', 'message': {
                        'role': 'assistant', 'content': pilot.encoded(answer).decode()}}]}
            return SimpleNamespace(content=pilot.encoded(content), status_code=200, raise_for_status=lambda: None)

        manifest = {'source_hashes': {b: spec[1] for b, spec in pilot.SOURCES.items()}}
        session = SimpleNamespace(trust_env=False, post=post)
        with patch.object(pilot, 'ROOT', root), patch.object(pilot, 'resident', side_effect=lambda path, *args: path), \
                patch.object(pilot, 'fingerprints', return_value=manifest['source_hashes']), \
                patch.object(runner, 'load', return_value=(cases, pilot.PROMPTS, manifest, {})), \
                patch.object(runner, 'service', return_value={'context_limit': 1_000_000}), \
                patch.object(runner.requests, 'Session', return_value=session), patch('builtins.print'):
            code = runner.run(SimpleNamespace(prompt_token_limit=131072, no_thinking=False))
        output = root / 'outputs' / pilot.RUN
        marker = output / ('FAILED.json' if code else 'COMPLETE.json')
        return code, pilot.strict_json(marker.read_bytes()), output

    def test_full_driver_accounts_for_controls_actual_proposals_and_usage(self):
        code, summary, output = self.simulate()
        self.assertEqual(code, 0)
        self.assertEqual(summary['completed'], 248)
        self.assertEqual(summary['known_usage']['total_tokens'], 248 * 520)
        self.assertTrue(summary['usage_complete_for_attempted_calls'])
        self.assertTrue(summary['source_immutable'])
        for bundle, metrics in summary['datasets'].items():
            self.assertEqual(metrics['synthetic_passed'], 4)
            self.assertEqual(metrics['masked_exact'], 20)
            self.assertEqual(metrics['actual_completed'], 100)
            self.assertEqual(metrics['actual_proposals'], 100)
            self.assertIsNone(metrics['actual_semantic_accuracy'])
            self.assertEqual(len((output / (bundle + '-proposals.jsonl')).read_text().splitlines()), 100)

    def test_failed_safety_control_prevents_real_inference(self):
        code, summary, output = self.simulate(bad_first=True)
        self.assertEqual(code, 1)
        self.assertEqual(summary['completed'], 4)
        self.assertEqual(summary['error_category'], 'SYNTHETIC_GATE_FAILED')
        self.assertEqual(summary['datasets']['esites']['actual_completed'], 0)
        self.assertFalse((output / 'esites-proposals.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
