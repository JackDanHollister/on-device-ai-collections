import json
from pathlib import Path
import tempfile
import unittest

from collections_ai import runner as e


def rows(task='country'):
    values = {'country': 'Exampleland', 'bibliography': 'An invented paper', 'locality': [12.0, 34.0]}
    value = values[task]
    return [dict(id=f'synthetic-{i}', context={'description': 'Artificial test record',
                 'candidates': [{'title': 'An invented paper'}]},
                 reference=value if i % 10 == 9 else None) for i in range(100)]


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class ExperimentTests(unittest.TestCase):
    def test_balanced_plan_is_deterministic_and_has_no_reused_records(self):
        plan = e.plan_records(rows(), seed=7, max_records=50)
        self.assertEqual(plan, e.plan_records(rows(), seed=7, max_records=50))
        self.assertEqual(len({r['id'] for r in plan}), 50)
        for offset in range(0, 50, 10):
            self.assertEqual([r['cohort'] for r in plan[offset:offset+10]], ['blank'] * 9 + ['held_out'])
        self.assertNotEqual(plan, e.plan_records(rows(), seed=8, max_records=50))

    def test_insufficient_check_pool_is_not_silently_called_ten_percent(self):
        with self.assertRaises(ValueError):
            e.plan_records(rows()[:9], seed=1, max_records=10)

    def test_reference_and_target_fields_never_enter_model_packet(self):
        for task, target_keys in e.TARGET_KEYS.items():
            row = rows(task)[9]
            row['context'].update({k: 'SECRET_TARGET' for k in target_keys})
            sent = e.packet(task, row)
            self.assertNotIn('SECRET_TARGET', json.dumps(sent))
            self.assertNotIn('reference', sent)
            self.assertNotIn('id', sent)
            if task == 'bibliography':
                self.assertEqual(sent['context']['candidates'][0]['title'], 'An invented paper')

    def execute(self, task, infer, clock, hours=24):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'run'
        return e.run(task, e.plan_records(rows(task), seed=7, max_records=10), self.output,
                     hours=hours, infer=infer, model='synthetic-test', input_sha256='synthetic',
                     seed=7, clock=clock)

    def test_three_tasks_record_actual_runtime_and_aggregate_validation(self):
        for task in e.TASKS:
            with self.subTest(task=task):
                clock = FakeClock()
                expected = rows(task)[9]['reference']
                def infer(task, packet, remaining):
                    clock.value += 2
                    return dict(value=expected, usage=dict(prompt_tokens=10, completion_tokens=4))
                result = self.execute(task, infer, clock)
                self.assertEqual(result['configured_max_hours'], 24)
                self.assertEqual(result['elapsed_seconds'], 20)
                self.assertEqual(result['stop_reason'], 'dataset_exhausted')
                self.assertEqual(result['observed_held_out_fraction'], .1)
                self.assertEqual(result['counts']['held_out_answered'], 1)
                self.assertEqual(result['counts']['blank_answered'], 9)
                if task == 'locality':
                    self.assertEqual(result['held_out_median_distance_km'], 0)
                else:
                    self.assertEqual(result['held_out_exact_fraction'], 1)
                for row in rows(task):
                    self.assertNotIn('"' + row['id'] + '"', json.dumps(result))

    def test_time_limit_reports_actual_partial_validation_fraction(self):
        clock = FakeClock()
        def infer(task, packet, remaining):
            clock.value += 1
            return {'value': None}
        result = self.execute('country', infer, clock, hours=3/3600)
        self.assertEqual(result['counts']['attempted'], 3)
        self.assertEqual(result['stop_reason'], 'time_limit')
        self.assertEqual(result['observed_held_out_fraction'], 0)
        self.assertIsNone(result['held_out_exact_fraction'])

    def test_titles_cannot_be_invented_and_errors_do_not_leak_record_content(self):
        clock = FakeClock()
        def infer(task, packet, remaining):
            clock.value += 1
            return {'value': 'A title absent from the candidates'}
        result = self.execute('bibliography', infer, clock)
        self.assertEqual(result['stop_reason'], 'consecutive_errors')
        self.assertEqual(result['counts']['failed'], 5)
        self.assertEqual(result['counts']['usage_missing'], 5)
        self.assertNotIn('absent from the candidates', (self.output / 'records.jsonl').read_text())

    def test_interruption_keeps_an_honest_summary(self):
        clock = FakeClock()
        def infer(task, packet, remaining):
            clock.value += 1
            raise KeyboardInterrupt()
        result = self.execute('country', infer, clock)
        self.assertEqual(result['stop_reason'], 'interrupted')
        self.assertEqual(result['elapsed_seconds'], 1)
        self.assertEqual(result['counts']['failed'], 1)

    def test_coordinate_bounds_and_distance(self):
        for value in [[91, 0], [0, 181], [True, 2], [float('nan'), 0]]:
            with self.assertRaises(ValueError):
                e.validate_value('locality', value)
        self.assertAlmostEqual(e.distance_km([0, 0], [0, 1]), 111.195, places=2)


if __name__ == '__main__':
    unittest.main()
