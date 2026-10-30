"""Offline demonstration of the record-processing interface."""
from __future__ import annotations

from html import escape
from importlib.resources import files
import json
from pathlib import Path

from . import runner


def create_demo(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    cards = []
    for task in runner.TASKS:
        raw = files('collections_ai').joinpath('data', task + '.jsonl').read_bytes()
        input_file = output / (task + '-input.jsonl')
        input_file.write_bytes(raw)
        rows, fingerprint = runner.load_records(input_file, task)
        plan = runner.plan_records(rows, seed=2026, max_records=10)

        def simulated_response(task, payload, remaining):
            values = {'country': 'Exampleland', 'bibliography': 'An invented paper',
                      'locality': [12.0, 34.0]}
            return {'value': values[task]}

        summary = runner.run(task, plan, output / task, hours=24,
                             infer=simulated_response, model='offline artificial example',
                             input_sha256=fingerprint, seed=2026,
                             execution_mode='offline_demonstration')
        payload = runner.packet(task, plan[0])
        answer = simulated_response(task, payload, 1)['value']
        title = {'country': 'Missing country', 'bibliography': 'Possible article title',
                 'locality': 'Locality to coordinates'}[task]
        cards.append(f'''<section id="{task}"><h2>{title}</h2>
<div class="pair"><div><h3>Artificial input</h3><pre>{escape(json.dumps(payload['context'], indent=2))}</pre></div>
<div><h3>Simulated response</h3><pre>{escape(json.dumps({'value': answer}, indent=2))}</pre></div></div>
<p>Processed {summary['counts']['blank_attempted']} missing-value records and {summary['counts']['held_out_attempted']} masked check.
<a href="{task}/summary.json">Summary</a> · <a href="{task}/records.jsonl">Individual demonstration records</a></p></section>''')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local AI for collections data — offline demonstration</title>
<style>body{font:17px/1.55 system-ui,sans-serif;color:#213132;background:#f3f5f3;margin:0}main{max-width:1050px;margin:45px auto;padding:0 24px}h1{font-size:36px;line-height:1.15}h2{font-size:24px}h3{font-size:15px}section{background:white;border:1px solid #d8dfda;border-radius:10px;padding:24px;margin:22px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}pre{font:14px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f3;padding:15px;border-radius:6px}a{color:#205e55}.notice{background:#e4eee9;padding:18px;border-radius:8px}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style>
<main><h1>Local AI for collections data</h1>
<p>Three small examples show how records become candidates that can be checked.</p>
<p class="notice"><strong>Offline demonstration.</strong> These records are artificial and the responses are simulated.
No model, GPU, network connection or museum data is used. This demonstrates the interface; it does not measure model performance.</p>
''' + '\n'.join(cards) + '</main></html>'
    (output / 'index.html').write_text(html)
    return output / 'index.html'
