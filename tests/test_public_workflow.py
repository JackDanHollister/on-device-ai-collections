from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys

import duckdb
import pytest

from collections_ai import cli, demo, gb10, gb300, runner

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    path = ROOT / 'scripts/gb300' / (name + '.py')
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def test_offline_demo_uses_no_network_or_performance_claims(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Offline demo attempted a network call')
    monkeypatch.setattr(gb10, 'urlopen', forbidden)
    index = demo.create_demo(tmp_path / 'demo')
    assert 'responses are simulated' in index.read_text()
    for task in runner.TASKS:
        summary = json.loads((index.parent / task / 'summary.json').read_text())
        assert summary['counts']['attempted'] == 10
        assert summary['counts']['held_out_attempted'] == 1
        assert summary['execution_mode'] == 'offline_demonstration'
        for key in ['projected_records_per_24h', 'held_out_exact_fraction', 'held_out_median_distance_km']:
            assert summary[key] is None


def test_example_export_never_overwrites_a_users_file(tmp_path):
    target = tmp_path / 'data.jsonl'
    cli.main(['example', '--task', 'country', '--output', str(target)])
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(rows) == 10
    assert all(row['synthetic'] for row in rows)
    original = target.read_bytes()
    with pytest.raises(SystemExit) as stopped:
        cli.main(['example', '--task', 'country', '--output', str(target)])
    assert stopped.value.code == 2
    assert target.read_bytes() == original


def test_staged_runtime_satisfies_the_original_freeze_contract(tmp_path):
    target = tmp_path / 'runtime'
    gb300.stage_runtime(target)
    manifest = json.loads((target / 'runtime-source.json').read_text())['files']
    control = module('gb300_experiment_control_v2')
    frozen = control.runtime_files(target)
    assert all(manifest[k] == v for k, v in frozen.items())
    assert not list(target.rglob('*.sqlite3'))
    assert not list(target.rglob('.env'))


@pytest.mark.parametrize('suffix,delimiter', [('.csv', ','), ('.tsv', '\t')])
def test_tabular_adapter_and_original_blind_preparation(tmp_path, suffix, delimiter):
    source = tmp_path / ('input' + suffix)
    source.write_text(delimiter.join(['gbifid','locality','countrycode','stateprovince','decimallatitude','decimallongitude']) + '\n' +
                      delimiter.join(['fake-1','Example Wood','GB','England','','']) + '\n' +
                      delimiter.join(['fake-2','Example Wood','GB','England','51.5','-0.1']) + '\n')
    prepared_source = tmp_path / 'source'
    gb300.prepare_source(source, prepared_source)
    prep = module('prepare_gb300_combined_experiment')
    result = prep.prepare(prepared_source, tmp_path / 'prepared')
    assert result['counts']['records_by_cohort'] == {'production': 1, 'evaluation': 1}
    assert result['counts']['unique_candidates'] == 1
    with duckdb.connect() as db:
        columns = db.read_parquet(str(tmp_path / 'prepared/candidates.parquet')).columns
        assert not {'decimallatitude', 'decimallongitude', 'latitude', 'longitude'} & set(columns)


def test_bad_source_identifiers_leave_no_complete_output(tmp_path):
    source = tmp_path / 'input.csv'
    source.write_text('gbifid,locality,countrycode,stateprovince,decimallatitude,decimallongitude\na,One,GB,,,\na,Two,GB,,,\n')
    out = tmp_path / 'output'
    with pytest.raises(ValueError, match='unique'):
        gb300.prepare_source(source, out)
    assert not out.exists()


def test_reference_aliases_are_removed_but_peer_titles_are_kept():
    row = {'context': {'Country_Code': 'hidden', 'loccountry': 'hidden', 'description': 'visible'}}
    assert runner.packet('country', row)['context'] == {'description': 'visible'}
    row = {'context': {'Decimal_Latitude': 'hidden', 'DECIMALLONGITUDE': 'hidden', 'locality': 'visible'}}
    assert runner.packet('locality', row)['context'] == {'locality': 'visible'}


def test_gb300_service_uses_initialized_experiment_paths(tmp_path):
    root = tmp_path / 'run with spaces'
    root.mkdir()
    (root/'experiment.json').write_text(json.dumps({'unit':'collections-gb300.service',
        'runner_command':['/example/python']}))
    output = tmp_path/'units/collections-gb300.service'
    result = gb300.write_service(root,output)
    text = output.read_text()
    assert '"/example/python"' in text
    assert 'gb300_experiment_control_v2.py" "--root"' in text
    assert '"' + str(root) + '" "supervise"' in text
    assert 'KillMode=mixed' in text
    assert result['installed_or_started'] is False
    with pytest.raises(FileExistsError):
        gb300.write_service(root,output)
    with pytest.raises(ValueError,match='filename'):
        gb300.write_service(root,tmp_path/'different.service')


def test_http_request_uses_only_the_masked_packet_and_remaining_timeout(monkeypatch):
    observed = []
    def response(request, timeout):
        observed.append({'body':json.loads(request.data), 'timeout':timeout, 'url':request.full_url})
        answer = ({'decision':'COPY_SPAN','block_id':'b0','quote':'Exampleland'} if len(observed) == 1
                  else {'assessment':'SUPPORTED','explanation':'Explicit country of the collecting site.'})
        return io.BytesIO(json.dumps({'model': 'test-model', 'choices': [{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(answer)}}]}).encode())
    monkeypatch.setattr(gb10, 'urlopen', response)
    payload = runner.packet('country', {'id':'private-id','reference':'hidden-reference','context':{'country':'hidden-reference','description':'An artificial site in Exampleland'}})
    value = gb10.requester('http://127.0.0.1:8000/v1', 'test-model', max_tokens=123)('country', payload, 8)
    assert value['value'] == 'Exampleland'
    assert len(observed) == 2
    assert 0 < observed[1]['timeout'] <= observed[0]['timeout'] <= 8
    assert observed[0]['body']['max_tokens'] == 123
    p, _, reviewer, _ = gb10.original_modules()
    assert observed[0]['body']['messages'][0]['content'] == p.PROMPTS['esites']
    assert observed[1]['body']['messages'][0]['content'] == reviewer.REVIEW_PROMPT
    assert 'hidden-reference' not in json.dumps(observed)
    assert 'private-id' not in json.dumps(observed)


@pytest.mark.parametrize('finish,model', [('length','test-model'),('stop','different-model')])
def test_truncated_or_wrong_model_responses_cannot_be_scored(monkeypatch, finish, model):
    response = {'model':model,'choices':[{'finish_reason':finish,'message':{'role':'assistant','content':'{"decision":"UNRESOLVED","block_id":"","quote":""}'}}]}
    monkeypatch.setattr(gb10, 'urlopen', lambda *a, **k: io.BytesIO(json.dumps(response).encode()))
    with pytest.raises(ValueError):
        gb10.requester('http://127.0.0.1:8000/v1', 'test-model')('country', {'context':{'locality':'An artificial site'}}, 8)
