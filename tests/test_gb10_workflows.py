"""Check the public entry points against the retained task contracts."""
import io
import json
from pathlib import Path
import signal
import time

import pytest

from collections_ai import cli, gb10, runner


def reply(answer, model='test-model'):
    return io.BytesIO(json.dumps({'model':model, 'choices':[{'finish_reason':'stop',
        'message':{'role':'assistant','content':json.dumps(answer)}}],
        'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}}).encode())


def test_no_5000_record_cap_and_no_incomplete_tail_discarded():
    rows = [{'id':str(i), 'context':{}, 'reference':None if i % 10 < 9 else 'known'} for i in range(7010)]
    planned = runner.plan_records(rows, seed=1)
    assert len(planned) == 7010
    assert len({r['id'] for r in planned}) == 7010
    rows.append({'id':'last-blank','context':{},'reference':None})
    rows.append({'id':'extra-check','context':{},'reference':'known'})
    assert len(runner.plan_records(rows, seed=1)) == 7012


def test_short_check_pool_is_an_error_instead_of_a_silent_limit():
    rows = [{'id':str(i),'context':{},'reference':None} for i in range(18)]
    rows.append({'id':'check','context':{},'reference':'known'})
    with pytest.raises(ValueError, match='no blank records were discarded'):
        runner.plan_records(rows, seed=1)


@pytest.mark.parametrize('assessment', ['AMBIGUOUS','NOT_SUPPORTED'])
def test_country_literal_span_still_needs_supported_review(monkeypatch, assessment):
    answers = iter([{'decision':'COPY_SPAN','block_id':'b0','quote':'France'},
                    {'assessment':assessment,'explanation':'Country attribution is uncertain.'}])
    monkeypatch.setattr(gb10, 'urlopen', lambda *a, **k: reply(next(answers)))
    result = gb10.requester('http://127.0.0.1:8000/v1','test-model')(
        'country', {'context':{'locality':'France, or possibly another collecting site.'}}, 10)
    assert result['value'] is None
    assert result['usage']['prompt_tokens'] == 20
    assert result['details']['review']['assessment'] == assessment


def test_country_cannot_copy_a_country_absent_from_source(monkeypatch):
    monkeypatch.setattr(gb10,'urlopen',lambda *a,**k:reply({'decision':'COPY_SPAN','block_id':'b0','quote':'France'}))
    p, _, _, _ = gb10.original_modules()
    with pytest.raises(p.PilotError, match='QUOTE_NOT_UNIQUE_LITERAL_SPAN'):
        gb10.requester('http://127.0.0.1:8000/v1','test-model')(
            'country', {'context':{'locality':'An unnamed island'}}, 10)


def bib_context(pages='10-20'):
    target = {'record_type':'Article', 'author_names':['A. Example'], 'publication_date':'2000',
              'container_title':'Invented Journal','volume':'1','pages':pages,'title':'HIDDEN_REFERENCE'}
    donor = dict(target, candidate_id='c1', title='An invented paper')
    return {'target':target, 'candidates':[donor]}


def test_bibliography_reuses_peer_evidence_and_fresh_review(monkeypatch):
    observed = []
    def respond(request, **kwargs):
        observed.append(json.loads(request.data))
        return reply({'decision':'SAME_PUBLICATION','candidate_id':'c1','explanation':'Metadata agrees.'})
    monkeypatch.setattr(gb10, 'urlopen', respond)
    payload = runner.packet('bibliography', {'context':bib_context()})
    result = gb10.requester('http://127.0.0.1:8000/v1','test-model')('bibliography',payload,10)
    assert result['value'] == 'An invented paper'
    assert result['details']['outcome'] == 'TITLE_AND_LINK_PROPOSAL'
    assert len(observed) == 2
    assert 'HIDDEN_REFERENCE' not in json.dumps(observed)
    _, _, _, original = gb10.original_modules()
    assert observed[0]['messages'][0]['content'] == original.PROMPT
    assert 'fresh assessment' in observed[1]['messages'][0]['content']


def test_bibliography_review_disagreement_withholds_title(monkeypatch):
    answers = iter([{'decision':'SAME_PUBLICATION','candidate_id':'c1','explanation':'Possible peer.'},
                    {'decision':'REVIEW','candidate_id':'','explanation':'Insufficient evidence.'}])
    monkeypatch.setattr(gb10,'urlopen',lambda *a,**k:reply(next(answers)))
    result = gb10.requester('http://127.0.0.1:8000/v1','test-model')(
        'bibliography',runner.packet('bibliography',{'context':bib_context()}),10)
    assert result['value'] is None


def georeference():
    return dict(country='Exampleland',state='',region='',sector='',exact_site='Artificial site',
        latitude=12.0,longitude=34.0,coordinate_source='model estimate',coordinate_radius_meters=None,
        elevation_meters=None,elevation_original='',confidence='low',notes='',collection_notes='')


def test_locality_uses_complete_placebot_contract_without_reference_coordinates(monkeypatch):
    observed = {}
    def respond(request, **kwargs):
        observed.update(json.loads(request.data))
        return reply(georeference())
    monkeypatch.setattr(gb10, 'urlopen', respond)
    payload = runner.packet('locality', {'reference':[85.123456789,179.123456789], 'context':{
        'locality':'Artificial site','country':'Exampleland','latitude':85.123456789,'longitude':179.123456789}})
    result = gb10.requester('http://127.0.0.1:8000/v1','test-model')('locality',payload,10)
    assert result['value'] == [12.0,34.0]
    assert observed['chat_template_kwargs']['enable_thinking'] is False
    assert observed['max_tokens'] == 1024
    assert '85.123456789' not in observed['messages'][0]['content']
    assert '179.123456789' not in observed['messages'][0]['content']
    assert 'Locality: "Artificial site"' in observed['messages'][0]['content']
    assert observed['response_format']['json_schema']['name'] == 'placebot_georeference'


@pytest.mark.parametrize('task', runner.TASKS)
def test_three_commands_default_to_24_hours_and_plan_without_network(tmp_path, monkeypatch, capsys, task):
    def forbidden(*args, **kwargs):
        pytest.fail('Planning must not contact a model')
    monkeypatch.setattr(gb10, 'urlopen', forbidden)
    source = tmp_path / 'input.jsonl'
    cli.main(['example','--task',task,'--output',str(source)])
    capsys.readouterr()
    gb10.main(task, ['--input',str(source),'--output',str(tmp_path/'run'),'--plan-only'])
    result = json.loads(capsys.readouterr().out)
    assert result['configured_max_hours'] == 24
    assert result['record_limit'] is None
    assert result['planned_records'] == 10
    assert result['held_out_records'] == 1
    assert not (tmp_path/'run').exists()


@pytest.mark.skipif(not hasattr(signal,'setitimer'), reason='Linux/macOS wall-time interrupt')
def test_deadline_interrupts_a_blocked_request_and_keeps_receipts(tmp_path):
    planned = [{'id':'one','context':{},'cohort':'blank','reference':None}]
    def blocked(*args):
        time.sleep(10)
    started = time.monotonic()
    result = runner.run('country', planned,tmp_path/'run',hours=.1/3600,
        infer=blocked, model='synthetic',input_sha256='synthetic',seed=1)
    assert time.monotonic()-started < 3
    assert result['stop_reason'] == 'time_limit'
    assert result['counts']['failed'] == 1
    assert (tmp_path/'run/records.jsonl').read_text()
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0,0.0)
