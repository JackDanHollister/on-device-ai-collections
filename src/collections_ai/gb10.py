"""GB10 entry points using the retained country, bibliography and PlaceBot logic."""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib.resources import files
import json
import math
import os
from pathlib import Path
import sys
import sysconfig
import time
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator

from . import runner


def original_modules():
    checkout = Path(__file__).resolve().parents[2] / 'scripts/gb10'
    installed = Path(sysconfig.get_path('data')) / 'share/on-device-ai-collections/scripts/gb10'
    for directory in (checkout, installed):
        if (directory / 'ebib_peer_overnight.py').is_file():
            sys.path.insert(0, str(directory))
            try:
                return tuple(importlib.import_module(name) for name in
                    ('local_span_pilot', 'run_local_span_pilots',
                     'run_esites_review_comparison', 'ebib_peer_overnight'))
            finally:
                sys.path.pop(0)
    raise FileNotFoundError('The original GB10 task modules are missing from this installation.')


def country_packet(context: dict) -> dict:
    p, _, _, _ = original_modules()
    fields = ('locpreciselocation', 'locnhmlabellocality', 'locality',
              'verbatimLocality', 'description', 'notes')
    blocks = [(key, context[key]) for key in fields if isinstance(context.get(key), str)]
    extra = {key: context[key] for key in ('loccontinent', 'locprovincestateterritory',
        'locdistrictcountyshire', 'locislandname', 'locislandgrouping', 'lococean', 'locseagulf')
        if isinstance(context.get(key), str)}
    result = p.packet('esites', 'loccountry', blocks, extra)
    if not result['evidence']:
        raise ValueError('Country records need locality text in locpreciselocation or locality.')
    return result


BIB_FIELDS = ('title', 'publication_date', 'distribution_date', 'volume', 'issue', 'pages',
    'parent_reference', 'container_title', 'container_issn', 'et_al', 'edition',
    'publisher_reference', 'institution', 'language', 'publication_city', 'series',
    'fascicle', 'chapter', 'issn', 'isbn')


def bibliography_packet(context: dict) -> dict:
    def record(raw):
        if not isinstance(raw, dict):
            raise ValueError('Bibliography targets and candidates must be objects.')
        result = {key: raw.get(key, '') for key in BIB_FIELDS}
        result['record_type'] = raw.get('record_type', 'Article')
        result['author_names'] = raw.get('author_names', [])
        result['author_refs'] = raw.get('author_refs', [])
        if any(not isinstance(result[k], str) for k in (*BIB_FIELDS, 'record_type')):
            raise ValueError('Bibliography metadata fields must be strings.')
        if not isinstance(result['author_names'], list) or any(not isinstance(v, str) for v in result['author_names']):
            raise ValueError('author_names must be a list of names.')
        if not isinstance(result['author_refs'], list) or any(not isinstance(v, dict) or
                any(not isinstance(v.get(k), str) for k in ('reference', 'position', 'role'))
                for v in result['author_refs']):
            raise ValueError('author_refs needs reference, position and role strings.')
        return result
    target = record(context.get('target', context))
    target['title'] = ''
    supplied = context.get('candidates')
    if not isinstance(supplied, list) or not supplied:
        raise ValueError('Supply retrieved bibliography peer records in candidates.')
    candidates = []
    for index, raw in enumerate(supplied, 1):
        candidate = record(raw)
        candidate['candidate_id'] = raw.get('candidate_id', f'c{index}')
        if not isinstance(candidate['candidate_id'], str) or not 1 <= len(candidate['candidate_id']) <= 32:
            raise ValueError('Candidate identifiers need 1–32 characters.')
        candidates.append(candidate)
    if len({c['candidate_id'] for c in candidates}) != len(candidates):
        raise ValueError('Bibliography candidate identifiers must be unique.')
    return {'target': target, 'candidates': candidates}


def locality_prompt(context: dict) -> str:
    # Use the included PlaceBot instructions and its original prompt formatter.
    from placebot.core.ai_processor import AIProcessor
    locality = context.get('locality') or context.get('locpreciselocation') or context.get('description')
    if not isinstance(locality, str) or not locality.strip():
        raise ValueError('Locality records need a non-empty locality string.')
    country = context.get('country') or context.get('countrycode') or context.get('loccountry') or ''
    if not isinstance(country, str):
        raise ValueError('Country context must be a string.')
    extra = {k: context[k] for k in ('stateprovince', 'state', 'province') if context.get(k)}
    if extra:
        locality += '\nAdditional source context (JSON):\n' + json.dumps(extra, ensure_ascii=False)
    processor = AIProcessor.__new__(AIProcessor)
    processor._prompt_cache = None
    return processor._build_prompt(locality, country,
        '\n\nNO EXISTING COORDINATES: Please estimate coordinates from locality if possible')


def final_answer(raw: dict, model: str) -> dict:
    if raw.get('model') != model or len(raw.get('choices', [])) != 1:
        raise ValueError('The server returned an unexpected model or number of choices.')
    choice = raw['choices'][0]
    message = choice.get('message', {})
    if choice.get('finish_reason') != 'stop' or message.get('role') != 'assistant' or message.get('tool_calls'):
        raise ValueError('The model response did not finish normally.')
    p, _, _, _ = original_modules()
    return p.strict_json(message.get('content'))


def requester(endpoint: str, model: str, *, max_tokens: int | None = None):
    p, base, reviewer, bibliography = original_modules()
    from placebot.models import nemotron_3_super_local as placebot

    def infer(task: str, payload: dict, remaining: float) -> dict:
        deadline = time.monotonic() + remaining
        usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        usage_complete = True
        calls = []

        def call(body):
            nonlocal usage_complete
            body['model'] = model
            if max_tokens is not None:
                body['max_tokens'] = max_tokens
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError('The processing window has ended.')
            headers = {'Content-Type': 'application/json'}
            if os.environ.get('LOCAL_MODEL_API_KEY'):
                headers['Authorization'] = 'Bearer ' + os.environ['LOCAL_MODEL_API_KEY']
            request = Request(endpoint.rstrip('/') + '/chat/completions',
                data=json.dumps(body).encode(), headers=headers)
            with urlopen(request, timeout=min(900.0, left)) as response:
                raw = json.load(response)
            answer = final_answer(raw, model)
            tokens = raw.get('usage') or {}
            for key in usage:
                if type(tokens.get(key)) is int and tokens[key] >= 0:
                    usage[key] += tokens[key]
                else:
                    usage_complete = False
            calls.append({'answer': answer, 'usage': tokens})
            return raw, answer

        context = payload['context']
        value, details = None, {}
        if task == 'country':
            packet = country_packet(context)
            case = {'packet': packet, 'expected': None}
            _, answer = call(base.request(case, p.PROMPTS['esites'], True))
            proposal = p.validate(packet, answer)
            details = {'extraction': answer, 'proposal': proposal}
            if proposal is not None:
                _, assessment = call(reviewer.review_body(packet, proposal, []))
                Draft202012Validator(reviewer.REVIEW_SCHEMA).validate(assessment)
                if not assessment['explanation'].strip():
                    raise ValueError('Country review needs a non-empty explanation.')
                details['review'] = assessment
                if assessment['assessment'] == 'SUPPORTED':
                    value = proposal['new_value']
        elif task == 'bibliography':
            packet = bibliography_packet(context)
            case = {'packet': packet}
            _, answer = call(bibliography.request(case))
            Draft202012Validator(bibliography.SCHEMA).validate(answer)
            if not answer['explanation'].strip():
                raise ValueError('Bibliography selection needs a non-empty explanation.')
            candidate, checks = bibliography.selection(packet, answer)
            details = {'selection': answer, 'evidence_checks': checks, 'outcome': 'UNRESOLVED'}
            if candidate is not None and checks.get('minimum_evidence_met'):
                _, assessment = call(bibliography.request(case, True))
                Draft202012Validator(bibliography.SCHEMA).validate(assessment)
                if not assessment['explanation'].strip():
                    raise ValueError('Bibliography review needs a non-empty explanation.')
                bibliography.selection(packet, assessment)
                details['review'] = assessment
                accepted = (assessment['decision'] in ('SAME_PUBLICATION', 'POSSIBLE_MATCH')
                            and assessment['candidate_id'] == answer['candidate_id'])
                strong = bool(checks.get('title_transfer_eligible') and
                              assessment['decision'] == 'SAME_PUBLICATION')
                if accepted:
                    details['outcome'] = 'LINK_ONLY_PROPOSAL'
                    if candidate['title'] and (strong or checks.get('review_candidate_eligible')):
                        value = candidate['title']
                        details['outcome'] = 'TITLE_AND_LINK_PROPOSAL' if strong else 'TITLE_REVIEW_CANDIDATE'
        else:
            body = placebot.format_request(locality_prompt(context), reasoning_mode='fast')
            raw, _ = call(body)
            record = json.loads(placebot.parse_response(raw))
            value = [record['latitude'], record['longitude']]
            details = {'georeference': record}
        details['calls'] = calls
        return {'value': value, 'details': details, 'usage': usage if usage_complete else {}}
    return infer


def main(task: str | None = None, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Run a GB10 collections workflow for 24 hours, with no record cap.')
    if task is None:
        parser.add_argument('--task', required=True, choices=runner.TASKS)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', default='nemotron_3_super_cluster')
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-tokens', type=int, help='Override the task output-token setting.')
    parser.add_argument('--plan-only', action='store_true', help='Check input and queue without contacting a model.')
    args = parser.parse_args(argv)
    task = task or args.task
    try:
        if not math.isfinite(args.hours) or args.hours <= 0:
            raise ValueError('Hours must be a positive finite number.')
        if args.max_tokens is not None and args.max_tokens <= 0:
            raise ValueError('The output-token setting must be positive.')
        rows, fingerprint = runner.load_records(args.input, task)
        planned = runner.plan_records(rows, seed=args.seed)
        for row in planned:
            context = runner.packet(task, row)['context']
            {'country': country_packet, 'bibliography': bibliography_packet,
             'locality': locality_prompt}[task](context)
        held_out = sum(row['cohort'] == 'held_out' for row in planned)
        if args.plan_only:
            print(json.dumps({'task': task, 'planned_records': len(planned),
                'held_out_records': held_out, 'record_limit': None,
                'configured_max_hours': args.hours, 'model_calls': 0}, indent=2))
            return
        p, _, reviewer, bibliography = original_modules()
        prompt = p.PROMPTS['esites'] + reviewer.REVIEW_PROMPT if task == 'country' else (
            bibliography.PROMPT if task == 'bibliography' else files('placebot').joinpath('data/prompt.md').read_text())
        result = runner.run(task, planned, args.output, hours=args.hours,
            infer=requester(args.endpoint, args.model, max_tokens=args.max_tokens),
            model=args.model, input_sha256=fingerprint, seed=args.seed,
            configuration={'workflow': 'gb10-' + task, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'thinking': task != 'locality',
                'max_output_tokens': args.max_tokens or {'country': 16384, 'bibliography': 8192, 'locality': 1024}[task]})
        print(json.dumps(result, indent=2))
        if result['stop_reason'] in {'consecutive_errors', 'interrupted'}:
            parser.exit(1)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    main()
