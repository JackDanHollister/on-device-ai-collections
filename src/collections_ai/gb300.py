"""Portable preparation helpers for the preserved GB300 scripts."""
from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path
import shutil
import sysconfig


def script_directory() -> Path:
    checkout = Path(__file__).resolve().parents[2] / 'scripts/gb300'
    installed = Path(sysconfig.get_path('data')) / 'share/on-device-ai-collections/scripts/gb300'
    for root in [checkout, installed]:
        if (root / 'gb300_experiment_control.py').is_file():
            return root
    raise FileNotFoundError('The preserved GB300 scripts are missing from this installation.')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage_runtime(output: Path) -> dict:
    """Recreate the layout hashed by the original supervisor, without run data."""
    source = Path(str(files('placebot')))
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    selected = list(source.rglob('*.py')) + [source / 'data/prompt.md', source / 'LICENSE']
    manifest = {}
    for file in selected:
        relative = Path('placebot') / file.relative_to(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
        manifest[str(relative)] = digest(target)
    target = output / 'research/scripts/gb300_experiment_control.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(script_directory() / target.name, target)
    manifest['research/scripts/' + target.name] = digest(target)
    summary = {'schema': 'collections-gb300-runtime-v1', 'files': manifest,
               'museum_records_included': False, 'model_weights_included': False}
    (output / 'runtime-source.json').write_text(json.dumps(summary, indent=2) + '\n')
    return {'runtime': str(output), 'source_files': len(manifest), 'model_calls': 0}


def write_service(root: Path, output: Path) -> dict:
    """Write the user service expected by the retained GB300 controller."""
    root = root.resolve()
    config = json.loads((root / 'experiment.json').read_text())
    if output.name != config['unit']:
        raise ValueError('The service filename must match the unit selected during init.')
    def quote(value):
        value = str(value)
        if '\n' in value or '\r' in value:
            raise ValueError('Service paths cannot contain line breaks.')
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%').replace('$', '$$') + '"'
    python = config['runner_command'][0]
    command = ' '.join(map(quote, [python, script_directory() / 'gb300_experiment_control_v2.py',
                                    '--root', root, 'supervise']))
    output.parent.mkdir(parents=True, exist_ok=True)
    content = ('[Unit]\nDescription=Collections locality experiment on GB300\n\n'
        '[Service]\nType=simple\nExecStart=' + command + '\n'
        'Restart=no\nTimeoutStopSec=1800\nKillMode=mixed\nUMask=0077\n'
        'Environment=PYTHONUNBUFFERED=1\n\n[Install]\nWantedBy=default.target\n')
    with output.open('x', encoding='utf8') as stream:
        stream.write(content)
    return {'service': str(output), 'root': str(root), 'installed_or_started': False}


def prepare_source(source: Path, output: Path) -> dict:
    """Adapt local tabular data to the original blind-union preparation input."""
    import duckdb

    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('Use a separate, fresh output directory.')
    required = {'gbifid', 'locality', 'countrycode', 'stateprovince',
                'decimallatitude', 'decimallongitude'}
    # Validate before creating output, so malformed input cannot look complete.
    with duckdb.connect() as db:
        if source.suffix.lower() == '.parquet':
            relation = db.read_parquet(str(source))
        elif source.suffix.lower() in {'.csv', '.tsv'}:
            relation = db.read_csv(str(source), header=True, all_varchar=True)
        else:
            raise ValueError('Use CSV, TSV or Parquet input.')
        if not required <= set(relation.columns):
            raise ValueError('Required columns: ' + ', '.join(sorted(required)))
        relation.create_view('supplied')
        count, unique, blank = db.execute("SELECT count(*),count(DISTINCT gbifid),count(*) FILTER (WHERE gbifid IS NULL OR trim(cast(gbifid AS VARCHAR))='') FROM supplied").fetchone()
        if not count or blank or count != unique:
            raise ValueError('Supply non-empty data with unique, non-empty gbifid values.')
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        target = output / 'source.parquet'
        relation.write_parquet(str(target), compression='zstd')
    summary = {'status': 'complete', 'schema': 'collections-gb300-source-v1',
               'input_sha256': digest(source), 'source_identity_sha256': digest(source),
               'compacted_source_sha256': digest(target), 'input_records': count,
               'preparation': 'Local tabular adapter for a new experiment; no model calls.'}
    (output / 'COMPLETE.json').write_text(json.dumps(summary, indent=2) + '\n')
    return {'input_records': count, 'source': str(target), 'model_calls': 0}
