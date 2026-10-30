"""Commands for the collections-data experiments."""
from __future__ import annotations

import argparse
from importlib.resources import files
import json
from pathlib import Path
import sys

from . import __version__, runner


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'run':
        from .gb10 import main as run_gb10
        run_gb10(argv=argv[1:])
        return
    parser = argparse.ArgumentParser(description='Local AI for collections data')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Try all three tasks offline with artificial records')
    demo.add_argument('--output', type=Path, default=Path('runs/demo'))
    example = commands.add_parser('example', help='Save an artificial JSONL input template')
    example.add_argument('--task', required=True, choices=runner.TASKS)
    example.add_argument('--output', type=Path, required=True)
    commands.add_parser('run', help='Run a new benchmark; use run --help for settings')
    stage = commands.add_parser('gb300-runtime', help='Stage the preserved runtime for the GB300 supervisor')
    stage.add_argument('--output', type=Path, required=True)
    prepare = commands.add_parser('gb300-source', help='Prepare a local CSV/TSV/Parquet file for the GB300 workflow')
    prepare.add_argument('--input', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    service = commands.add_parser('gb300-service', help='Write the user service for an initialized GB300 experiment')
    service.add_argument('--root', type=Path, required=True)
    service.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'demo':
            from .demo import create_demo
            print(create_demo(args.output))
        elif args.command == 'example':
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('xb') as out:
                out.write(files('collections_ai').joinpath('data', args.task + '.jsonl').read_bytes())
            print(args.output)
        elif args.command == 'gb300-runtime':
            from .gb300 import stage_runtime
            print(json.dumps(stage_runtime(args.output), indent=2))
        elif args.command == 'gb300-source':
            from .gb300 import prepare_source
            print(json.dumps(prepare_source(args.input, args.output), indent=2))
        elif args.command == 'gb300-service':
            from .gb300 import write_service
            print(json.dumps(write_service(args.root, args.output), indent=2))
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, str(exc) + '\n')
