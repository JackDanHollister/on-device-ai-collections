# Setup and run

## Run the three GB10 experiments

Install the package as shown in the README. The model service must already be
running on the GB10 systems and expose an OpenAI-compatible chat endpoint. Model
weights and GPU-serving software are obtained separately from NVIDIA. The
[model page](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
provides model and deployment information.

The three entry points are:

```sh
python scripts/gb10/run_country.py --input data/private/country.jsonl --output runs/country
python scripts/gb10/run_bibliography.py --input data/private/bibliography.jsonl --output runs/bibliography
python scripts/gb10/run_locality.py --input data/private/locality.jsonl --output runs/locality
```

Each defaults to 24 hours. There is no record-count cap. It processes each queued
record once and stops when the input runs out, the time ends, you interrupt it,
or five consecutive records fail. A blocked request is also interrupted at the
time boundary on Linux and macOS. Run these tasks separately when they share a
model service. Starting another command creates a separate experiment.

The default endpoint is `http://127.0.0.1:8000/v1` and the served model ID is
`nemotron_3_super_cluster`. Set `--endpoint` and `--model` to match your server.
Optional authentication reads `LOCAL_MODEL_API_KEY` from the environment.
`--hours` changes the duration; `--plan-only` checks the input and queue without
calling the model.

### Input files

Use JSONL: one JSON object per line, with a unique string `id`, a `context` object,
and a `reference`. Use `null` for a genuinely missing target. Known references
are withheld checks: strings for countries and titles, or `[latitude, longitude]`
for coordinates. Supply one check for each nine missing-value records. The
planner keeps every missing-value record, including a final incomplete block;
if there are too few checks, it reports the problem instead of dropping blanks.
No source file is edited.

Export artificial examples to see the formats:

```sh
collections-ai example --task country --output runs/country-input.jsonl
collections-ai example --task bibliography --output runs/bibliography-input.jsonl
collections-ai example --task locality --output runs/locality-input.jsonl
python scripts/gb10/run_country.py --input runs/country-input.jsonl --output runs/input-check --plan-only
```

| Task | Context fields | What the code does |
|---|---|---|
| Country | `locpreciselocation` and/or `locnhmlabellocality`; `locality` and `description` are also accepted | Uses the original extraction prompt, checks the exact source span, then runs the original country review prompt. Only a supported proposal is returned. |
| Bibliography | `target` metadata and retrieved peer records in `candidates` | Uses the original publication-matching prompt, identity checks and a fresh model review. Copies a title only from a selected peer; keeps strong proposals, review candidates and links distinct. |
| Locality | `locality`, with optional `country`, `countrycode` and `stateprovince` | Uses the included PlaceBot instructions, Nemotron fast-mode request and complete coordinate-output checks. |

Bibliography metadata includes `record_type`, `author_names`, `publication_date`,
`container_title`, `volume`, `issue` and `pages`. Candidates can carry a
`candidate_id`; otherwise the code assigns one. The artificial example includes
a complete matching pair. The retained `ebib_peer_overnight.py` also contains the
original broad peer-retrieval keys and museum-export preparation code.

The model never receives the separate `reference` field. Target country/title or
coordinate columns are removed from the request; bibliography peer titles remain
available as evidence. Do not duplicate a held-out answer in unrelated context
fields. The command supports interleaved checks in its input file.

Country extraction and bibliography matching use reasoning. Their default output
settings are 16,384 and 8,192 tokens respectively. Locality uses fast mode with
reasoning disabled and a 1,024-token output setting. These are the retained task
settings, rather than record limits. The server context must fit both the prompt
and requested output. `--max-tokens` provides an explicit override.

### Saved results

Each output folder contains `records.jsonl` with answers, source-span or peer
checks, review decisions and available token usage, plus an aggregate
`summary.json` updated after each completed record. The summary records the
configured window, actual elapsed time, processed counts, masked-check matches
and coordinate distances. A partial run reports its actual check fraction.
The input and prompt hashes identify what was used.

Keep individual records and responses private. The historical museum exports,
responses and references are not included in this repository. The dated harnesses
and service templates preserve the earlier audit logic; their deployment paths
have been made generic. The current entry points reuse the task functions without
requiring the old private folders, fixed dates or earlier run receipts.

An offline demonstration is available with `collections-ai demo`. It uses
artificial records and simulated responses, with performance metrics disabled.

## Prepare and run the GB300 workflow

The preserved workflow needs Linux, Docker, a running NVIDIA model service and
local model weights. The service must expose the model ID
`placebot-nemotron-super-fast` at `http://127.0.0.1:8000/v1/chat/completions`.
The runner uses 100 parallel requests, a 16,384-token context setting and
1,024-token fast-mode outputs. It has no record-count cap.

The saved GB300 run processed 4,183,811 distinct inputs representing 9,122,868
specimens across six processing windows. Its average throughput was 22,493.562
requests/hour over 669,601.353 seconds of processing-window time, including
startup and recovery within those windows and excluding pauses between them.
The aggregate receipt is in `results/gb300-summary.json`.

First stage the original runtime and prepare the input. The supplied CSV is
artificial; replace it with your own source for an experiment.

```sh
collections-ai gb300-runtime --output runs/gb300-runtime
collections-ai gb300-source --input examples/gb300_records.csv --output runs/gb300-source
python scripts/gb300/prepare_gb300_combined_experiment.py --source-root runs/gb300-source --output runs/gb300-prepared
```

CSV/TSV/Parquet inputs need `gbifid`, `locality`, `countrycode`, `stateprovince`,
`decimallatitude` and `decimallongitude` columns. Identifiers must be unique.
The adapter records a source hash. The original preparation code then creates
shared model inputs, the specimen mapping, counts and withheld reference files.
The model-input file contains no reference coordinate columns. This preparation
runs locally without model calls.

For the original large GBIF source route, `stage_gb300_gbif.py`, its downloader
and eligibility audit are included. The source citation is in `docs/gbif-source-citation.txt`.

Set the container name and model directory to your Linux installation, then
initialize the experiment:

```sh
COLLECTIONS_CONTAINER=your-running-model-container
COLLECTIONS_MODEL_DIRECTORY=/path/to/model-weights
python scripts/gb300/gb300_experiment_control_v2.py --root runs/gb300 init --preparation runs/gb300-prepared --runtime runs/gb300-runtime --python "$(pwd)/.venv/bin/python" --container "$COLLECTIONS_CONTAINER" --model-directory "$COLLECTIONS_MODEL_DIRECTORY"
collections-ai gb300-service --root runs/gb300 --output ~/.config/systemd/user/placebot-combined-experiment.service
systemctl --user daemon-reload
```

The service writer fills in the local paths; it does not start a process or
replace an existing service file. Initialization records and checks the source,
runtime, model assets and container configuration. The controller then starts
and stops the experiment through that service:

```sh
python scripts/gb300/gb300_experiment_control_v2.py --root runs/gb300 start --hours 24
python scripts/gb300/gb300_experiment_control_v2.py --root runs/gb300 status
python scripts/gb300/gb300_experiment_control_v2.py --root runs/gb300 pause
```

Use `start --continuous` instead of `start --hours 24` to process the full dataset
without a time window. Later starts resume the same append-only ledger. Window
boundaries do not create a new cohort or repeat completed inputs. If the full
input finishes, the supervisor stops. The service should run under a Linux user
session that remains active for the experiment.

`monitor_gb300_experiment.py` records resource telemetry;
`report_gb300_experiment_window.py` generates window and cumulative reports.
Their command-line help lists required paths. Runtime folders contain private
records, responses and machine details and stay out of Git.

## Checks and data availability

```sh
python -m pip install ".[test]"
python -m pytest -q
python analysis/verify_archive.py
```

Tests use artificial records and simulated services/processes. They check source
spans, candidate selection, withheld references, coordinate validity,
deduplication, checkpoints, pause/restart behaviour and report arithmetic. They
use artificial data rather than measuring model accuracy.

Museum exports, protected reference files, per-record prompts/responses and the
original large database are not distributed. The repository contains code,
artificial examples and tests. GBIF source
data should be obtained through its providers and cited using the parent data
DOI and applicable source licences. The source-selection code records its filters
and source identity.

## References

Dell Technologies (n.d.-a) *Dell Pro Max with GB10*. Product brochure and specification sheet. https://www.delltechnologies.com/asset/en-gb/products/workstations/technical-support/dell-pro-max-with-gb10-workstation-brochure-and-spec-sheet.pdf.external (accessed 30 October 2026).

Dell Technologies (n.d.-b) *Dell Pro Max Performance PCs and Precision Fixed Workstation Family*. GB300 specifications. https://www.delltechnologies.com/asset/en-us/products/workstations/briefs-summaries/dell-pro-max-family-brochure.pdf (accessed 30 October 2026).

GBIF.org (2026) GBIF Occurrence Download, 1 July 2026. https://doi.org/10.15468/dl.teccpa

Hollister JD, Price BW (2026) PlaceBot: A Python tool for AI-assisted georeferencing of museum specimen locality data, with cross-vendor model evaluation. *Natural History Collections and Museomics* 3: e203521. https://doi.org/10.3897/nhcm.3.203521

NVIDIA (2026) Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning. arXiv preprint. https://doi.org/10.48550/arXiv.2604.12374

Skende A, Rosseel G, Pinckney N (2026) 17.1 NVIDIA GB10: SoC Built for AI Acceleration. In: *2026 IEEE International Solid-State Circuits Conference (ISSCC)*. IEEE, 298–300. https://doi.org/10.1109/ISSCC49663.2026.11409351

## Licences and credits

The experiment code and original documentation are MIT licensed. The included
PlaceBot source retains its existing MIT notice at `src/placebot/LICENSE`.
Jack Dan Hollister's hardware photograph is CC BY 4.0. Museum source data and model
weights are excluded and their rights are not changed by the code licence.
