"""NVIDIA Nemotron 3 Super served by a local OpenAI-compatible vLLM API.

The profile defaults to a vLLM server on this machine.  A remote GB10 service
can be selected without putting a site-specific address in the repository:

    PLACEBOT_NEMOTRON_BASE_URL=http://gb10-host:8000

``PLACEBOT_NEMOTRON_API_ENDPOINT`` may be used instead when the full chat
completions URL is already known.  The model id and optional bearer token are
configurable in the same way.
"""

import json
import math
import os
from typing import Dict, List, Optional


def _env_bool(name: str, default: bool) -> bool:
    """Read a conventional boolean environment override."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_nonnegative_int(name: str, default: int) -> int:
    """Read a non-negative integer environment override."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _chat_completions_endpoint(base_url: str) -> str:
    """Return a chat-completions endpoint from a host, API base, or endpoint."""
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


MODEL_NAME = "NVIDIA Nemotron 3 Super (Local vLLM)"
MODEL_PROVIDER = "NVIDIA (via local vLLM)"
MODEL_ID = os.environ.get("PLACEBOT_NEMOTRON_MODEL_ID", "nemotron_3_super_cluster")

_base_url = os.environ.get("PLACEBOT_NEMOTRON_BASE_URL", "http://localhost:8000")
API_ENDPOINT = os.environ.get(
    "PLACEBOT_NEMOTRON_API_ENDPOINT",
    _chat_completions_endpoint(_base_url),
).strip()
API_VERSION = "v1"

# Explicit metadata lets PlaceBot distinguish this keyless local vLLM profile
# from Ollama profiles, which have their own model-discovery/status probe.
MODEL_TYPE = "local"
MODEL_BACKEND = "vllm"
LOCAL_READY = True
LOCAL_STATUS = "Configured"
LOCAL_STATUS_DETAIL = API_ENDPOINT

# vLLM can be configured with an API token, but the GB10 service is keyless.
API_KEY = os.environ.get("PLACEBOT_NEMOTRON_API_KEY", "")

# Local inference has no per-token API charge.
COST_PER_1K_INPUT_TOKENS = 0.0
COST_PER_1K_OUTPUT_TOKENS = 0.0
ESTIMATED_COST_PER_RECORD = 0.0

# The proven GB10 service used an 8,192-token context.  Accuracy experiments
# can raise the served context and profile metadata together via this override.
MAX_TOKENS = int(os.environ.get("PLACEBOT_NEMOTRON_CONTEXT_WINDOW", "8192"))
MAX_OUTPUT_TOKENS = int(
    os.environ.get("PLACEBOT_NEMOTRON_MAX_OUTPUT_TOKENS", "1024")
)
CONTEXT_WINDOW = MAX_TOKENS
REQUESTS_PER_MINUTE = 999999
REQUEST_TIMEOUT = int(os.environ.get("PLACEBOT_NEMOTRON_REQUEST_TIMEOUT", "300"))
# A full-reasoning request may consume hours. Never silently repeat it unless
# an operator explicitly opts into retries for a particular run.
MAX_RETRIES = _env_nonnegative_int("PLACEBOT_NEMOTRON_MAX_RETRIES", 0)

SPEED = "GB10 cluster dependent"
ACCURACY = "Very high (120B-A12B Nemotron 3 Super)"
COST_EFFICIENCY = "Perfect (local inference)"
BEST_FOR = "Private, reproducible high-quality locality georeferencing"

SEED = int(os.environ.get("PLACEBOT_NEMOTRON_SEED", "123"))
TEMPERATURE = float(os.environ.get("PLACEBOT_NEMOTRON_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("PLACEBOT_NEMOTRON_TOP_P", "0.95"))
STREAM = False
# A generic JSON object permits syntactically valid but unusable responses such
# as ``{"name": "Wales"}``.  Constrain the final answer to one complete
# PlaceBot record while leaving the reasoning trace unconstrained.
_STRING_FIELDS = {
    "country",
    "state",
    "region",
    "sector",
    "exact_site",
    "coordinate_source",
    "elevation_original",
    "confidence",
    "notes",
    "collection_notes",
}
_NULLABLE_NUMBER_FIELDS = {
    "coordinate_radius_meters",
    "elevation_meters",
}
_COORDINATE_FIELDS = {"latitude", "longitude"}
_REQUIRED_FIELDS = _STRING_FIELDS | _NULLABLE_NUMBER_FIELDS | _COORDINATE_FIELDS
_PLACEBOT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "country": {"type": "string"},
        "state": {"type": "string"},
        "region": {"type": "string"},
        "sector": {"type": "string"},
        "exact_site": {"type": "string"},
        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
        "longitude": {"type": "number", "minimum": -180, "maximum": 180},
        "coordinate_source": {"type": "string"},
        "coordinate_radius_meters": {
            "type": ["number", "null"],
            "minimum": 0,
        },
        "elevation_meters": {"type": ["number", "null"]},
        "elevation_original": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "notes": {"type": "string"},
        "collection_notes": {"type": "string"},
    },
    "required": sorted(_REQUIRED_FIELDS),
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "placebot_georeference",
        "strict": True,
        "schema": _PLACEBOT_OUTPUT_SCHEMA,
    },
}
ENABLE_THINKING = _env_bool("PLACEBOT_NEMOTRON_ENABLE_THINKING", False)
LOW_EFFORT = _env_bool("PLACEBOT_NEMOTRON_LOW_EFFORT", False)
if LOW_EFFORT and not ENABLE_THINKING:
    raise ValueError(
        "PLACEBOT_NEMOTRON_LOW_EFFORT requires "
        "PLACEBOT_NEMOTRON_ENABLE_THINKING"
    )

FAST_REASONING_MODE = "fast"
FULL_REASONING_MODE = "full"
REASONING_MODE_SETTINGS = {
    FAST_REASONING_MODE: {
        "reasoning_mode": FAST_REASONING_MODE,
        "max_output_tokens": 1_024,
        "request_timeout_seconds": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    FULL_REASONING_MODE: {
        "reasoning_mode": FULL_REASONING_MODE,
        "max_output_tokens": 65_536,
        "request_timeout_seconds": 7_200,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "low_effort": False,
        },
    },
}

# These IDs were verified against the byte-identical tokenizer.json published
# by the pinned NVFP4 and FP8 model revisions. The NVFP4 revision remains the
# default; set the override to the exact served checkpoint revision so result
# metadata identifies which official model supplied the token IDs.
SUPER_V3_DEFAULT_TOKENIZER_REVISION = (
    "4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6"
)
SUPER_V3_TOKENIZER_REVISION = os.environ.get(
    "PLACEBOT_NEMOTRON_TOKENIZER_REVISION",
    SUPER_V3_DEFAULT_TOKENIZER_REVISION,
).strip()
if not SUPER_V3_TOKENIZER_REVISION:
    raise ValueError("PLACEBOT_NEMOTRON_TOKENIZER_REVISION must not be empty")

# Both official snapshots expose this same tokenizer.json SHA-256. It maps
# ``<think>`` to 12 and ``</think>`` to 13.
SUPER_V3_TOKENIZER_SHA256 = (
    "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7"
)
SUPER_V3_THINK_START_TOKEN_ID = 12
SUPER_V3_THINK_END_TOKEN_ID = 13
REASONING_TOKEN_SOURCE = (
    f"vllm_token_ids:super_v3@{SUPER_V3_TOKENIZER_REVISION}"
)
IMPLICIT_START_REASONING_TOKEN_SOURCE = (
    f"vllm_token_ids:super_v3_implicit_start@{SUPER_V3_TOKENIZER_REVISION}"
)
NO_THINKING_TOKEN_SOURCE = (
    f"vllm_token_ids:super_v3_no_thinking@{SUPER_V3_TOKENIZER_REVISION}"
)


def get_headers(api_key: str = None) -> dict:
    """Return vLLM headers, adding bearer auth only when configured."""
    headers = {"Content-Type": "application/json"}
    token = api_key or API_KEY
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_reasoning_mode_settings(reasoning_mode: str) -> dict:
    """Return an isolated copy of one explicit adaptive-workflow mode."""
    try:
        settings = REASONING_MODE_SETTINGS[reasoning_mode]
    except (KeyError, TypeError) as exc:
        choices = ", ".join(sorted(REASONING_MODE_SETTINGS))
        raise ValueError(
            f"reasoning_mode must be one of: {choices}"
        ) from exc
    return {
        **settings,
        "chat_template_kwargs": dict(settings["chat_template_kwargs"]),
    }


def _effective_generation_settings(
    max_tokens: Optional[int], reasoning_mode: Optional[str]
) -> tuple[int, dict]:
    """Resolve legacy environment settings or an explicit request mode."""
    if reasoning_mode is None:
        cap = MAX_OUTPUT_TOKENS
        requested = cap if max_tokens is None else max_tokens
        chat_template_kwargs = {"enable_thinking": ENABLE_THINKING}
        if LOW_EFFORT:
            chat_template_kwargs["low_effort"] = True
    else:
        mode_settings = get_reasoning_mode_settings(reasoning_mode)
        cap = mode_settings["max_output_tokens"]
        requested = cap if max_tokens is None else max_tokens
        chat_template_kwargs = mode_settings["chat_template_kwargs"]

    if isinstance(requested, bool) or not isinstance(requested, int):
        raise ValueError("max_tokens must be a positive integer")
    if requested <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return min(requested, cap), chat_template_kwargs


def get_reproducibility_metadata(
    max_tokens: Optional[int] = None,
    *,
    reasoning_mode: Optional[str] = None,
) -> dict:
    """Return the effective, non-secret generation settings for run metadata.

    Omitting ``reasoning_mode`` preserves the environment-driven legacy
    profile. Supplying ``fast`` or ``full`` makes the request independent of
    module-global environment state, which prevents mode leakage between rows.
    """
    effective_max_tokens, chat_template_kwargs = _effective_generation_settings(
        max_tokens, reasoning_mode
    )
    return {
        "max_tokens": effective_max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "stream": STREAM,
        "response_format": dict(RESPONSE_FORMAT),
        "chat_template_kwargs": chat_template_kwargs,
        # vLLM extension used only for observability; it does not change the
        # generated sequence.
        "return_token_ids": True,
    }


def split_reasoning_token_ids(
    token_ids: List[int],
    *,
    reasoning_mode: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    """Count no-thinking output or split one unambiguous reasoning sequence.

    A no-thinking response must contain no thinking delimiters, so every token
    is final output. Reasoning counts exclude generated delimiter tokens. vLLM
    0.20 may omit the opening token because the chat template supplied
    ``<think>`` before generation; in that observed form, one closing token
    still provides an exact split. Any ambiguous form returns no count.
    """
    if reasoning_mode is None:
        enable_thinking = ENABLE_THINKING
    else:
        enable_thinking = get_reasoning_mode_settings(reasoning_mode)[
            "chat_template_kwargs"
        ]["enable_thinking"]

    if not enable_thinking:
        think_delimiters = {
            SUPER_V3_THINK_START_TOKEN_ID,
            SUPER_V3_THINK_END_TOKEN_ID,
        }
        if any(token_id in think_delimiters for token_id in token_ids):
            return None
        return {
            "reasoning_tokens": 0,
            "final_output_tokens": len(token_ids),
            "reasoning_token_source": NO_THINKING_TOKEN_SOURCE,
        }

    starts = [
        index
        for index, token_id in enumerate(token_ids)
        if token_id == SUPER_V3_THINK_START_TOKEN_ID
    ]
    ends = [
        index
        for index, token_id in enumerate(token_ids)
        if token_id == SUPER_V3_THINK_END_TOKEN_ID
    ]
    if len(ends) != 1 or len(starts) > 1:
        return None

    end = ends[0]
    if starts:
        start = starts[0]
        if start >= end:
            return None
        reasoning_tokens = end - start - 1
        source = REASONING_TOKEN_SOURCE
    else:
        reasoning_tokens = end
        source = IMPLICIT_START_REASONING_TOKEN_SOURCE

    return {
        "reasoning_tokens": reasoning_tokens,
        "final_output_tokens": len(token_ids) - end - 1,
        "reasoning_token_source": source,
    }


def format_request(
    prompt: str,
    max_tokens: Optional[int] = None,
    cached_content_name: str = None,
    *,
    reasoning_mode: Optional[str] = None,
) -> dict:
    """Build a JSON-only request using NVIDIA's recommended sampling values."""
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        **get_reproducibility_metadata(
            max_tokens,
            reasoning_mode=reasoning_mode,
        ),
    }


def parse_response(response_json: dict) -> str:
    """Return only a complete, naturally stopped final PlaceBot record."""
    try:
        choice = response_json["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected vLLM response format: {exc}") from exc

    if choice.get("finish_reason") != "stop":
        raise ValueError(
            "Unexpected vLLM response format: final output did not stop naturally"
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Unexpected vLLM response format: empty assistant content")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unexpected vLLM response format: invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict) or set(parsed) != _REQUIRED_FIELDS:
        raise ValueError(
            "Unexpected vLLM response format: incomplete PlaceBot final record"
        )
    if any(not isinstance(parsed[field], str) for field in _STRING_FIELDS):
        raise ValueError(
            "Unexpected vLLM response format: invalid PlaceBot string field"
        )
    if any(
        parsed[field] is not None and not _is_finite_number(parsed[field])
        for field in _NULLABLE_NUMBER_FIELDS
    ):
        raise ValueError(
            "Unexpected vLLM response format: invalid PlaceBot numeric field"
        )
    if any(not _is_finite_number(parsed[field]) for field in _COORDINATE_FIELDS):
        raise ValueError(
            "Unexpected vLLM response format: missing or invalid coordinate pair"
        )
    if not -90 <= parsed["latitude"] <= 90:
        raise ValueError("Unexpected vLLM response format: latitude out of range")
    if not -180 <= parsed["longitude"] <= 180:
        raise ValueError("Unexpected vLLM response format: longitude out of range")
    radius = parsed["coordinate_radius_meters"]
    if radius is not None and radius < 0:
        raise ValueError(
            "Unexpected vLLM response format: negative coordinate radius"
        )
    if parsed["confidence"] not in {"high", "medium", "low"}:
        raise ValueError("Unexpected vLLM response format: invalid confidence")
    return content.strip()


def _is_finite_number(value: object) -> bool:
    """Return true for finite JSON numbers, excluding booleans."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )
