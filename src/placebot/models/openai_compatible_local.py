"""Strict PlaceBot profile for an arbitrary OpenAI-compatible local server.

This profile is deliberately model-neutral.  It is useful on an HPC cluster
when the selected vLLM-served checkpoint does not have a dedicated PlaceBot
profile yet.  Every setting is controlled through ``PLACEBOT_LOCAL_*``
environment variables, and the final response is accepted only when it is a
complete, naturally stopped 14-field PlaceBot record.
"""

from __future__ import annotations

import json
import math
import os


def _chat_completions_endpoint(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


MODEL_NAME = "OpenAI-compatible local model (strict PlaceBot JSON)"
MODEL_PROVIDER = "Local OpenAI-compatible server"
MODEL_ID = os.environ.get("PLACEBOT_LOCAL_MODEL_ID", "replace-with-served-model-id")

_base_url = os.environ.get("PLACEBOT_LOCAL_BASE_URL", "http://127.0.0.1:8000")
API_ENDPOINT = os.environ.get(
    "PLACEBOT_LOCAL_API_ENDPOINT",
    _chat_completions_endpoint(_base_url),
).strip()
API_KEY = os.environ.get("PLACEBOT_LOCAL_API_KEY", "")

MODEL_TYPE = "local"
MODEL_BACKEND = "openai-compatible"
LOCAL_READY = True
LOCAL_STATUS = "Configured"
LOCAL_STATUS_DETAIL = API_ENDPOINT

COST_PER_1K_INPUT_TOKENS = 0.0
COST_PER_1K_OUTPUT_TOKENS = 0.0
ESTIMATED_COST_PER_RECORD = 0.0

MAX_TOKENS = int(os.environ.get("PLACEBOT_LOCAL_CONTEXT_WINDOW", "32768"))
MAX_OUTPUT_TOKENS = int(os.environ.get("PLACEBOT_LOCAL_MAX_OUTPUT_TOKENS", "4096"))
REQUESTS_PER_MINUTE = int(
    os.environ.get("PLACEBOT_LOCAL_REQUESTS_PER_MINUTE", "999999")
)
REQUEST_TIMEOUT = int(os.environ.get("PLACEBOT_LOCAL_REQUEST_TIMEOUT", "900"))
MAX_RETRIES = int(os.environ.get("PLACEBOT_LOCAL_MAX_RETRIES", "0"))

SEED = int(os.environ.get("PLACEBOT_LOCAL_SEED", "123"))
TEMPERATURE = float(os.environ.get("PLACEBOT_LOCAL_TEMPERATURE", "0.2"))
TOP_P = float(os.environ.get("PLACEBOT_LOCAL_TOP_P", "0.95"))
ENABLE_THINKING = _env_bool("PLACEBOT_LOCAL_ENABLE_THINKING", False)

SPEED = "HPC and checkpoint dependent"
ACCURACY = "Must be qualified on an exposed PlaceBot canary"
COST_EFFICIENCY = "Local inference"
BEST_FOR = "Portable strict-schema HPC inference"

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
_NULLABLE_NUMBER_FIELDS = {"coordinate_radius_meters", "elevation_meters"}
_COORDINATE_FIELDS = {"latitude", "longitude"}
_REQUIRED_FIELDS = _STRING_FIELDS | _NULLABLE_NUMBER_FIELDS | _COORDINATE_FIELDS
_OUTPUT_SCHEMA = {
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
        "coordinate_radius_meters": {"type": ["number", "null"], "minimum": 0},
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
        "schema": _OUTPUT_SCHEMA,
    },
}


def get_headers(api_key: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = api_key or API_KEY
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_reasoning_mode_settings(reasoning_mode: str) -> dict:
    if reasoning_mode == "fast":
        return {
            "max_output_tokens": min(1024, MAX_OUTPUT_TOKENS),
            "request_timeout_seconds": min(300, REQUEST_TIMEOUT),
            "chat_template_kwargs": {"enable_thinking": False},
        }
    if reasoning_mode == "full":
        return {
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "request_timeout_seconds": REQUEST_TIMEOUT,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    raise ValueError("reasoning_mode must be fast or full")


def get_reproducibility_metadata(
    max_tokens: int = MAX_OUTPUT_TOKENS,
    *,
    reasoning_mode: str | None = None,
) -> dict:
    if reasoning_mode is None:
        cap = MAX_OUTPUT_TOKENS
        thinking = ENABLE_THINKING
    else:
        settings = get_reasoning_mode_settings(reasoning_mode)
        cap = settings["max_output_tokens"]
        thinking = settings["chat_template_kwargs"]["enable_thinking"]
    metadata = {
        "max_tokens": min(max_tokens, cap),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "stream": False,
        "response_format": RESPONSE_FORMAT,
    }
    metadata["chat_template_kwargs"] = {"enable_thinking": thinking}
    return metadata


def format_request(
    prompt: str,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    cached_content_name: str | None = None,
    *,
    reasoning_mode: str | None = None,
) -> dict:
    del cached_content_name
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        **get_reproducibility_metadata(max_tokens, reasoning_mode=reasoning_mode),
    }


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def parse_response(response_json: dict) -> str:
    try:
        choice = response_json["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected OpenAI-compatible response: {exc}") from exc
    if choice.get("finish_reason") != "stop":
        raise ValueError("Final output did not stop naturally")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Final output is empty")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Final output is not JSON: {exc}") from exc
    if not isinstance(parsed, dict) or set(parsed) != _REQUIRED_FIELDS:
        raise ValueError("Final output is not a complete PlaceBot record")
    if any(not isinstance(parsed[field], str) for field in _STRING_FIELDS):
        raise ValueError("A PlaceBot string field has the wrong type")
    if any(
        parsed[field] is not None and not _finite_number(parsed[field])
        for field in _NULLABLE_NUMBER_FIELDS
    ):
        raise ValueError("A PlaceBot numeric field has the wrong type")
    if any(not _finite_number(parsed[field]) for field in _COORDINATE_FIELDS):
        raise ValueError("The PlaceBot coordinate pair is missing or invalid")
    if not -90 <= parsed["latitude"] <= 90:
        raise ValueError("Latitude is out of range")
    if not -180 <= parsed["longitude"] <= 180:
        raise ValueError("Longitude is out of range")
    radius = parsed["coordinate_radius_meters"]
    if radius is not None and radius < 0:
        raise ValueError("Coordinate radius is negative")
    if parsed["confidence"] not in {"high", "medium", "low"}:
        raise ValueError("Confidence is invalid")
    return content.strip()
