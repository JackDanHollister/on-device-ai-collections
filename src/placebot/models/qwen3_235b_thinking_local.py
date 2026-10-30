"""Qwen3 235B-A22B Thinking served by the two-GB10 vLLM API."""

import json
import os


def _chat_completions_endpoint(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


MODEL_NAME = "Qwen3 235B-A22B Thinking (Two-GB10 vLLM)"
MODEL_PROVIDER = "Qwen (via local vLLM)"
MODEL_ID = os.environ.get(
    "PLACEBOT_QWEN235_MODEL_ID",
    "qwen3_235b_a22b_nvfp4",
)

_base_url = os.environ.get("PLACEBOT_QWEN235_BASE_URL", "http://localhost:8000")
API_ENDPOINT = os.environ.get(
    "PLACEBOT_QWEN235_API_ENDPOINT",
    _chat_completions_endpoint(_base_url),
).strip()
API_KEY = os.environ.get("PLACEBOT_QWEN235_API_KEY", "")
API_VERSION = "v1"

MODEL_TYPE = "local"
MODEL_BACKEND = "vllm"
LOCAL_READY = True
LOCAL_STATUS = "Configured"
LOCAL_STATUS_DETAIL = API_ENDPOINT

COST_PER_1K_INPUT_TOKENS = 0.0
COST_PER_1K_OUTPUT_TOKENS = 0.0
ESTIMATED_COST_PER_RECORD = 0.0

MAX_TOKENS = int(os.environ.get("PLACEBOT_QWEN235_CONTEXT_WINDOW", "16384"))
MAX_OUTPUT_TOKENS = int(
    os.environ.get("PLACEBOT_QWEN235_MAX_OUTPUT_TOKENS", "8192")
)
CONTEXT_WINDOW = MAX_TOKENS
REQUESTS_PER_MINUTE = 999999
REQUEST_TIMEOUT = int(os.environ.get("PLACEBOT_QWEN235_REQUEST_TIMEOUT", "1800"))
MAX_RETRIES = int(os.environ.get("PLACEBOT_QWEN235_MAX_RETRIES", "0"))

SPEED = "Two-GB10 cluster dependent; full reasoning enabled"
ACCURACY = "Very high (Qwen3 235B-A22B NVIDIA NVFP4)"
COST_EFFICIENCY = "Perfect (local inference)"
BEST_FOR = "Accuracy-first locality georeferencing"

# Qwen3 thinking-mode sampling guidance.
SEED = int(os.environ.get("PLACEBOT_QWEN235_SEED", "123"))
TEMPERATURE = float(os.environ.get("PLACEBOT_QWEN235_TEMPERATURE", "0.6"))
TOP_P = float(os.environ.get("PLACEBOT_QWEN235_TOP_P", "0.95"))
TOP_K = int(os.environ.get("PLACEBOT_QWEN235_TOP_K", "20"))
MIN_P = float(os.environ.get("PLACEBOT_QWEN235_MIN_P", "0"))
STREAM = False

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
    "latitude",
    "longitude",
    "coordinate_radius_meters",
    "elevation_meters",
}
_REQUIRED_FIELDS = _STRING_FIELDS | _NULLABLE_NUMBER_FIELDS
_PLACEBOT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "country": {"type": "string"},
        "state": {"type": "string"},
        "region": {"type": "string"},
        "sector": {"type": "string"},
        "exact_site": {"type": "string"},
        "latitude": {"type": ["number", "null"], "minimum": -90, "maximum": 90},
        "longitude": {
            "type": ["number", "null"],
            "minimum": -180,
            "maximum": 180,
        },
        "coordinate_source": {"type": "string"},
        "coordinate_radius_meters": {
            "type": ["number", "null"],
            "minimum": 0,
        },
        "elevation_meters": {"type": ["number", "null"]},
        "elevation_original": {"type": "string"},
        "confidence": {"type": "string"},
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


def get_headers(api_key: str = None) -> dict:
    headers = {"Content-Type": "application/json"}
    token = api_key or API_KEY
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_reproducibility_metadata(max_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    return {
        "max_tokens": min(max_tokens, MAX_OUTPUT_TOKENS),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "min_p": MIN_P,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": SEED,
        "stream": STREAM,
        "response_format": dict(RESPONSE_FORMAT),
    }


def format_request(
    prompt: str,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    cached_content_name: str = None,
) -> dict:
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        **get_reproducibility_metadata(max_tokens),
    }


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_response(response_json: dict) -> str:
    """Return only a complete, naturally stopped final PlaceBot record."""
    try:
        choice = response_json["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected vLLM response format: {exc}") from exc

    if choice.get("finish_reason") != "stop":
        raise ValueError(
            "Unexpected vLLM response format: final output did not stop naturally"
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            "Unexpected vLLM response format: empty final content "
            "(reasoning is never used as the answer)"
        )

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
        parsed[field] is not None and not _is_number(parsed[field])
        for field in _NULLABLE_NUMBER_FIELDS
    ):
        raise ValueError(
            "Unexpected vLLM response format: invalid PlaceBot numeric field"
        )
    if (parsed["latitude"] is None) != (parsed["longitude"] is None):
        raise ValueError(
            "Unexpected vLLM response format: incomplete coordinate pair"
        )
    if parsed["latitude"] is not None and not -90 <= parsed["latitude"] <= 90:
        raise ValueError("Unexpected vLLM response format: latitude out of range")
    if parsed["longitude"] is not None and not -180 <= parsed["longitude"] <= 180:
        raise ValueError("Unexpected vLLM response format: longitude out of range")
    radius = parsed["coordinate_radius_meters"]
    if radius is not None and radius < 0:
        raise ValueError(
            "Unexpected vLLM response format: negative coordinate radius"
        )
    return content.strip()
