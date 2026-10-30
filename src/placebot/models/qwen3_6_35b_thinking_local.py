"""Qwen3.6 35B-A3B Thinking served by a local OpenAI-compatible vLLM API."""

import os


def _chat_completions_endpoint(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


MODEL_NAME = "Qwen3.6 35B-A3B Thinking (Local vLLM)"
MODEL_PROVIDER = "Qwen (via local vLLM)"
MODEL_ID = os.environ.get(
    "PLACEBOT_QWEN36_MODEL_ID",
    "qwen3_6_35b_a3b_nvfp4",
)

_base_url = os.environ.get("PLACEBOT_QWEN36_BASE_URL", "http://localhost:8000")
API_ENDPOINT = os.environ.get(
    "PLACEBOT_QWEN36_API_ENDPOINT",
    _chat_completions_endpoint(_base_url),
).strip()
API_KEY = os.environ.get("PLACEBOT_QWEN36_API_KEY", "")
API_VERSION = "v1"

MODEL_TYPE = "local"
MODEL_BACKEND = "vllm"
LOCAL_READY = True
LOCAL_STATUS = "Configured"
LOCAL_STATUS_DETAIL = API_ENDPOINT

COST_PER_1K_INPUT_TOKENS = 0.0
COST_PER_1K_OUTPUT_TOKENS = 0.0
ESTIMATED_COST_PER_RECORD = 0.0

# Qwen recommends at least 131,072 tokens of context and up to 81,920 output
# tokens for complex reasoning benchmarks. The NVIDIA checkpoint supports the
# model's native 262,144-token context.
MAX_TOKENS = int(os.environ.get("PLACEBOT_QWEN36_CONTEXT_WINDOW", "262144"))
MAX_OUTPUT_TOKENS = int(
    os.environ.get("PLACEBOT_QWEN36_MAX_OUTPUT_TOKENS", "81920")
)
CONTEXT_WINDOW = MAX_TOKENS
REQUESTS_PER_MINUTE = 999999
REQUEST_TIMEOUT = int(os.environ.get("PLACEBOT_QWEN36_REQUEST_TIMEOUT", "7200"))
MAX_RETRIES = int(os.environ.get("PLACEBOT_QWEN36_MAX_RETRIES", "0"))

SPEED = "DGX Spark dependent; full reasoning enabled"
ACCURACY = "High (Qwen3.6 35B-A3B NVFP4)"
COST_EFFICIENCY = "Perfect (local inference)"
BEST_FOR = "Maximum-reasoning locality georeferencing"

# Official general-reasoning recipe from the Qwen3.6 model card.
SEED = int(os.environ.get("PLACEBOT_QWEN36_SEED", "123"))
TEMPERATURE = float(os.environ.get("PLACEBOT_QWEN36_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("PLACEBOT_QWEN36_TOP_P", "0.95"))
TOP_K = int(os.environ.get("PLACEBOT_QWEN36_TOP_K", "20"))
MIN_P = float(os.environ.get("PLACEBOT_QWEN36_MIN_P", "0"))
PRESENCE_PENALTY = float(
    os.environ.get("PLACEBOT_QWEN36_PRESENCE_PENALTY", "1.5")
)
REPETITION_PENALTY = float(
    os.environ.get("PLACEBOT_QWEN36_REPETITION_PENALTY", "1.0")
)
STREAM = False
# ``json_object`` permits an empty object, which is syntactically valid but not
# a usable PlaceBot result.  Keep Qwen's full reasoning unconstrained, then
# constrain only its final answer to the complete PlaceBot record contract.
_PLACEBOT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "country": {"type": "string"},
        "state": {"type": "string"},
        "region": {"type": "string"},
        "sector": {"type": "string"},
        "exact_site": {"type": "string"},
        "latitude": {"type": ["number", "null"]},
        "longitude": {"type": ["number", "null"]},
        "coordinate_source": {"type": "string"},
        "coordinate_radius_meters": {"type": ["number", "null"]},
        "elevation_meters": {"type": ["number", "null"]},
        "elevation_original": {"type": "string"},
        "confidence": {"type": "string"},
        "notes": {"type": "string"},
        "collection_notes": {"type": "string"},
    },
    "required": [
        "country",
        "state",
        "region",
        "sector",
        "exact_site",
        "latitude",
        "longitude",
        "coordinate_source",
        "coordinate_radius_meters",
        "elevation_meters",
        "elevation_original",
        "confidence",
        "notes",
        "collection_notes",
    ],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "placebot_georeference",
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
    """Return Qwen's official general-thinking generation settings."""
    return {
        "max_tokens": min(max_tokens, MAX_OUTPUT_TOKENS),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "min_p": MIN_P,
        "presence_penalty": PRESENCE_PENALTY,
        "frequency_penalty": 0.0,
        "repetition_penalty": REPETITION_PENALTY,
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


def parse_response(response_json: dict) -> str:
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected vLLM response format: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            "Unexpected vLLM response format: empty final content "
            "(reasoning may have exhausted max_tokens)"
        )
    return content.strip()
