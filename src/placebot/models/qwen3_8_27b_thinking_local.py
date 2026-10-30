"""Qwen3.8 27B Thinking served by a local OpenAI-compatible vLLM API."""

import os


def _chat_completions_endpoint(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


MODEL_NAME = "Qwen3.8 27B Thinking BF16 (Local vLLM)"
MODEL_PROVIDER = "Qwen (via local vLLM)"
MODEL_ID = os.environ.get("PLACEBOT_QWEN38_MODEL_ID", "qwen3_8_27b_bf16")

_base_url = os.environ.get("PLACEBOT_QWEN38_BASE_URL", "http://localhost:8000")
API_ENDPOINT = os.environ.get(
    "PLACEBOT_QWEN38_API_ENDPOINT",
    _chat_completions_endpoint(_base_url),
).strip()
API_KEY = os.environ.get("PLACEBOT_QWEN38_API_KEY", "")
API_VERSION = "v1"

MODEL_TYPE = "local"
MODEL_BACKEND = "vllm"
LOCAL_READY = True
LOCAL_STATUS = "Configured"
LOCAL_STATUS_DETAIL = API_ENDPOINT

COST_PER_1K_INPUT_TOKENS = 0.0
COST_PER_1K_OUTPUT_TOKENS = 0.0
ESTIMATED_COST_PER_RECORD = 0.0

MAX_TOKENS = int(os.environ.get("PLACEBOT_QWEN38_CONTEXT_WINDOW", "65536"))
MAX_OUTPUT_TOKENS = int(
    os.environ.get("PLACEBOT_QWEN38_MAX_OUTPUT_TOKENS", "49152")
)
CONTEXT_WINDOW = MAX_TOKENS
REQUESTS_PER_MINUTE = 999999
REQUEST_TIMEOUT = int(os.environ.get("PLACEBOT_QWEN38_REQUEST_TIMEOUT", "7200"))
MAX_RETRIES = int(os.environ.get("PLACEBOT_QWEN38_MAX_RETRIES", "0"))

SPEED = "DGX Spark dependent; xhigh reasoning enabled"
ACCURACY = "High (Qwen3.8 27B BF16)"
COST_EFFICIENCY = "Perfect (local inference)"
BEST_FOR = "Maximum-reasoning locality georeferencing"

# The checkpoint's generation_config.json specifies these sampling values.
SEED = int(os.environ.get("PLACEBOT_QWEN38_SEED", "123"))
TEMPERATURE = float(os.environ.get("PLACEBOT_QWEN38_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("PLACEBOT_QWEN38_TOP_P", "0.95"))
TOP_K = int(os.environ.get("PLACEBOT_QWEN38_TOP_K", "20"))
REASONING_EFFORT = os.environ.get(
    "PLACEBOT_QWEN38_REASONING_EFFORT", "xhigh"
).strip()
if REASONING_EFFORT not in {"xhigh", "medium", "low"}:
    raise ValueError(
        "PLACEBOT_QWEN38_REASONING_EFFORT must be xhigh, medium, or low"
    )
STREAM = False

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
    """Return the pinned Qwen3.8 full-thinking request settings."""
    return {
        "max_tokens": min(max_tokens, MAX_OUTPUT_TOKENS),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "stream": STREAM,
        "response_format": dict(RESPONSE_FORMAT),
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": REASONING_EFFORT,
            "preserve_thinking": True,
        },
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
