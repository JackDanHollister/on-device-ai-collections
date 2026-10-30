"""Qwen3-Next 80B Instruct served by a local OpenAI-compatible vLLM API."""

import os


def _chat_completions_endpoint(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


MODEL_NAME = "Qwen3-Next 80B Instruct (Local vLLM)"
MODEL_PROVIDER = "Qwen (via local vLLM)"
MODEL_ID = os.environ.get(
    "PLACEBOT_QWEN_NEXT_MODEL_ID", "qwen3_next_80b_instruct_cluster"
)

_base_url = os.environ.get("PLACEBOT_QWEN_NEXT_BASE_URL", "http://localhost:8000")
API_ENDPOINT = os.environ.get(
    "PLACEBOT_QWEN_NEXT_API_ENDPOINT",
    _chat_completions_endpoint(_base_url),
).strip()
API_KEY = os.environ.get("PLACEBOT_QWEN_NEXT_API_KEY", "")
API_VERSION = "v1"

MODEL_TYPE = "local"
MODEL_BACKEND = "vllm"
LOCAL_READY = True
LOCAL_STATUS = "Configured"
LOCAL_STATUS_DETAIL = API_ENDPOINT

COST_PER_1K_INPUT_TOKENS = 0.0
COST_PER_1K_OUTPUT_TOKENS = 0.0
ESTIMATED_COST_PER_RECORD = 0.0

MAX_TOKENS = 8192
MAX_OUTPUT_TOKENS = int(
    os.environ.get("PLACEBOT_QWEN_NEXT_MAX_OUTPUT_TOKENS", "1024")
)
CONTEXT_WINDOW = 8192
REQUESTS_PER_MINUTE = 999999
REQUEST_TIMEOUT = 300

SPEED = "GB10 cluster dependent"
ACCURACY = "High (80B-A3B Qwen3-Next Instruct)"
COST_EFFICIENCY = "Perfect (local inference)"
BEST_FOR = "Private locality-georeferencing comparison"

SEED = int(os.environ.get("PLACEBOT_QWEN_NEXT_SEED", "123"))
TEMPERATURE = float(os.environ.get("PLACEBOT_QWEN_NEXT_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("PLACEBOT_QWEN_NEXT_TOP_P", "0.8"))
TOP_K = int(os.environ.get("PLACEBOT_QWEN_NEXT_TOP_K", "20"))
MIN_P = float(os.environ.get("PLACEBOT_QWEN_NEXT_MIN_P", "0"))
STREAM = False
RESPONSE_FORMAT = {"type": "json_object"}


def get_headers(api_key: str = None) -> dict:
    headers = {"Content-Type": "application/json"}
    token = api_key or API_KEY
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_reproducibility_metadata(max_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    """Return Qwen's official non-thinking sampling settings."""
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


def parse_response(response_json: dict) -> str:
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected vLLM response format: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Unexpected vLLM response format: empty assistant content")
    return content.strip()
