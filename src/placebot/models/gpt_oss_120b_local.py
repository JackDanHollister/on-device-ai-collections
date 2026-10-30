"""OpenAI gpt-oss-120b served by a local OpenAI-compatible vLLM API."""

import os


def _chat_completions_endpoint(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


MODEL_NAME = "gpt-oss-120b High Reasoning (Local vLLM)"
MODEL_PROVIDER = "OpenAI (via local vLLM)"
MODEL_ID = os.environ.get(
    "PLACEBOT_GPT_OSS_MODEL_ID",
    "gpt_oss_120b_cluster",
)

_base_url = os.environ.get("PLACEBOT_GPT_OSS_BASE_URL", "http://localhost:8000")
API_ENDPOINT = os.environ.get(
    "PLACEBOT_GPT_OSS_API_ENDPOINT",
    _chat_completions_endpoint(_base_url),
).strip()
API_KEY = os.environ.get("PLACEBOT_GPT_OSS_API_KEY", "")
API_VERSION = "v1"

MODEL_TYPE = "local"
MODEL_BACKEND = "vllm"
LOCAL_READY = True
LOCAL_STATUS = "Configured"
LOCAL_STATUS_DETAIL = API_ENDPOINT

COST_PER_1K_INPUT_TOKENS = 0.0
COST_PER_1K_OUTPUT_TOKENS = 0.0
ESTIMATED_COST_PER_RECORD = 0.0

MAX_TOKENS = int(os.environ.get("PLACEBOT_GPT_OSS_CONTEXT_WINDOW", "131072"))
MAX_OUTPUT_TOKENS = int(
    os.environ.get("PLACEBOT_GPT_OSS_MAX_OUTPUT_TOKENS", "65536")
)
CONTEXT_WINDOW = MAX_TOKENS
REQUESTS_PER_MINUTE = 999999
REQUEST_TIMEOUT = int(os.environ.get("PLACEBOT_GPT_OSS_REQUEST_TIMEOUT", "7200"))
# Avoid repeating a potentially multi-hour high-reasoning generation after a
# client timeout. A failed record remains visible to the benchmark runner.
MAX_RETRIES = int(os.environ.get("PLACEBOT_GPT_OSS_MAX_RETRIES", "0"))

SPEED = "GB10 cluster dependent; high reasoning enabled"
ACCURACY = "Very high (gpt-oss-120b, high reasoning)"
COST_EFFICIENCY = "Perfect (local inference)"
BEST_FOR = "Maximum-reasoning locality georeferencing"

SEED = int(os.environ.get("PLACEBOT_GPT_OSS_SEED", "123"))
TEMPERATURE = float(os.environ.get("PLACEBOT_GPT_OSS_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("PLACEBOT_GPT_OSS_TOP_P", "1.0"))
REASONING_EFFORT = os.environ.get(
    "PLACEBOT_GPT_OSS_REASONING_EFFORT",
    "high",
).strip().lower()
if REASONING_EFFORT not in {"low", "medium", "high"}:
    raise ValueError(
        "PLACEBOT_GPT_OSS_REASONING_EFFORT must be low, medium, or high"
    )
STREAM = False
RESPONSE_FORMAT = {"type": "json_object"}


def get_headers(api_key: str = None) -> dict:
    headers = {"Content-Type": "application/json"}
    token = api_key or API_KEY
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_reproducibility_metadata(max_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    """Return the effective high-reasoning Harmony generation settings."""
    return {
        "max_tokens": min(max_tokens, MAX_OUTPUT_TOKENS),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "stream": STREAM,
        "response_format": dict(RESPONSE_FORMAT),
        "reasoning_effort": REASONING_EFFORT,
    }


def format_request(
    prompt: str,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    cached_content_name: str = None,
) -> dict:
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        # vLLM's GPT-OSS Harmony path consumes this top-level field and builds
        # the corresponding system message.  Do not rewrite it client-side.
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
