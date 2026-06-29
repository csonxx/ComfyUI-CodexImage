"""Core image generation logic — pure Python, no third-party dependencies.

Can be imported and used standalone without any package manager:
  pip install (torch, etc.) not required.

Usage:
    from generator import generate_image
    img_bytes, img_path = generate_image(prompt="a cat", model="gpt-5.5", ...)
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

# Simple print-based error logger
def _log_error(msg: str, exc: Exception | None = None) -> None:
    timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[codex_image][{timestamp}] ERROR"
    print(f"{prefix} {msg}", file=sys.stderr)
    if exc:
        import traceback
        traceback.print_exc(file=sys.stderr)

# ── Constants ─────────────────────────────────────────────────────────────────

# Environment variable overrides
DEFAULT_MODEL = os.environ.get("CODEX_IMAGE_MODEL", "gpt-5.5")
DEFAULT_SIZE = os.environ.get("CODEX_IMAGE_SIZE", "1024x1024")
DEFAULT_QUALITY = os.environ.get("CODEX_IMAGE_QUALITY", "medium")
DEFAULT_FORMAT = os.environ.get("CODEX_IMAGE_FORMAT", "png")
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_RETRIES = int(os.environ.get("CODEX_IMAGE_MAX_RETRIES", "8"))
DEFAULT_RETRY_BASE_SECONDS = float(os.environ.get("CODEX_IMAGE_RETRY_BASE_SECONDS", "2"))
DEFAULT_RETRY_MAX_SECONDS = float(os.environ.get("CODEX_IMAGE_RETRY_MAX_SECONDS", "90"))
DEFAULT_RATE_LIMIT_FLOOR_SECONDS = float(os.environ.get("CODEX_IMAGE_RATE_LIMIT_FLOOR_SECONDS", "65"))
DEFAULT_BASE_URL = os.environ.get("CODEX_IMAGE_BASE_URL", "https://chatgpt.com/backend-api/codex")
DEFAULT_CODEX_SCRIPT = os.environ.get("CODEX_IMAGE_SCRIPT", "~/.codex-image/scripts/codex_image.py")
DEFAULT_OPENROUTER_BASE_URL = os.environ.get("CODEX_IMAGE_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/images")
DEFAULT_OPENROUTER_MODEL = os.environ.get("CODEX_IMAGE_OPENROUTER_MODEL", "openai/gpt-image-2")
DEFAULT_LITELLM_BASE_URL = os.environ.get("CODEX_IMAGE_LITELLM_BASE_URL", "http://localhost:4000")
DEFAULT_LITELLM_MODEL = os.environ.get("CODEX_IMAGE_LITELLM_MODEL", "gpt-image-2")

# Supported image sizes for GPT Image 2
SUPPORTED_SIZES = (
    "1024x1024",    # 1:1
    "1536x1024",    # 3:2 landscape
    "1024x1536",    # 2:3 portrait
    "1792x1024",    # 16:9 landscape
    "1024x1792",    # 9:16 portrait
    "1920x1080",    # 16:9 landscape HD
    "1080x1920",    # 9:16 portrait HD
    "2048x2048",    # 1:1 high-res
    "3840x2160",    # 4K 16:9
    "2160x3840",    # 4K 9:16
)

# ── Auth ─────────────────────────────────────────────────────────────────────

def _load_auth_from_codex_home() -> str:
    """Load credentials from ~/.codex/auth.json.

    Priority:
      1. OPENAI_API_KEY environment variable
      2. OPENAI_API_KEY field in ~/.codex/auth.json
      3. ChatGPT OAuth access_token in ~/.codex/auth.json (after `codex login`)
    """
    codex_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    auth_path = codex_home / "auth.json"
    if not auth_path.exists():
        msg = (
            f"auth.json not found at {auth_path}. "
            "Run `codex login` or set OPENAI_API_KEY environment variable."
        )
        _log_error(msg)
        raise FileNotFoundError(msg)
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Failed to read {auth_path}: {exc}"
        _log_error(msg, exc)
        raise RuntimeError(msg) from exc

    field_key = (data.get("OPENAI_API_KEY") or "").strip()
    if field_key:
        return field_key

    if data.get("auth_mode") == "chatgpt":
        tokens = data.get("tokens") or {}
        at = (tokens.get("access_token") or "").strip()
        if at:
            return at

    msg = (
        "No credentials found in ~/.codex/auth.json. "
        "Run `codex login` or set OPENAI_API_KEY in auth.json."
    )
    _log_error(msg)
    raise ValueError(msg)


def _resolve_api_key(api_key: str) -> str:
    if api_key and api_key.strip():
        return api_key.strip()
    return _load_auth_from_codex_home()


def _resolve_api_url(base_url: str) -> str:
    """Resolve base_url to a full API endpoint URL.

    Handles:
      - Full URL with scheme, e.g. https://chatgpt.com/backend-api/codex
      - Just a path starting with /, e.g. /v1/responses  → join with DEFAULT_BASE_URL
      - Empty / whitespace                              → use DEFAULT_BASE_URL
    """
    base_url = (base_url or "").strip()
    if not base_url:
        base_url = DEFAULT_BASE_URL
    elif base_url.startswith("/"):
        # Relative path — join with DEFAULT_BASE_URL's origin
        base = DEFAULT_BASE_URL.rstrip("/")
        base_url = base + base_url
    else:
        base_url = base_url.rstrip("/")
        if "responses" not in base_url:
            base_url = f"{base_url}/responses"
    return base_url


def _resolve_openrouter_url(base_url: str) -> str:
    """Resolve OpenRouter base URL to its dedicated Image API endpoint."""
    base_url = (base_url or "").strip() or DEFAULT_OPENROUTER_BASE_URL
    base_url = base_url.rstrip("/")
    if base_url.endswith("/api/v1/images"):
        return base_url
    if base_url.endswith("/api/v1"):
        return f"{base_url}/images"
    return f"{base_url}/api/v1/images"


def _resolve_litellm_url(base_url: str, edit: bool = False) -> str:
    """Resolve LiteLLM base URL to /v1/images/generations or /v1/images/edits."""
    endpoint = "edits" if edit else "generations"
    other_endpoint = "generations" if edit else "edits"
    base_url = (base_url or "").strip() or DEFAULT_LITELLM_BASE_URL
    base_url = base_url.rstrip("/")
    if base_url.endswith(f"/v1/images/{endpoint}"):
        return base_url
    if base_url.endswith(f"/v1/images/{other_endpoint}"):
        return base_url.rsplit("/", 1)[0] + f"/{endpoint}"
    if base_url.endswith("/v1/images"):
        return f"{base_url}/{endpoint}"
    if base_url.endswith("/v1"):
        return f"{base_url}/images/{endpoint}"
    return f"{base_url}/v1/images/{endpoint}"


def _resolve_env_api_key(api_key: str, env_names: tuple[str, ...], label: str) -> str:
    """Resolve a provider API key from an explicit value or environment variables."""
    if api_key and api_key.strip():
        return _normalize_bearer_token(api_key)
    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            return _normalize_bearer_token(value)
    names = " or ".join(env_names)
    raise ValueError(f"{label} API key not found. Set {names}.")


def _normalize_bearer_token(value: str) -> str:
    """Accept either a raw token or an accidentally pasted Bearer token."""
    token = (value or "").strip()
    if token.lower().startswith("bearer "):
        return token[7:].strip()
    return token


def _image_extension_from_bytes(img_bytes: bytes, fallback: str) -> str:
    fallback = (fallback or "png").lower().lstrip(".")
    if img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if img_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if img_bytes.startswith(b"RIFF") and img_bytes[8:12] == b"WEBP":
        return "webp"
    return fallback


def _write_temp_image(img_bytes: bytes, fmt: str) -> str:
    ext = _image_extension_from_bytes(img_bytes, fmt)
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
        f.write(img_bytes)
        return f.name


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    match = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", data_url, re.S)
    if not match:
        raise ValueError("Invalid data URL")
    media_type = match.group(1) or "application/octet-stream"
    is_base64 = bool(match.group(2))
    payload = match.group(3)
    if is_base64:
        return base64.b64decode(payload), media_type

    from urllib.parse import unquote_to_bytes

    return unquote_to_bytes(payload), media_type


def _download_image_url(url: str, token: str = "") -> bytes:
    from urllib import request

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(url, headers=headers, method="GET")
    with request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        return resp.read()


def _augment_http_error_message(url: str, status: int, body_text: str) -> str:
    """Add provider-specific setup hints without hiding the original response."""
    hints: list[str] = []
    lower_body = body_text.lower()
    if status == 401 and "openrouter.ai" in url:
        hints.append(
            "OpenRouter authentication failed. Make sure the ComfyUI process has "
            "CODEX_IMAGE_OPENROUTER_API_KEY or OPENROUTER_API_KEY set to a valid "
            "OpenRouter key, usually starting with sk-or-. In Docker, check the "
            "running ComfyUI Python process environment, not only a docker exec shell."
        )
    if status == 403 and "key_model_access_denied" in lower_body:
        hints.append(
            "LiteLLM rejected the model name for this key. Use one of the exact "
            "model aliases listed in the error response."
        )
    if not hints:
        return body_text
    return f"{body_text}\n\nHints:\n- " + "\n- ".join(hints)


def _normalize_litellm_model(model: str) -> str:
    """Resolve the LiteLLM model name without rewriting provider aliases."""
    model = (model or "").strip()
    if not model:
        return DEFAULT_LITELLM_MODEL
    return model


def _post_streaming(url: str, token: str, payload: dict, timeout: int) -> list[dict]:
    """POST JSON with SSE streaming, collect all data events."""
    from urllib import error, request

    body = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    req = request.Request(url, data=body, headers=headers, method="POST")
    events: list[dict] = []

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            buf = ""
            for chunk in iter(lambda: resp.read(4096), b""):
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line == "data: [DONE]":
                        break
                    if line.startswith("data: "):
                        try:
                            events.append(json.loads(line[6:]))
                        except json.JSONDecodeError:
                            pass
    except error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        body_text = _augment_http_error_message(url, exc.code, body_text)
        msg = f"POST {url} failed: status={exc.code}\n{body_text}"
        _log_error(msg, exc)
        if exc.code == 429:
            raise CodexImageRateLimitError(
                msg,
                _parse_retry_after_seconds(body_text),
            ) from exc
        if exc.code >= 500:
            raise CodexImageTransientAPIError(str(exc.code), msg) from exc
        raise RuntimeError(msg) from exc
    except error.URLError as exc:
        msg = f"POST {url} failed: {exc}"
        _log_error(msg, exc)
        raise RuntimeError(msg) from exc

    return events


class CodexImageRateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class CodexImageTransientAPIError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code or "unknown"
        self.message = message
        super().__init__(f"Codex image API error ({self.code}): {message}")


def _parse_retry_after_seconds(message: str) -> float | None:
    match = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|s)", message, re.I)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    return value / 1000.0 if unit == "ms" else value


def _rate_limit_floor_seconds(message: str) -> float:
    if "per min" in message.lower():
        return DEFAULT_RATE_LIMIT_FLOOR_SECONDS
    return 0.0


def _build_rate_limit_message(exc: CodexImageRateLimitError, attempts: int) -> str:
    return (
        f"{exc}\n"
        f"Codex image generation is still rate limited after {attempts} attempts. "
        "Wait a few minutes, stop other image-generation jobs that share this "
        "account or organization, or lower ComfyUI queue concurrency before retrying. "
        "You can tune CODEX_IMAGE_MAX_RETRIES, CODEX_IMAGE_RETRY_MAX_SECONDS, and "
        "CODEX_IMAGE_RATE_LIMIT_FLOOR_SECONDS if you want ComfyUI to wait longer."
    )


def _is_retryable_api_code(code: str) -> bool:
    code = code.lower()
    return code in {
        "server_error",
        "internal_error",
        "temporarily_unavailable",
        "service_unavailable",
        "server_is_overloaded",
        "overloaded",
        "capacity_exceeded",
        "timeout",
        "gateway_timeout",
    } or code.startswith("5")


def _build_transient_api_message(exc: CodexImageTransientAPIError, attempts: int) -> str:
    return (
        f"{exc}\n"
        f"Codex image API kept returning a temporary server error after {attempts} attempts. "
        "Retry later. If the same request ID keeps appearing, include it when contacting support."
    )


def _raise_api_error_if_present(events: list[dict]) -> None:
    for ev in events:
        err = ev.get("error")
        if not err and isinstance(ev.get("response"), dict):
            err = ev["response"].get("error")
        if not isinstance(err, dict):
            continue

        code = str(err.get("code") or "")
        message = str(err.get("message") or err)
        if code == "rate_limit_exceeded":
            raise CodexImageRateLimitError(message, _parse_retry_after_seconds(message))
        if _is_retryable_api_code(code):
            raise CodexImageTransientAPIError(code, message)
        raise RuntimeError(f"Codex image API error ({code or 'unknown'}): {message}")


def _extract_image(events: list[dict]) -> str:
    """Pull base64-encoded image string from parsed SSE events.

    The API returns image data in three possible shapes — try all of them.
    """
    for ev in events:
        if ev.get("type") == "response.image_generation_call.done":
            r = ev.get("result")
            if r:
                return str(r)

        for key in ("item", "output_item"):
            item = ev.get(key, {})
            if item.get("type") == "image_generation_call" and item.get("result"):
                return str(item["result"])

        resp = ev.get("response", {})
        if isinstance(resp, dict):
            for o in resp.get("output") or []:
                if o.get("type") == "image_generation_call" and o.get("result"):
                    return str(o["result"])

    _raise_api_error_if_present(events)

    tail = events[-3:] if len(events) > 3 else events
    raise RuntimeError(
        f"No generated image found in SSE events:\n"
        f"{json.dumps(tail, ensure_ascii=False, indent=2)[:2000]}"
    )


# ── Payload ───────────────────────────────────────────────────────────────────

def _build_payload(
    prompt: str,
    model: str,
    size: str,
    quality: str,
    input_image_urls: list[str] | None = None,
    action: str = "auto",
) -> dict[str, Any]:
    """Build the request body for the Codex Responses API."""
    actual_model = (model or "").strip() or DEFAULT_MODEL
    input_image_urls = input_image_urls or []

    tool = {"type": "image_generation", "size": size, "quality": quality}
    if input_image_urls:
        tool["action"] = action or "edit"
        content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
        content.extend(
            {"type": "input_image", "image_url": image_url}
            for image_url in input_image_urls
            if image_url
        )
        input_value: str | list[dict[str, Any]] = [
            {"role": "user", "content": content}
        ]
    else:
        input_value = [{"role": "user", "content": prompt}]

    return {
        "model": actual_model,
        "instructions": "Generate the requested image using the image_generation tool.",
        "input": input_value,
        "tools": [tool],
        "store": False,
        "stream": True,
    }


# ── API mode ──────────────────────────────────────────────────────────────────

def _generate_api(
    prompt: str,
    model: str,
    size: str,
    quality: str,
    fmt: str,
    base_url: str,
    api_key: str,
    input_image_urls: list[str] | None = None,
    action: str = "auto",
) -> tuple[bytes, str]:
    """Generate an image via direct HTTP to the Codex Responses API.

    Args:
        prompt:    Image description
        model:     Model name (e.g. "gpt-5.5")
        size:      Dimensions string (e.g. "1024x1024")
        quality:   "low" | "medium" | "high"
        fmt:       Output format ("png" | "jpeg" | "webp")
        base_url:  API base URL
        api_key:   Bearer token (leave empty to auto-load from ~/.codex/auth.json)

    Returns:
        (raw_image_bytes, path_to_temp_file)
    """
    api_url = _resolve_api_url(base_url)

    token = _resolve_api_key(api_key)
    payload = _build_payload(
        prompt,
        model,
        size,
        quality,
        input_image_urls=input_image_urls,
        action=action,
    )
    last_retryable_error: RuntimeError | None = None

    for attempt in range(max(1, DEFAULT_MAX_RETRIES)):
        try:
            events = _post_streaming(api_url, token, payload, DEFAULT_TIMEOUT)
            img_b64 = _extract_image(events)
            break
        except (CodexImageRateLimitError, CodexImageTransientAPIError) as exc:
            last_retryable_error = exc
            if attempt >= DEFAULT_MAX_RETRIES - 1:
                if isinstance(exc, CodexImageRateLimitError):
                    raise CodexImageRateLimitError(
                        _build_rate_limit_message(exc, attempt + 1),
                        exc.retry_after_seconds,
                    ) from exc
                raise RuntimeError(_build_transient_api_message(exc, attempt + 1)) from exc
            is_rate_limit = isinstance(exc, CodexImageRateLimitError)
            retry_after = exc.retry_after_seconds if is_rate_limit else None
            backoff = DEFAULT_RETRY_BASE_SECONDS * (2 ** attempt)
            delay = max(
                retry_after or 0.0,
                backoff,
                _rate_limit_floor_seconds(str(exc)) if is_rate_limit else 0.0,
            )
            delay = min(delay, DEFAULT_RETRY_MAX_SECONDS)
            reason = "rate limited" if is_rate_limit else "temporary API error"
            print(
                f"[codex_image] {reason}; retrying in {delay:.1f}s "
                f"({attempt + 2}/{DEFAULT_MAX_RETRIES})",
                file=sys.stderr,
            )
            time.sleep(delay)
    else:
        raise last_retryable_error or RuntimeError("Codex image generation failed")

    img_bytes = base64.b64decode(img_b64)

    return img_bytes, _write_temp_image(img_bytes, fmt)


# ── OpenRouter / LiteLLM image APIs ───────────────────────────────────────────

def _post_json(
    url: str,
    token: str,
    payload: dict[str, Any],
    timeout: int,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST JSON and return a parsed JSON response."""
    from urllib import error, request

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    headers.update(extra_headers or {})

    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        body_text = _augment_http_error_message(url, exc.code, body_text)
        msg = f"POST {url} failed: status={exc.code}\n{body_text}"
        _log_error(msg, exc)
        if exc.code == 429:
            raise CodexImageRateLimitError(msg, _parse_retry_after_seconds(body_text)) from exc
        if exc.code >= 500:
            raise CodexImageTransientAPIError(str(exc.code), msg) from exc
        raise RuntimeError(msg) from exc
    except error.URLError as exc:
        msg = f"POST {url} failed: {exc}"
        _log_error(msg, exc)
        raise RuntimeError(msg) from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"POST {url} returned non-JSON response:\n{text[:2000]}") from exc


def _post_multipart(
    url: str,
    token: str,
    fields: dict[str, str],
    files: list[tuple[str, str, bytes, str]],
    timeout: int,
) -> dict[str, Any]:
    """POST multipart/form-data and return a parsed JSON response.

    files entries are: (field_name, filename, bytes, content_type).
    """
    from urllib import error, request
    import uuid

    boundary = f"----CodexImage{uuid.uuid4().hex}"
    parts: list[bytes] = []

    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    for field_name, filename, content, content_type in files:
        parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{field_name}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type or 'application/octet-stream'}\r\n\r\n".encode("utf-8"),
                content,
                b"\r\n",
            ]
        )

    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    req = request.Request(url, data=body, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        body_text = _augment_http_error_message(url, exc.code, body_text)
        msg = f"POST {url} failed: status={exc.code}\n{body_text}"
        _log_error(msg, exc)
        if exc.code == 429:
            raise CodexImageRateLimitError(msg, _parse_retry_after_seconds(body_text)) from exc
        if exc.code >= 500:
            raise CodexImageTransientAPIError(str(exc.code), msg) from exc
        raise RuntimeError(msg) from exc
    except error.URLError as exc:
        msg = f"POST {url} failed: {exc}"
        _log_error(msg, exc)
        raise RuntimeError(msg) from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"POST {url} returned non-JSON response:\n{text[:2000]}") from exc


def _extract_image_bytes_from_images_response(response: dict[str, Any], token: str = "") -> bytes:
    """Extract image bytes from OpenAI/OpenRouter-style Images API responses."""
    data = response.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError(
            "No image data found in response:\n"
            f"{json.dumps(response, ensure_ascii=False, indent=2)[:2000]}"
        )

    first = data[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"Unexpected image response item: {first!r}")

    b64_json = first.get("b64_json")
    if b64_json:
        return base64.b64decode(str(b64_json))

    url = (first.get("url") or "").strip()
    if url:
        if url.startswith("data:"):
            image_bytes, _ = _decode_data_url(url)
            return image_bytes
        return _download_image_url(url, token="")

    raise RuntimeError(
        "Image response item did not contain b64_json or url:\n"
        f"{json.dumps(first, ensure_ascii=False, indent=2)[:2000]}"
    )


def _build_openrouter_payload(
    prompt: str,
    model: str,
    size: str,
    quality: str,
    fmt: str,
    input_image_urls: list[str] | None = None,
    background: str = "opaque",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model or DEFAULT_OPENROUTER_MODEL,
        "prompt": prompt,
        "n": 1,
    }
    if size:
        payload["size"] = size
    if quality:
        payload["quality"] = quality
    if fmt:
        payload["output_format"] = fmt
    if background:
        payload["background"] = background

    refs = []
    for image_url in input_image_urls or []:
        if image_url:
            refs.append({"type": "image_url", "image_url": {"url": image_url}})
    if refs:
        payload["input_references"] = refs

    return payload


def _generate_openrouter(
    prompt: str,
    model: str,
    size: str,
    quality: str,
    fmt: str,
    base_url: str,
    api_key: str,
    input_image_urls: list[str] | None = None,
    background: str = "opaque",
) -> tuple[bytes, str]:
    url = _resolve_openrouter_url(base_url)
    token = _resolve_env_api_key(
        api_key,
        ("CODEX_IMAGE_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
        "OpenRouter",
    )
    payload = _build_openrouter_payload(
        prompt=prompt,
        model=model or DEFAULT_OPENROUTER_MODEL,
        size=size,
        quality=quality,
        fmt=fmt,
        input_image_urls=input_image_urls,
        background=background,
    )
    extra_headers = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    if referer:
        extra_headers["HTTP-Referer"] = referer
    title = os.environ.get("OPENROUTER_X_TITLE", "").strip()
    if title:
        extra_headers["X-Title"] = title
    response = _post_json(url, token, payload, DEFAULT_TIMEOUT, extra_headers=extra_headers)
    img_bytes = _extract_image_bytes_from_images_response(response, token)
    return img_bytes, _write_temp_image(img_bytes, fmt)


def _build_litellm_generation_payload(
    prompt: str,
    model: str,
    size: str,
    quality: str,
) -> dict[str, Any]:
    model = _normalize_litellm_model(model)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    if quality:
        payload["quality"] = quality
    return payload


def _data_urls_to_multipart_files(
    input_image_urls: list[str],
    mask_image_url: str | None = None,
) -> list[tuple[str, str, bytes, str]]:
    files: list[tuple[str, str, bytes, str]] = []
    for idx, image_url in enumerate(input_image_urls):
        if not image_url:
            continue
        image_bytes, media_type = _decode_data_url(image_url)
        ext = "png" if media_type.endswith("png") else "jpg"
        files.append(("image", f"image_{idx}.{ext}", image_bytes, media_type))

    if mask_image_url:
        mask_bytes, media_type = _decode_data_url(mask_image_url)
        ext = "png" if media_type.endswith("png") else "jpg"
        files.append(("mask", f"mask.{ext}", mask_bytes, media_type))

    return files


def _generate_litellm(
    prompt: str,
    model: str,
    size: str,
    quality: str,
    fmt: str,
    base_url: str,
    api_key: str,
    input_image_urls: list[str] | None = None,
    mask_image_url: str | None = None,
) -> tuple[bytes, str]:
    token = _resolve_env_api_key(
        api_key,
        ("CODEX_IMAGE_LITELLM_API_KEY", "LITELLM_API_KEY", "LITELLM_MASTER_KEY"),
        "LiteLLM",
    )

    model = _normalize_litellm_model(model)
    input_image_urls = input_image_urls or []
    if input_image_urls:
        url = _resolve_litellm_url(base_url, edit=True)
        fields = {
            "model": model,
            "prompt": prompt,
            "n": "1",
            "size": size,
        }
        if quality:
            fields["quality"] = quality
        files = _data_urls_to_multipart_files(input_image_urls, mask_image_url)
        response = _post_multipart(url, token, fields, files, DEFAULT_TIMEOUT)
    else:
        url = _resolve_litellm_url(base_url, edit=False)
        payload = _build_litellm_generation_payload(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
        )
        response = _post_json(url, token, payload, DEFAULT_TIMEOUT)

    img_bytes = _extract_image_bytes_from_images_response(response, token)
    return img_bytes, _write_temp_image(img_bytes, fmt)


# ── CLI mode ─────────────────────────────────────────────────────────────────

def _generate_cli(
    prompt: str,
    model: str,
    size: str,
    quality: str,
    fmt: str,
    codex_cmd: str,
) -> tuple[bytes, str]:
    """Generate an image by calling `codex exec`.

    Runs the bundled codex_image.py script through Codex's own exec mechanism,
    which uses the user's logged-in Codex credentials automatically.

    Args:
        prompt:     Image description
        model:      Model name
        size:       Dimensions
        quality:    Quality tier
        fmt:        Output format
        codex_cmd:  Command template. {CMD} is replaced with the inner script call.
                    Default: "codex exec -- sh -c {CMD}"

    Returns:
        (raw_image_bytes, path_to_output_file)
    """
    cmd = (codex_cmd or "").strip()
    if not cmd:
        raise ValueError("codex_cmd cannot be empty")

    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
        tmp_out = f.name

    script_path = os.environ.get(
        "CODEX_IMAGE_SCRIPT",
        str(Path(DEFAULT_CODEX_SCRIPT).expanduser()),
    )

    script_cmd = (
        f"python {script_path} "
        f"{prompt!r} "
        f"--size {size} "
        f"--quality {quality} "
        f"--format {fmt} "
        f"--out {tmp_out}"
    )
    if model:
        script_cmd += f" --model {model}"

    if "{CMD}" in cmd:
        full_cmd_str = cmd.replace("{CMD}", script_cmd)
    else:
        full_cmd_str = f"{cmd} {script_cmd!r}"

    exec_parts = [p.strip() for p in full_cmd_str.split()]

    env = dict(os.environ)
    result = subprocess.run(
        exec_parts,
        capture_output=True,
        text=True,
        timeout=DEFAULT_TIMEOUT,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"codex exec failed (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    # codex_image.py prints the output path to stdout (last non-log line)
    output_lines = [
        l for l in result.stdout.strip().splitlines()
        if l and not l.startswith("[codex-image]")
    ]
    img_path = output_lines[-1].strip() if output_lines else tmp_out

    if not Path(img_path).exists():
        raise FileNotFoundError(
            f"codex exec did not produce an output file. stdout: {result.stdout}"
        )

    img_bytes = Path(img_path).read_bytes()
    return img_bytes, img_path


# ── Public API ───────────────────────────────────────────────────────────────

def generate_image(
    prompt: str,
    model: str = "",
    size: str = DEFAULT_SIZE,
    quality: str = DEFAULT_QUALITY,
    fmt: str = DEFAULT_FORMAT,
    mode: Literal["api", "auth", "cli", "openrouter", "litellm"] = "auth",
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "",
    codex_cmd: str = "codex exec -- sh -c {CMD}",
    input_image_urls: list[str] | None = None,
    mask_image_url: str | None = None,
    action: str = "auto",
    background: str = "opaque",
) -> tuple[bytes, str]:
    """Generate an image.

    Args:
        prompt:    Image description (required)
        model:     Model name. Empty uses the default for the selected mode.
        size:      Dimensions (default: "1024x1024")
        quality:   "low" | "medium" | "high" (default: "medium")
        fmt:       "png" | "jpeg" | "webp" (default: "png")
        mode:      "api" (user URL+key) | "auth" (auto from ~/.codex/auth.json)
                   | "cli" (codex exec) | "openrouter" | "litellm"
        base_url:  API base URL (mode "api" only)
        api_key:   Bearer token (mode "api": required; mode "auth": ignored)
        codex_cmd: codex exec command template (mode "cli" only, {CMD} is replaced)

    Returns:
        (raw_image_bytes, path_to_file)
    """
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")

    if mode == "cli":
        if input_image_urls:
            raise ValueError("image input is not supported in cli mode")
        return _generate_cli(
            prompt=prompt,
            model=(model or "").strip() or DEFAULT_MODEL,
            size=size,
            quality=quality,
            fmt=fmt,
            codex_cmd=codex_cmd,
        )
    if mode == "openrouter":
        provider_base_url = "" if base_url == DEFAULT_BASE_URL else base_url
        provider_model = (model or "").strip() or DEFAULT_OPENROUTER_MODEL
        return _generate_openrouter(
            prompt=prompt,
            model=provider_model,
            size=size,
            quality=quality,
            fmt=fmt,
            base_url=provider_base_url,
            api_key=api_key,
            input_image_urls=input_image_urls,
            background=background,
        )
    if mode == "litellm":
        provider_base_url = "" if base_url == DEFAULT_BASE_URL else base_url
        provider_model = (model or "").strip() or DEFAULT_LITELLM_MODEL
        return _generate_litellm(
            prompt=prompt,
            model=provider_model,
            size=size,
            quality=quality,
            fmt=fmt,
            base_url=provider_base_url,
            api_key=api_key,
            input_image_urls=input_image_urls,
            mask_image_url=mask_image_url,
        )
    else:
        # "api" or "auth": use direct HTTP
        resolved_key = "" if mode == "auth" else api_key
        return _generate_api(
            prompt=prompt,
            model=(model or "").strip() or DEFAULT_MODEL,
            size=size,
            quality=quality,
            fmt=fmt,
            base_url=base_url,
            api_key=resolved_key,
            input_image_urls=input_image_urls,
            action=action,
        )
