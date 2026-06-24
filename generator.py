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

# Regex for validating arbitrary size strings (kept for reference)
SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*[xX×]\s*(\d+)\s*$")

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
    actual_model = DEFAULT_MODEL if model.startswith("gpt-image") else model
    input_image_urls = input_image_urls or []

    dim = SIZE_PATTERN.match(size)
    if dim:
        w, h = int(dim.group(1)), int(dim.group(2))
        orient = "square" if w == h else ("landscape" if w > h else "portrait")
        prompt = f"{prompt}\n\nFinal output: {w}x{h} pixel {orient} canvas."

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

    ext = f".{fmt}"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(img_bytes)
        tmp_path = f.name

    return img_bytes, tmp_path


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
    model: str = DEFAULT_MODEL,
    size: str = DEFAULT_SIZE,
    quality: str = DEFAULT_QUALITY,
    fmt: str = DEFAULT_FORMAT,
    mode: Literal["api", "auth", "cli"] = "auth",
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "",
    codex_cmd: str = "codex exec -- sh -c {CMD}",
    input_image_urls: list[str] | None = None,
    action: str = "auto",
) -> tuple[bytes, str]:
    """Generate an image.

    Args:
        prompt:    Image description (required)
        model:     Model name (default: "gpt-5.5")
        size:      Dimensions (default: "1024x1024")
        quality:   "low" | "medium" | "high" (default: "medium")
        fmt:       "png" | "jpeg" | "webp" (default: "png")
        mode:      "api" (user URL+key) | "auth" (auto from ~/.codex/auth.json) | "cli" (codex exec)
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
            raise ValueError("image input is only supported in api/auth mode")
        return _generate_cli(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            fmt=fmt,
            codex_cmd=codex_cmd,
        )
    else:
        # "api" or "auth": use direct HTTP
        resolved_key = "" if mode == "auth" else api_key
        return _generate_api(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            fmt=fmt,
            base_url=base_url,
            api_key=resolved_key,
            input_image_urls=input_image_urls,
            action=action,
        )
