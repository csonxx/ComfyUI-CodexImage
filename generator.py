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
from pathlib import Path
from typing import Any, Literal

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_SIZE = "1024x1024"
DEFAULT_QUALITY = "medium"
DEFAULT_FORMAT = "png"
DEFAULT_TIMEOUT = 600
DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_SCRIPT = "~/.codex-image/scripts/codex_image.py"

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
        raise FileNotFoundError(
            f"auth.json not found at {auth_path}. "
            "Run `codex login` or set OPENAI_API_KEY environment variable."
        )
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read {auth_path}: {exc}") from exc

    field_key = (data.get("OPENAI_API_KEY") or "").strip()
    if field_key:
        return field_key

    if data.get("auth_mode") == "chatgpt":
        tokens = data.get("tokens") or {}
        at = (tokens.get("access_token") or "").strip()
        if at:
            return at

    raise ValueError(
        "No credentials found in ~/.codex/auth.json. "
        "Run `codex login` or set OPENAI_API_KEY in auth.json."
    )


def _resolve_api_key(api_key: str) -> str:
    if api_key and api_key.strip():
        return api_key.strip()
    return _load_auth_from_codex_home()


# ── HTTP ──────────────────────────────────────────────────────────────────────

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
        raise RuntimeError(f"Request failed: status={exc.code}\n{body_text}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc

    return events


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

    tail = events[-3:] if len(events) > 3 else events
    raise RuntimeError(
        f"No generated image found in SSE events:\n"
        f"{json.dumps(tail, ensure_ascii=False, indent=2)[:2000]}"
    )


# ── Payload ───────────────────────────────────────────────────────────────────

def _build_payload(prompt: str, model: str, size: str, quality: str) -> dict[str, Any]:
    """Build the request body for the Codex Responses API."""
    actual_model = DEFAULT_MODEL if model.startswith("gpt-image") else model

    dim = SIZE_PATTERN.match(size)
    if dim:
        w, h = int(dim.group(1)), int(dim.group(2))
        orient = "square" if w == h else ("landscape" if w > h else "portrait")
        prompt = f"{prompt}\n\nFinal output: {w}x{h} pixel {orient} canvas."

    return {
        "model": actual_model,
        "instructions": "Generate the requested image using the image_generation tool.",
        "input": [{"role": "user", "content": prompt}],
        "tools": [{"type": "image_generation", "size": size, "quality": quality}],
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
    base_url = base_url.rstrip("/")
    if "backend-api/codex" in base_url:
        api_url = f"{base_url}/responses"
    elif base_url.endswith("/v1"):
        api_url = f"{base_url}/responses"
    else:
        api_url = f"{base_url}/v1/responses"

    token = _resolve_api_key(api_key)
    payload = _build_payload(prompt, model, size, quality)
    events = _post_streaming(api_url, token, payload, DEFAULT_TIMEOUT)
    img_b64 = _extract_image(events)
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
        )
