# ComfyUI CodexImage

> **Chinese version**: [README_CN.md](README_CN.md)

## Overview

A ComfyUI custom node + standalone CLI for generating images via **GPT Image 2 (gpt-5.5)**. It wraps the ChatGPT Responses API with the built-in `image_generation` tool.

The key design goal: **reuse your existing Codex/ChatGPT authentication** — no new API key required if you already have a logged-in Codex session.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    codex_image_node.py                  │
│         ComfyUI node class + tensor conversion          │
│         (torch, numpy, PIL — ComfyUI environment)      │
└──────────────────────┬──────────────────────────────────┘
                       │ imports
                       ▼
┌─────────────────────────────────────────────────────────┐
│                      generator.py                       │
│     Core logic: HTTP/SSE/Auth — zero dependencies     │
│              (pure Python stdlib only)                 │
└──────────────────────┬──────────────────────────────────┘
                       │ imports
                       ▼
┌─────────────────────────────────────────────────────────┐
│                        cli.py                          │
│         Standalone CLI entry point — no deps           │
│         (pure Python stdlib, no torch needed)         │
└─────────────────────────────────────────────────────────┘
```

- **`generator.py`**: No third-party imports. Does HTTP POST, SSE streaming, auth resolution, SSE event parsing, base64 decoding.
- **`codex_image_node.py`**: Imports `generator.py`. Adds ComfyUI tensor conversion (`[B, H, W, C]` float32 in `[0,1]`) and the `CodexImageNode` class.
- **`cli.py`**: Imports `generator.py` directly. Provides a plain CLI without any ComfyUI dependency.

---

## Three Modes

| Mode | How it works |
|------|-------------|
| `api` | Direct HTTP POST to a user-provided `base_url` + `api_key` |
| `auth` | Same HTTP POST, but credentials are auto-loaded from `~/.codex/auth.json` (zero config needed) |
| `cli` | Spawns `codex exec` as a subprocess; the codex CLI handles auth internally |

## Additional Provider Nodes

This package also provides separate provider nodes:

| Node | API target | API key |
|------|------------|---------|
| `OpenRouter Image (GPT Image 2)` | OpenRouter dedicated Images API | `CODEX_IMAGE_OPENROUTER_API_KEY` or `OPENROUTER_API_KEY` |
| `Mix Codex Copycat Image I2I (GPT Image 2)` | Codex-style I2I packing, then Responses API `image_generation` tool call through OpenRouter or LiteLLM selected by `mode` | OpenRouter or LiteLLM environment variables |
| `LiteLLM Image (GPT Image 2)` | LiteLLM OpenAI-compatible `/v1/images/*` proxy | `CODEX_IMAGE_LITELLM_API_KEY`, `LITELLM_API_KEY`, or `LITELLM_MASTER_KEY` |

Provider nodes let you enter the `model` name directly in the node. `base_url` is not shown in the UI; it defaults from environment variables:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export CODEX_IMAGE_OPENROUTER_BASE_URL="https://openrouter.ai/api/v1/images"
export CODEX_IMAGE_OPENROUTER_MODEL="openai/gpt-image-2"

export LITELLM_API_KEY="sk-..."
export CODEX_IMAGE_LITELLM_BASE_URL="http://localhost:4000"
export CODEX_IMAGE_LITELLM_MODEL="gpt-image-2"
```

Provider nodes send the `model` string exactly as configured in the node or the matching environment default. Use the model alias exposed by your provider, for example `openai/gpt-image-2`, `gpt-image-2`, `openrouter/gpt-image-2`, or a Vertex/Gemini alias depending on your proxy's configuration.

For Docker deployments, set these variables before starting ComfyUI. A later
`docker exec` shell can show variables that the already-running ComfyUI process
does not have. To check the process environment without printing the full key:

```bash
pid=$(pgrep -f "python.*main.py" | head -1)
tr '\0' '\n' < /proc/$pid/environ | grep -E '^(OPENROUTER_API_KEY|CODEX_IMAGE_OPENROUTER_API_KEY)=' | wc -l
tr '\0' '\n' < /proc/$pid/environ | sed -n -E 's/^(OPENROUTER_API_KEY|CODEX_IMAGE_OPENROUTER_API_KEY)=//p' | head -1 | awk '{print length($0), substr($0,1,8)}'
```

The provider nodes support prompt-only generation. If you connect `image`, `image_2`, or `mask`, they send the request as an image edit/reference-image request when the provider endpoint supports it. The prompt text is sent as entered; size, quality, and mask are represented through API fields or image alpha/multipart data instead of prompt text.

`Mix Codex Copycat Image I2I (GPT Image 2)` is image-input-only and mirrors `Codex Image I2I (GPT Image 2)` reference packing: the main image is always sent, `image_2` is a second reference, and a ComfyUI `mask` is baked into the first image alpha channel. Its `mode` selects `openrouter` or `litellm`, using the same environment-variable API keys/base URLs as the dedicated provider nodes. Unlike the dedicated provider nodes, this node posts a Responses API payload with `tools: [{"type": "image_generation", ...}]` to the provider's `/responses` endpoint.

---

## Implementation Principles

### 1. API Request Flow (modes: `api`, `auth`)

```
User prompt
    │
    ▼
_build_payload()
    │ builds JSON body:
    │ {
    │   "model": "gpt-5.5",
    │   "instructions": "Generate the requested image...",
    │   "input": [{"role": "user", "content": prompt}],
    │   "tools": [{"type": "image_generation", "size": "...", "quality": "..."}],
    │   "stream": true
    │ }
    ▼
urllib.request.Request
    │ POST with headers:
    │   Authorization: Bearer <token>
    │   Accept: text/event-stream
    ▼
POST to {base_url}/responses
    │
    ▼
SSE stream (chunked transfer)
    │ response arrives as repeated "data: {...}" lines
    │
    ▼
_post_streaming() — SSE parsing
    │ reads 4 KB chunks, accumulates in line buffer,
    │ splits on "\n", strips "data: " prefix,
    │ parses each as JSON → list of event dicts
    │
    ▼
_extract_image() — find the image event
    │ looks for:
    │   ev["type"] == "response.image_generation_call.done"
    │   ev["result"]  ← base64-encoded image string
    │
    ▼
base64.b64decode(img_b64) → raw image bytes
    │
    ▼
_bytes_to_tensor() → ComfyUI IMAGE tensor [1, H, W, C] float32 [0,1]
```

### 2. Auth Resolution (`_resolve_api_key`)

```
api_key provided by user?
    │
    ├─ YES → use it directly as Bearer token
    │
    └─ NO (auth mode):
           │
           ▼
       1. OPENAI_API_KEY env var
           │ found? → use it
           └─ not found
              │
              ▼
           2. ~/.codex/auth.json → "OPENAI_API_KEY" field
              │ found? → use it
              └─ not found
                 │
                 ▼
              3. ~/.codex/auth.json → "tokens.access_token"
                 │ (ChatGPT OAuth token from `codex login`)
                 │ found? → use it
                 └─ not found → raise ValueError
```

### 3. CLI Mode (`_generate_cli`)

```
Build inner script command:
  python <script_path> <prompt> --size X --quality X --format X --out /tmp/xxx.png

Wrap with codex exec template:
  codex exec -- sh -c "python <script> ..."

subprocess.run() → captures stdout
  codex_image.py prints the output path as last stdout line:
    /tmp/xxx.png

Read the file at that path → raw bytes
```

### 4. Tensor Format

ComfyUI IMAGE tensors follow `[B, H, W, C]` with float32 in `[0, 1]`:

```
PIL.Image.open(bytes)  →  RGB PIL image
numpy.array(pil)       →  [H, W, C] uint8 [0, 255]
astype(np.float32)/255.0 →  [H, W, C] float32 [0, 1]
torch.from_numpy()[None, ] →  [1, H, W, C]
to(dtype=torch.float32)   →  final tensor
```

---

## File Manifest

| File | Purpose | Dependencies |
|------|---------|-------------|
| `generator.py` | Core logic: payload building, HTTP POST, SSE parsing, auth, base64 decode | None (stdlib) |
| `codex_image_node.py` | ComfyUI node + tensor conversion | torch, numpy, Pillow |
| `cli.py` | Standalone CLI | None |

---

## Environment Variables

All read at import time:

| Variable | Default | Description |
|---------|---------|-------------|
| `CODEX_IMAGE_BASE_URL` | `https://chatgpt.com/backend-api/codex` | API endpoint |
| `CODEX_IMAGE_MODEL` | `gpt-5.5` | Model name |
| `CODEX_IMAGE_SIZE` | `1024x1024` | Default size |
| `CODEX_IMAGE_QUALITY` | `medium` | Default quality |
| `CODEX_IMAGE_FORMAT` | `png` | Default format |
| `CODEX_IMAGE_SCRIPT` | `~/.codex-image/scripts/codex_image.py` | CLI mode script path |
| `OPENAI_API_KEY` | _(empty)_ | Overrides all auth (highest priority) |
| `CODEX_HOME` | `~/.codex` | Codex auth directory |
| `CODEX_IMAGE_OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1/images` | OpenRouter Images API endpoint |
| `CODEX_IMAGE_OPENROUTER_MODEL` | `openai/gpt-image-2` | Default model for the OpenRouter node |
| `CODEX_IMAGE_OPENROUTER_API_KEY` / `OPENROUTER_API_KEY` | _(empty)_ | OpenRouter API key |
| `CODEX_IMAGE_LITELLM_BASE_URL` | `http://localhost:4000` | LiteLLM proxy base URL or images endpoint |
| `CODEX_IMAGE_LITELLM_MODEL` | `gpt-image-2` | Default model for the LiteLLM node |
| `CODEX_IMAGE_LITELLM_API_KEY` / `LITELLM_API_KEY` / `LITELLM_MASTER_KEY` | _(empty)_ | LiteLLM proxy API key |

---

## Node Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `mode` | `auth` | `api` / `auth` / `cli` |
| `prompt` | — | Image description (required) |
| `model` | mode-specific | Model name. Empty CLI value uses the selected mode's default. |
| `size` | `1024x1024` | `1024x1024` / `1536x1024` / `1024x1536` / `1792x1024` / `1024x1792` / `1920x1080` / `1080x1920` / `2048x2048` / `3840x2160` / `2160x3840` |
| `quality` | `medium` | `low` / `medium` / `high` |
| `format` | `png` | `png` / `jpeg` / `webp`; passed to providers that support it. Saved output paths use the actual returned image type when it can be detected. |
| `output_path` | _(empty)_ | Save a copy to this path |

**Hidden params:**

| Param | Default | Mode |
|-------|---------|------|
| `base_url` | `https://chatgpt.com/backend-api/codex` | `api` |
| `api_key` | _(empty)_ | `api` |
| `codex_cmd` | `codex exec -- sh -c {CMD}` | `cli` |

---

## CLI Usage

```bash
# Auth mode (default — reads ~/.codex/auth.json)
python cli.py "a cat"

# API mode (user-provided URL + key)
python cli.py "a cat" --mode api \
  --base-url https://chatgpt.com/backend-api/codex \
  --api-key sk-xxxx

# CLI mode (via codex exec)
python cli.py "a cat" --mode cli

# OpenRouter mode (uses OPENROUTER_API_KEY)
python cli.py "a cat" --mode openrouter --model openai/gpt-image-2

# LiteLLM mode (uses LITELLM_API_KEY)
python cli.py "a cat" --mode litellm --model gpt-image-2

# Specify output path
python cli.py "a cat" --out ./output.png
```

## Workflow

```
[CLIP Text Encode] → [CodexImageNode] → [PreviewImage / SaveImage]
```
