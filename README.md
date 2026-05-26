# ComfyUI CodexImage

Custom ComfyUI node for image generation via GPT Image 2 (gpt-5.5).

> **Chinese version**: [README_CN.md](README_CN.md)

## Quick Start

```bash
# ComfyUI plugin
cp -r . <ComfyUI>/custom_nodes/ComfyUI-CodexImage/
# Restart ComfyUI, search for "CodexImage" in the node browser

# Standalone CLI (no ComfyUI needed)
python cli.py "a cute cat"
```

## Three Modes

| Mode | `mode` value | Auth |
|------|-------------|------|
| **API** | `api` | User provides `base_url` + `api_key` |
| **Codex Auth** | `auth` | Auto-reads `~/.codex/auth.json` (OAuth token or `OPENAI_API_KEY` field) |
| **CLI** | `cli` | Calls `codex exec` — uses locally logged-in Codex credentials |

## CLI Usage

```bash
# Auth mode (default — reads ~/.codex/auth.json)
python cli.py "a cat"

# API mode (user-provided URL + key)
python cli.py "a cat" --mode api \
  --base-url https://chatgpt.com/backend-api/codex \
  --api-key sk-xxxx

# CLI mode (codex exec)
python cli.py "a cat" --mode cli
```

## Files

| File | Purpose | Dependencies |
|------|---------|--------------|
| `generator.py` | Core generation logic (HTTP/SSE/Auth) | None (stdlib only) |
| `codex_image_node.py` | ComfyUI node class + tensor conversion | torch, numpy, Pillow |
| `cli.py` | Standalone CLI entry point | None |

## Workflow

```
[CLIP Text Encode] → [CodexImageNode] → [PreviewImage / SaveImage]
```

## Node Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `prompt` | — | Image description (required) |
| `model` | `gpt-5.5` | Model name |
| `size` | `1024x1024` | `1024x1024` / `1792x1024` / `1024x1792` |
| `quality` | `medium` | `low` / `medium` / `high` |
| `format` | `png` | `png` / `jpeg` / `webp` |
| `output_path` | _(empty)_ | Optional — save a copy to this path |

**Hidden params:**

| Param | Default | Mode |
|-------|---------|------|
| `base_url` | `https://chatgpt.com/backend-api/codex` | `api` |
| `api_key` | _(empty)_ | `api` |
| `codex_cmd` | `codex exec -- sh -c {CMD}` | `cli` |

## Implementation

### Auth priority (auth mode)

1. `OPENAI_API_KEY` env var
2. `OPENAI_API_KEY` field in `~/.codex/auth.json`
3. ChatGPT OAuth `access_token` in `~/.codex/auth.json` (from `codex login`)

### API request flow

1. Build JSON payload with `model`, `instructions`, `input`, `tools` (image_generation)
2. POST to `/{base_url}/v1/responses` with `Accept: text/event-stream`
3. Read SSE stream chunk by chunk, parse `data: {...}` lines
4. Extract base64 image from `response.image_generation_call.done` event
5. Decode base64 → raw image bytes
