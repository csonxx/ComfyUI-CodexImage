"""ComfyUI custom node for Codex Image generation (gpt-5.5 / GPT Image 2).

Core generation logic lives in generator.py (no third-party dependencies).
This file adds the ComfyUI integration: tensor conversion and the node class.

Usage:
  - ComfyUI: copy folder to <ComfyUI>/custom_nodes/, restart
  - CLI:     python cli.py "a cute cat" --size 1024x1024
             (cli.py uses generator.py directly — no torch or ComfyUI needed)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add this directory to path so generator.py is importable
sys.path.insert(0, str(Path(__file__).parent))

from generator import (
    DEFAULT_MODEL,
    DEFAULT_SIZE,
    DEFAULT_QUALITY,
    DEFAULT_FORMAT,
    DEFAULT_BASE_URL,
    generate_image,
)

# ComfyUI-only imports — not available in standalone CLI
try:
    import numpy as np
    import torch
    import comfy.model_management
    from PIL import Image
    _HAS_COMFYU = True
except ImportError:
    _HAS_COMFYU = False


# ── Tensor conversion (ComfyUI only) ─────────────────────────────────────────

def _image_bytes_to_tensor(img_bytes: bytes) -> "torch.Tensor":
    """Convert raw image bytes to a ComfyUI IMAGE tensor [B, H, W, C] float32 in [0,1]."""
    if not _HAS_COMFYU:
        raise RuntimeError("ComfyUI dependencies (torch, numpy, PIL) not available.")
    pil = Image.open(img_bytes if isinstance(img_bytes, bytes) else open(img_bytes, "rb")).convert("RGB")
    np_img = np.array(pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_img)[None,]   # [1, H, W, C]
    tensor = tensor.to(dtype=comfy.model_management.intermediate_dtype())
    return tensor


# ── ComfyUI Node ─────────────────────────────────────────────────────────────

class CodexImageNode:
    """Generate images using GPT Image 2.

    Three modes:
      - "api":  call the Codex Responses REST API directly (base_url + api_key)
      - "auth": same API call, but api_key is auto-loaded from ~/.codex/auth.json
      - "cli":  call `codex exec` which pipes through your local Codex CLI login

    Outputs:
      - image:      ComfyUI IMAGE tensor [B, H, W, C] float32 in [0, 1]
      - image_path: File path where the image was saved
    """

    CATEGORY = "image/generation"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["api", "auth", "cli"], {"default": "auth", "label": "mode"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": ("STRING", {"default": DEFAULT_MODEL}),
                "size": (
                    ["1024x1024", "1792x1024", "1024x1792"],
                    {"default": DEFAULT_SIZE, "label": "size"},
                ),
                "quality": (["low", "medium", "high"], {"default": DEFAULT_QUALITY}),
                "format": (["png", "jpeg", "webp"], {"default": DEFAULT_FORMAT}),
            },
            "optional": {
                "output_path": ("STRING", {"default": "", "label": "output_path"}),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL, "label": "base_url"}),
            },
            "hidden": {
                "api_key": ("STRING", {"default": ""}),
                "codex_cmd": ("STRING", {"default": "codex exec -- sh -c {CMD}"}),
            },
        }

    def generate(
        self,
        mode: str,
        prompt: str,
        model: str,
        size: str,
        quality: str,
        format: str,
        output_path: str = "",
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        codex_cmd: str = "codex exec -- sh -c {CMD}",
    ) -> tuple:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        if not _HAS_COMFYU:
            raise RuntimeError("ComfyUI dependencies not available. Use cli.py instead.")

        img_bytes, img_path = generate_image(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            fmt=format,
            mode=mode,
            base_url=base_url,
            api_key=api_key,
            codex_cmd=codex_cmd,
        )

        tensor = _image_bytes_to_tensor(img_bytes)

        if output_path:
            out_dir = Path(output_path).expanduser()
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            out_path = out_dir.with_suffix(f".{format}")
            out_path.write_bytes(img_bytes)
            img_path = str(out_path)

        return (tensor, img_path)


# ── Standalone CLI (uses generator.py directly — no torch needed) ─────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CodexImage standalone generator")
    p.add_argument("prompt", help="Image description")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--size", default=DEFAULT_SIZE)
    p.add_argument("--quality", default=DEFAULT_QUALITY, choices=["low", "medium", "high"])
    p.add_argument("--format", default=DEFAULT_FORMAT, choices=["png", "jpeg", "webp"])
    p.add_argument("--out", default="", help="Output file path")
    p.add_argument(
        "--mode", default="auth", choices=["api", "auth", "cli"],
        help="api: URL+key | auth: auto ~/.codex/auth.json | cli: codex exec"
    )
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default="")
    p.add_argument("--codex-cmd", default="codex exec -- sh -c {CMD}")

    args = p.parse_args()

    try:
        img_bytes, path = generate_image(
            prompt=args.prompt,
            model=args.model,
            size=args.size,
            quality=args.quality,
            fmt=args.format,
            mode=args.mode,
            base_url=args.base_url,
            api_key=args.api_key,
            codex_cmd=args.codex_cmd,
        )
        if args.out:
            out_path = Path(args.out).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(img_bytes)
            print(f"Saved to: {out_path}")
        else:
            print(f"Saved to: {path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
