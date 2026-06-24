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
import base64
import sys
from io import BytesIO
from pathlib import Path

# Add this directory to path so generator.py is importable
sys.path.insert(0, str(Path(__file__).parent))

from generator import (
    DEFAULT_MODEL,
    DEFAULT_SIZE,
    DEFAULT_QUALITY,
    DEFAULT_FORMAT,
    DEFAULT_BASE_URL,
    SUPPORTED_SIZES,
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

def _image_bytes_to_tensor(img_bytes_or_path) -> "torch.Tensor":
    """Convert raw image bytes or file path to a ComfyUI IMAGE tensor [B, H, W, C] float32 in [0,1]."""
    if not _HAS_COMFYU:
        raise RuntimeError("ComfyUI dependencies (torch, numpy, PIL) not available.")
    from io import BytesIO
    if isinstance(img_bytes_or_path, bytes):
        pil = Image.open(BytesIO(img_bytes_or_path)).convert("RGB")
    else:
        pil = Image.open(img_bytes_or_path).convert("RGB")
    np_img = np.array(pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_img)[None,]   # [1, H, W, C]
    # Use intermediate_dtype if available (some ComfyUI versions), otherwise fall back to float32
    dtype_fn = getattr(comfy.model_management, "intermediate_dtype", None)
    tensor = tensor.to(dtype=dtype_fn() if dtype_fn else torch.float32)
    return tensor


def _pil_to_png_data_url(pil: "Image.Image") -> str:
    buffer = BytesIO()
    pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_tensor_to_pil_rgb(image: "torch.Tensor") -> "Image.Image":
    """Convert a ComfyUI IMAGE tensor to an RGB PIL image."""
    if not _HAS_COMFYU:
        raise RuntimeError("ComfyUI dependencies (torch, numpy, PIL) not available.")
    if image is None:
        raise ValueError("image input is required")

    img_tensor = image[0] if len(image.shape) == 4 else image
    img_np = img_tensor.detach().cpu().numpy()
    img_np = np.clip(img_np, 0.0, 1.0)
    img_np = (img_np * 255.0).round().astype(np.uint8)

    if img_np.ndim == 2:
        pil = Image.fromarray(img_np, mode="L").convert("RGB")
    elif img_np.shape[-1] == 1:
        pil = Image.fromarray(img_np[..., 0], mode="L").convert("RGB")
    else:
        pil = Image.fromarray(img_np[..., :3]).convert("RGB")

    return pil


def _image_tensor_to_data_url(image: "torch.Tensor") -> str:
    """Convert a ComfyUI IMAGE tensor to a PNG data URL for Responses input_image."""
    return _pil_to_png_data_url(_image_tensor_to_pil_rgb(image))


def _mask_tensor_to_pil_l(mask: "torch.Tensor", size: tuple[int, int]) -> "Image.Image":
    """Convert a ComfyUI MASK tensor to an L image resized to the target image size."""
    if not _HAS_COMFYU:
        raise RuntimeError("ComfyUI dependencies (torch, numpy, PIL) not available.")
    if mask is None:
        raise ValueError("mask input is required")

    mask_tensor = mask[0] if len(mask.shape) in (3, 4) else mask
    mask_np = mask_tensor.detach().cpu().numpy()
    mask_np = np.squeeze(mask_np)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    if mask_np.ndim != 2:
        raise ValueError(f"mask must be 2D after squeezing, got shape {mask_np.shape}")

    mask_np = np.clip(mask_np, 0.0, 1.0)
    mask_np = (mask_np * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(mask_np, mode="L")
    if pil.size != size:
        resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
        pil = pil.resize(size, resampling)
    return pil


def _image_tensor_and_mask_to_data_url(image: "torch.Tensor", mask: "torch.Tensor") -> str:
    """Convert an image plus ComfyUI MASK to RGBA PNG.

    ComfyUI convention is used: white mask pixels are edited, black pixels are
    preserved. The image API convention is transparent pixels are edited, so the
    mask is inverted into the alpha channel.
    """
    pil = _image_tensor_to_pil_rgb(image).convert("RGBA")
    mask_l = _mask_tensor_to_pil_l(mask, pil.size)
    alpha = Image.eval(mask_l, lambda px: 255 - px)
    pil.putalpha(alpha)
    return _pil_to_png_data_url(pil)


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
                "size": (list(SUPPORTED_SIZES), {"default": DEFAULT_SIZE, "label": "size"}),
                "quality": (["low", "medium", "high"], {"default": DEFAULT_QUALITY}),
                "format": (["png", "jpeg", "webp"], {"default": DEFAULT_FORMAT}),
            },
            "optional": {
                "output_path": ("STRING", {"default": "", "label": "output_path"}),
            },
            "hidden": {
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
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


class CodexImageI2INode:
    """Generate or edit an image using one or two IMAGE tensors as visual context."""

    CATEGORY = "image/generation"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["api", "auth"], {"default": "auth", "label": "mode"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": ("STRING", {"default": DEFAULT_MODEL}),
                "size": (list(SUPPORTED_SIZES), {"default": DEFAULT_SIZE, "label": "size"}),
                "quality": (["low", "medium", "high"], {"default": DEFAULT_QUALITY}),
                "format": (["png", "jpeg", "webp"], {"default": DEFAULT_FORMAT}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "mask": ("MASK",),
                "output_path": ("STRING", {"default": "", "label": "output_path"}),
            },
            "hidden": {
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    def generate(
        self,
        image,
        mode: str,
        prompt: str,
        model: str,
        size: str,
        quality: str,
        format: str,
        output_path: str = "",
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        image_2=None,
        mask=None,
    ) -> tuple:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        if not _HAS_COMFYU:
            raise RuntimeError("ComfyUI dependencies not available.")

        edit_prompt = prompt
        if mask is not None:
            input_image_urls = [_image_tensor_and_mask_to_data_url(image, mask)]
            edit_prompt = (
                f"{prompt}\n\n"
                "Mask guidance: the first input image contains transparency derived "
                "from the ComfyUI mask. Edit only the transparent/white-masked area "
                "and preserve the opaque/black-masked area as much as possible."
            )
        else:
            input_image_urls = [_image_tensor_to_data_url(image)]
        if image_2 is not None:
            input_image_urls.append(_image_tensor_to_data_url(image_2))

        img_bytes, img_path = generate_image(
            prompt=edit_prompt,
            model=model,
            size=size,
            quality=quality,
            fmt=format,
            mode=mode,
            base_url=base_url,
            api_key=api_key,
            input_image_urls=input_image_urls,
            action="edit",
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
