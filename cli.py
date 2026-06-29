#!/usr/bin/env python3
"""Standalone CLI for CodexImage — pure Python, no third-party dependencies.

Requires only Python 3 standard library.

Usage:
    python cli.py "a cute cat" --size 1024x1024
    python cli.py "a cat" --mode api --api-key sk-xxx --base-url https://chatgpt.com/backend-api/codex
    python cli.py "a cat" --mode cli
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def main() -> int:
    p = argparse.ArgumentParser(description="CodexImage standalone generator")
    p.add_argument("prompt", help="Image description")
    p.add_argument("--model", default="")
    p.add_argument("--size", default=DEFAULT_SIZE, choices=SUPPORTED_SIZES)
    p.add_argument("--quality", default=DEFAULT_QUALITY, choices=["low", "medium", "high"])
    p.add_argument("--format", default=DEFAULT_FORMAT, choices=["png", "jpeg", "webp"])
    p.add_argument("--out", default="", help="Output file path")
    p.add_argument(
        "--mode", default="auth", choices=["api", "auth", "cli", "openrouter", "litellm"],
        help=(
            "api: URL+key | auth: auto ~/.codex/auth.json | cli: codex exec | "
            "openrouter: OPENROUTER_API_KEY | litellm: LITELLM_API_KEY"
        )
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
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
