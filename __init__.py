from .codex_image_node import CodexImageNode

NODE_CLASS_MAPPINGS = {
    "CodexImageNode": CodexImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CodexImageNode": "Codex Image (GPT Image 2)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
