from .codex_image_node import CodexImageI2INode, CodexImageNode

NODE_CLASS_MAPPINGS = {
    "CodexImageNode": CodexImageNode,
    "CodexImageI2INode": CodexImageI2INode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CodexImageNode": "Codex Image (GPT Image 2)",
    "CodexImageI2INode": "Codex Image I2I (GPT Image 2)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
