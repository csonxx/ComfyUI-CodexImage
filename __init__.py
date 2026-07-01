from .codex_image_node import (
    CodexImageI2INode,
    CodexImageNode,
    LiteLLMImageNode,
    MixCodexCopycatImageI2INode,
    OpenRouterImageNode,
)

NODE_CLASS_MAPPINGS = {
    "CodexImageNode": CodexImageNode,
    "CodexImageI2INode": CodexImageI2INode,
    "OpenRouterImageNode": OpenRouterImageNode,
    "MixCodexCopycatImageI2INode": MixCodexCopycatImageI2INode,
    "LiteLLMImageNode": LiteLLMImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CodexImageNode": "Codex Image (GPT Image 2)",
    "CodexImageI2INode": "Codex Image I2I (GPT Image 2)",
    "OpenRouterImageNode": "OpenRouter Image (GPT Image 2)",
    "MixCodexCopycatImageI2INode": "Mix Codex Copycat Image I2I (GPT Image 2)",
    "LiteLLMImageNode": "LiteLLM Image (GPT Image 2)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
