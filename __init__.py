from .codex_image_node import (
    CodexImageI2INode,
    CodexImageNode,
    GPTImage2ResponseI2INode,
    LiteLLMImageNode,
    MixCodexCopycatImageI2INode,
    OpenRouterImageNode,
    RequestyImageEditI2INode,
    RequestyResponseI2INode,
    WaveSpeedImageEditI2INode,
)

NODE_CLASS_MAPPINGS = {
    "CodexImageNode": CodexImageNode,
    "CodexImageI2INode": CodexImageI2INode,
    "OpenRouterImageNode": OpenRouterImageNode,
    "MixCodexCopycatImageI2INode": MixCodexCopycatImageI2INode,
    "GPTImage2ResponseI2INode": GPTImage2ResponseI2INode,
    "RequestyImageEditI2INode": RequestyImageEditI2INode,
    "RequestyResponseI2INode": RequestyResponseI2INode,
    "WaveSpeedImageEditI2INode": WaveSpeedImageEditI2INode,
    "LiteLLMImageNode": LiteLLMImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CodexImageNode": "Codex Image (GPT Image 2)",
    "CodexImageI2INode": "Codex Image I2I (GPT Image 2)",
    "OpenRouterImageNode": "OpenRouter Image (GPT Image 2)",
    "MixCodexCopycatImageI2INode": "Mix Codex Copycat Image I2I (GPT Image 2)",
    "GPTImage2ResponseI2INode": "GPT-Image-2 Response i2i",
    "RequestyImageEditI2INode": "Requesty I2I (gpt-image-2 edit)",
    "RequestyResponseI2INode": "Requesty Response I2I (GPT Image 2)",
    "WaveSpeedImageEditI2INode": "WaveSpeed I2I (gpt-image-2 edit)",
    "LiteLLMImageNode": "LiteLLM Image (GPT Image 2)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
