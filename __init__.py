from .codex_image_node import (
    ComfyProxyImageMaskToRGBA,
    ComfyProxyMaskToTransparentImage,
    ComfyProxyValueOutput,
    CodexImageI2INode,
    CodexImageNode,
    GPTImage2ResponseI2INode,
    LiteLLMImageNode,
    MixCodexCopycatImageI2INode,
    OpenRouterGeminiImageNode,
    OpenRouterImageNode,
    RequestyImageEditI2INode,
    WaveSpeedImageEditI2INode,
)

NODE_CLASS_MAPPINGS = {
    "CodexImageNode": CodexImageNode,
    "CodexImageI2INode": CodexImageI2INode,
    "OpenRouterImageNode": OpenRouterImageNode,
    "OpenRouterGeminiImageNode": OpenRouterGeminiImageNode,
    "MixCodexCopycatImageI2INode": MixCodexCopycatImageI2INode,
    "GPTImage2ResponseI2INode": GPTImage2ResponseI2INode,
    "RequestyImageEditI2INode": RequestyImageEditI2INode,
    "WaveSpeedImageEditI2INode": WaveSpeedImageEditI2INode,
    "LiteLLMImageNode": LiteLLMImageNode,
    "ComfyProxyImageMaskToRGBA": ComfyProxyImageMaskToRGBA,
    "ComfyProxyMaskToTransparentImage": ComfyProxyMaskToTransparentImage,
    "ComfyProxyValueOutput": ComfyProxyValueOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CodexImageNode": "Codex Image (GPT Image 2)",
    "CodexImageI2INode": "Codex Image I2I (GPT Image 2)",
    "OpenRouterImageNode": "OpenRouter Image (GPT Image 2)",
    "OpenRouterGeminiImageNode": "OpenRouter Gemini Image",
    "MixCodexCopycatImageI2INode": "Mix Codex Copycat Image I2I (GPT Image 2)",
    "GPTImage2ResponseI2INode": "GPT-Image-2 Response i2i",
    "RequestyImageEditI2INode": "Requesty I2I (gpt-image-2 edit)",
    "WaveSpeedImageEditI2INode": "WaveSpeed I2I (gpt-image-2 edit)",
    "LiteLLMImageNode": "LiteLLM Image (GPT Image 2)",
    "ComfyProxyImageMaskToRGBA": "ComfyProxy Image Mask To RGBA",
    "ComfyProxyMaskToTransparentImage": "ComfyProxy Mask To Transparent Image",
    "ComfyProxyValueOutput": "ComfyProxy Value Output",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
