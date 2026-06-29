import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import codex_image_node  # noqa: E402
import generator  # noqa: E402


class GeneratorContractsTest(unittest.TestCase):
    def test_codex_payload_preserves_prompt_and_model(self):
        payload = generator._build_payload(
            prompt="plain prompt",
            model="gpt-image-2",
            size="1024x1536",
            quality="medium",
        )

        self.assertEqual(payload["model"], "gpt-image-2")
        self.assertEqual(payload["input"][0]["content"], "plain prompt")
        self.assertEqual(payload["tools"][0]["size"], "1024x1536")

    def test_openrouter_payload_preserves_prompt_and_model(self):
        payload = generator._build_openrouter_payload(
            prompt="plain prompt",
            model="openrouter/custom-image-model",
            size="1024x1536",
            quality="medium",
            fmt="png",
            input_image_urls=["data:image/png;base64,abc"],
            background="opaque",
        )

        self.assertEqual(payload["model"], "openrouter/custom-image-model")
        self.assertEqual(payload["prompt"], "plain prompt")
        self.assertEqual(payload["size"], "1024x1536")
        self.assertEqual(payload["output_format"], "png")

    def test_litellm_payload_preserves_prompt_and_model(self):
        payload = generator._build_litellm_generation_payload(
            prompt="plain prompt",
            model="openrouter/gpt-image-2",
            size="1024x1536",
            quality="medium",
        )

        self.assertEqual(payload["model"], "openrouter/gpt-image-2")
        self.assertEqual(payload["prompt"], "plain prompt")
        self.assertEqual(payload["size"], "1024x1536")

    def test_litellm_edit_preserves_prompt_and_model(self):
        seen = []
        original_resolve_key = generator._resolve_env_api_key
        original_post_multipart = generator._post_multipart
        original_extract = generator._extract_image_bytes_from_images_response

        def fake_post_multipart(url, token, fields, files, timeout):
            seen.append({"url": url, "token": token, "fields": fields, "files": files})
            return {"data": [{"b64_json": "ignored"}]}

        try:
            generator._resolve_env_api_key = lambda *args, **kwargs: "token"
            generator._post_multipart = fake_post_multipart
            generator._extract_image_bytes_from_images_response = lambda response, token: b"\x89PNG\r\n\x1a\n"

            generator._generate_litellm(
                prompt="plain prompt",
                model="openrouter/gpt-image-2",
                size="1024x1536",
                quality="medium",
                fmt="png",
                base_url="http://litellm",
                api_key="",
                input_image_urls=["data:image/png;base64,aGVsbG8="],
            )
        finally:
            generator._resolve_env_api_key = original_resolve_key
            generator._post_multipart = original_post_multipart
            generator._extract_image_bytes_from_images_response = original_extract

        self.assertEqual(seen[0]["fields"]["model"], "openrouter/gpt-image-2")
        self.assertEqual(seen[0]["fields"]["prompt"], "plain prompt")
        self.assertEqual(seen[0]["fields"]["size"], "1024x1536")

    def test_provider_defaults_only_apply_to_empty_model(self):
        openrouter_seen = []
        litellm_seen = []
        original_openrouter = generator._generate_openrouter
        original_litellm = generator._generate_litellm

        def fake_openrouter(**kwargs):
            openrouter_seen.append(kwargs["model"])
            return b"\x89PNG\r\n\x1a\n", "/tmp/out.png"

        def fake_litellm(**kwargs):
            litellm_seen.append(kwargs["model"])
            return b"\x89PNG\r\n\x1a\n", "/tmp/out.png"

        try:
            generator._generate_openrouter = fake_openrouter
            generator._generate_litellm = fake_litellm
            generator.generate_image("p", mode="openrouter", model="gpt-5.5")
            generator.generate_image("p", mode="openrouter", model="")
            generator.generate_image("p", mode="litellm", model="openrouter/gpt-image-2")
            generator.generate_image("p", mode="litellm", model="")
        finally:
            generator._generate_openrouter = original_openrouter
            generator._generate_litellm = original_litellm

        self.assertEqual(openrouter_seen[0], "gpt-5.5")
        self.assertEqual(openrouter_seen[1], generator.DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(litellm_seen[0], "openrouter/gpt-image-2")
        self.assertEqual(litellm_seen[1], generator.DEFAULT_LITELLM_MODEL)


class NodeContractsTest(unittest.TestCase):
    def test_legacy_node_class_keys_are_registered(self):
        spec = importlib.util.spec_from_file_location(
            "ComfyUI_CodexImage_test",
            ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        self.assertEqual(module.NODE_CLASS_MAPPINGS["OpenRouterImageNode"].__name__, "OpenRouterImageNode")
        self.assertEqual(module.NODE_CLASS_MAPPINGS["LiteLLMImageNode"].__name__, "LiteLLMImageNode")
        self.assertEqual(
            module.NODE_DISPLAY_NAME_MAPPINGS["OpenRouterImageNode"],
            "OpenRouter Image (GPT Image 2)",
        )
        self.assertEqual(
            module.NODE_DISPLAY_NAME_MAPPINGS["LiteLLMImageNode"],
            "LiteLLM Image (GPT Image 2)",
        )

    def test_output_copy_uses_actual_image_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = codex_image_node._write_output_copy(
                b"\x89PNG\r\n\x1a\n",
                "/tmp/provider-returned.png",
                str(Path(tmpdir) / "requested.jpeg"),
                "jpeg",
            )

            self.assertTrue(out.endswith(".png"))
            self.assertEqual(Path(out).read_bytes(), b"\x89PNG\r\n\x1a\n")

    def test_i2i_mask_does_not_modify_prompt(self):
        seen = []
        original_has_comfyu = codex_image_node._HAS_COMFYU
        original_mask_to_url = codex_image_node._image_tensor_and_mask_to_data_url
        original_image_to_url = codex_image_node._image_tensor_to_data_url
        original_image_to_tensor = codex_image_node._image_bytes_to_tensor
        original_generate = codex_image_node.generate_image

        def fake_generate(**kwargs):
            seen.append(kwargs)
            return b"\x89PNG\r\n\x1a\n", "/tmp/out.png"

        try:
            codex_image_node._HAS_COMFYU = True
            codex_image_node._image_tensor_and_mask_to_data_url = lambda image, mask: "data:image/png;base64,mask"
            codex_image_node._image_tensor_to_data_url = lambda image: "data:image/png;base64,image"
            codex_image_node._image_bytes_to_tensor = lambda img_bytes: "tensor"
            codex_image_node.generate_image = fake_generate

            codex_image_node.CodexImageI2INode().generate(
                prompt="plain prompt",
                model="gpt-image-2",
                size="1024x1024",
                quality="medium",
                format="png",
                mode="api",
                image=object(),
                mask=object(),
            )
        finally:
            codex_image_node._HAS_COMFYU = original_has_comfyu
            codex_image_node._image_tensor_and_mask_to_data_url = original_mask_to_url
            codex_image_node._image_tensor_to_data_url = original_image_to_url
            codex_image_node._image_bytes_to_tensor = original_image_to_tensor
            codex_image_node.generate_image = original_generate

        self.assertEqual(seen[0]["prompt"], "plain prompt")
        self.assertEqual(seen[0]["input_image_urls"], ["data:image/png;base64,mask"])

    def test_openrouter_mask_does_not_modify_prompt(self):
        seen = []
        original_has_comfyu = codex_image_node._HAS_COMFYU
        original_mask_to_url = codex_image_node._image_tensor_and_mask_to_data_url
        original_image_to_url = codex_image_node._image_tensor_to_data_url
        original_image_to_tensor = codex_image_node._image_bytes_to_tensor
        original_generate = codex_image_node.generate_image

        def fake_generate(**kwargs):
            seen.append(kwargs)
            return b"\x89PNG\r\n\x1a\n", "/tmp/out.png"

        try:
            codex_image_node._HAS_COMFYU = True
            codex_image_node._image_tensor_and_mask_to_data_url = lambda image, mask: "data:image/png;base64,mask"
            codex_image_node._image_tensor_to_data_url = lambda image: "data:image/png;base64,image"
            codex_image_node._image_bytes_to_tensor = lambda img_bytes: "tensor"
            codex_image_node.generate_image = fake_generate

            codex_image_node.OpenRouterImageNode().generate(
                prompt="plain prompt",
                model="openai/gpt-image-2",
                size="1024x1024",
                quality="medium",
                background="opaque",
                format="png",
                image=object(),
                mask=object(),
            )
        finally:
            codex_image_node._HAS_COMFYU = original_has_comfyu
            codex_image_node._image_tensor_and_mask_to_data_url = original_mask_to_url
            codex_image_node._image_tensor_to_data_url = original_image_to_url
            codex_image_node._image_bytes_to_tensor = original_image_to_tensor
            codex_image_node.generate_image = original_generate

        self.assertEqual(seen[0]["prompt"], "plain prompt")
        self.assertEqual(seen[0]["input_image_urls"], ["data:image/png;base64,mask"])


if __name__ == "__main__":
    unittest.main()
