import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import codex_image_node  # noqa: E402
import generator  # noqa: E402


class GeneratorContractsTest(unittest.TestCase):
    def test_openai_api_key_environment_variable_does_not_require_auth_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "environment-key",
                    "CODEX_HOME": directory,
                },
                clear=False,
            ):
                self.assertEqual(generator._resolve_api_key(""), "environment-key")

    def test_explicit_api_key_overrides_openai_api_key_environment_variable(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "environment-key"}, clear=False):
            self.assertEqual(generator._resolve_api_key("node-key"), "node-key")

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

    def test_openrouter_gemini_payload_uses_gemini_fields_only(self):
        payload = generator._build_openrouter_gemini_payload(
            prompt="plain prompt",
            model="google/gemini-3.1-flash-image",
            resolution="2k",
            aspect_ratio="16:9",
            input_image_urls=["data:image/png;base64,abc"],
        )

        self.assertEqual(payload["model"], "google/gemini-3.1-flash-image")
        self.assertEqual(payload["prompt"], "plain prompt")
        self.assertEqual(payload["resolution"], "2K")
        self.assertEqual(payload["aspect_ratio"], "16:9")
        self.assertEqual(
            payload["input_references"],
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}],
        )
        self.assertNotIn("size", payload)
        self.assertNotIn("quality", payload)
        self.assertNotIn("background", payload)
        self.assertNotIn("output_format", payload)

    def test_openrouter_gemini_payload_validates_model_capabilities(self):
        with self.assertRaisesRegex(ValueError, "does not support resolution"):
            generator._build_openrouter_gemini_payload(
                prompt="plain prompt",
                model="google/gemini-2.5-flash-image",
                resolution="2K",
                aspect_ratio="1:1",
            )

        with self.assertRaisesRegex(ValueError, "does not support aspect_ratio"):
            generator._build_openrouter_gemini_payload(
                prompt="plain prompt",
                model="google/gemini-3-pro-image",
                resolution="2K",
                aspect_ratio="1:8",
            )

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

    def test_openrouter_gemini_generation_posts_to_images_api(self):
        seen = []
        original_resolve_key = generator._resolve_env_api_key
        original_post_json = generator._post_json
        original_extract = generator._extract_image_bytes_from_images_response

        def fake_post_json(url, token, payload, timeout, extra_headers=None):
            seen.append(
                {
                    "url": url,
                    "token": token,
                    "payload": payload,
                    "extra_headers": extra_headers,
                }
            )
            return {"data": [{"b64_json": "ignored"}]}

        try:
            generator._resolve_env_api_key = lambda api_key, env_names, label: f"{label}-token"
            generator._post_json = fake_post_json
            generator._extract_image_bytes_from_images_response = lambda response, token: b"\x89PNG\r\n\x1a\n"

            generator.generate_openrouter_gemini_image(
                prompt="plain prompt",
                model="google/gemini-3.1-flash-image",
                resolution="4K",
                aspect_ratio="9:16",
                base_url="https://openrouter.ai/api/v1/images",
                api_key="",
                input_image_urls=["data:image/png;base64,abc"],
            )
        finally:
            generator._resolve_env_api_key = original_resolve_key
            generator._post_json = original_post_json
            generator._extract_image_bytes_from_images_response = original_extract

        self.assertEqual(seen[0]["url"], "https://openrouter.ai/api/v1/images")
        self.assertEqual(seen[0]["token"], "OpenRouter-token")
        self.assertEqual(seen[0]["payload"]["model"], "google/gemini-3.1-flash-image")
        self.assertEqual(seen[0]["payload"]["resolution"], "4K")
        self.assertEqual(seen[0]["payload"]["aspect_ratio"], "9:16")
        self.assertNotIn("size", seen[0]["payload"])
        self.assertNotIn("quality", seen[0]["payload"])

    def test_openrouter_responses_uses_openrouter_server_tool_payload(self):
        seen = []
        original_run = generator._run_responses_image_request
        original_resolve_key = generator._resolve_env_api_key

        def fake_run(api_url, token, payload, fmt, extra_headers=None):
            seen.append(
                {
                    "api_url": api_url,
                    "token": token,
                    "payload": payload,
                    "fmt": fmt,
                    "extra_headers": extra_headers,
                }
            )
            return b"\x89PNG\r\n\x1a\n", "/tmp/out.png"

        try:
            generator._run_responses_image_request = fake_run
            generator._resolve_env_api_key = lambda api_key, env_names, label: f"{label}-token"

            generator.generate_responses_image(
                prompt="plain prompt",
                model="openai/gpt-5.5-20260423",
                image_model="openai/gpt-image-2",
                size="1024x1536",
                quality="medium",
                fmt="png",
                mode="openrouter",
                input_image_urls=["data:image/png;base64,abc"],
                action="edit",
            )
        finally:
            generator._run_responses_image_request = original_run
            generator._resolve_env_api_key = original_resolve_key

        self.assertEqual(seen[0]["api_url"], "https://openrouter.ai/api/v1/responses")
        self.assertEqual(seen[0]["token"], "OpenRouter-token")
        self.assertEqual(seen[0]["payload"]["model"], "openai/gpt-5.5-20260423")
        self.assertEqual(seen[0]["payload"]["tools"][0]["type"], "openrouter:image_generation")
        self.assertEqual(seen[0]["payload"]["tools"][0]["parameters"]["model"], "openai/gpt-image-2")
        self.assertEqual(seen[0]["payload"]["tools"][0]["parameters"]["size"], "1024x1536")
        self.assertNotIn("action", seen[0]["payload"]["tools"][0])
        self.assertEqual(seen[0]["payload"]["input"][0]["content"][0]["text"], "plain prompt")
        self.assertEqual(seen[0]["payload"]["input"][0]["content"][1]["image_url"], "data:image/png;base64,abc")

    def test_openrouter_responses_defaults_split_chat_and_image_models(self):
        payload = generator._build_openrouter_responses_payload(
            prompt="plain prompt",
            model="",
            image_model="",
            size="1024x1536",
            quality="medium",
            fmt="png",
            input_image_urls=None,
        )

        self.assertEqual(payload["model"], generator.DEFAULT_OPENROUTER_RESPONSES_MODEL)
        self.assertEqual(payload["input"], "plain prompt")
        self.assertEqual(payload["tools"][0]["type"], "openrouter:image_generation")
        self.assertEqual(payload["tools"][0]["parameters"]["model"], generator.DEFAULT_OPENROUTER_IMAGE_MODEL)

    def test_native_openrouter_responses_uses_image_generation_action_edit_payload(self):
        seen = []
        original_run = generator._run_responses_image_request
        original_resolve_key = generator._resolve_env_api_key

        def fake_run(api_url, token, payload, fmt, extra_headers=None):
            seen.append(
                {
                    "api_url": api_url,
                    "token": token,
                    "payload": payload,
                    "fmt": fmt,
                    "extra_headers": extra_headers,
                }
            )
            return b"\x89PNG\r\n\x1a\n", "/tmp/out.png"

        try:
            generator._run_responses_image_request = fake_run
            generator._resolve_env_api_key = lambda api_key, env_names, label: f"{label}-token"

            generator.generate_native_responses_image(
                prompt="plain prompt",
                model="openai/gpt-5.5",
                image_model="openai/gpt-image-2",
                size="1024x1536",
                quality="medium",
                fmt="png",
                mode="openrouter",
                input_image_urls=["data:image/png;base64,abc"],
                action="edit",
            )
        finally:
            generator._run_responses_image_request = original_run
            generator._resolve_env_api_key = original_resolve_key

        self.assertEqual(seen[0]["api_url"], "https://openrouter.ai/api/v1/responses")
        self.assertEqual(seen[0]["token"], "OpenRouter-token")
        self.assertEqual(seen[0]["payload"]["model"], "openai/gpt-5.5")
        tool = seen[0]["payload"]["tools"][0]
        self.assertEqual(tool["type"], "image_generation")
        self.assertEqual(tool["action"], "edit")
        self.assertEqual(tool["parameters"]["model"], "openai/gpt-image-2")
        self.assertEqual(tool["parameters"]["size"], "1024x1536")
        self.assertEqual(tool["parameters"]["quality"], "medium")
        self.assertEqual(tool["parameters"]["output_format"], "png")
        self.assertEqual(seen[0]["payload"]["input"][0]["content"][0]["text"], "plain prompt")
        self.assertEqual(seen[0]["payload"]["input"][0]["content"][1]["image_url"], "data:image/png;base64,abc")

    def test_native_litellm_responses_keeps_plain_image_generation_tool(self):
        seen = []
        original_run = generator._run_responses_image_request
        original_resolve_key = generator._resolve_env_api_key

        def fake_run(api_url, token, payload, fmt, extra_headers=None):
            seen.append(
                {
                    "api_url": api_url,
                    "token": token,
                    "payload": payload,
                    "fmt": fmt,
                    "extra_headers": extra_headers,
                }
            )
            return b"\x89PNG\r\n\x1a\n", "/tmp/out.png"

        try:
            generator._run_responses_image_request = fake_run
            generator._resolve_env_api_key = lambda api_key, env_names, label: f"{label}-token"

            generator.generate_native_responses_image(
                prompt="plain prompt",
                model="openrouter/gpt-5.5",
                image_model="openai/gpt-image-2",
                size="1024x1536",
                quality="medium",
                fmt="png",
                mode="litellm",
                base_url="http://litellm.local/v1/images/edits",
                input_image_urls=["data:image/png;base64,abc"],
                action="edit",
            )
        finally:
            generator._run_responses_image_request = original_run
            generator._resolve_env_api_key = original_resolve_key

        self.assertEqual(seen[0]["api_url"], "http://litellm.local/v1/responses")
        self.assertEqual(seen[0]["token"], "LiteLLM-token")
        self.assertEqual(seen[0]["payload"]["tools"][0]["type"], "image_generation")
        self.assertEqual(seen[0]["payload"]["tools"][0]["action"], "edit")
        self.assertNotIn("parameters", seen[0]["payload"]["tools"][0])

    def test_litellm_responses_uses_image_generation_tool_payload(self):
        seen = []
        original_run = generator._run_responses_image_request
        original_resolve_key = generator._resolve_env_api_key

        def fake_run(api_url, token, payload, fmt, extra_headers=None):
            seen.append(
                {
                    "api_url": api_url,
                    "token": token,
                    "payload": payload,
                    "fmt": fmt,
                    "extra_headers": extra_headers,
                }
            )
            return b"\x89PNG\r\n\x1a\n", "/tmp/out.png"

        try:
            generator._run_responses_image_request = fake_run
            generator._resolve_env_api_key = lambda api_key, env_names, label: f"{label}-token"

            generator.generate_responses_image(
                prompt="plain prompt",
                model="openrouter/gpt-image-2",
                size="1024x1536",
                quality="medium",
                fmt="png",
                mode="litellm",
                base_url="http://litellm.local/v1/images/edits",
                input_image_urls=["data:image/png;base64,abc"],
                action="edit",
            )
        finally:
            generator._run_responses_image_request = original_run
            generator._resolve_env_api_key = original_resolve_key

        self.assertEqual(seen[0]["api_url"], "http://litellm.local/v1/responses")
        self.assertEqual(seen[0]["token"], "LiteLLM-token")
        self.assertEqual(seen[0]["payload"]["model"], "openrouter/gpt-image-2")
        self.assertEqual(seen[0]["payload"]["tools"][0]["type"], "image_generation")
        self.assertEqual(seen[0]["payload"]["tools"][0]["action"], "edit")
        self.assertEqual(seen[0]["payload"]["input"][0]["content"][0]["text"], "plain prompt")
        self.assertEqual(seen[0]["payload"]["input"][0]["content"][1]["image_url"], "data:image/png;base64,abc")

    def test_responses_extracts_openrouter_image_url(self):
        original_download = generator._download_image_url
        seen = []

        def fake_download(url, token=""):
            seen.append({"url": url, "token": token})
            return b"\x89PNG\r\n\x1a\n"

        try:
            generator._download_image_url = fake_download
            img_bytes = generator._extract_image_bytes_from_responses_events(
                [
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "tool_result",
                            "output": {"status": "ok", "imageUrl": "https://example.test/generated.png"},
                        },
                    }
                ],
                token="provider-token",
            )
        finally:
            generator._download_image_url = original_download

        self.assertEqual(img_bytes, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(seen, [{"url": "https://example.test/generated.png", "token": ""}])

    def test_responses_extracts_openrouter_image_generation_result(self):
        img_bytes = generator._extract_image_bytes_from_responses_events(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "output": [
                            {
                                "type": "openrouter:image_generation",
                                "result": "iVBORw0KGgo=",
                                "imageUrl": "https://example.test/generated.png",
                            }
                        ]
                    },
                }
            ]
        )

        self.assertEqual(img_bytes, b"\x89PNG\r\n\x1a\n")

    def test_responses_extracts_openrouter_markdown_image_url(self):
        original_download = generator._download_image_url
        seen = []

        def fake_download(url, token=""):
            seen.append({"url": url, "token": token})
            return b"\x89PNG\r\n\x1a\n"

        try:
            generator._download_image_url = fake_download
            img_bytes = generator._extract_image_bytes_from_responses_events(
                [
                    {
                        "type": "response.content_part.done",
                        "part": {
                            "type": "output_text",
                            "text": "Done ![image](https://example.test/generated)",
                        },
                    }
                ]
            )
        finally:
            generator._download_image_url = original_download

        self.assertEqual(img_bytes, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(seen, [{"url": "https://example.test/generated", "token": ""}])

    def test_responses_does_not_treat_input_image_as_generated_image(self):
        with self.assertRaisesRegex(RuntimeError, "No generated image"):
            generator._extract_image_bytes_from_responses_events(
                [
                    {
                        "type": "response.completed",
                        "response": {
                            "input": [
                                {
                                    "type": "input_image",
                                    "image_url": "data:image/png;base64,aW5wdXQ=",
                                }
                            ],
                        },
                    }
                ]
            )

    def test_wavespeed_edit_submits_json_and_extracts_base64_output(self):
        seen = []
        original_post = generator._post_json
        original_resolve_key = generator._resolve_env_api_key

        def fake_post(url, token, payload, timeout, extra_headers=None):
            seen.append(
                {
                    "url": url,
                    "token": token,
                    "payload": payload,
                    "timeout": timeout,
                    "extra_headers": extra_headers,
                }
            )
            return {
                "data": {
                    "status": "completed",
                    "outputs": ["iVBORw0KGgo="],
                }
            }

        try:
            generator._post_json = fake_post
            generator._resolve_env_api_key = lambda api_key, env_names, label: f"{label}-token"

            img_bytes, _ = generator.generate_wavespeed_edit(
                prompt="plain prompt",
                model="openai/gpt-image-2/edit",
                size="1024x1536",
                quality="medium",
                fmt="png",
                input_image_urls=["data:image/png;base64,abc"],
            )
        finally:
            generator._post_json = original_post
            generator._resolve_env_api_key = original_resolve_key

        self.assertEqual(img_bytes, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(
            seen[0]["url"],
            "https://api.wavespeed.ai/api/v3/openai/gpt-image-2/edit",
        )
        self.assertEqual(seen[0]["token"], "WaveSpeed-token")
        self.assertEqual(seen[0]["payload"]["prompt"], "plain prompt")
        self.assertEqual(seen[0]["payload"]["images"], ["data:image/png;base64,abc"])
        self.assertEqual(seen[0]["payload"]["aspect_ratio"], "2:3")
        self.assertEqual(seen[0]["payload"]["resolution"], "1k")
        self.assertEqual(seen[0]["payload"]["quality"], "medium")
        self.assertEqual(seen[0]["payload"]["output_format"], "png")
        self.assertIs(seen[0]["payload"]["enable_sync_mode"], True)
        self.assertIs(seen[0]["payload"]["enable_base64_output"], True)


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
        self.assertEqual(
            module.NODE_CLASS_MAPPINGS["OpenRouterGeminiImageNode"].__name__,
            "OpenRouterGeminiImageNode",
        )
        self.assertEqual(module.NODE_CLASS_MAPPINGS["LiteLLMImageNode"].__name__, "LiteLLMImageNode")
        self.assertEqual(
            module.NODE_CLASS_MAPPINGS["MixCodexCopycatImageI2INode"].__name__,
            "MixCodexCopycatImageI2INode",
        )
        self.assertEqual(
            module.NODE_CLASS_MAPPINGS["GPTImage2ResponseI2INode"].__name__,
            "GPTImage2ResponseI2INode",
        )
        self.assertEqual(
            module.NODE_CLASS_MAPPINGS["RequestyImageEditI2INode"].__name__,
            "RequestyImageEditI2INode",
        )
        self.assertEqual(
            module.NODE_CLASS_MAPPINGS["WaveSpeedImageEditI2INode"].__name__,
            "WaveSpeedImageEditI2INode",
        )
        self.assertEqual(
            module.NODE_CLASS_MAPPINGS["ComfyProxyValueOutput"].__name__,
            "ComfyProxyValueOutput",
        )
        self.assertEqual(
            module.NODE_DISPLAY_NAME_MAPPINGS["OpenRouterImageNode"],
            "OpenRouter Image (GPT Image 2)",
        )
        self.assertEqual(
            module.NODE_DISPLAY_NAME_MAPPINGS["OpenRouterGeminiImageNode"],
            "OpenRouter Gemini Image",
        )
        self.assertEqual(
            module.NODE_DISPLAY_NAME_MAPPINGS["LiteLLMImageNode"],
            "LiteLLM Image (GPT Image 2)",
        )
        self.assertEqual(
            module.NODE_DISPLAY_NAME_MAPPINGS["MixCodexCopycatImageI2INode"],
            "Mix Codex Copycat Image I2I (GPT Image 2)",
        )
        self.assertEqual(
            module.NODE_DISPLAY_NAME_MAPPINGS["GPTImage2ResponseI2INode"],
            "GPT-Image-2 Response i2i",
        )
        self.assertEqual(
            module.NODE_DISPLAY_NAME_MAPPINGS["RequestyImageEditI2INode"],
            "Requesty I2I (gpt-image-2 edit)",
        )
        self.assertEqual(
            module.NODE_DISPLAY_NAME_MAPPINGS["WaveSpeedImageEditI2INode"],
            "WaveSpeed I2I (gpt-image-2 edit)",
        )
        self.assertEqual(
            module.NODE_DISPLAY_NAME_MAPPINGS["ComfyProxyValueOutput"],
            "ComfyProxy Value Output",
        )

    def test_comfyproxy_value_output_history_payload(self):
        result = codex_image_node.ComfyProxyValueOutput().save("hello")
        self.assertEqual(result, {"ui": {"comfyproxy_value": ["hello"]}})

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

    def test_mix_copycat_routes_openrouter_with_responses_tool_payload(self):
        seen = []
        original_has_comfyu = codex_image_node._HAS_COMFYU
        original_mask_to_url = codex_image_node._image_tensor_and_mask_to_data_url
        original_image_to_url = codex_image_node._image_tensor_to_data_url
        original_image_to_tensor = codex_image_node._image_bytes_to_tensor
        original_generate = codex_image_node.generate_responses_image

        def fake_generate(**kwargs):
            seen.append(kwargs)
            return b"\x89PNG\r\n\x1a\n", "/tmp/out.png"

        try:
            codex_image_node._HAS_COMFYU = True
            codex_image_node._image_tensor_and_mask_to_data_url = lambda image, mask: "data:image/png;base64,mask"
            codex_image_node._image_tensor_to_data_url = lambda image: "data:image/png;base64,image"
            codex_image_node._image_bytes_to_tensor = lambda img_bytes: "tensor"
            codex_image_node.generate_responses_image = fake_generate

            codex_image_node.MixCodexCopycatImageI2INode().generate(
                image=object(),
                mode="openrouter",
                prompt="plain prompt",
                model="openai/gpt-image-2",
                image_model="openai/gpt-image-2",
                size="1024x1024",
                quality="medium",
                format="png",
                image_2=object(),
                mask=object(),
            )
        finally:
            codex_image_node._HAS_COMFYU = original_has_comfyu
            codex_image_node._image_tensor_and_mask_to_data_url = original_mask_to_url
            codex_image_node._image_tensor_to_data_url = original_image_to_url
            codex_image_node._image_bytes_to_tensor = original_image_to_tensor
            codex_image_node.generate_responses_image = original_generate

        self.assertEqual(seen[0]["mode"], "openrouter")
        self.assertEqual(seen[0]["prompt"], "plain prompt")
        self.assertEqual(seen[0]["model"], "openai/gpt-image-2")
        self.assertEqual(seen[0]["image_model"], "openai/gpt-image-2")
        self.assertEqual(
            seen[0]["input_image_urls"],
            ["data:image/png;base64,mask", "data:image/png;base64,image"],
        )
        self.assertEqual(seen[0]["action"], "edit")
        self.assertNotIn("mask_image_url", seen[0])

    def test_mix_copycat_routes_litellm_with_responses_tool_payload(self):
        seen = []
        original_has_comfyu = codex_image_node._HAS_COMFYU
        original_mask_to_url = codex_image_node._image_tensor_and_mask_to_data_url
        original_image_to_url = codex_image_node._image_tensor_to_data_url
        original_image_to_tensor = codex_image_node._image_bytes_to_tensor
        original_generate = codex_image_node.generate_responses_image

        def fake_generate(**kwargs):
            seen.append(kwargs)
            return b"\x89PNG\r\n\x1a\n", "/tmp/out.png"

        try:
            codex_image_node._HAS_COMFYU = True
            codex_image_node._image_tensor_and_mask_to_data_url = lambda image, mask: "data:image/png;base64,mask"
            codex_image_node._image_tensor_to_data_url = lambda image: "data:image/png;base64,image"
            codex_image_node._image_bytes_to_tensor = lambda img_bytes: "tensor"
            codex_image_node.generate_responses_image = fake_generate

            codex_image_node.MixCodexCopycatImageI2INode().generate(
                image=object(),
                mode="litellm",
                prompt="plain prompt",
                model="openrouter/gpt-image-2",
                image_model="openai/gpt-image-2",
                size="1024x1024",
                quality="medium",
                format="png",
                mask=object(),
            )
        finally:
            codex_image_node._HAS_COMFYU = original_has_comfyu
            codex_image_node._image_tensor_and_mask_to_data_url = original_mask_to_url
            codex_image_node._image_tensor_to_data_url = original_image_to_url
            codex_image_node._image_bytes_to_tensor = original_image_to_tensor
            codex_image_node.generate_responses_image = original_generate

        self.assertEqual(seen[0]["mode"], "litellm")
        self.assertEqual(seen[0]["prompt"], "plain prompt")
        self.assertEqual(seen[0]["model"], "openrouter/gpt-image-2")
        self.assertEqual(seen[0]["image_model"], "openai/gpt-image-2")
        self.assertEqual(seen[0]["input_image_urls"], ["data:image/png;base64,mask"])
        self.assertEqual(seen[0]["action"], "edit")
        self.assertNotIn("mask_image_url", seen[0])

    def test_gpt_image_2_response_i2i_routes_provider_with_native_responses_payload(self):
        seen = []
        original_has_comfyu = codex_image_node._HAS_COMFYU
        original_mask_to_url = codex_image_node._image_tensor_and_mask_to_data_url
        original_image_to_url = codex_image_node._image_tensor_to_data_url
        original_image_to_tensor = codex_image_node._image_bytes_to_tensor
        original_generate = codex_image_node.generate_native_responses_image

        def fake_generate(**kwargs):
            seen.append(kwargs)
            return b"\x89PNG\r\n\x1a\n", "/tmp/out.png"

        try:
            codex_image_node._HAS_COMFYU = True
            codex_image_node._image_tensor_and_mask_to_data_url = lambda image, mask: "data:image/png;base64,mask"
            codex_image_node._image_tensor_to_data_url = lambda image: "data:image/png;base64,image"
            codex_image_node._image_bytes_to_tensor = lambda img_bytes: "tensor"
            codex_image_node.generate_native_responses_image = fake_generate

            codex_image_node.GPTImage2ResponseI2INode().generate(
                image=object(),
                mode="openrouter",
                prompt="plain prompt",
                model="openai/gpt-5.5",
                image_model="openai/gpt-image-2",
                size="1024x1024",
                quality="medium",
                format="png",
                image_2=object(),
                mask=object(),
            )
        finally:
            codex_image_node._HAS_COMFYU = original_has_comfyu
            codex_image_node._image_tensor_and_mask_to_data_url = original_mask_to_url
            codex_image_node._image_tensor_to_data_url = original_image_to_url
            codex_image_node._image_bytes_to_tensor = original_image_to_tensor
            codex_image_node.generate_native_responses_image = original_generate

        self.assertEqual(seen[0]["mode"], "openrouter")
        self.assertEqual(seen[0]["prompt"], "plain prompt")
        self.assertEqual(seen[0]["model"], "openai/gpt-5.5")
        self.assertEqual(seen[0]["image_model"], "openai/gpt-image-2")
        self.assertEqual(
            seen[0]["input_image_urls"],
            ["data:image/png;base64,mask", "data:image/png;base64,image"],
        )
        self.assertEqual(seen[0]["action"], "edit")

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

    def test_openrouter_gemini_node_routes_to_gemini_generator(self):
        seen = []
        original_has_comfyu = codex_image_node._HAS_COMFYU
        original_mask_to_url = codex_image_node._image_tensor_and_mask_to_data_url
        original_image_to_url = codex_image_node._image_tensor_to_data_url
        original_image_to_tensor = codex_image_node._image_bytes_to_tensor
        original_generate = codex_image_node.generate_openrouter_gemini_image

        def fake_generate(**kwargs):
            seen.append(kwargs)
            return b"\x89PNG\r\n\x1a\n", "/tmp/out.png"

        try:
            codex_image_node._HAS_COMFYU = True
            codex_image_node._image_tensor_and_mask_to_data_url = lambda image, mask: "data:image/png;base64,mask"
            codex_image_node._image_tensor_to_data_url = lambda image: "data:image/png;base64,image"
            codex_image_node._image_bytes_to_tensor = lambda img_bytes: "tensor"
            codex_image_node.generate_openrouter_gemini_image = fake_generate

            codex_image_node.OpenRouterGeminiImageNode().generate(
                prompt="plain prompt",
                model="google/gemini-3.1-flash-image",
                resolution="2K",
                aspect_ratio="16:9",
                image=object(),
                image_2=object(),
                mask=object(),
            )
        finally:
            codex_image_node._HAS_COMFYU = original_has_comfyu
            codex_image_node._image_tensor_and_mask_to_data_url = original_mask_to_url
            codex_image_node._image_tensor_to_data_url = original_image_to_url
            codex_image_node._image_bytes_to_tensor = original_image_to_tensor
            codex_image_node.generate_openrouter_gemini_image = original_generate

        self.assertEqual(seen[0]["prompt"], "plain prompt")
        self.assertEqual(seen[0]["model"], "google/gemini-3.1-flash-image")
        self.assertEqual(seen[0]["resolution"], "2K")
        self.assertEqual(seen[0]["aspect_ratio"], "16:9")
        self.assertEqual(
            seen[0]["input_image_urls"],
            ["data:image/png;base64,mask", "data:image/png;base64,image"],
        )


if __name__ == "__main__":
    unittest.main()
