# ComfyUI-CodexImage 安装说明

## 安装

1. 解压压缩包。
2. 把整个 `ComfyUI-CodexImage` 文件夹放到 ComfyUI 的 `custom_nodes` 目录。
3. 重启 ComfyUI。
4. 在节点菜单里找到：
   - `Codex Image (GPT Image 2)`
   - `Codex Image I2I (GPT Image 2)`
   - `OpenRouter Image (GPT Image 2)`
   - `Mix Codex Copycat Image I2I (GPT Image 2)`
   - `GPT-Image-2 Response i2i`
   - `Requesty I2I (gpt-image-2 edit)`
   - `WaveSpeed I2I (gpt-image-2 edit)`
   - `LiteLLM Image (GPT Image 2)`

## 认证

节点默认使用 `auth` 模式，会读取当前用户的 `~/.codex/auth.json`。

如果接收方没有 Codex 登录状态，可以改用 `api` 模式，并在隐藏输入里配置 `base_url` 和 `api_key`。

OpenRouter 节点不在 UI 里填写 key，启动 ComfyUI 前设置环境变量：

```bash
export OPENROUTER_API_KEY="sk-or-..."
export CODEX_IMAGE_OPENROUTER_BASE_URL="https://openrouter.ai/api/v1/images"
export CODEX_IMAGE_OPENROUTER_MODEL="openai/gpt-image-2"
```

LiteLLM 节点同样通过环境变量配置：

```bash
export LITELLM_API_KEY="sk-..."
export CODEX_IMAGE_LITELLM_BASE_URL="http://localhost:4000"
export CODEX_IMAGE_LITELLM_MODEL="gpt-image-2"
```

Requesty edit 节点可以在节点上临时填 `api_key`，不填时读取环境变量：

```bash
export REQUESTY_API_KEY="sk-..."
export CODEX_IMAGE_REQUESTY_BASE_URL="https://router.requesty.ai/v1"
export CODEX_IMAGE_REQUESTY_MODEL="azure/openai/gpt-image-2"
```

WaveSpeed edit 节点也可以在节点上临时填 `api_key`，不填时读取环境变量：

```bash
export WAVESPEED_API_KEY="ws_..."
export CODEX_IMAGE_WAVESPEED_BASE_URL="https://api.wavespeed.ai/api/v3"
export CODEX_IMAGE_WAVESPEED_MODEL="openai/gpt-image-2/edit"
```

`CODEX_IMAGE_OPENROUTER_MODEL`、`CODEX_IMAGE_LITELLM_MODEL`、`CODEX_IMAGE_REQUESTY_MODEL` 和 `CODEX_IMAGE_WAVESPEED_MODEL` 只是默认值，节点上仍然可以手动填写模型名。

Provider 节点会原样发送节点里填写的 model 或对应环境变量默认值，不会自动改写 provider 前缀。这里要填写你的 provider 实际暴露的 model alias，例如 `openai/gpt-image-2`、`gpt-image-2`、`openrouter/gpt-image-2`，或对应的 Vertex/Gemini alias。

如果在 Docker 里排查 OpenRouter 401，不要只看 `docker exec` shell 的 `env`。要确认 ComfyUI 进程本身有 key：

```bash
pid=$(pgrep -f "python.*main.py" | head -1)
tr '\0' '\n' < /proc/$pid/environ | grep -E '^(OPENROUTER_API_KEY|CODEX_IMAGE_OPENROUTER_API_KEY)=' | wc -l
tr '\0' '\n' < /proc/$pid/environ | sed -n -E 's/^(OPENROUTER_API_KEY|CODEX_IMAGE_OPENROUTER_API_KEY)=//p' | head -1 | awk '{print length($0), substr($0,1,8)}'
```

## I2I 和 Mask

`Codex Image I2I (GPT Image 2)` 支持：

- `image`: 主输入图。
- `image_2`: 可选第二参考图。
- `mask`: 可选遮罩。

`OpenRouter Image (GPT Image 2)` 和 `LiteLLM Image (GPT Image 2)` 也支持可选 `image` / `image_2` / `mask`。不接图片时就是纯 prompt 生图。

`Mix Codex Copycat Image I2I (GPT Image 2)` 只做 I2I，主图必填，并用 `mode` 选择 `openrouter` 或 `litellm`。它复用对应 provider 的环境变量 key/base URL，但图片组织方式对齐 `Codex Image I2I (GPT Image 2)`：mask 会烘到第一张图的 alpha 通道。请求会走 provider 的 `/responses` 端点。`litellm` mode 发送 OpenAI 兼容的 `tools: [{"type": "image_generation", ...}]`；`openrouter` mode 发送 OpenRouter server tool 方言 `tools: [{"type": "openrouter:image_generation", "parameters": {...}}]`。OpenRouter mode 里节点的 `model` 是负责看图和调用工具的 Responses 模型，`image_model` 是实际生图模型。

`GPT-Image-2 Response i2i` 也只做 I2I，主图必填，并用 `mode` 选择 `openrouter` 或 `litellm`。它同样复用 provider 环境变量 key/base URL，但会向 `/responses` 发送原生 OpenAI-style `tools: [{"type": "image_generation", "action": "edit", ...}]`，用于复刻 Codex I2I 的 Responses tool 调用形态。OpenRouter mode 里，`model` 是负责看图和调用工具的 Responses 模型，`image_model` 会作为 tool 参数发送，默认是 `openai/gpt-image-2`；LiteLLM mode 保持纯原生 tool 形态并忽略 `image_model`。

`Requesty I2I (gpt-image-2 edit)` 只做 I2I，主图必填。它会把参考图解成 multipart 文件后 POST 到 Requesty 的 `/images/edits` 路由；默认模型是 `azure/openai/gpt-image-2`，返回图从 `data[0].b64_json` 读取。

`WaveSpeed I2I (gpt-image-2 edit)` 只做 I2I，主图必填。它会向 WaveSpeed 的 `openai/gpt-image-2/edit` prediction API 发送 JSON，开启 sync mode 和 base64 output，并把节点里的像素 `size` 映射成 WaveSpeed 的 `aspect_ratio` 与 `resolution` 参数。

Mask 规则：

- 白色区域：编辑或重绘。
- 黑色区域：尽量保留。

使用 mask 时，节点会把白色遮罩区域转换成透明区域发给图像编辑接口，不会额外拼接 mask 说明到 prompt 文本。

## 依赖

这个节点没有额外 pip 依赖。它只使用 ComfyUI 自带的 `torch`、`numpy` 和 `Pillow`。
