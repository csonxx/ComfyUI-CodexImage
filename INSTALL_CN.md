# ComfyUI-CodexImage 安装说明

## 安装

1. 解压压缩包。
2. 把整个 `ComfyUI-CodexImage` 文件夹放到 ComfyUI 的 `custom_nodes` 目录。
3. 重启 ComfyUI。
4. 在节点菜单里找到：
   - `Codex Image (GPT Image 2)`
   - `Codex Image I2I (GPT Image 2)`
   - `OpenRouter Image (GPT Image 2)`
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

`CODEX_IMAGE_OPENROUTER_MODEL` 和 `CODEX_IMAGE_LITELLM_MODEL` 只是默认值，节点上仍然可以手动填写模型名。

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

Mask 规则：

- 白色区域：编辑或重绘。
- 黑色区域：尽量保留。

使用 mask 时，节点会把白色遮罩区域转换成透明区域发给图像编辑接口，不会额外拼接 mask 说明到 prompt 文本。

## 依赖

这个节点没有额外 pip 依赖。它只使用 ComfyUI 自带的 `torch`、`numpy` 和 `Pillow`。
