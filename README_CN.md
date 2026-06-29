# ComfyUI CodexImage

GPT Image 2 (gpt-5.5) 生图 ComfyUI 自定义节点，同时提供独立 CLI。

## 核心设计目标

**复用已有的 Codex/ChatGPT 认证体系**，不需要额外部署 API Key。如果你本地已经有登录好的 Codex session，开箱即用。

---

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     codex_image_node.py                    │
│            ComfyUI 节点类 + tensor 格式转换                 │
│              (torch, numpy, PIL — ComfyUI 环境)           │
└─────────────────────────┬─────────────────────────────────┘
                          │ 导入
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                        generator.py                        │
│         核心逻辑：HTTP / SSE 流 / 认证解析 / base64 解码    │
│                    （纯 Python 标准库，无第三方依赖）         │
└─────────────────────────┬─────────────────────────────────┘
                          │ 导入
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                          cli.py                           │
│               独立 CLI 入口 — 无任何第三方依赖              │
│                 （纯 stdlib，不需要 torch）                │
└─────────────────────────────────────────────────────────────┘
```

- **`generator.py`**：无第三方导入。负责 HTTP POST、SSE 流式读取解析、认证解析、从 SSE event 中提取 base64 图片数据。
- **`codex_image_node.py`**：导入 `generator.py`。额外负责将 raw bytes 转换为 ComfyUI 的 IMAGE tensor 格式 `[B, H, W, C]` float32 `[0,1]`。
- **`cli.py`**：直接导入 `generator.py`。提供独立命令行入口，不依赖 ComfyUI 环境。

---

## 三种模式

| 模式 | 实现方式 |
|------|---------|
| `api` | 直接 HTTP POST 到用户指定的 `base_url`，带上 `api_key` |
| `auth` | 同样的 HTTP POST，但 credentials 自动从 `~/.codex/auth.json` 读取（零配置） |
| `cli` | 启动 `codex exec` 子进程，认证逻辑完全由 codex CLI 处理 |

## 额外 Provider 节点

现在还提供两个独立 provider 节点：

| 节点 | API 目标 | API Key |
|------|----------|---------|
| `OpenRouter Image (GPT Image 2)` | OpenRouter dedicated Images API | `CODEX_IMAGE_OPENROUTER_API_KEY` 或 `OPENROUTER_API_KEY` |
| `LiteLLM Image (GPT Image 2)` | LiteLLM OpenAI-compatible `/v1/images/*` proxy | `CODEX_IMAGE_LITELLM_API_KEY`、`LITELLM_API_KEY` 或 `LITELLM_MASTER_KEY` |

两个节点都可以直接在节点上填写 `model`。`api_key` 不进 UI，只读环境变量；`base_url` 也不进 UI，默认从环境变量读取：

```bash
export OPENROUTER_API_KEY="sk-or-..."
export CODEX_IMAGE_OPENROUTER_BASE_URL="https://openrouter.ai/api/v1/images"
export CODEX_IMAGE_OPENROUTER_MODEL="openai/gpt-image-2"

export LITELLM_API_KEY="sk-..."
export CODEX_IMAGE_LITELLM_BASE_URL="http://localhost:4000"
export CODEX_IMAGE_LITELLM_MODEL="gpt-image-2"
```

LiteLLM 节点会原样发送节点里填写的 model 或 `CODEX_IMAGE_LITELLM_MODEL`，不会自动改写 provider 前缀。这里要填写你的 LiteLLM proxy 实际暴露的 model alias，例如 `gpt-image-2`、`openrouter/gpt-image-2`，或对应的 Vertex/Gemini alias。

Docker 部署时，这些环境变量需要在启动 ComfyUI 前注入。后续 `docker exec` 进入容器看到的 shell 环境变量，不一定等于已经运行中的 ComfyUI Python 进程环境。可以用下面命令检查进程环境，且不打印完整 key：

```bash
pid=$(pgrep -f "python.*main.py" | head -1)
tr '\0' '\n' < /proc/$pid/environ | grep -E '^(OPENROUTER_API_KEY|CODEX_IMAGE_OPENROUTER_API_KEY)=' | wc -l
tr '\0' '\n' < /proc/$pid/environ | sed -n -E 's/^(OPENROUTER_API_KEY|CODEX_IMAGE_OPENROUTER_API_KEY)=//p' | head -1 | awk '{print length($0), substr($0,1,8)}'
```

这两个节点支持纯 prompt 生图。接入 `image`、`image_2` 或 `mask` 时，会按 provider 能力走图像编辑/参考图请求。

---

## 实现原理

### 1. API 请求流程（模式：`api` / `auth`）

```
用户输入 prompt
        │
        ▼
_build_payload()
        │ 构建 JSON 请求体：
        │ {
        │   "model": "gpt-5.5",
        │   "instructions": "Generate the requested image...",
        │   "input": [{"role": "user", "content": prompt}],
        │   "tools": [{"type": "image_generation", "size": "...", "quality": "..."}],
        │   "stream": true
        │ }
        ▼
urllib.request.Request
        │ POST，HTTP headers：
        │   Authorization: Bearer <token>
        │   Accept: text/event-stream
        ▼
POST 到 {base_url}/responses
        │
        ▼
SSE 流（分块传输）
        │ 响应内容为重复的 "data: {...}" 行
        │
        ▼
_post_streaming() — SSE 解析
        │ 每次读 4KB chunk，累积到行缓冲区，
        │ 按 "\n" 切割，去掉 "data: " 前缀，
        │ 每行 JSON → event dict list
        │
        ▼
_extract_image() — 找到图片 event
        │ 查找：
        │   ev["type"] == "response.image_generation_call.done"
        │   ev["result"]  ← base64 编码的图片字符串
        │
        ▼
base64.b64decode(img_b64) → 原始图片 bytes
        │
        ▼
_bytes_to_tensor() → ComfyUI IMAGE tensor [1, H, W, C] float32 [0,1]
```

### 2. 认证解析（`_resolve_api_key`）

```
用户提供了 api_key？
        │
        ├─ 是 → 直接作为 Bearer token 使用
        │
        └─ 否（auth 模式）：
                │
                ▼
            1. 检查 OPENAI_API_KEY 环境变量
                │ 存在？ → 使用它
                │ 不存在
                ▼
            2. ~/.codex/auth.json → "OPENAI_API_KEY" 字段
                │ 存在？ → 使用它
                │ 不存在
                ▼
            3. ~/.codex/auth.json → "tokens.access_token"
                │ （`codex login` 时生成的 ChatGPT OAuth token）
                │ 存在？ → 使用它
                │ 不存在 → 抛出 ValueError
```

### 3. CLI 模式（`_generate_cli`）

```
构建内部脚本命令：
  python <script_path> <prompt> --size X --quality X --format X --out /tmp/xxx.png

用 codex exec 模板包装：
  codex exec -- sh -c "python <script> ..."

subprocess.run() → 捕获 stdout
  codex_image.py 在 stdout 最后一行打印输出路径：
    /tmp/xxx.png

读取该路径的文件 → raw bytes
```

### 4. Tensor 格式

ComfyUI 的 IMAGE tensor 格式为 `[B, H, W, C]`，float32 值域 `[0, 1]`：

```
PIL.Image.open(bytes)  →  RGB PIL image
numpy.array(pil)       →  [H, W, C] uint8 [0, 255]
astype(np.float32)/255.0 →  [H, W, C] float32 [0, 1]
torch.from_numpy()[None, ] →  [1, H, W, C]
to(dtype=torch.float32)   →  最终 tensor
```

---

## 文件说明

| 文件 | 说明 | 依赖 |
|------|------|------|
| `generator.py` | 核心逻辑：payload 构建、HTTP POST、SSE 解析、认证、base64 解码 | 无（纯标准库） |
| `codex_image_node.py` | ComfyUI 节点 + tensor 转换 | torch, numpy, Pillow |
| `cli.py` | 独立 CLI 入口 | 无 |

---

## 环境变量

所有环境变量在模块导入时读取：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CODEX_IMAGE_BASE_URL` | `https://chatgpt.com/backend-api/codex` | API 端点 |
| `CODEX_IMAGE_MODEL` | `gpt-5.5` | 模型名 |
| `CODEX_IMAGE_SIZE` | `1024x1024` | 默认尺寸 |
| `CODEX_IMAGE_QUALITY` | `medium` | 默认质量 |
| `CODEX_IMAGE_FORMAT` | `png` | 默认格式 |
| `CODEX_IMAGE_SCRIPT` | `~/.codex-image/scripts/codex_image.py` | CLI 模式脚本路径 |
| `OPENAI_API_KEY` | _(空)_ | 最高优先级认证覆盖 |
| `CODEX_HOME` | `~/.codex` | Codex auth.json 目录 |
| `CODEX_IMAGE_OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1/images` | OpenRouter Images API 端点 |
| `CODEX_IMAGE_OPENROUTER_MODEL` | `openai/gpt-image-2` | OpenRouter 节点默认模型 |
| `CODEX_IMAGE_OPENROUTER_API_KEY` / `OPENROUTER_API_KEY` | _(空)_ | OpenRouter API key |
| `CODEX_IMAGE_LITELLM_BASE_URL` | `http://localhost:4000` | LiteLLM proxy base URL 或 images endpoint |
| `CODEX_IMAGE_LITELLM_MODEL` | `gpt-image-2` | LiteLLM 节点默认模型 |
| `CODEX_IMAGE_LITELLM_API_KEY` / `LITELLM_API_KEY` / `LITELLM_MASTER_KEY` | _(空)_ | LiteLLM proxy API key |

---

## 节点参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mode` | `auth` | `api` / `auth` / `cli` |
| `prompt` | — | 图片描述（必填） |
| `model` | `gpt-5.5` | 模型名 |
| `size` | `1024x1024` | `1024x1024` / `1536x1024` / `1024x1536` / `1792x1024` / `1024x1792` / `1920x1080` / `1080x1920` / `2048x2048` / `3840x2160` / `2160x3840` |
| `quality` | `medium` | `low` / `medium` / `high` |
| `format` | `png` | `png` / `jpeg` / `webp` |
| `output_path` | _(空)_ | 保存副本到指定路径 |

**隐藏字段（ComfyUI UI 不显示）：**

| 字段 | 默认值 | 适用模式 |
|------|--------|---------|
| `base_url` | `https://chatgpt.com/backend-api/codex` | `api` |
| `api_key` | _(空)_ | `api` |
| `codex_cmd` | `codex exec -- sh -c {CMD}` | `cli` |

---

## CLI 用法

```bash
# Codex Auth 模式（默认，自动读 ~/.codex/auth.json）
python cli.py "a cat"

# API 模式（自己填 URL + Key）
python cli.py "a cat" --mode api \
  --base-url https://chatgpt.com/backend-api/codex \
  --api-key sk-xxxx

# CLI 模式（通过 codex exec）
python cli.py "a cat" --mode cli

# OpenRouter 模式（使用 OPENROUTER_API_KEY）
python cli.py "a cat" --mode openrouter --model openai/gpt-image-2

# LiteLLM 模式（使用 LITELLM_API_KEY）
python cli.py "a cat" --mode litellm --model gpt-image-2

# 指定输出路径
python cli.py "a cat" --out ./output.png
```

## 工作流示例

```
[CLIP Text Encode] → [CodexImageNode] → [PreviewImage / SaveImage]
```
