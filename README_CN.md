# ComfyUI CodexImage

ComfyUI 自定义节点，通过 GPT Image 2 (gpt-5.5) 生成图片。

## 核心思路

复用本地已有的认证体系生成图片，不依赖额外的 API Key 配置。

图片生成走的是 Codex Responses API（`/responses` 端点），底层是 ChatGPT 的 `image_generation` 工具。认证则通过三种方式解决：

1. **API 模式** — 用户直接提供 `base_url` + `api_key`
2. **Codex Auth 模式** — 自动读 `~/.codex/auth.json`（`codex login` 时生成的 OAuth token 或手动写入的 `OPENAI_API_KEY`）
3. **CLI 模式** — 调用本机 `codex exec` 命令，透传本地已 login 的 Codex 认证

## 文件结构

| 文件 | 说明 | 依赖 |
|------|------|------|
| `generator.py` | 核心生成逻辑：HTTP SSE 流式请求、认证解析、SSE event 解析 | 无（纯标准库） |
| `codex_image_node.py` | ComfyUI 节点类 + tensor 转换 + CLI 入口 | torch, numpy, Pillow |
| `cli.py` | 独立 CLI 入口，可直接运行不依赖 ComfyUI | 无 |

## 安装

### ComfyUI 插件
1. 把整个文件夹复制到 `<ComfyUI>/custom_nodes/ComfyUI-CodexImage/`
2. 重启 ComfyUI 或点 Refresh
3. 在节点搜索器找 **`image/generation → CodexImage`**

### 独立 CLI（不需要 ComfyUI）
```bash
python cli.py "a cute cat"
```

## 使用方式

### ComfyUI 节点参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mode` | `auth` | `api` / `auth` / `cli` |
| `prompt` | — | 图片描述（必填） |
| `model` | `gpt-5.5` | 模型名 |
| `size` | `1024x1024` | 尺寸，如 `1024x1024`、`1792x1024`、`1024x1792` |
| `quality` | `medium` | `low` / `medium` / `high` |
| `format` | `png` | `png` / `jpeg` / `webp` |
| `output_path` | _(空)_ | 可选，保存副本到指定路径 |

**隐藏字段（ComfyUI UI 不显示）：**

| 字段 | 默认值 | 适用模式 |
|------|--------|---------|
| `base_url` | `https://chatgpt.com/backend-api/codex` | `api` |
| `api_key` | _(空)_ | `api` |
| `codex_cmd` | `codex exec -- sh -c {CMD}` | `cli` |

### CLI 用法

```bash
# Codex Auth 模式（默认，自动读 ~/.codex/auth.json）
python cli.py "a cat"

# API 模式（自己填 URL + Key）
python cli.py "a cat" --mode api \
  --base-url https://chatgpt.com/backend-api/codex \
  --api-key sk-xxxx

# CLI 模式（通过 codex exec）
python cli.py "a cat" --mode cli

# 指定输出路径
python cli.py "a cat" --out ./output.png
```

## 工作流示例

```
[CLIP Text Encode] → [CodexImageNode] → [PreviewImage / SaveImage]
```

## 实现细节

### 认证优先级（auth 模式）

1. `OPENAI_API_KEY` 环境变量
2. `~/.codex/auth.json` 中的 `OPENAI_API_KEY` 字段
3. `~/.codex/auth.json` 中的 ChatGPT OAuth `access_token`（`codex login` 时生成）

### API 请求流程

1. 构建 JSON payload（含 `model`、`instructions`、`input`、`tools`）
2. POST 到 `/{base_url}/v1/responses`，请求头带 `Accept: text/event-stream`
3. 分块读取 SSE 流，解析每行 `data: {...}` JSON
4. 从 event 中提取 `response.image_generation_call.done` 的 `result` 字段（base64 编码的图片）
5. base64 解码得到原始图片 bytes

### CLI 模式原理

把生成命令包装成：
```
codex exec -- sh -c "python ~/.codex-image/scripts/codex_image.py 'prompt' --size X ..."
```
`codex exec` 会用本机已 login 的身份执行命令，认证逻辑完全由它处理。
