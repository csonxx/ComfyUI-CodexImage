# ComfyUI CodexImage

GPT Image 2 (gpt-5.5) 生图节点，支持三种认证方式。

## 三种模式

| 模式 | 值 | 认证方式 |
|------|-----|---------|
| **API** | `api` | 用户填 `base_url` + `api_key` |
| **Codex Auth** | `auth` | 自动读 `~/.codex/auth.json`（`OPENAI_API_KEY` 字段或 `codex login` 的 OAuth token） |
| **CLI** | `cli` | 调用本机 `codex exec` 命令，使用本地已 login 的 Codex 认证 |

## 安装

### ComfyUI 插件
1. 把整个文件夹复制到 `<ComfyUI>/custom_nodes/ComfyUI-CodexImage/`
2. 重启 ComfyUI 或点 Refresh
3. 在节点搜索器里找 **`image/generation → CodexImage`**

### 独立 CLI
```bash
cd ComfyUI-CodexImage
python cli.py "a cute cat" --size 1024x1024
```

## 依赖

| 文件 | 用途 | 依赖 |
|------|------|------|
| `generator.py` | 核心生成逻辑（纯标准库） | 无 |
| `codex_image_node.py` | ComfyUI 节点 + CLI 入口 | torch, numpy, Pillow |
| `cli.py` | 独立 CLI 入口 | 无 |

**注意**：把插件放进 ComfyUI 的 `custom_nodes/` 后，ComfyUI 自带的 Python 环境里已经有 torch 和 numpy，只需确保 Pillow 已安装即可。独立 CLI 模式不依赖任何第三方库。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | — | 图片描述（必填） |
| `model` | `gpt-5.5` | 模型名 |
| `size` | `1024x1024` | 尺寸，如 `1024x1024`、`1792x1024`、`1024x1792` |
| `quality` | `medium` | `low` / `medium` / `high` |
| `format` | `png` | `png` / `jpeg` / `webp` |
| `output_path` | _(空)_ | 可选，填了就保存一份副本到指定路径 |

**隐藏字段（ComfyUI UI 不显示，但可通过节点链调用）：**

| 字段 | 默认值 | 适用模式 |
|------|--------|---------|
| `base_url` | `https://chatgpt.com/backend-api/codex` | `api` |
| `api_key` | _(空)_ | `api` |
| `codex_cmd` | `codex exec -- sh -c {CMD}` | `cli` |

## CLI 用法

```bash
# Codex Auth 模式（默认，自动读 ~/.codex/auth.json）
python cli.py "a cat" --size 1024x1024

# API 模式（自己填 URL + Key）
python cli.py "a cat" --mode api \
  --base-url https://chatgpt.com/backend-api/codex \
  --api-key sk-xxxx

# CLI 模式（通过 codex exec 调用本地脚本）
python cli.py "a cat" --mode cli

# 指定输出路径
python cli.py "a cat" --out ./output.png
```

## 工作流示例

```
[CLIP Text Encode] → [CodexImageNode] → [PreviewImage / SaveImage]
```

1. 文本节点连到 `prompt`
2. 选择 `mode`（`auth` 最省事，本地有 `~/.codex/auth.json` 就够）
3. 运行，生成图片后可接入任意 ComfyUI 图像节点
