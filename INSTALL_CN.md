# ComfyUI-CodexImage 安装说明

## 安装

1. 解压压缩包。
2. 把整个 `ComfyUI-CodexImage` 文件夹放到 ComfyUI 的 `custom_nodes` 目录。
3. 重启 ComfyUI。
4. 在节点菜单里找到：
   - `Codex Image (GPT Image 2)`
   - `Codex Image I2I (GPT Image 2)`

## 认证

节点默认使用 `auth` 模式，会读取当前用户的 `~/.codex/auth.json`。

如果接收方没有 Codex 登录状态，可以改用 `api` 模式，并在隐藏输入里配置 `base_url` 和 `api_key`。

## I2I 和 Mask

`Codex Image I2I (GPT Image 2)` 支持：

- `image`: 主输入图。
- `image_2`: 可选第二参考图。
- `mask`: 可选遮罩。

Mask 规则：

- 白色区域：编辑或重绘。
- 黑色区域：尽量保留。

使用 mask 时，节点会把白色遮罩区域转换成透明区域发给图像编辑接口。

## 依赖

这个节点没有额外 pip 依赖。它只使用 ComfyUI 自带的 `torch`、`numpy` 和 `Pillow`。
