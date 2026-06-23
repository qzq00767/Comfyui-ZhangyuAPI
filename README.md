# Comfyui-ZhangyuAPI

ComfyUI 自定义节点包，提供 **13 个节点**，覆盖图片生成、视频生成、提示词优化等场景。

## 环境要求

- **Python**: 3.10+
- **ComfyUI**: 最新版

## 依赖库

| 库 | 版本 | 说明 |
|----|------|------|
| `httpx` | >=0.27 | HTTP 客户端（支持 HTTP/2）|
| `numpy` | >=1.20 | 数组处理（ComfyUI 自带）|
| `Pillow` | >=9.0 | 图片处理（ComfyUI 自带）|
| `torch` | >=2.0 | 张量处理（ComfyUI 自带）|

大部分依赖 ComfyUI 已内置，通常无需额外安装。

## 安装

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-repo/Comfyui-ZhangyuAPI.git
cd Comfyui-ZhangyuAPI
pip install -r requirements.txt
```

重启 ComfyUI，搜索 `zhangyuapi` 即可。

## 节点列表

### 图片生成

| 节点 | 说明 |
|------|------|
| **Comfyui-ZhangyuAPI-image-2** | GPT Image 2 专用节点，支持 text2img/img2img/inpainting |
| **ComfyUI-zhangyuapi-通用openai格式** | 通用 OpenAI 格式，兼容 DALL-E/FLUX/SD 等 |
| **ComfyUI-zhangyuapi-通用Gemini格式** | Gemini 原生 generateContent 协议 |
| **ComfyUI-zhangyuapi-NanoBanana生图** | Nano Banana 系列模型 |
| **ComfyUI-zhangyuapi-通用LLM接口** | 通用 LLM 调用（支持文本/图片/视频）|

### 视频生成

| 节点 | 说明 |
|------|------|
| **ComfyUI-zhangyuapi-Sora格式** | OpenAI Sora 视频生成 |
| **ComfyUI-zhangyuapi-可灵视频** | 可灵视频（5合1：文生/图生/多图/延长/唇形同步）|
| **ComfyUI-zhangyuapi-即梦格式** | 即梦视频生成 |
| **ComfyUI-zhangyuapi-Veo3视频** | Google Veo3 视频生成 |
| **ComfyUI-zhangyuapi-通义万相视频** | 阿里云通义万相视频生成 |

### 工具

| 节点 | 说明 |
|------|------|
| **ComfyUI-zhangyuapi-提示词优化器** | LLM 提示词优化/反推 |
| **ComfyUI-zhangyuapi-文本停留编辑器** | 工作流暂停编辑文本 |
| **ComfyUI-zhangyuapi-中译英** | 中文提示词翻译为英文 |

## 可灵节点功能

通过"功能模式"下拉切换：
- **文生视频** — 输入提示词生成视频
- **图生视频** — 参考图 + 提示词生成视频
- **多图转视频** — 最多4张参考图生成视频
- **视频延长** — 延长已有视频
- **唇形同步** — 语音驱动口型

## 通用参数

| 参数 | 说明 |
|------|------|
| `api_key` | API 密钥 |
| `api_base` | 接口地址（默认 zhangyuapi.com）|
| `model` | 模型名称 |
| `timeout` | 超时时间 |
| `retry_times` | 重试次数 |

## API 端点

| 端点 | 用途 |
|------|------|
| `POST /v1/images/generations` | 文生图 |
| `POST /v1/images/edits` | 图生图/编辑 |
| `POST /v1/chat/completions` | LLM/Nano Banana |
| `POST /v1/videos` | Sora 视频 |
| `POST /kling/v1/videos/*` | 可灵视频 |
| `POST /v2/videos/generations` | Veo3 视频 |
