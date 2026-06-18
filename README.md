# Comfyui-ZhangyuAPI

ComfyUI 自定义节点，对接 **zhangyuapi.com** — 它允许你在 ComfyUI 工作流中直接调用任意兼容 OpenAI Images API 格式的第三方服务。

## 📖 简介

Comfyui-ZhangyuAPI 提供 **9 个节点**，覆盖从"想生成什么"到"怎么生成"的完整链路——提示词优化 → 生图（GPT-Image-2 / OpenAI / Gemini）→ 视频生成（Sora / Kling / 即梦）→ 文本编辑 → 中译英翻译。所有节点通过 REST API 通信，支持 HTTP/2 直连、自适应轮询和自动重试。

**核心亮点：**

- ✅ **GPT-Image-2 原生节点** — 支持 size / quality / format / mask 全参数
- ✅ **通用 OpenAI 生图** — 兼容 DALL-E / FLUX / SD / Midjourney 等任意模型
- ✅ **Gemini 格式生图** — 适配 `generateContent` 协议，支持图生图
- ✅ **Sora 格式视频** — 严格遵循 OpenAI Sora 规范，multipart/form-data
- ✅ **可灵格式视频** — 文生视频 / 图生视频，支持 Kling 兼容端点
- ✅ **即梦格式视频** — 支持 `CVSync2Async` 提交流程
- ✅ **提示词优化** — LLM 根据用户需求自动生成结构化生图提示词
- ✅ **文本停留编辑** — 工作流执行中暂停，手动编辑文本后继续
- ✅ **中译英翻译** — 中文提示词 → 英文，适配英文响应更好的模型

## ✨ 特性

| 特性 | 说明 |
|------|------|
| 通用兼容 | 兼容 OpenAI、Sora、Kling、Jimeng、Gemini 多种 API 协议格式 |
| 异步轮询 | API 返回任务 ID 后自动自适应轮询（1s → 3s → 8s → 15s），最多等待超时上限 |
| 多图输出 | 支持一次生成多张图片（n 参数，最多 10 张） |
| 多图输入 | 图生图模式支持最多 8 张参考图 + mask 遮罩 |
| 灵活参数 | 支持 model、size、quality、aspect_ratio、output_format、seed、steps、cfg_scale 等 |
| 模型感知校验 | 自动检测模型家族（GPT / DALL-E / SD / FLUX / Midjourney），适配合法参数 |
| 进度条 | 前端实时显示生成进度，自适应曲线加速感知 |
| 自动重试 | 408 / 429 / 5xx 自动退避重试 |
| 中译英翻译 | 中文提示词 → 英文，适配英文响应更好的生图模型 |

## 🔧 安装

进入 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-repo/Comfyui-ZhangyuAPI.git
cd Comfyui-ZhangyuAPI
pip install -r requirements.txt
```

重启 ComfyUI，搜索 `zhangyuapi` 即可找到所有节点。

## 🎮 节点说明

### 总览（9 个节点）

| 节点 | 分类 | 协议 | 说明 |
|------|------|------|------|
| `Comfyui-ZhangyuAPI-image-2` | 生图 | OpenAI | GPT-Image-2 原生节点，支持 size / quality / aspect_ratio / mask |
| `ComfyUI-zhangyuapi-通用openai格式` | 生图 | OpenAI | 严格遵循 `/v1/images/generations`，支持 style / 参考图 |
| `ComfyUI-zhangyuapi-通用Gemini格式` | 生图 | Gemini | Gemini `generateContent` 协议生图 |
| `ComfyUI-zhangyuapi-Sora格式` | 视频 | Sora | Sora 视频生成，严格遵循 OpenAI Sora 规范 |
| `ComfyUI-zhangyuapi-可灵格式` | 视频 | Kling | 可灵文生视频 / 图生视频 |
| `ComfyUI-zhangyuapi-即梦格式` | 视频 | Jimeng | 即梦 `CVSync2Async` 视频生成 |
| `ComfyUI-zhangyuapi-提示词优化器` | 文本 | OpenAI | 文字需求 → 结构化生图提示词，支持参考图模式 |
| `ComfyUI-zhangyuapi-文本停留编辑器` | 文本 | — | 工作流执行中暂停，手动编辑文本后继续 |
| `ComfyUI-zhangyuapi-中译英` | 工具 | LLM | 中文提示词 → 英文，适配英文响应更好的模型 |

---

### Comfyui-ZhangyuAPI-image-2

**输入参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key (API密钥)` | STRING | — | zhangyuapi.com 的 API Key |
| `prompt (提示词)` | STRING | — | 正向提示词（多行文本） |
| `mode (模式)` | ENUM | `AUTO` | `AUTO` / `text2img` / `img2img` |
| `model (模型)` | ENUM | `gpt-image-2` | 模型选择 |
| `custom_model (自定义模型名)` | STRING | — | 自定义模型名，填写后覆盖下拉选择 |
| `n (生成数量)` | INT | 1 | 生成图片数量（1~5） |
| `api_base (接口域名)` | STRING | `https://zhangyuapi.com/v1` | API 基础地址 |
| `image_size (分辨率)` | ENUM | `1K` | `auto` / `1K` / `2K` / `4K` |
| `aspect_ratio (宽高比)` | ENUM | `1:1` | `AUTO` / `1:1` / `16:9` / `9:16` 等 |
| `quality (画质)` | ENUM | `auto` | `auto` / `low` / `medium` / `high` |
| `response_format (响应格式)` | ENUM | `b64_json` | `b64_json` / `url` |
| `output_format (输出格式)` | ENUM | `jpeg` | `png` / `jpeg` / `webp` |
| `output_compression (压缩率)` | INT | 85 | 0~100 |
| `seed (种子)` | INT | 0 | 0 表示随机 |
| `timeout_seconds (超时秒数)` | INT | 360 | 60~1800 |
| `retry_times (重试次数)` | INT | 3 | 1~10 |

**可选输入**：`image_01` ~ `image_08`（参考图）、`mask`（遮罩）

**输出端口**

| 端口 | 类型 | 说明 |
|------|------|------|
| `image` | IMAGE | 生成的图像张量 |
| `response` | STRING | 请求摘要（模型、耗时、URL 等关键信息） |
| `image_urls` | STRING | 图片 URL 列表 |
| `chats` | STRING | API 原始响应（已剔除 base64 数据） |
| `model_list` | STRING | 当前可用模型列表 |

---

### ComfyUI-zhangyuapi-通用openai格式

严格遵循 OpenAI `POST /v1/images/generations` 接口规范，参数布局对齐 Image-2 节点。适合通过 NewAPI 中继调用 DALL-E / FLUX / SD 等兼容 OpenAI 格式的后端模型。

**输入参数**（与 Image-2 基本一致，额外支持 `style`）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key (API密钥)` | STRING | — | zhangyuapi.com 的 API Key |
| `prompt (提示词)` | STRING | — | 正向提示词（多行文本） |
| `model (模型)` | STRING | `gpt-image-2` | 模型名，自动过滤兼容模型列表 |
| `api_base (接口域名)` | STRING | `https://zhangyuapi.com/v1` | API 基础地址 |
| `image_size (分辨率)` | ENUM | `1K` | `auto` / `1K` / `2K` / `4K` |
| `aspect_ratio (宽高比)` | ENUM | `1:1` | `1:1` / `16:9` / `9:16` / `4:3` / `3:4` / `3:2` / `2:3` / `21:9` |
| `quality (画质)` | ENUM | `auto` | `auto` / `low` / `medium` / `high` |
| `response_format (响应格式)` | ENUM | `b64_json` | `b64_json` / `url` |
| `output_format (输出格式)` | ENUM | `jpeg` | `png` / `jpeg` / `webp` |
| `output_compression (压缩率)` | INT | 85 | 0~100 |
| `n (生成数量)` | INT | 1 | 1~5 |
| `seed (种子)` | INT | 0 | 0 表示随机 |
| `timeout_seconds (超时秒数)` | INT | 360 | 60~1800 |
| `retry_times (重试次数)` | INT | 2 | 1~10 |

**可选输入**：`style (风格)`（`vivid` / `natural`）、`image_01` ~ `image_08`（参考图，触发图生图）

**输出端口**

| 端口 | 类型 | 说明 |
|------|------|------|
| `image` | IMAGE | 生成的图像张量 |
| `response` | STRING | 请求摘要（模型、耗时、URL 等关键信息） |
| `image_urls` | STRING | 图片 URL 列表 |
| `chats` | STRING | API 原始响应（已剔除 base64 数据） |
| `model_list` | STRING | 当前可用模型列表 |

> 该节点与 Image-2 共享同一套基础设施（`zhangyu_gpt_img2.py`），支持异步轮询、自适应退避和前端进度条。与 Image-2 的主要区别：不支持 `mode` / `mask` 参数，但额外提供 `style`（vivid / natural）选项。

---

### ComfyUI-zhangyuapi-通用Gemini格式

适配 Gemini `generateContent` 协议，通过 NewAPI 中继调用原生 Gemini 模型生成图片。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt (提示词)` | STRING | — | 正向提示词（多行文本） |
| `model (模型)` | STRING | `nano-banana` | Gemini 模型名 |
| `api_base (接口域名)` | STRING | `https://zhangyuapi.com` | API 基础地址 |
| `image_size (分辨率)` | ENUM | `1K` | `1K` / `2K` / `4K` |
| `aspect_ratio (宽高比)` | ENUM | `1:1` | `1:1` / `16:9` / `9:16` 等 |
| `n (生成数量)` | INT | 1 | 1~4 |
| `seed (种子)` | INT | 0 | 0 表示随机 |
| `temperature (创造性)` | FLOAT | 0.7 | 0.0~2.0 |
| `output_format (输出格式)` | ENUM | `jpeg` | `png` / `jpeg` / `webp` |

**可选输入**：`image_01` ~ `image_08`（参考图，触发图生图模式）

---

### ComfyUI-zhangyuapi-Sora格式

严格遵循 OpenAI Sora API 规范（`POST /v1/videos` multipart/form-data）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt (提示词)` | STRING | — | 视频描述 |
| `model (模型)` | STRING | `sora` | 模型选择 |
| `api_base (接口域名)` | STRING | `https://zhangyuapi.com/v1` | API 基础地址 |
| `size (分辨率)` | ENUM | `1280x720` | 视频分辨率 |
| `duration (时长秒数)` | INT | 8 | 4~60 |
| `seed (种子)` | INT | 0 | 0 表示随机 |
| `fps (帧率)` | INT | 24 | 1~120 |
| `negative_prompt (反向提示词)` | STRING | — | 反向提示词 |
| `n (生成数量)` | INT | 1 | 1~4 |

**可选输入**：`image_01`（参考图）

**输出**：`video`（视频文件）、`response`（JSON 摘要）

---

### ComfyUI-zhangyuapi-可灵格式

适配 Kling 兼容端点，支持文生视频 + 图生视频。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt (提示词)` | STRING | — | 视频描述 |
| `model (模型)` | STRING | `kling-v1` | 模型选择 |
| `api_base (接口域名)` | STRING | `https://zhangyuapi.com/v1` | API 基础地址 |
| `size (分辨率)` | ENUM | `1280x720` | 视频分辨率 |
| `duration (时长秒数)` | INT | 8 | 4~60 |
| `seed (种子)` | INT | 0 | 0 表示随机 |
| `fps (帧率)` | INT | 24 | 1~120 |
| `negative_prompt (反向提示词)` | STRING | — | 反向提示词 |
| `n (生成数量)` | INT | 1 | 1~4 |

**可选输入**：`image_01`（参考图，触发图生视频）

**输出**：`video`（视频文件）、`response`（JSON 摘要）

---

### ComfyUI-zhangyuapi-即梦格式

适配即梦 (Jimeng) `CVSync2Async` 端点，通过 Action query param 区分提交/查询。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt (提示词)` | STRING | — | 视频描述 |
| `model (模型)` | STRING | `jimeng` | 模型选择 |
| `api_base (接口域名)` | STRING | `https://zhangyuapi.com/v1` | API 基础地址 |
| `req_key (请求类型)` | ENUM | `jimeng_t2v` | 文生视频 / 图生视频 等 |
| `version (API版本)` | STRING | `2024-02-28` | 即梦 API 版本号 |
| `seed (种子)` | INT | 0 | 0 表示随机 |
| `duration (时长秒数)` | INT | 5 | 2~60 |
| `fps (帧率)` | INT | 24 | 1~120 |
| `negative_prompt (反向提示词)` | STRING | — | 反向提示词 |

**可选输入**：`image_01` ~ `image_04`（参考图，用于图生视频）

**输出**：`video`（视频文件）、`response`（JSON 摘要）

---

### ComfyUI-zhangyuapi-提示词优化器

将用户的自然语言需求转化为结构化生图提示词。支持：

- **文本模式**：输入文字需求 → LLM 生成优化后的提示词
- **参考图模式**：上传 1~5 张参考图 → LLM 分析风格/构图/版式 → 生成约束提示词
- **流式输出**：可开启 `stream` 实时查看 LLM 生成过程
- **预设系统提示词**：内置「复刻提示词」「通用反推模板」等预设

| 关键参数 | 说明 |
|----------|------|
| `user_prompt (用户需求)` | 自然语言描述想要生成什么 |
| `model (模型)` | LLM 模型选择 |
| `stream (流式输出)` | 是否开启流式 |
| `temperature (创造性)` | 0.0~2.0 |
| `max_tokens (最大长度)` | 64~32768 |
| `preset (预设)` | 选择内置系统提示词预设 |
| `reference_image_01~05` | 参考图（触发参考图模式） |

---

### ComfyUI-zhangyuapi-文本停留编辑器

工作流执行到该节点时暂停，弹出编辑器让用户手动修改文本，确认后继续执行。适合需要人工审核或调整中间文本的场景。

---

### ComfyUI-zhangyuapi-中译英

将中文提示词翻译为英文，适配对英文响应更好的生图模型。支持自定义模型选择，可与其他节点组合使用。

| 关键参数 | 说明 |
|----------|------|
| `prompt_cn (中文提示词)` | 中文描述 |
| `model (模型)` | LLM 模型选择 |
| `api_base (接口域名)` | API 基础地址 |

**输出**：`prompt_en`（英文提示词）

## 📋 工作流示例

### 文生图

```
[提示词优化器] → [Image-2 生图] → [Save Image]
```

1. 提示词优化器将"一只戴帽子的猫"转为结构化提示词
2. Image-2 节点用优化后的提示词生成图片

### 图生图 + 参考图

```
[Load Image] → [提示词优化器] → [通用生图] → [Save Image]
                     ↑ 参考图输入
```

参考图传入优化器，LLM 分析图片风格后生成匹配的提示词，再送入生图节点。

### Gemini 生图

```
[提示词优化器] → [Gemini 生图] → [Save Image]
```

通过 `generateContent` 协议调用 Gemini 模型生图，支持图生图参考。

### 中译英 + 生图

```
[中译英翻译] → [Image-2 生图] → [Save Image]
```

1. 中译英节点将中文提示词翻译为英文
2. 英文提示词送入生图节点生成图片

### 视频生成

```
[提示词优化器] → [Sora格式 / 可灵格式 / 即梦格式] → [Save Video]
```

根据目标视频 API 格式选择对应节点，提示词优化器 + 视频节点组合使用。

### 带人工审核

```
[提示词优化器] → [文本停留编辑器] → [通用生图] → [Save Image]
                       ↑ 人工修改提示词
```

## 🌐 API 端点

所有节点使用以下端点（`{api_base}` 默认 `https://zhangyuapi.com/v1`）：

| 端点 | 方法 | 用途 | 节点 |
|------|------|------|------|
| `/v1/models` | GET | 获取可用模型列表 | Image-2 / 通用 OpenAI / Sora / Kling |
| `/v1beta/models` | GET | Gemini 模型列表 | Gemini |
| `/v1/images/generations` | POST | 文生图 | Image-2 / 通用 OpenAI |
| `/v1/images/edits` | POST | 图生图 / 重绘 | Image-2 / 通用 OpenAI |
| `/v1beta/models/{model}:generateContent` | POST | Gemini 生图 | Gemini |
| `/v1/videos` | POST | Sora 视频生成 (multipart) | Sora 格式 |
| `/v1/videos/{id}` | GET | Sora 视频状态查询 | Sora 格式 |
| `/v1/videos/{id}/content` | GET | Sora 视频下载 | Sora 格式 |
| `/kling/v1/videos/text2video` | POST | 可灵文生视频 | 可灵格式 |
| `/kling/v1/videos/image2video` | POST | 可灵图生视频 | 可灵格式 |
| `/kling/v1/videos/text2video/{id}` | GET | 可灵状态查询 | 可灵格式 |
| `/jimeng/?Action=CVSync2AsyncSubmitTask` | POST | 即梦提交任务 | 即梦格式 |
| `/jimeng/?Action=CVSync2AsyncGetResult` | POST | 即梦查询结果 | 即梦格式 |
| `/v1/chat/completions` | POST | 提示词优化（LLM） | 提示词优化器 |
| `/v1/chat/completions` | POST | 中译英翻译（LLM） | 中译英 |

鉴权格式：`Authorization: Bearer YOUR_API_KEY`

## 🧩 架构

```
用户输入 → INPUT_TYPES 校验 → generate()
                │
                ├─ 提示词优化 → POST /v1/chat/completions → 流式/非流式
                ├─ 中译英翻译 → POST /v1/chat/completions → 英文提示词
                │
                ├─ 文生图 → POST /v1/images/generations
                ├─ 图生图 → POST /v1/images/edits
                ├─ Gemini → POST /v1beta/models/{model}:generateContent
                ├─ Sora   → POST /v1/videos (multipart)
                ├─ 可灵   → POST /kling/v1/videos/text2video
                │          POST /kling/v1/videos/image2video
                ├─ 即梦   → POST /jimeng/?Action=CVSync2AsyncSubmitTask
                │
                ├─ 同步响应 → 解码图片/视频 → 返回
                └─ 异步响应 → 自适应轮询(1s→3s→8s→15s) → 返回
                              │
                              └─ 前端进度条实时更新
```

**模块依赖图**：`zhangyu_gpt_img2.py` 是共享基础设施，所有其他节点模块均从它导入 HTTP 客户端、重试/退避、轮询、图像转换等公共函数。

## 🔧 常见问题

**旧工作流报错** — 视频节点已从统一的「视频生成」拆分为 Sora、可灵、即梦三个独立格式节点，请重新添加节点或使用最新 `example_workflow.json`。

**模型列表为空** — 检查 API Key 是否正确，接口域名是否可达。

**408 / 429 / 5xx 错误** — 节点会自动重试（`retry_times` 次），无需手动干预。

**新模型不在下拉列表中** — 在 `model (模型)` 字段直接填写模型名即可。

**中译英翻译失败** — 检查 API Key 和模型名称是否正确，确保使用支持翻译的 LLM 模型。

## 📄 许可证

MIT

## 📬 联系

如有问题或建议，欢迎提交 Issue。
