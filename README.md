# Comfyui-ZhangyuAPI

ComfyUI 自定义节点，对接 **zhangyuapi.com** 它允许你在 ComfyUI 工作流中直接调用任意兼容 OpenAI Images API 格式的第三方服务。

## 📖 简介

Comfyui-ZhangyuAPI 提供 5 个节点，覆盖从"想生成什么"到"怎么生成"的完整链路——提示词优化 → 图像生成 → 视频生成 → 文本编辑。所有节点通过 OpenAI 兼容 REST API 通信，支持 HTTP/2 直连、自适应轮询和自动重试。

**核心亮点：**

- ✅ **GPT-Image-2 原生节点** — 支持 size / quality / format / mask 全参数
- ✅ **通用生图接口** — 兼容 DALL-E / FLUX / SD / Midjourney 等任意模型
- ✅ **视频生成** — Sora / Veo / Kling 等视频模型，自适应异步轮询
- ✅ **提示词优化** — LLM 根据用户需求自动生成结构化生图提示词
- ✅ **文本停留编辑** — 工作流执行中暂停，手动编辑文本后继续

## ✨ 特性

| 特性 | 说明 |
|------|------|
| 通用兼容 | 兼容 `/v1/images/generations`、`/v1/images/edits`、`/v1/videos`、`/v1/chat/completions` 端点 |
| 异步轮询 | API 返回任务 ID 后自动自适应轮询（1s → 3s → 8s → 15s），最多等待超时上限 |
| HTTP/2 直连 | 强制直连绕过系统代理，低延迟高吞吐 |
| 多图输出 | 支持一次生成多张图片（n 参数，最多 10 张） |
| 多图输入 | 图生图模式支持最多 8 张参考图 + mask 遮罩 |
| 灵活参数 | 支持 model、size、quality、aspect_ratio、output_format、seed、steps、cfg_scale 等 |
| 模型感知校验 | 自动检测模型家族（GPT / DALL-E / SD / FLUX / Midjourney），适配合法参数 |
| 进度条 | 前端实时显示生成进度，自适应曲线加速感知 |
| 自动重试 | 408 / 429 / 5xx 自动退避重试 |

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

### 生图节点

| 节点 | 分类 | 说明 |
|------|------|------|
| `Comfyui-ZhangyuAPI-image-2` | 生图 | GPT-Image-2 原生节点，支持 size / quality / aspect_ratio / mask |
| `ComfyUI-zhangyuapi-通用生图接口` | 生图 | 全平台通用，DALL-E / FLUX / SD / Midjourney 兼容 |

### 视频节点

| 节点 | 分类 | 说明 |
|------|------|------|
| `ComfyUI-zhangyuapi-视频生成 🧪测试中` | 视频 | Sora / Veo / Kling 等视频模型，异步轮询 |

### 文本节点

| 节点 | 分类 | 说明 |
|------|------|------|
| `ComfyUI-zhangyuapi-提示词优化器 🧪测试中` | 文本 | 文字需求 → 结构化生图提示词，支持参考图模式 |
| `ComfyUI-zhangyuapi-文本停留编辑器 🧪测试中` | 文本 | 工作流执行中暂停，手动编辑文本后继续 |

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

### ComfyUI-zhangyuapi-通用生图接口

除 GPT-Image-2 专属参数外，额外支持：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `negative_prompt (反向提示词)` | STRING | — | 反向提示词 |
| `quality (画质)` | ENUM | `standard` | `standard` / `hd` |
| `style (风格)` | ENUM | `vivid` | `vivid` / `natural` |
| `steps (采样步数)` | INT | 30 | 1~150（SD/FLUX） |
| `cfg_scale (提示词引导强度)` | FLOAT | 7.0 | 1.0~30.0（SD/FLUX） |
| `sampler (采样器)` | ENUM | `auto` | 采样器选择 |
| `denoising_strength (重绘强度)` | FLOAT | 1.0 | 0.0~1.0（img2img） |
| `mj_ar (MJ宽高比)` | ENUM | `auto` | Midjourney `--ar` 映射 |
| `mj_stylize (MJ风格化)` | INT | 100 | Midjourney `--stylize` 映射 |
| `mj_chaos (MJ混乱度)` | INT | 0 | Midjourney `--chaos` 映射 |
| `mj_weird (MJ怪异度)` | INT | 0 | Midjourney `--weird` 映射 |

> 节点会自动检测模型家族并适配合法参数：DALL-E 3 限制 n=1、Midjourney 映射到对应 `--` 参数等。

---

### ComfyUI-zhangyuapi-视频生成 🧪测试中

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt (提示词)` | STRING | — | 视频描述 |
| `model (模型)` | ENUM | `auto (自动选择)` | 模型选择 |
| `seconds (时长秒数)` | INT | 8 | 4~60 |
| `size (分辨率)` | ENUM | `1280x720` | 视频分辨率 |
| `aspect_ratio (画面比例)` | ENUM | `16:9` | `16:9` / `9:16` / `1:1` |
| `negative_prompt (反向提示词)` | STRING | — | 反向提示词 |

**输出**：`video`（视频文件）、`response`（JSON 摘要）

---

### ComfyUI-zhangyuapi-提示词优化器 🧪测试中

将用户的自然语言需求转化为结构化生图提示词。支持：

- **文本模式**：输入文字需求 → LLM 生成优化后的提示词
- **参考图模式**：上传 1~5 张参考图 → LLM 分析风格/构图/版式 → 生成约束提示词
- **流式输出**：可开启 `stream` 实时查看 LLM 生成过程
- **预设系统提示词**：内置电商海报、社交媒体封面、纯视觉画面等预设

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

### ComfyUI-zhangyuapi-文本停留编辑器 🧪测试中

工作流执行到该节点时暂停，弹出编辑器让用户手动修改文本，确认后继续执行。适合需要人工审核或调整中间文本的场景。

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

### 视频生成

```
[提示词优化器] → [视频生成] → [Save Video]
```

### 带人工审核

```
[提示词优化器] → [文本停留编辑器] → [通用生图] → [Save Image]
                       ↑ 人工修改提示词
```

## 🌐 API 端点

所有节点使用以下端点（`{api_base}` 默认 `https://zhangyuapi.com/v1`）：

| 端点 | 方法 | 用途 |
|------|------|------|
| `/v1/models` | GET | 获取可用模型列表 |
| `/v1/images/generations` | POST | 文生图 |
| `/v1/images/edits` | POST | 图生图 / 重绘 |
| `/v1/videos` | POST | 视频生成 |
| `/v1/tasks/{id}` | GET | 查询任务状态 |
| `/v1/videos/{id}/content` | GET | 下载视频 |
| `/v1/chat/completions` | POST | 提示词优化（LLM） |

鉴权格式：`Authorization: Bearer YOUR_API_KEY`

## 🧩 架构

```
用户输入 → INPUT_TYPES 校验 → generate()
                │
                ├─ 提示词优化 → POST /v1/chat/completions → 流式/非流式
                │
                ├─ 文生图 → POST /v1/images/generations
                ├─ 图生图 → POST /v1/images/edits
                ├─ 视频   → POST /v1/videos
                │
                ├─ 同步响应 → 解码图片/视频 → 返回
                └─ 异步响应 → 自适应轮询(1s→3s→8s→15s) → 返回
                              │
                              └─ 前端进度条实时更新
```

## 🔧 常见问题

**旧工作流报错** — 重新添加节点或使用最新 `example_workflow.json`。

**模型列表为空** — 检查 API Key 是否正确，接口域名是否可达。

**408 / 429 / 5xx 错误** — 节点会自动重试（`retry_times` 次），无需手动干预。

**新模型不在下拉列表中** — 在 `custom_model (自定义模型名)` 字段直接填写模型名即可，无需修改代码。


## 📄 许可证

MIT

## 📬 联系

如有问题或建议，欢迎提交 Issue。
