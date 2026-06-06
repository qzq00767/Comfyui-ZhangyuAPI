# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ComfyUI custom node package for the ZhangyuAPI GPT image generation service. Provides image generation (GPT-Image-2), video generation, prompt optimization (via chat-completions LLM), and a text-list pause-editor node — all backed by OpenAI-compatible REST APIs.

## Architecture

```
__init__.py              # Aggregates NODE_CLASS_MAPPINGS from all modules
zhangyu_gpt_img2.py      # Core: shared HTTP/retry/polling/image utilities + ComfyuiZhangyuAPIImage2Node
zhangyu_img.py           # ComfyuiZhangyuAPIUniversalImageNode — generic OpenAI-compatible image gen
video_node.py            # ComfyuiZhangyuAPIVideoNode — video generation with async polling
zhangyuapi_prompt_control_nodes.py  # ZhangyuAPIPromptOptimizer + ZhangyuAPITextListEditor
prompts/                 # LLM system prompt templates (txt) + presets (JSON)
web/extensions/          # Frontend JS: runtime status bar + preset helper
```

### Module dependency graph

`zhangyu_gpt_img2.py` is the **shared foundation module** — it defines the per-thread HTTP/2 client, retry/backoff helpers, adaptive polling, image tensor conversion, size tables, and frontend status emitter. **All other node modules import from it.**

- `zhangyu_img.py` imports ~20 shared functions from `zhangyu_gpt_img2`
- `video_node.py` imports ~15 shared functions from `zhangyu_gpt_img2`
- `zhangyuapi_prompt_control_nodes.py` imports HTTP helpers + status emitter from `zhangyu_gpt_img2`

### Threading model (critical context)

ComfyUI invokes `generate()` / `optimize()` from a **ThreadPoolExecutor**. Since `httpx.Client` is not thread-safe, each thread gets its own `httpx.Client` via `threading.local()`. The `_run_async_coroutine()` bridge in `zhangyu_gpt_img2.py` handles running async code (concurrent image downloads) from these sync thread-pool contexts.

### Backend ↔ Frontend communication

- **Model input**: All nodes use a STRING widget for manual model name input (auto-fetch was removed).
- **Runtime status**: Python calls `emit_runtime_status()` → `PromptServer.instance.send_sync("comfyui_zhangyuapi_status", ...)`. The JS status extension listens for these events and renders a progress bar per node.
- **Text-list editor**: Uses a session-based pause/confirm pattern with `_pending_text_lists` dict + routes like `POST /zhangyuapi_text_list_edit/confirm`.

## No build/lint/test pipeline

This is a ComfyUI custom node package — there is **no build step, no linter config, and no test suite**. "Testing" means loading the nodes in ComfyUI and running a workflow. The `example_workflow.json` in the repo root serves as both demo and manual test harness.

## Key patterns when modifying code

### Adding a new node

1. Create a new `.py` module with a node class, `NODE_CLASS_MAPPINGS`, and `NODE_DISPLAY_NAME_MAPPINGS`.
2. Import and merge into `__init__.py`.
3. If the node needs a runtime status bar, add its class name to `TARGET_NODE_TYPES` in the JS status extension.
4. Register any new PromptServer route in the module's top-level `try/except` block (copy the existing route-registration pattern).

### Widget naming convention

Widget keys in `INPUT_TYPES` use the pattern `display_name (中文描述)` — e.g. `"api_key (API密钥)"`, `"timeout_seconds (超时秒数)"`. These exact strings are used both as kwargs keys in `generate()` and as `widget.name` lookups in the frontend JS. **Never rename widgets without updating both Python kwargs access and JS `find*Widget()` functions.**

### Sharing code between nodes

Import from `zhangyu_gpt_img2.py` — that's where shared infrastructure lives. Don't duplicate retry logic, polling, image conversion, or HTTP helpers.

### Prompt templates

The `prompts/` directory contains:
- `.txt` files: system prompts loaded at module import time via `pathlib.Path.read_text()`.
- `presets/*.json`: each has `name`, `system_prompt` (currently unused at runtime), and `default_params` which map to widget values via `_PRESET_KEY_MAP`.

### API endpoints used

| Endpoint | Used by |
|---|---|
| `POST /v1/images/generations` | Image-2 node (text2img), Universal node |
| `POST /v1/images/edits` | Image-2 node (img2img / inpainting) |
| `POST /v1/videos` or `/v1/videos/generations` | Video node (auto-detects) |
| `GET /v1/tasks/{id}` | Async task polling |
| `GET /v1/videos/{id}` | Video task polling |
| `GET /v1/videos/{id}/content` | Video download |
| `POST /v1/chat/completions` | Prompt optimizer (stream + non-stream) |

Auth is always `Authorization: Bearer <api_key>`. The node uses HTTP/2 via `httpx` with `trust_env=False` (bypasses system proxies).
