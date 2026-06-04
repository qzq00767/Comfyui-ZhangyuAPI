#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — 提示词优化 + 文本停留编辑节点.

合并了原来的「图生图提示词控制器」和「GPT-Image-2 文生图提示词控制器」
为一个节点，根据是否连接参考图自动切换模式。支持流式 / 非流式对话生成。

文本能力核心：
- Endpoint: ``POST /v1/chat/completions``
- Auth: ``Authorization: Bearer <api_key>``
- 非流式: ``{"stream": false}`` → ``choices[0].message.content``
- 流式: ``{"stream": true}`` → SSE ``data: {...}`` 事件流

模型列表由前端自动从 ``GET /v1/models`` 获取，无需硬编码。
"""

import hashlib
import json
import pathlib
import re
import time
import uuid

import httpx

from .zhangyu_gpt_img2 import (
    ZHANGYUAPI_post,
    ZHANGYUAPI_timeout,
    _get_http_client,
    _RETRYABLE_EXCEPTIONS,
    _jittered_sleep,
    is_retryable_http_status,
    normalize_api_base,
    tensor_to_data_url,
    emit_runtime_status,
    DEFAULT_API_BASE_URL,
    # Model resolution & validation
    resolve_and_validate_model,
    _filter_chat_models,
)


# ===================================================================
# Constants
# ===================================================================

CATEGORY = "Comfyui-ZhangyuAPI/文本"

# Aspect ratios — kept for prompt optimisation context
ASPECT_RATIO_OPTIONS = [
    "auto",
    "1:1", "2:3", "3:2", "3:4", "4:5",
    "9:16", "16:9", "21:9",
]

_LANDSCAPE = {"16:9", "3:2", "2:1", "21:9", "3:1", "4:1", "8:1"}
_PORTRAIT = {"9:16", "2:3", "3:4", "1:2", "9:21", "1:3", "1:4", "1:8"}
_SQUARE = {"1:1"}

_REFERENCE_MODE_MAP = {
    "自动判断": "auto",
    "综合参考": "full_reference",
    "只参考风格": "style_only",
    "只参考构图": "composition_only",
    "只参考色彩光影": "color_lighting_only",
    "只参考版式": "layout_only",
}

_TEXT_POLICY_MAP = {
    "不加文字": "none",
    "保留原文": "preserve",
    "优化原文": "enhance",
    "自动生成": "generate",
}

_STRENGTH_MAP = {"light": "标准", "standard": "标准", "strong": "增强"}

# Map preset default_params keys → ComfyUI kwarg display names
_PRESET_KEY_MAP = {
    "layout_type": "layout_type (版式类型)",
    "text_policy": "text_policy (文字策略)",
    "optimize_strength": "optimize_strength (优化强度)",
}

_BANNED_TEXT_PHRASES = [
    "预留标题", "标题展示", "展示标题", "显示标题",
    "预留文字", "文字展示", "展示文字", "添加文案",
    "按钮文字", "具体文案", "标题区", "卖点栏", "品牌区", "信息栏",
    "主标题", "副标题", "卖点", "文字", "文案", "标签",
]

# Session storage for text-list editor
import threading as _threading
_pending_text_lists = {}
_pending_text_lists_lock = _threading.Lock()
_SESSION_TIMEOUT_SECONDS = 300


# ===================================================================
# Prompt files
# ===================================================================

_PROMPT_DIR = pathlib.Path(__file__).parent / "prompts"

_REFERENCE_SYSTEM_PROMPT = (_PROMPT_DIR / "reference_image_optimizer_system.txt").read_text(encoding="utf-8-sig")
_SCHEMA_PARSER_PROMPT = (_PROMPT_DIR / "gpt-image-2_schema_parser_v1.txt").read_text(encoding="utf-8-sig")
_RENDERER_PROMPT = (_PROMPT_DIR / "gpt-image-2_prompt_renderer_v1.txt").read_text(encoding="utf-8-sig")


# ===================================================================
# Presets
# ===================================================================

# Presets cache — loaded once at first access, reused thereafter
_PRESETS_CACHE = None


def _load_presets():
    """Load system-prompt presets from ``prompts/presets/*.json``.

    Results are cached after the first call since preset files only
    change on restart.

    Returns:
        ``dict`` — ``{display_name: {name, system_prompt, default_params}}``.
        Always contains at least ``"默认"``.
    """
    global _PRESETS_CACHE
    if _PRESETS_CACHE is not None:
        return _PRESETS_CACHE
    presets = {"默认": {}}
    presets_dir = _PROMPT_DIR / "presets"
    if not presets_dir.is_dir():
        _PRESETS_CACHE = presets
        return presets
    for f in sorted(presets_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            name = data.get("name", f.stem)
            presets[name] = data
        except Exception:
            pass
    _PRESETS_CACHE = presets
    return presets


def _get_preset_names():
    """Return a sorted list of preset display names for the widget."""
    return list(_load_presets().keys())


# ===================================================================
# Utility functions
# ===================================================================

def _chat_url(api_base):
    base = normalize_api_base(api_base)
    return f"{base}/v1/chat/completions"


def _api_headers(api_key):
    key = (api_key or "").strip()
    if not key:
        raise ValueError("API Key 不能为空")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _extract_chat_content(data):
    """Extract ``choices[0].message.content`` from an OpenAI chat response."""
    try:
        choices = data.get("choices") or []
        message = choices[0].get("message") or {}
        content = message.get("content", "")
    except Exception as exc:
        raise RuntimeError(f"API 响应格式异常: {str(data)[:500]}") from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def _call_chat_stream(api_base, api_key, model, messages,
                      timeout_seconds=600, temperature=0.7, max_tokens=4096):
    """Streaming call to ``POST /v1/chat/completions`` (SSE).

    Parses ``data: {...}`` lines, concatenates ``delta.content`` chunks,
    and returns the complete text.

    Args:
        api_base: Normalized base URL.
        api_key: Bearer token.
        model: Model ID.
        messages: Chat message list.
        timeout_seconds: Read timeout.
        temperature: Sampling temperature (0–2).
        max_tokens: Max output tokens.

    Returns:
        ``(full_text: str, raw_data: dict)`` — concatenated content and
        a synthetic response dict mimicking the non-streaming format.
    """
    url = _chat_url(api_base)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    content_parts = []
    finish_reason = None
    model_used = model

    with _get_http_client().stream(
        "POST", url,
        json=payload,
        headers=_api_headers(api_key),
        timeout=ZHANGYUAPI_timeout(timeout_seconds),
    ) as response:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"API 请求失败: HTTP {response.status_code}; "
                f"response={response.text[:2000] if response.text else '<empty>'}"
            ) from exc
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if "content" in delta and delta["content"] is not None:
                    content_parts.append(delta["content"])
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
            if chunk.get("model"):
                model_used = chunk["model"]

    full_text = "".join(content_parts)
    # Build a synthetic non-streaming-style response for downstream parsers
    synthetic_data = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": full_text,
            },
            "finish_reason": finish_reason or "stop",
        }],
        "model": model_used,
    }
    return full_text, synthetic_data


def _call_chat_nonstream(api_base, api_key, model, messages,
                          timeout_seconds=600, temperature=0.7, max_tokens=4096):
    """Non-streaming call to ``POST /v1/chat/completions``.

    Returns:
        ``(content_text: str, response_data: dict)``.
    """
    url = _chat_url(api_base)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = ZHANGYUAPI_post(
        url,
        timeout_seconds,
        headers=_api_headers(api_key),
        json=payload,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text[:2000] if response.text else "<empty>"
        raise RuntimeError(
            f"API 请求失败: HTTP {response.status_code}; response={body}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"API 返回非 JSON: {response.text[:1000]}"
        ) from exc

    raw = _extract_chat_content(data).strip()
    if not raw:
        raise RuntimeError(f"模型未返回有效内容: {str(data)[:500]}")
    return raw, data


def _call_chat(api_base, api_key, model, messages, timeout_seconds=600,
               stream=False, temperature=0.7, max_tokens=4096):
    """Unified chat entry point — dispatches to stream or non-stream.

    Returns:
        ``(content_text: str, response_data: dict)``.
    """
    if stream:
        return _call_chat_stream(
            api_base, api_key, model, messages,
            timeout_seconds, temperature, max_tokens,
        )
    return _call_chat_nonstream(
        api_base, api_key, model, messages,
        timeout_seconds, temperature, max_tokens,
    )


def _call_chat_with_retry(api_base, api_key, model, messages,
                           timeout_seconds=600, stream=False,
                           temperature=0.7, max_tokens=4096,
                           retry_times=2):
    """Call ``_call_chat`` with retry on transient errors.

    Retries on ``_RETRYABLE_EXCEPTIONS`` and retryable HTTP status codes
    (408, 429, 5xx).  Non-retryable errors propagate immediately.
    """
    last_error = None
    for attempt in range(1, retry_times + 1):
        try:
            return _call_chat(
                api_base, api_key, model, messages,
                timeout_seconds, stream, temperature, max_tokens,
            )
        except _RETRYABLE_EXCEPTIONS as exc:
            last_error = str(exc)
            if attempt < retry_times:
                _jittered_sleep(attempt)
                continue
            break
        except RuntimeError as exc:
            # Both _call_chat_stream and _call_chat_nonstream wrap
            # HTTPStatusError in RuntimeError before re-raising.
            # Re-raise non-HTTP errors directly.
            msg = str(exc)
            if "HTTP " not in msg:
                raise
            # Extract status code from the wrapped RuntimeError message.
            # Format: "API 请求失败: HTTP {code}; response=..."
            # NOTE: this relies on the error message format.  If the
            # downstream functions change their error phrasing, retry
            # logic may silently stop working.
            m = re.search(r"HTTP (\d{3})", msg)
            if m and is_retryable_http_status(int(m.group(1))):
                last_error = msg
                if attempt < retry_times:
                    _jittered_sleep(attempt)
                    continue
            raise
    raise RuntimeError(
        f"LLM 调用连续 {retry_times} 次失败，最后错误: {last_error}"
    )


def _parse_tagged_output(raw):
    """Parse ``optimized_prompt: ...`` and ``reference_summary: ...`` tags."""
    def extract(tag):
        pattern = rf"{tag}:\s*(.*?)(?=\n\w+_\w+:|$)"
        match = re.search(pattern, raw, re.DOTALL)
        return match.group(1).strip() if match else ""

    optimized_prompt = extract("optimized_prompt")
    reference_summary = extract("reference_summary")
    return optimized_prompt, reference_summary


def _build_reference_message(ref_urls, user_prompt, reference_mode,
                              target_aspect_ratio, subject_url=None):
    """Build a multimodal message for reference-image analysis."""
    content = []
    if subject_url is not None:
        content.append({"type": "text", "text": "以下是 subject_image（主体图）："})
        content.append({"type": "image_url", "image_url": {"url": subject_url}})
    content.append({
        "type": "text",
        "text": (
            f"以下是 {len(ref_urls)} 张 reference_image（参考图）。请综合分析这些参考图的构图、光影、色彩、"
            "人物/产品关系、画面风格、文字版式和共同视觉规律；如果多张图存在差异，请优先提炼可迁移的共性，"
            "不要机械拼接互相冲突的细节。"
        ),
    })
    for index, ref_url in enumerate(ref_urls, start=1):
        content.append({"type": "text", "text": f"reference_image_{index:02d}："})
        content.append({"type": "image_url", "image_url": {"url": ref_url}})
    has_subject = "是" if subject_url is not None else "否"
    content.append({
        "type": "text",
        "text": (
            f"用户需求：{user_prompt}\n"
            f"是否提供 subject_image：{has_subject}\n"
            f"reference_image_count：{len(ref_urls)}\n"
            f"reference_mode：{reference_mode}\n"
            f"target_aspect_ratio：{target_aspect_ratio}"
        ),
    })
    return content


def _clean_json_block(raw):
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    return text


def _parse_json_response(raw):
    text = _clean_json_block(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"schema 解析失败，模型输出非 JSON: {raw[:500]}")


def _ratio_to_direction(aspect_ratio):
    if aspect_ratio in _LANDSCAPE:
        return "横版构图"
    if aspect_ratio in _PORTRAIT:
        return "竖版构图"
    if aspect_ratio in _SQUARE:
        return "方形构图"
    return "由画面内容决定"


def _build_input_payload(layout_type, optimize_strength, aspect_ratio,
                         user_prompt, exact_text, text_policy):
    return {
        "layout_type": layout_type,
        "optimize_strength": optimize_strength,
        "aspect_ratio": aspect_ratio,
        "direction": _ratio_to_direction(aspect_ratio),
        "user_prompt": user_prompt or "",
        "exact_text": exact_text or "",
        "text_policy": text_policy,
    }


def _remove_text_hints(schema):
    for key in ("composition", "ui_layout", "constraints", "layout_plan",
                "information_hierarchy"):
        value = schema.get(key)
        if isinstance(value, str):
            for phrase in _BANNED_TEXT_PHRASES:
                value = value.replace(phrase, "")
            schema[key] = value.strip(" ，,；;。")
        elif isinstance(value, list):
            schema[key] = [
                item for item in value
                if not any(phrase in str(item) for phrase in _BANNED_TEXT_PHRASES)
            ]
    schema["text_requirements"] = []
    schema["typography_plan"] = ""
    schema["copy_strategy"] = ""
    return schema


def _normalize_schema(schema, aspect_ratio, exact_text, text_policy,
                      optimize_strength="", layout_type=""):
    schema["aspect_ratio"] = aspect_ratio
    schema["direction"] = _ratio_to_direction(aspect_ratio)
    schema["text_policy"] = text_policy
    schema["optimize_strength"] = optimize_strength
    schema["layout_type"] = layout_type

    if not isinstance(schema.get("constraints"), list):
        schema["constraints"] = []
    schema["constraints"] = schema["constraints"][:3]

    if not isinstance(schema.get("named_entities"), list):
        schema["named_entities"] = []

    exact = exact_text.strip() if exact_text else ""
    if text_policy == "none":
        schema = _remove_text_hints(schema)
    elif text_policy == "preserve":
        schema["text_requirements"] = [exact] if exact else []
    elif text_policy == "enhance":
        if not schema.get("text_requirements"):
            schema["text_requirements"] = [exact] if exact else []
    elif text_policy != "generate":
        schema["text_requirements"] = (
            [exact] if exact else schema.get("text_requirements", [])
        )

    return schema


# ===================================================================
# Merged prompt optimizer node
# ===================================================================

class ZhangyuAPIPromptOptimizer:
    """ComfyUI 提示词优化器 — 合并参考图模式 & 纯文本模式.

    根据是否连接参考图自动切换：

    * **参考图模式** (有 ``reference_image_*`` 输入):
      多模态模型分析参考图的构图/光影/色彩/风格 → 输出结构化提示词。

    * **纯文本模式** (无参考图):
      两阶段处理 — schema 解析 → 提示词渲染 → GPT-Image-2 优化提示词。

    支持流式 / 非流式、系统提示词预设、模型自动获取。
    """

    # Widget option pools
    LAYOUT_TYPES = ["自动判断", "纯画面", "图文混排海报", "电商主图", "社媒封面"]
    TEXT_POLICIES = ["不加文字", "保留原文", "优化原文", "自动生成"]
    STRENGTH_OPTIONS = ["标准", "增强"]
    REFERENCE_MODES = [
        "自动判断", "综合参考", "只参考风格",
        "只参考构图", "只参考色彩光影", "只参考版式",
    ]

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("optimized_prompt", "debug_info")
    FUNCTION = "optimize"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "提示词优化器：域名+Key 即用，模型自动获取，"
        "有参考图→多模态分析，无参考图→两阶段schema优化，"
        "支持流式/非流式、系统提示词预设"
    )

    @classmethod
    def INPUT_TYPES(cls):
        presets = _get_preset_names()
        return {
            "required": {
                "api_key (API密钥)": (
                    "STRING", {"default": "", "multiline": False}),
                "api_base (接口域名)": (
                    "STRING", {"default": DEFAULT_API_BASE_URL, "multiline": False}),
                "user_prompt (用户需求)": (
                    "STRING", {"multiline": True, "default": ""}),
                "model (模型)": (
                    ["auto (自动选择)"],),
                "aspect_ratio (目标比例)": (
                    ASPECT_RATIO_OPTIONS, {"default": "auto"}),
                "seed (种子)": (
                    "INT", {"default": 0, "min": 0, "max": 2147483647,
                            "control_after_generate": True}),
                "timeout_seconds (超时秒数)": (
                    "INT", {"default": 600, "min": 30, "max": 1800}),
                "stream (流式输出)": (
                    "BOOLEAN", {"default": False}),
                "temperature (创造性)": (
                    "FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0,
                              "step": 0.1}),
                "max_tokens (最大长度)": (
                    "INT", {"default": 4096, "min": 64, "max": 32768}),
                "retry_times (重试次数)": (
                    "INT", {"default": 2, "min": 1, "max": 5}),
            },
            "optional": {
                # Reference image inputs (triggers reference mode)
                "reference_image_01": ("IMAGE",),
                "reference_image_02": ("IMAGE",),
                "reference_image_03": ("IMAGE",),
                "reference_image_04": ("IMAGE",),
                "reference_image_05": ("IMAGE",),
                "subject_image": ("IMAGE",),
                # Reference-mode only
                "reference_mode (参考模式)": (
                    cls.REFERENCE_MODES, {"default": "自动判断"}),
                # Text-mode only
                "layout_type (版式类型)": (
                    cls.LAYOUT_TYPES, {"default": "自动判断"}),
                "text_policy (文字策略)": (
                    cls.TEXT_POLICIES, {"default": "保留原文"}),
                "optimize_strength (优化强度)": (
                    cls.STRENGTH_OPTIONS, {"default": "标准"}),
                "exact_text (精确文字)": (
                    "STRING", {"multiline": True, "default": ""}),
                # Preset
                "preset (预设)": (
                    presets, {"default": "默认"}),
                # Custom model override — if filled, takes precedence over the combo
                "custom_model (自定义模型名)": (
                    "STRING", {"default": "", "multiline": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Include a sentinel for whether reference images are connected —
        # connecting/disconnecting refs changes the entire execution path
        # (reference mode vs. text mode).
        has_refs = cls._has_reference_images(kwargs)
        key = json.dumps(
            {
                k: str(v) if not isinstance(v, (list, tuple))
                   else f"<tensor_{len(v)}>"
                for k, v in kwargs.items()
                if k not in (
                    "reference_image_01", "reference_image_02",
                    "reference_image_03", "reference_image_04",
                    "reference_image_05", "subject_image",
                )
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.md5(
            f"{key}|has_refs={has_refs}".encode()
        ).hexdigest()

    # ------------------------------------------------------------------
    # Mode detection
    # ------------------------------------------------------------------

    @staticmethod
    def _has_reference_images(kwargs):
        """Return True if any reference image input is connected."""
        for i in range(1, 6):
            if kwargs.get(f"reference_image_{i:02d}") is not None:
                return True
        return False

    # ------------------------------------------------------------------
    # Reference-image mode
    # ------------------------------------------------------------------

    def _optimize_from_references(self, api_base, api_key, model, kwargs,
                                   timeout_seconds, stream, temperature,
                                   max_tokens, retry_times, unique_id,
                                   start_ts):
        """Analyse reference images → optimized prompt."""
        user_prompt = kwargs.get("user_prompt (用户需求)", "")
        target_aspect_ratio = kwargs.get("aspect_ratio (目标比例)", "auto")
        reference_mode = kwargs.get("reference_mode (参考模式)", "自动判断")
        subject_image = kwargs.get("subject_image")
        ref_mode_en = _REFERENCE_MODE_MAP.get(reference_mode, reference_mode)

        emit_runtime_status(unique_id, "running", "准备参考图",
                            0.0, 1, 1, timeout_seconds)

        # Collect reference image URLs (all slots guarded against None)
        ref_images = []
        for i in range(1, 6):
            img = kwargs.get(f"reference_image_{i:02d}")
            if img is not None:
                ref_images.append(img)

        ref_urls = [tensor_to_data_url(img) for img in ref_images]
        subject_url = tensor_to_data_url(subject_image) if subject_image is not None else None

        user_content = _build_reference_message(
            ref_urls, user_prompt, ref_mode_en,
            target_aspect_ratio, subject_url=subject_url,
        )
        messages = [
            {"role": "system", "content": _REFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        emit_runtime_status(
            unique_id, "running",
            f"分析 {len(ref_urls)} 张参考图{' (流式)' if stream else ''}",
            time.time() - start_ts, 1, 1, timeout_seconds,
        )
        print(
            f"[ZhangyuAPI Prompt Optimizer] reference mode, "
            f"images={len(ref_urls)}, model={model}, stream={stream}"
        )

        raw, _data = _call_chat_with_retry(
            api_base, api_key, model, messages,
            timeout_seconds, stream, temperature, max_tokens,
            retry_times=retry_times,
        )

        emit_runtime_status(unique_id, "running", "解析提示词",
                            time.time() - start_ts, 1, 1, timeout_seconds)

        optimized_prompt, reference_summary = _parse_tagged_output(raw)
        if not optimized_prompt:
            optimized_prompt = raw

        elapsed = time.time() - start_ts
        emit_runtime_status(
            unique_id, "success",
            f"提示词生成完成 (耗时 {elapsed:.1f}s)",
            elapsed, 1, 1, timeout_seconds,
        )
        return optimized_prompt, reference_summary

    # ------------------------------------------------------------------
    # Text-only mode (two-stage schema → render)
    # ------------------------------------------------------------------

    def _optimize_from_text(self, api_base, api_key, model, kwargs,
                             timeout_seconds, stream, temperature,
                             max_tokens, retry_times, unique_id, start_ts):
        """Two-stage text optimisation: schema parse → prompt render."""
        user_prompt = kwargs.get("user_prompt (用户需求)", "")
        layout_type = kwargs.get("layout_type (版式类型)", "自动判断")
        text_policy_raw = kwargs.get("text_policy (文字策略)", "保留原文")
        optimize_strength_raw = kwargs.get("optimize_strength (优化强度)", "标准")
        aspect_ratio = kwargs.get("aspect_ratio (目标比例)", "16:9")
        exact_text = kwargs.get("exact_text (精确文字)", "") or ""

        optimize_strength = _STRENGTH_MAP.get(optimize_strength_raw, optimize_strength_raw)
        text_policy = _TEXT_POLICY_MAP.get(text_policy_raw, text_policy_raw)

        # ---- Stage 1: Schema parsing ----
        emit_runtime_status(unique_id, "running", "解析需求结构",
                            0.0, 1, 2, timeout_seconds)
        payload = _build_input_payload(
            layout_type, optimize_strength, aspect_ratio,
            user_prompt, exact_text, text_policy,
        )
        schema_messages = [
            {"role": "system", "content": _SCHEMA_PARSER_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        print(
            f"[ZhangyuAPI Prompt Optimizer] text mode stage-1, "
            f"model={model}, stream={stream}"
        )
        schema_raw, _data = _call_chat_with_retry(
            api_base, api_key, model, schema_messages,
            timeout_seconds, stream, temperature, max_tokens,
            retry_times=retry_times,
        )

        emit_runtime_status(unique_id, "running", "整理 Schema",
                            time.time() - start_ts, 1, 2, timeout_seconds)
        schema = _parse_json_response(schema_raw)
        schema = _normalize_schema(
            schema, aspect_ratio, exact_text, text_policy,
            optimize_strength, layout_type,
        )

        # ---- Stage 2: Prompt rendering ----
        renderer_messages = [
            {"role": "system", "content": _RENDERER_PROMPT},
            {"role": "user", "content": json.dumps(schema, ensure_ascii=False)},
        ]
        emit_runtime_status(unique_id, "running", "渲染最终提示词",
                            time.time() - start_ts, 2, 2, timeout_seconds)
        print(
            f"[ZhangyuAPI Prompt Optimizer] text mode stage-2, "
            f"model={model}"
        )
        optimized, _data = _call_chat_with_retry(
            api_base, api_key, model, renderer_messages,
            timeout_seconds, stream, temperature, max_tokens,
            retry_times=retry_times,
        )

        # Post-processing
        if optimize_strength == "增强" and text_policy == "generate":
            optimized = optimized.replace("【限制条件】", "【创作自由】")
        if optimize_strength == "标准" and text_policy == "generate":
            texts = schema.get("text_requirements", [])
            text_list = "、".join([f"\"{t}\"" for t in texts])
            strict_text_block = (
                f"画面必须且只能显示以下文字：{text_list}。\n"
                "不得出现除此之外的任何可读文字、装饰文字、章印文字、小标签、英文补充或无意义文字。"
                "文字应清晰、准确、层级明确，排版稳定，不做创意发散。"
            )
            optimized = re.sub(
                r"【文字要求】.*?(?=【|$)",
                f"【文字要求】\n{strict_text_block}\n",
                optimized,
                flags=re.DOTALL,
            )

        direction = _ratio_to_direction(aspect_ratio)
        debug_info = (
            f"model={model}\n"
            f"api_base={api_base}\n"
            f"layout_type={layout_type}\n"
            f"optimize_strength={optimize_strength}\n"
            f"aspect_ratio={aspect_ratio}\n"
            f"direction={direction}\n"
            f"text_policy={text_policy}\n"
            f"has_exact_text={str(bool(exact_text.strip())).lower()}\n"
            f"stream={stream}\n"
            f"temperature={temperature}\n"
            f"max_tokens={max_tokens}\n"
            f"resolved_layout_type={schema.get('image_type', '')}\n"
            f"resolved_text_policy={schema.get('text_policy', '')}\n"
            f"schema_result={json.dumps(schema, ensure_ascii=False)}\n"
            f"final_prompt={optimized}"
        )
        elapsed = time.time() - start_ts
        emit_runtime_status(
            unique_id, "success",
            f"提示词生成完成 (耗时 {elapsed:.1f}s)",
            elapsed, 2, 2, timeout_seconds,
        )
        return optimized, debug_info

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def optimize(self, **kwargs):
        """Main entry point — auto-detects mode and runs optimisation.

        Args:
            **kwargs: ComfyUI widget values.

        Returns:
            ``(optimized_prompt: str, debug_info: str)``.
        """
        api_key = kwargs.get("api_key (API密钥)", "")
        api_base = normalize_api_base(
                kwargs.get("api_base (接口域名)", DEFAULT_API_BASE_URL)
            )
        model = kwargs.get("model (模型)", "auto (自动选择)")
        custom_model = (kwargs.get("custom_model (自定义模型名)") or "").strip()
        if custom_model:
            model = custom_model
        timeout_seconds = kwargs.get("timeout_seconds (超时秒数)", 600)
        stream = kwargs.get("stream (流式输出)", False)
        temperature = float(kwargs.get("temperature (创造性)", 0.7))
        max_tokens = int(kwargs.get("max_tokens (最大长度)", 4096))
        retry_times = int(kwargs.get("retry_times (重试次数)", 2))
        unique_id = kwargs.get("unique_id")
        start_ts = time.time()

        # Validate
        if not api_key.strip():
            emit_runtime_status(unique_id, "error", "API Key 为空",
                                0.0, 0, 1, timeout_seconds)
            raise ValueError("API Key 不能为空")

        # -- resolve & validate model (placeholder → auto-detect) -----------
        try:
            model, _model_list = resolve_and_validate_model(
                model, api_base, api_key.strip(), unique_id,
                placeholder="auto (自动选择)",
                filter_func=_filter_chat_models,
            )
        except ValueError as exc:
            emit_runtime_status(unique_id, "error", str(exc),
                                0.0, 0, 1, timeout_seconds)
            raise

        try:
            # Apply preset if selected
            preset_name = kwargs.get("preset (预设)", "默认")
            if preset_name != "默认":
                presets = _load_presets()
                preset = presets.get(preset_name, {})
                if preset.get("default_params"):
                    opt_inputs = self.INPUT_TYPES().get("optional", {})
                    for preset_key, preset_val in preset["default_params"].items():
                        # Map internal preset key → ComfyUI kwarg display name
                        kwarg_key = _PRESET_KEY_MAP.get(preset_key, preset_key)
                        if kwarg_key not in kwargs:
                            continue
                        widget_def = opt_inputs.get(kwarg_key)
                        if not (isinstance(widget_def, (list, tuple))
                                and len(widget_def) >= 2
                                and isinstance(widget_def[1], dict)):
                            continue
                        widget_default = widget_def[1].get("default")
                        # Apply preset value only when user hasn't changed from default
                        if (widget_default is not None
                                and kwargs[kwarg_key] == widget_default):
                            kwargs[kwarg_key] = preset_val

            if self._has_reference_images(kwargs):
                return self._optimize_from_references(
                    api_base, api_key, model, kwargs,
                    timeout_seconds, stream, temperature, max_tokens,
                    retry_times, unique_id, start_ts,
                )
            else:
                return self._optimize_from_text(
                    api_base, api_key, model, kwargs,
                    timeout_seconds, stream, temperature, max_tokens,
                    retry_times, unique_id, start_ts,
                )
        except Exception as exc:
            emit_runtime_status(
                unique_id, "error", str(exc),
                time.time() - start_ts, 1, 2, timeout_seconds,
            )
            raise


# ===================================================================
# Text list editor node (unchanged from original)
# ===================================================================

class ZhangyuAPITextListEditor:
    """ComfyUI 文本停留编辑器.

    工作流执行到该节点时暂停，弹出编辑框让用户手动修改文本列表，
    确认后继续执行。超时 300 秒自动取消。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_list": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("edited_text", "edited_texts")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "edit_text_list"
    CATEGORY = CATEGORY

    def edit_text_list(self, text_list, unique_id=None):
        try:
            import server as _comfy_server
            from aiohttp import web as _aiohttp_web
            from nodes import interrupt_processing as _interrupt
        except Exception:
            raise RuntimeError("文本停留编辑器需要在 ComfyUI 服务环境中运行")

        if _comfy_server is None or _comfy_server.PromptServer.instance is None:
            raise RuntimeError("文本停留编辑器需要在 ComfyUI 服务环境中运行")

        if isinstance(unique_id, list):
            unique_id = unique_id[0] if unique_id else None

        texts = text_list if isinstance(text_list, list) else [str(text_list)]
        cleaned_texts = [str(t).strip() for t in texts]
        session_id = str(uuid.uuid4())
        with _pending_text_lists_lock:
            _pending_text_lists[session_id] = {
                "edited_texts": cleaned_texts.copy(),
                "confirmed": False,
                "cancelled": False,
                "_created_at": time.time(),
            }

        _comfy_server.PromptServer.instance.send_sync(
            "zhangyuapi_text_list_edit_session",
            {
                "session_id": session_id,
                "node_id": unique_id,
                "texts": cleaned_texts,
            },
        )

        # Clean up orphaned sessions before waiting
        _cleanup_orphaned_sessions()

        start_time = time.time()
        while True:
            time.sleep(0.1)
            with _pending_text_lists_lock:
                if session_id not in _pending_text_lists:
                    _interrupt()
                    return ("", [])

                session = dict(_pending_text_lists[session_id])  # shallow copy
                if session.get("confirmed"):
                    break
                if session.get("cancelled"):
                    del _pending_text_lists[session_id]
                    _interrupt()
                    return ("", [])
                if time.time() - start_time > _SESSION_TIMEOUT_SECONDS:
                    print(
                        f"ZhangyuAPITextListEditor: session {session_id} "
                        f"timed out after {_SESSION_TIMEOUT_SECONDS}s"
                    )
                    del _pending_text_lists[session_id]
                    _interrupt()
                    return ("", [])

        with _pending_text_lists_lock:
            edited_texts = list(_pending_text_lists[session_id].get("edited_texts", []))
            del _pending_text_lists[session_id]
        edited_text = "\n".join(
            str(text).strip() for text in edited_texts if str(text).strip()
        )
        return (edited_text, edited_texts)


def _cleanup_orphaned_sessions():
    """Remove stale sessions that were never confirmed or cancelled."""
    now = time.time()
    with _pending_text_lists_lock:
        stale = [
            sid for sid, session in _pending_text_lists.items()
            if now - session.get("_created_at", 0) > _SESSION_TIMEOUT_SECONDS
        ]
        for sid in stale:
            del _pending_text_lists[sid]


# ===================================================================
# Server routes for text-list editor
# ===================================================================

def _add_text_editor_routes(routes):
    try:
        from aiohttp import web as _aiohttp_web
    except Exception:
        return

    @routes.post("/zhangyuapi_text_list_edit/confirm")
    async def _confirm(request):
        try:
            data = await request.json()
            session_id = data.get("session_id")
            edited_texts = data.get("edited_texts", [])

            with _pending_text_lists_lock:
                if session_id not in _pending_text_lists:
                    return _aiohttp_web.json_response(
                        {"status": "error", "message": "Session not found"},
                        status=404,
                    )

                if not isinstance(edited_texts, list):
                    edited_texts = [edited_texts] if edited_texts else []

                _pending_text_lists[session_id]["edited_texts"] = edited_texts
                _pending_text_lists[session_id]["confirmed"] = True
            return _aiohttp_web.json_response({"status": "success"})
        except Exception as exc:
            print(f"ZhangyuAPITextListEditor: confirm error: {exc}")
            return _aiohttp_web.json_response(
                {"status": "error", "message": str(exc)}, status=500,
            )

    @routes.post("/zhangyuapi_text_list_edit/cancel")
    async def _cancel(request):
        try:
            data = await request.json()
            session_id = data.get("session_id")

            with _pending_text_lists_lock:
                if session_id not in _pending_text_lists:
                    return _aiohttp_web.json_response(
                        {"status": "error", "message": "Session not found"},
                        status=404,
                    )

                _pending_text_lists[session_id]["cancelled"] = True
            return _aiohttp_web.json_response({"status": "success"})
        except Exception as exc:
            print(f"ZhangyuAPITextListEditor: cancel error: {exc}")
            return _aiohttp_web.json_response(
                {"status": "error", "message": str(exc)}, status=500,
            )


try:
    import server as _comfy_server
    from aiohttp import web as _aiohttp_web
    if (_comfy_server is not None
            and _comfy_server.PromptServer.instance is not None):
        _add_text_editor_routes(_comfy_server.PromptServer.instance.routes)
except Exception as exc:
    print(f"Warning: Could not register ZhangyuAPITextListEditor routes: {exc}")


# ===================================================================
# ComfyUI node registration
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ZhangyuAPIPromptOptimizer": ZhangyuAPIPromptOptimizer,
    "ZhangyuAPITextListEditor": ZhangyuAPITextListEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZhangyuAPIPromptOptimizer": "ComfyUI-zhangyuapi-提示词优化器 🧪测试中",
    "ZhangyuAPITextListEditor": "ComfyUI-zhangyuapi-文本停留编辑器 🧪测试中",
}
