#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — 提示词优化 + 文本停留编辑节点.

- 提示词反推：多模态 LLM 分析参考图 → 输出提示词
- 提示词优化：两阶段 LLM（Schema 解析 → 提示词渲染）
- 文本停留编辑器：暂停工作流让用户手动编辑文本

Endpoint: ``POST /v1/chat/completions``
Auth: ``Authorization: Bearer <api_key>``
"""

import hashlib
import json
import pathlib
import random
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
    denormalize_api_base,
    tensor_to_data_url,
    emit_runtime_status,
    DEFAULT_API_BASE_URL,
    # Model list (output port only, no validation)
    fetch_available_models_cached,
    _filter_chat_models,
    _log,
    _on_retryable_error,
    _skip_error_return,
    safe_int,
)


# ===================================================================
# Constants
# ===================================================================

CATEGORY = "Comfyui-ZhangyuAPI/📝文本 Text"

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
    "layout_type": "layout_type (画面版式)",
    "text_policy": "text_policy (画面文字)",
    "optimize_strength": "optimize_strength (强度)",
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


def _build_chat_payload(model, messages, stream, temperature, max_tokens,
                         seed=0, enable_web_search=False, json_output=False):
    """Build the JSON body for ``POST /v1/chat/completions``."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed:
        payload["seed"] = seed
    if enable_web_search:
        payload["web_search_options"] = {"search_context_size": "medium"}
    if json_output:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _call_chat_stream(api_base, api_key, model, messages,
                      timeout_seconds=600, temperature=0.7, max_tokens=4096,
                      seed=0, enable_web_search=False,
                      json_output=False):
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
    payload = _build_chat_payload(
        model, messages, True, temperature, max_tokens,
        seed=seed, enable_web_search=enable_web_search,
        json_output=json_output,
    )
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
                          timeout_seconds=600, temperature=0.7, max_tokens=4096,
                          seed=0, enable_web_search=False,
                          json_output=False):
    """Non-streaming call to ``POST /v1/chat/completions``.

    Returns:
        ``(content_text: str, response_data: dict)``.
    """
    url = _chat_url(api_base)
    payload = _build_chat_payload(
        model, messages, False, temperature, max_tokens,
        seed=seed, enable_web_search=enable_web_search,
        json_output=json_output,
    )
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
               stream=False, temperature=0.7, max_tokens=4096, seed=0,
               enable_web_search=False, json_output=False):
    """Unified chat entry point — dispatches to stream or non-stream.

    Returns:
        ``(content_text: str, response_data: dict)``.
    """
    if stream:
        return _call_chat_stream(
            api_base, api_key, model, messages,
            timeout_seconds, temperature, max_tokens, seed=seed,
            enable_web_search=enable_web_search,
            json_output=json_output,
        )
    return _call_chat_nonstream(
        api_base, api_key, model, messages,
        timeout_seconds, temperature, max_tokens, seed=seed,
        enable_web_search=enable_web_search,
        json_output=json_output,
    )


def _call_chat_with_retry(api_base, api_key, model, messages,
                           timeout_seconds=600, stream=False,
                           temperature=0.7, max_tokens=4096,
                           retry_times=2, seed=0, enable_web_search=False,
                           json_output=False):
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
                seed=seed, enable_web_search=enable_web_search,
                json_output=json_output,
            )
        except _RETRYABLE_EXCEPTIONS as exc:
            last_error = str(exc)
            _log("warn",
                 f"LLM 调用失败 (attempt={attempt}/{retry_times}, "
                 f"type={type(exc).__name__}): {last_error}")
            if attempt < retry_times:
                _on_retryable_error(exc)
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


def _build_input_payload(layout_type, optimize_strength,
                         user_prompt, text_policy):
    return {
        "layout_type": layout_type,
        "optimize_strength": optimize_strength,
        "aspect_ratio": "auto",
        "direction": "由画面内容决定",
        "user_prompt": user_prompt or "",
        "exact_text": "",
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


def _normalize_schema(schema, text_policy,
                      optimize_strength="", layout_type=""):
    schema["aspect_ratio"] = "auto"
    schema["direction"] = "由画面内容决定"
    schema["text_policy"] = text_policy
    schema["optimize_strength"] = optimize_strength
    schema["layout_type"] = layout_type

    if not isinstance(schema.get("constraints"), list):
        schema["constraints"] = []
    schema["constraints"] = schema["constraints"][:3]

    if not isinstance(schema.get("named_entities"), list):
        schema["named_entities"] = []

    if text_policy == "none":
        schema = _remove_text_hints(schema)
    elif text_policy == "preserve":
        schema["text_requirements"] = []
    elif text_policy == "enhance":
        if not schema.get("text_requirements"):
            schema["text_requirements"] = []
    elif text_policy != "generate":
        schema["text_requirements"] = schema.get("text_requirements", [])

    return schema


# ===================================================================
# Merged prompt optimizer node
# ===================================================================

class ZhangyuAPIPromptOptimizer:
    """ComfyUI 提示词优化器 — 两种模式手动切换.

    * **提示词反推**：上传参考图 → 多模态 LLM 分析 → 反推出提示词。
    * **提示词优化**：输入文字需求 → 两阶段 LLM（Schema 解析 → 渲染）→ 输出优化提示词。
    """

    # Widget option pools
    MODE_OPTIONS = ["提示词反推", "提示词优化"]
    LAYOUT_TYPES = ["自动判断", "纯画面", "图文混排海报", "电商主图", "社媒封面"]
    TEXT_POLICIES = ["不加文字", "保留原文", "优化原文", "自动生成"]
    STRENGTH_OPTIONS = ["标准", "增强"]
    REFERENCE_MODES = [
        "自动判断", "综合参考", "只参考风格",
        "只参考构图", "只参考色彩光影", "只参考版式",
    ]

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("optimized_prompt", "debug_info", "model_list")
    FUNCTION = "optimize"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "提示词优化器：域名+Key 即用，模型自动获取，"
        "提示词反推→多模态分析参考图，提示词优化→两阶段LLM优化"
    )

    @classmethod
    def INPUT_TYPES(cls):
        presets = _get_preset_names()
        return {
            "required": {
                "api_key (密钥)": (
                    "STRING", {"default": "", "multiline": False}),
                "api_base (API地址)": (
                    "STRING", {"default": DEFAULT_API_BASE_URL, "multiline": False}),
                "prompt (提示词)": (
                    "STRING", {"multiline": True, "default": ""}),
                "mode (模式)": (
                    cls.MODE_OPTIONS, {"default": "提示词优化"}),
                "model (模型)": (
                    "STRING", {"default": "deepseek-v4-pro", "multiline": False,
                               "placeholder": "默认 deepseek-v4-pro，留空自动获取"}),
                "preset (预设)": (
                    presets, {"default": "默认"}),
            },
            "optional": {
                # ---- 参考图 ----
                "reference_image_01": ("IMAGE",),
                "reference_image_02": ("IMAGE",),
                "reference_image_03": ("IMAGE",),
                "reference_image_04": ("IMAGE",),
                "reference_image_05": ("IMAGE",),
                "subject_image": ("IMAGE",),
                "reference_mode (参考范围)": (
                    cls.REFERENCE_MODES, {"default": "自动判断"}),
                # ---- 文本模式参数 ----
                "layout_type (画面版式)": (
                    cls.LAYOUT_TYPES, {"default": "自动判断"}),
                "text_policy (画面文字)": (
                    cls.TEXT_POLICIES, {"default": "保留原文"}),
                "optimize_strength (强度)": (
                    cls.STRENGTH_OPTIONS, {"default": "标准"}),
                "enable_web_search (联网搜索)": (
                    "BOOLEAN", {"default": False}),
                "json_output (JSON输出)": (
                    "BOOLEAN", {"default": False}),
                # ---- 高级参数 ----
                "seed (种子)": (
                    "INT", {"default": 0, "min": 0, "max": 2147483647,
                            "control_after_generate": True}),
                "stream (流式)": (
                    "BOOLEAN", {"default": False}),
                "temperature (创造性)": (
                    "FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0,
                              "step": 0.1}),
                "max_tokens (最大输出)": (
                    "INT", {"default": 4096, "min": 64, "max": 32768}),
                "timeout_seconds (超时)": (
                    "INT", {"default": 600, "min": 30, "max": 1800}),
                "retry_times (重试)": (
                    "INT", {"default": 2, "min": 1, "max": 5}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "skip_error": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        mode = kwargs.get("mode (模式)", "提示词优化")
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
            f"{key}|mode={mode}".encode()
        ).hexdigest()

    # ------------------------------------------------------------------
    # 提示词反推模式（参考图 → 提示词）
    # ------------------------------------------------------------------

    def _optimize_from_references(self, opts, kwargs, unique_id, start_ts):
        """Analyse reference images → optimized prompt."""
        user_prompt = kwargs.get("prompt (提示词)", "")
        target_aspect_ratio = "auto"
        reference_mode = kwargs.get("reference_mode (参考范围)", "自动判断")
        subject_image = kwargs.get("subject_image")
        ref_mode_en = _REFERENCE_MODE_MAP.get(reference_mode, reference_mode)

        emit_runtime_status(unique_id, "running", "准备参考图",
                            0.0, 1, 1, opts["timeout_seconds"])

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
            f"分析 {len(ref_urls)} 张参考图{' (流式)' if opts['stream'] else ''}",
            time.time() - start_ts, 1, 1, opts["timeout_seconds"],
        )
        print(
            f"[ZhangyuAPI Prompt Optimizer] reference mode, "
            f"images={len(ref_urls)}, model={opts['model']}, stream={opts['stream']}"
        )

        raw, _data = _call_chat_with_retry(
            opts["api_base"], opts["api_key"], opts["model"], messages,
            opts["timeout_seconds"], opts["stream"], opts["temperature"], opts["max_tokens"],
            retry_times=opts["retry_times"], seed=opts["seed"],
            enable_web_search=opts["enable_web_search"],
            json_output=opts["json_output"],
        )

        emit_runtime_status(unique_id, "running", "解析提示词",
                            time.time() - start_ts, 1, 1, opts["timeout_seconds"])

        optimized_prompt, reference_summary = _parse_tagged_output(raw)
        if not optimized_prompt:
            optimized_prompt = raw

        elapsed = time.time() - start_ts
        emit_runtime_status(
            unique_id, "success",
            f"提示词生成完成 (耗时 {elapsed:.1f}s)",
            elapsed, 1, 1, opts["timeout_seconds"],
        )
        return optimized_prompt, reference_summary

    # ------------------------------------------------------------------
    # 提示词优化模式（文本 → 两阶段 schema → 渲染）
    # ------------------------------------------------------------------

    def _optimize_from_text(self, opts, kwargs, unique_id, start_ts):
        """Two-stage text optimisation: schema parse → prompt render."""
        user_prompt = kwargs.get("prompt (提示词)", "")
        layout_type = kwargs.get("layout_type (画面版式)", "自动判断")
        text_policy_raw = kwargs.get("text_policy (画面文字)", "保留原文")
        optimize_strength_raw = kwargs.get("optimize_strength (强度)", "标准")

        optimize_strength = _STRENGTH_MAP.get(optimize_strength_raw, optimize_strength_raw)
        text_policy = _TEXT_POLICY_MAP.get(text_policy_raw, text_policy_raw)

        # ---- Stage 1: Schema parsing ----
        emit_runtime_status(unique_id, "running", "解析需求结构",
                            0.0, 1, 2, opts["timeout_seconds"])
        payload = _build_input_payload(
            layout_type, optimize_strength,
            user_prompt, text_policy,
        )
        schema_messages = [
            {"role": "system", "content": _SCHEMA_PARSER_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        print(
            f"[ZhangyuAPI Prompt Optimizer] text mode stage-1, "
            f"model={opts['model']}, stream={opts['stream']}"
        )
        schema_raw, _data = _call_chat_with_retry(
            opts["api_base"], opts["api_key"], opts["model"], schema_messages,
            opts["timeout_seconds"], opts["stream"], opts["temperature"], opts["max_tokens"],
            retry_times=opts["retry_times"], seed=opts["seed"],
            enable_web_search=opts["enable_web_search"],
            json_output=opts["json_output"],
        )

        emit_runtime_status(unique_id, "running", "整理 Schema",
                            time.time() - start_ts, 1, 2, opts["timeout_seconds"])
        schema = _parse_json_response(schema_raw)
        schema = _normalize_schema(
            schema, text_policy,
            optimize_strength, layout_type,
        )

        # ---- Stage 2: Prompt rendering ----
        renderer_messages = [
            {"role": "system", "content": _RENDERER_PROMPT},
            {"role": "user", "content": json.dumps(schema, ensure_ascii=False)},
        ]
        emit_runtime_status(unique_id, "running", "渲染最终提示词",
                            time.time() - start_ts, 2, 2, opts["timeout_seconds"])
        print(
            f"[ZhangyuAPI Prompt Optimizer] text mode stage-2, "
            f"model={opts['model']}"
        )
        optimized, _data = _call_chat_with_retry(
            opts["api_base"], opts["api_key"], opts["model"], renderer_messages,
            opts["timeout_seconds"], opts["stream"], opts["temperature"], opts["max_tokens"],
            retry_times=opts["retry_times"], seed=opts["seed"],
            enable_web_search=opts["enable_web_search"],
            json_output=opts["json_output"],
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

        debug_info = (
            f"model={opts['model']}\n"
            f"api_base={denormalize_api_base(opts['api_base'])}\n"
            f"layout_type={layout_type}\n"
            f"optimize_strength={optimize_strength}\n"
            f"text_policy={text_policy}\n"
            f"has_exact_text=false\n"
            f"stream={opts['stream']}\n"
            f"temperature={opts['temperature']}\n"
            f"max_tokens={opts['max_tokens']}\n"
            f"resolved_layout_type={schema.get('image_type', '')}\n"
            f"resolved_text_policy={schema.get('text_policy', '')}\n"
            f"schema_result={json.dumps(schema, ensure_ascii=False)}\n"
            f"final_prompt={optimized}"
        )
        elapsed = time.time() - start_ts
        emit_runtime_status(
            unique_id, "success",
            f"提示词生成完成 (耗时 {elapsed:.1f}s)",
            elapsed, 2, 2, opts["timeout_seconds"],
        )
        return optimized, debug_info

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def optimize(self, **kwargs):
        """Thin wrapper with ``skip_error`` handling for workflow continuity."""
        skip_error = kwargs.get("skip_error", False)
        try:
            return self._optimize_impl(**kwargs)
        except Exception as exc:
            if not skip_error:
                raise
            error_msg = f"{type(exc).__name__}: {exc}"
            _log("warn", f"skip_error 模式，节点失败: {error_msg}")
            return _skip_error_return(
                error_msg, self.RETURN_TYPES,
                unique_id=kwargs.get("unique_id"),
                retry_times=kwargs.get("retry_times (重试次数)", 2),
                timeout_seconds=kwargs.get("timeout_seconds (超时秒数)", 600),
            )

    def _optimize_impl(self, **kwargs):
        """Main entry point — auto-detects mode and runs optimisation.

        Args:
            **kwargs: ComfyUI widget values.

        Returns:
            ``(optimized_prompt: str, debug_info: str)``.
        """
        api_key = kwargs.get("api_key (密钥)", "")
        api_base = normalize_api_base(
                kwargs.get("api_base (API地址)", DEFAULT_API_BASE_URL)
            )
        model = kwargs.get("model (模型)", "").strip()
        timeout_seconds = kwargs.get("timeout_seconds (超时)", 600)
        stream = kwargs.get("stream (流式)", False)
        temperature = float(kwargs.get("temperature (创造性)", 0.7))
        max_tokens = int(kwargs.get("max_tokens (最大输出)", 4096))
        retry_times = int(kwargs.get("retry_times (重试)", 2))
        enable_web_search = kwargs.get("enable_web_search (联网搜索)", False)
        json_output = kwargs.get("json_output (JSON输出)", False)
        seed = int(kwargs.get("seed (种子)", 0))
        if seed == 0:
            seed = random.randint(1, 2147483647)
        unique_id = kwargs.get("unique_id")
        start_ts = time.time()

        # Validate
        if not api_key.strip():
            emit_runtime_status(unique_id, "error", "API Key 为空",
                                0.0, 0, 1, timeout_seconds)
            raise ValueError("API Key 不能为空")

        # -- fetch model list for output port (best-effort) --------------------
        model_list = []
        try:
            all_models = fetch_available_models_cached(
                api_base, api_key.strip())
            model_list = _filter_chat_models(all_models)
        except Exception as exc:
            _log("warn", f"获取模型列表失败（不影响优化）: {exc}")

        if not model:
            model = "deepseek-v4-pro"

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

                # Auto-fill prompt from preset template if prompt is empty
                current_prompt = kwargs.get("prompt (提示词)", "").strip()
                if not current_prompt and preset.get("prompt_template"):
                    kwargs["prompt (提示词)"] = preset["prompt_template"]
                    _log("info", f"[预设模板] 已自动填充 prompt: {preset['prompt_template']}")

            mode = kwargs.get("mode (模式)", "提示词优化")
            opts = {
                "api_base": api_base, "api_key": api_key, "model": model,
                "timeout_seconds": timeout_seconds, "stream": stream,
                "temperature": temperature, "max_tokens": max_tokens,
                "retry_times": retry_times, "seed": seed,
                "enable_web_search": enable_web_search,
                "json_output": json_output,
            }
            if mode == "提示词反推":
                opt, dbg = self._optimize_from_references(
                    opts, kwargs, unique_id, start_ts,
                )
            else:
                opt, dbg = self._optimize_from_text(
                    opts, kwargs, unique_id, start_ts,
                )
            return opt, dbg, json.dumps(model_list, ensure_ascii=False)
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
# 节点：中译英翻译
# ===================================================================

class ZhangyuAPITranslateNode:
    """将中文提示词翻译为英文，适配对英文响应更好的生图模型。

    Node display name: **ComfyUI-zhangyuapi-中译英**
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_en",)
    FUNCTION = "translate"
    CATEGORY = "Comfyui-ZhangyuAPI/🔧工具 Tools"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key (API密钥)": (
                    "STRING", {"default": "", "multiline": False}),
                "prompt_cn (中文提示词)": (
                    "STRING", {"multiline": True, "default": ""}),
                "model (模型)": (
                    "STRING", {"default": "deepseek-v4-pro",
                               "multiline": False}),
                "api_base (接口域名)": (
                    "STRING", {"default": DEFAULT_API_BASE_URL,
                               "multiline": False}),
                "timeout_seconds (超时秒数)": (
                    "INT", {"default": 120, "min": 30, "max": 600}),
                "retry_times (重试次数)": (
                    "INT", {"default": 2, "min": 1, "max": 5}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "skip_error": ("BOOLEAN", {"default": False}),
            },
        }

    def translate(self, **kwargs):
        skip_error = kwargs.get("skip_error", False)
        try:
            return self._translate_impl(**kwargs)
        except Exception as exc:
            if not skip_error:
                raise
            error_msg = f"{type(exc).__name__}: {exc}"
            _log("warn", f"skip_error 模式，翻译节点失败: {error_msg}")
            return (f"skip_error: {error_msg}",)

    def _translate_impl(self, **kwargs):
        api_key = kwargs.get("api_key (API密钥)", "").strip()
        prompt_cn = kwargs.get("prompt_cn (中文提示词)", "").strip()
        model = (kwargs.get("model (模型)") or "").strip() or "deepseek-v4-pro"
        api_base = normalize_api_base(
            kwargs.get("api_base (接口域名)", DEFAULT_API_BASE_URL))
        timeout_seconds = safe_int(
            kwargs.get("timeout_seconds (超时秒数)", 120), 120, 30, 600)
        retry_times = safe_int(
            kwargs.get("retry_times (重试次数)", 2), 2, 1, 5)
        unique_id = kwargs.get("unique_id")

        if not prompt_cn:
            return ("",)
        if not api_key:
            emit_runtime_status(unique_id, "error", "API Key 为空",
                                0, 0, retry_times, timeout_seconds)
            raise ValueError("API Key 不能为空")

        emit_runtime_status(unique_id, "running", "翻译中…",
                            0, 0, retry_times, timeout_seconds)

        messages = [
            {"role": "system",
             "content": (
                 "你是一个专业的图像提示词翻译器。"
                 "将用户输入的中文提示词翻译成自然流畅的英文。"
                 "保持原意和细节，不添加不删减。"
                 "只输出翻译结果，不要加任何解释或注释。"
             )},
            {"role": "user", "content": prompt_cn},
        ]
        result, _ = _call_chat_with_retry(
            api_base, api_key, model, messages,
            timeout_seconds=timeout_seconds, stream=False, temperature=0.3,
            max_tokens=1024, retry_times=retry_times,
        )
        return (result.strip(),)


# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ZhangyuAPIPromptOptimizer": ZhangyuAPIPromptOptimizer,
    "ZhangyuAPITextListEditor": ZhangyuAPITextListEditor,
    "ZhangyuAPITranslateNode": ZhangyuAPITranslateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZhangyuAPIPromptOptimizer": "ComfyUI-zhangyuapi-提示词优化器",
    "ZhangyuAPITextListEditor": "ComfyUI-zhangyuapi-文本停留编辑器",
    "ZhangyuAPITranslateNode": "ComfyUI-zhangyuapi-中译英",
}
