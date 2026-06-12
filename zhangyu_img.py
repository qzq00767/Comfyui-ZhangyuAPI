#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — OpenAI 格式通用生图节点.

严格遵循 OpenAI ``POST /v1/images/generations`` 接口规范。
UI 布局对齐 Comfyui-ZhangyuAPI-image-2 节点，仅参数根据
OpenAI 原生接口做适配。

- Endpoint: ``POST /v1/images/generations``
- Auth: ``Authorization: Bearer <api_key>``
- Model list: ``GET /v1/models``
"""

import hashlib
import json
import time

import torch

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

from .zhangyu_gpt_img2 import (
    ZHANGYUAPI_post,
    _RETRYABLE_EXCEPTIONS,
    _safe_json_dumps,
    _skip_error_return,
    _log,
    DEFAULT_NODE_TIMEOUT,
    DEFAULT_MIN_NODE_TIMEOUT,
    DEFAULT_MAX_NODE_TIMEOUT,
    DEFAULT_API_BASE_URL,
    _jittered_sleep,
    is_retryable_http_status,
    is_async_task_response,
    _poll_async_task,
    _parse_response_images,
    _sanitize_api_response,
    _extract_api_error_message,
    safe_choice,
    safe_int,
    normalize_prompt_text,
    normalize_api_base,
    denormalize_api_base,
    normalize_size,
    tensor_to_data_url,
    _auto_downscale,
    _filter_image_models,
    fetch_available_models_cached,
    emit_runtime_status,
)


# ===================================================================
# Node class
# ===================================================================

class ComfyuiZhangyuAPIUniversalImageNode:
    """ComfyUI OpenAI 格式通用生图节点 — 对齐 Image-2 布局."""

    IMAGE_SIZES = ["auto (不传size)", "ratio_only (仅传比例)", "1K", "2K", "4K"]

    ASPECT_RATIOS = [
        "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9",
    ]

    QUALITIES = ["auto", "low", "medium", "high"]

    STYLES = ["vivid", "natural"]

    OUTPUT_FORMATS = ["png", "jpeg", "webp"]

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_urls", "chats", "model_list")
    FUNCTION = "generate"
    CATEGORY = "Comfyui-ZhangyuAPI/🖼️图片 Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key (API密钥)": (
                    "STRING", {"default": "", "multiline": False}),
                "prompt (提示词)": (
                    "STRING", {"default": "", "multiline": True}),
                "model (模型)": (
                    "STRING", {"default": "gpt-image-2", "multiline": False,
                               "placeholder": "gpt-image-2"}),
                "api_base (接口域名)": (
                    "STRING", {"default": DEFAULT_API_BASE_URL, "multiline": False}),
                "image_size (分辨率)": (
                    cls.IMAGE_SIZES, {"default": "auto (不传size)"}),
                "aspect_ratio (宽高比)": (
                    cls.ASPECT_RATIOS, {"default": "1:1"}),
                "quality (画质)": (
                    cls.QUALITIES, {"default": "auto"}),
                "response_format (响应格式)": (
                    ["b64_json", "url"], {"default": "b64_json"}),
                "output_format (输出格式)": (
                    cls.OUTPUT_FORMATS, {"default": "jpeg"}),
                "output_compression (压缩率)": (
                    "INT", {"default": 85, "min": 0, "max": 100}),
                "n (生成数量)": (
                    "INT", {"default": 1, "min": 1, "max": 5}),
                "seed (种子)": (
                    "INT", {
                        "default": 0, "min": 0, "max": 2147483647,
                        "control_after_generate": True,
                    }),
                "timeout_seconds (超时秒数)": (
                    "INT", {
                        "default": DEFAULT_NODE_TIMEOUT,
                        "min": DEFAULT_MIN_NODE_TIMEOUT,
                        "max": DEFAULT_MAX_NODE_TIMEOUT,
                    }),
                "retry_times (重试次数)": (
                    "INT", {"default": 3, "min": 1, "max": 10}),
            },
            "optional": {
                "style (风格)": (
                    cls.STYLES, {"default": "vivid"}),
                **{f"image_{i:02d}": ("IMAGE",) for i in range(1, 9)},
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "skip_error": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        key = json.dumps(
            {k: str(v) if not isinstance(v, (list, tuple))
               else f"<tensor_{len(v)}>"
             for k, v in kwargs.items()
             if k not in tuple(f"image_{i:02d}" for i in range(1, 9))},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.md5(key.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_images(kwargs):
        """Collect reference images from optional ``image_01``…``image_08`` inputs.

        Returns:
            ``list[str]`` — PNG data-URL strings for each non-None input.
        """
        data_urls = []
        for i in range(1, 9):
            tensor = kwargs.get(f"image_{i:02d}")
            if tensor is None:
                continue
            tensor = _auto_downscale(tensor)
            data_urls.append(tensor_to_data_url(tensor))
        return data_urls

    def _build_payload(self, model, prompt, n_images, effective_size,
                       quality, response_format, output_format,
                       output_compression, seed, style,
                       image_data_urls=None, aspect_ratio=None):
        """Build the JSON request body for ``POST /v1/images/generations``.

        *effective_size* must already be resolved via :func:`normalize_size`.
        When *effective_size* starts with ``"ratio:"``, the value after the
        colon is sent as ``aspect_ratio`` instead of ``size``.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n_images,
            "response_format": response_format,
            "seed": seed,
        }

        if isinstance(effective_size, str) and effective_size.startswith("ratio:"):
            ratio_val = effective_size[6:]
            if ratio_val and ratio_val.upper() != "AUTO":
                payload["aspect_ratio"] = ratio_val
        else:
            payload["size"] = effective_size

        if quality != "auto":
            payload["quality"] = quality

        # output format + compression
        payload["output_format"] = output_format
        if output_format == "png":
            # PNG is lossless — force compression to 100 regardless of widget
            payload["output_compression"] = 100
        elif output_compression != 85:
            payload["output_compression"] = output_compression

        if style != "vivid":
            payload["style"] = style

        # Reference images: send as base64 data URLs if present (img2img)
        if image_data_urls:
            payload["image_data"] = image_data_urls

        return payload

    def _request_generation(self, api_base, headers, payload, timeout_seconds):
        url = f"{api_base}/v1/images/generations"
        return ZHANGYUAPI_post(
            url,
            timeout_seconds,
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self, **kwargs):
        """Thin wrapper with ``skip_error`` handling for workflow continuity."""
        skip_error = kwargs.get("skip_error", False)
        try:
            return self._generate_impl(**kwargs)
        except Exception as exc:
            if not skip_error:
                raise
            error_msg = f"{type(exc).__name__}: {exc}"
            _log("warn", f"skip_error 模式，节点失败: {error_msg}")
            return _skip_error_return(
                error_msg, self.RETURN_TYPES,
                unique_id=kwargs.get("unique_id"),
                retry_times=kwargs.get("retry_times (重试次数)", 3),
                timeout_seconds=kwargs.get("timeout_seconds (超时秒数)", 360),
            )

    def _generate_impl(self, **kwargs):
        pbar = ProgressBar(100) if ProgressBar else None
        api_key = kwargs.get("api_key (API密钥)", "")
        api_base = normalize_api_base(
            kwargs.get("api_base (接口域名)", DEFAULT_API_BASE_URL))
        prompt = kwargs.get("prompt (提示词)", "")
        model = kwargs.get("model (模型)", "").strip() or "gpt-image-2"
        n_images = safe_int(kwargs.get("n (生成数量)", 1), 1, 1, 5)
        image_size = safe_choice(
            kwargs.get("image_size (分辨率)", "auto (不传size)"),
            self.IMAGE_SIZES, "1K")
        aspect_ratio = safe_choice(
            kwargs.get("aspect_ratio (宽高比)", "1:1"),
            self.ASPECT_RATIOS, "1:1")
        quality = safe_choice(
            kwargs.get("quality (画质)", "auto"),
            self.QUALITIES, "auto")
        response_format = safe_choice(
            kwargs.get("response_format (响应格式)", "b64_json"),
            ["b64_json", "url"], "b64_json")
        output_format = safe_choice(
            kwargs.get("output_format (输出格式)", "jpeg"),
            self.OUTPUT_FORMATS, "jpeg")
        output_compression = safe_int(
            kwargs.get("output_compression (压缩率)", 85), 85, 0, 100)
        seed = safe_int(kwargs.get("seed (种子)", 0), 0, 0, 2147483647)
        style = safe_choice(
            kwargs.get("style (风格)", "vivid"),
            self.STYLES, "vivid")
        timeout_seconds = safe_int(
            kwargs.get("timeout_seconds (超时秒数)", DEFAULT_NODE_TIMEOUT),
            DEFAULT_NODE_TIMEOUT,
            DEFAULT_MIN_NODE_TIMEOUT,
            DEFAULT_MAX_NODE_TIMEOUT,
        )
        retry_times = safe_int(
            kwargs.get("retry_times (重试次数)", 3),
            2, 1, 10)

        unique_id = kwargs.get("unique_id")
        start_ts = time.time()

        if not api_key.strip():
            emit_runtime_status(unique_id, "error", "API Key 为空",
                                0.0, 0, retry_times, timeout_seconds)
            raise ValueError("API Key 不能为空")

        clean_prompt = normalize_prompt_text(prompt)
        if not clean_prompt:
            emit_runtime_status(unique_id, "error", "prompt 不能为空",
                                0.0, 0, retry_times, timeout_seconds)
            raise ValueError("prompt 不能为空")

        # -- fetch model list for output port (best-effort) --------------------
        model_list = []
        try:
            all_models = fetch_available_models_cached(
                api_base, api_key.strip())
            model_list = _filter_image_models(all_models)
        except Exception as exc:
            print(f"[Comfyui-ZhangyuAPI-OpenAI] 获取模型列表失败（不影响生成）: {exc}")

        # -- prepare request -------------------------------------------------
        headers = {"Authorization": f"Bearer {api_key.strip()}"}
        image_data_urls = self._collect_images(kwargs)
        try:
            effective_size = normalize_size(image_size, aspect_ratio)
        except ValueError as exc:
            emit_runtime_status(unique_id, "error", str(exc),
                                0.0, 0, retry_times, timeout_seconds)
            raise
        payload = self._build_payload(
            model, clean_prompt, n_images, effective_size,
            quality, response_format, output_format, output_compression,
            seed, style, image_data_urls=image_data_urls,
            aspect_ratio=aspect_ratio,
        )

        print(
            f"[Comfyui-ZhangyuAPI-OpenAI] model={model}, n={n_images}, "
            f"image_size={image_size}, aspect_ratio={aspect_ratio} → {effective_size}, "
            f"quality={quality}, response_format={response_format}, "
            f"output_format={output_format}, output_compression={output_compression}, "
            f"style={style}, seed={seed}"
            + (f", ref_images={len(image_data_urls)}" if image_data_urls else "")
        )
        emit_runtime_status(unique_id, "running", "开始生成",
                            0.0, 0, retry_times, timeout_seconds)
        if pbar: pbar.update_absolute(10)

        # -- retry loop ------------------------------------------------------
        last_error = None
        for attempt in range(1, retry_times + 1):
            try:
                emit_runtime_status(
                    unique_id, "running",
                    f"请求生成中 ({attempt}/{retry_times})",
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )

                response = self._request_generation(
                    api_base, headers, payload, timeout_seconds,
                )

                if response.status_code != 200:
                    if response.status_code == 401:
                        raise RuntimeError(
                            "API Key 无效 (401 Unauthorized)，"
                            "请检查 API 密钥是否正确"
                        )
                    if response.status_code == 403:
                        raise RuntimeError(
                            "API 访问被拒绝 (403 Forbidden)，"
                            "请检查账户权限或余额"
                        )

                    try:
                        err_data = response.json()
                        err_msg = _extract_api_error_message(err_data)
                    except Exception:
                        err_msg = response.text[:500]
                    last_error = f"API 错误 {response.status_code}: {err_msg}"
                    if (is_retryable_http_status(response.status_code)
                            and attempt < retry_times):
                        emit_runtime_status(
                            unique_id, "running",
                            f"API 返回 {response.status_code}，"
                            f"重试中 ({attempt}/{retry_times})",
                            time.time() - start_ts,
                            attempt, retry_times, timeout_seconds,
                        )
                        _jittered_sleep(attempt)
                        continue
                    raise RuntimeError(last_error)

                data = response.json()
                if pbar: pbar.update_absolute(50)

                if is_async_task_response(data):
                    task_id = data.get("task_id") or data.get("id")
                    remaining = timeout_seconds - int(time.time() - start_ts)
                    emit_runtime_status(
                        unique_id, "running",
                        f"任务已提交 (id={task_id})，自适应轮询中",
                        time.time() - start_ts,
                        attempt, retry_times, timeout_seconds,
                    )

                    def _on_poll_tick(elapsed, poll_elapsed, interval):
                        emit_runtime_status(
                            unique_id, "running",
                            f"轮询任务中 · 间隔{interval:.0f}s · "
                            f"已等待{poll_elapsed:.0f}s",
                            elapsed,
                            attempt, retry_times, timeout_seconds,
                        )

                    data = _poll_async_task(
                        api_base, headers, task_id, remaining, retry_times,
                        on_tick=_on_poll_tick,
                    )

                image_tensor, image_urls, _failed = _parse_response_images(
                    data, timeout_seconds, error_prefix="OpenAI生图",
                    unique_id=unique_id, n_expected=n_images)
                elapsed = time.time() - start_ts

                if pbar: pbar.update_absolute(90)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                response_info = (
                    f"## ZhangyuAPI 生成结果 ({timestamp})\n\n"
                    f"- **模型**：{model}\n"
                    f"- **接口**：{denormalize_api_base(api_base)}\n"
                    f"- **分辨率**：{image_size}"
                    + (f" → {effective_size}" if effective_size != image_size else "") + "\n"
                    f"- **宽高比**：{aspect_ratio}\n"
                    f"- **画质**：{quality}\n"
                    f"- **输出格式**：{output_format}"
                    + (f" (压缩率 {output_compression})" if output_format != "png" else "") + "\n"
                    f"- **风格**：{style}\n"
                    f"- **响应格式**：{response_format}\n"
                    f"- **生成数量**：{n_images} 张\n"
                    f"- **成功**：{int(image_tensor.shape[0])} 张\n"
                    + (f"- **失败**：{_failed} 张\n"
                       f"- **警告**：请求 {n_images} 张，{_failed} 张下载/解码失败\n"
                       if _failed else "")
                    + (f"- **参考图**：{len(image_data_urls)} 张\n" if image_data_urls else "")
                    + (f"- **种子**：{seed}\n" if seed else "")
                    + (f"- **耗时**：{elapsed:.1f}s (attempt {attempt}/{retry_times})\n"
                       f"- **Usage**：{data.get('usage')}\n"
                       if data.get("usage") else "")
                )

                emit_runtime_status(
                    unique_id, "success",
                    f"生成成功 (耗时 {elapsed:.1f}s)"
                    + (f"，{_failed} 张失败" if _failed else ""),
                    elapsed, attempt, retry_times, timeout_seconds,
                )
                if pbar: pbar.update_absolute(100)
                return (
                    image_tensor,
                    response_info,
                    _safe_json_dumps(image_urls),
                    _safe_json_dumps(_sanitize_api_response(data), indent=2),
                    _safe_json_dumps(model_list),
                )

            except _RETRYABLE_EXCEPTIONS as exc:
                last_error = str(exc)
                if attempt < retry_times:
                    emit_runtime_status(
                        unique_id, "running",
                        f"网络/代理/超时，重试中 ({attempt}/{retry_times})",
                        time.time() - start_ts,
                        attempt, retry_times, timeout_seconds,
                    )
                    _jittered_sleep(attempt)
                    continue
                break
            except Exception as exc:
                last_error = str(exc)
                emit_runtime_status(
                    unique_id, "error", last_error,
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )
                raise

        elapsed = time.time() - start_ts
        emit_runtime_status(
            unique_id, "error",
            f"连续 {retry_times} 次失败",
            elapsed, retry_times, retry_times, timeout_seconds,
        )
        raise RuntimeError(
            f"Comfyui-ZhangyuAPI-OpenAI 连续 {retry_times} 次失败，"
            f"最后错误: {last_error}"
        )


NODE_CLASS_MAPPINGS = {
    "ComfyuiZhangyuAPIUniversalImageNode": ComfyuiZhangyuAPIUniversalImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiZhangyuAPIUniversalImageNode": "ComfyUI-zhangyuapi-通用openai格式",
}
