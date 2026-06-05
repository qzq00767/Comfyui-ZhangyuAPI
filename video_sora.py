#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — Sora 格式视频生成节点.

适配 NewAPI 的 Sora 兼容端点，严格遵循 OpenAI Sora API 规范
(``POST /v1/videos`` → ``GET /v1/videos/{id}`` → ``GET /v1/videos/{id}/content``)。

- Endpoint: ``POST /v1/videos`` (multipart/form-data)
- Status:   ``GET /v1/videos/{task_id}``
- Download: ``GET /v1/videos/{task_id}/content`` → video/mp4 二进制
- Auth:     ``Authorization: Bearer <api_key>``
- Model list: ``GET /v1/models`` → 客户端按视频关键词筛选
"""

import hashlib
import json
import os
import random
import time
import uuid

from .zhangyu_gpt_img2 import (
    # HTTP client
    _get_http_client,
    ZHANGYUAPI_timeout,
    ZHANGYUAPI_get,
    # URL helpers
    normalize_api_base,
    denormalize_api_base,
    # Input sanitizers
    safe_choice,
    safe_int,
    normalize_prompt_text,
    # Retry / backoff
    _RETRYABLE_EXCEPTIONS,
    is_retryable_http_status,
    _jittered_sleep,
    _jittered_backoff_seconds,
    _download_bytes_with_retry,
    # Frontend status
    emit_runtime_status,
    # Model resolution
    resolve_and_validate_model,
    _sanitize_api_response,
    _extract_api_error_message,
    _filter_models_by_patterns,
    fetch_available_models_cached,
    # Image conversion
    tensor_to_data_url,
    # Logging
    _log,
    # Constants
    DEFAULT_API_BASE_URL,
    DEFAULT_NODE_TIMEOUT,
    DEFAULT_MIN_NODE_TIMEOUT,
    DEFAULT_MAX_NODE_TIMEOUT,
    DEFAULT_RETRY_TIMES,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_READ_TIMEOUT,
    DEFAULT_POOL_TIMEOUT,
)

# ---------------------------------------------------------------------------
# VideoFromFile — for ComfyUI VIDEO output type
# ---------------------------------------------------------------------------
try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    try:
        from comfy_api.latest._input_impl.video_types import VideoFromFile
    except ImportError:
        VideoFromFile = None


# ===================================================================
# Constants
# ===================================================================

DEFAULT_SORA_BASE = DEFAULT_API_BASE_URL
DEFAULT_VIDEO_TIMEOUT = 900
DEFAULT_MIN_VIDEO_TIMEOUT = 120
DEFAULT_MAX_VIDEO_TIMEOUT = 3600
DEFAULT_VIDEO_RETRY_TIMES = 2

VIDEO_SIZES = [
    "1280x720",      # 720p 横屏
    "720x1280",      # 720p 竖屏
    "1920x1080",     # 1080p 横屏
    "1080x1920",     # 1080p 竖屏
    "1024x1792",     # HD 竖屏
    "1792x1024",     # HD 横屏
]

# Video model filter patterns (include mode)
_SORA_MODEL_PATTERNS = [
    "sora",
]


def _filter_sora_models(all_models):
    """Filter model list to Sora-compatible models."""
    return _filter_models_by_patterns(
        all_models, _SORA_MODEL_PATTERNS,
        mode="include", fallback_empty=True,
    )


# ---------------------------------------------------------------------------
# Backend route for Sora model fetching
# ---------------------------------------------------------------------------
try:
    import asyncio as _asyncio_import_check
    import server as _comfy_server
    from aiohttp import web as _aiohttp_web

    if (_comfy_server is not None
            and _comfy_server.PromptServer.instance is not None):
        _routes = _comfy_server.PromptServer.instance.routes

        @_routes.post("/zhangyuapi_fetch_sora_models")
        async def _zhangyuapi_fetch_sora_models_route(request):
            try:
                data = await request.json()
                api_base = data.get("api_base", "")
                api_key = data.get("api_key", "")

                if not api_key or not api_key.strip():
                    return _aiohttp_web.json_response(
                        {"status": "error", "message": "API Key 不能为空"},
                        status=400,
                    )

                loop = _asyncio_import_check.get_running_loop()
                all_models = await loop.run_in_executor(
                    None,
                    lambda: fetch_available_models_cached(
                        api_base, api_key.strip(), timeout_seconds=30,
                    ),
                )
                sora_models = _filter_sora_models(all_models)
                return _aiohttp_web.json_response(
                    {"status": "success", "models": sora_models},
                )
            except RuntimeError as exc:
                msg = str(exc)
                print(f"[Sora] fetch models error: {msg}")
                return _aiohttp_web.json_response(
                    {"status": "error", "message": msg}, status=502,
                )
            except Exception as exc:
                print(f"[Sora] fetch models error: {exc}")
                return _aiohttp_web.json_response(
                    {"status": "error", "message": str(exc)}, status=500,
                )
except Exception as _exc:
    print(f"Warning: Could not register Sora model-fetch route: {_exc}")


# ===================================================================
# Helper: multipart/form-data body builder
# ===================================================================

def _build_multipart_body(fields):
    """Build a ``multipart/form-data`` request body from a dict of fields.

    Args:
        fields: ``dict`` — field name → string value.  ``None`` values are
            skipped.

    Returns:
        ``(bytes, str)`` — raw body bytes and the ``Content-Type`` header
        value (including boundary).
    """
    boundary = "----ZhangyuAPISora" + uuid.uuid4().hex[:16]
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n'
            f"\r\n"
            f"{value}\r\n"
        )
    parts.append(f"--{boundary}--\r\n")
    body = "".join(parts).encode("utf-8")
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


# ===================================================================
# Helper: parse size string
# ===================================================================

def _parse_size(size_str):
    """Parse ``"WxH"`` size string into ``(width, height)`` integers.

    Args:
        size_str: e.g. ``"1280x720"``.

    Returns:
        ``(int, int)`` — defaults to ``(1280, 720)`` on parse failure.
    """
    try:
        w, h = str(size_str).split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return 1280, 720


# ===================================================================
# Node class
# ===================================================================

class ComfyuiZhangyuAPISoraNode:
    """ComfyUI Sora 格式视频生成节点 — OpenAI Sora API 兼容.

    Node display name: **ComfyUI-zhangyuapi-Sora格式**
    """

    SIZES = VIDEO_SIZES

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "response", "model_list")
    FUNCTION = "generate"
    CATEGORY = "Comfyui-ZhangyuAPI/视频"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key (API密钥)": (
                    "STRING", {"default": "", "multiline": False}),
                "prompt (提示词)": (
                    "STRING", {"default": "", "multiline": True}),
                "model (模型)": (
                    "STRING", {"default": "sora", "multiline": False,
                               "placeholder": "sora"}),
                "api_base (接口域名)": (
                    "STRING", {"default": DEFAULT_SORA_BASE, "multiline": False}),
                "size (分辨率)": (
                    cls.SIZES, {"default": "1280x720"}),
                "duration (时长秒数)": (
                    "INT", {"default": 8, "min": 4, "max": 60}),
                "seed (种子)": (
                    "INT", {"default": 0, "min": 0, "max": 2147483647,
                            "control_after_generate": True}),
                "timeout_seconds (超时秒数)": (
                    "INT", {"default": DEFAULT_VIDEO_TIMEOUT,
                            "min": DEFAULT_MIN_VIDEO_TIMEOUT,
                            "max": DEFAULT_MAX_VIDEO_TIMEOUT}),
                "retry_times (重试次数)": (
                    "INT", {"default": DEFAULT_VIDEO_RETRY_TIMES,
                            "min": 1, "max": 5}),
            },
            "optional": {
                "negative_prompt (反向提示词)": (
                    "STRING", {"default": "", "multiline": True}),
                "fps (帧率)": (
                    "INT", {"default": 24, "min": 1, "max": 120}),
                "n (生成数量)": (
                    "INT", {"default": 1, "min": 1, "max": 4}),
                "image_01 (参考图)": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
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
             if k != "image_01 (参考图)"},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.md5(key.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Payload builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_payload(model, prompt, width, height, duration,
                        negative_prompt=None, fps=None, n=1, seed=None,
                        image_data_url=None):
        """Build the multipart form fields for ``POST /v1/videos``.

        Args:
            model: Model ID.
            prompt: Cleaned prompt text.
            width: Video width in pixels.
            height: Video height in pixels.
            duration: Video duration in seconds.
            negative_prompt: Optional negative prompt → ``metadata.negative_prompt``.
            fps: Optional frame rate.
            n: Number of videos (usually 1).
            seed: Optional random seed.
            image_data_url: Optional base64 data URL for img2video.

        Returns:
            ``dict`` — field name → string value.
        """
        fields = {
            "model": model,
            "prompt": prompt,
            "width": str(width),
            "height": str(height),
            "duration": str(duration),
            "n": str(n),
        }

        if seed is not None and seed > 0:
            fields["seed"] = str(seed)

        if fps is not None and fps > 0:
            fields["fps"] = str(fps)

        if image_data_url:
            fields["image"] = image_data_url

        if negative_prompt:
            fields["metadata"] = json.dumps(
                {"negative_prompt": negative_prompt}, ensure_ascii=False,
            )

        return fields

    # ------------------------------------------------------------------
    # API request
    # ------------------------------------------------------------------

    @staticmethod
    def _request_generation(api_base, headers, fields, timeout_seconds):
        """POST to ``/v1/videos`` with multipart/form-data encoding."""
        url = f"{api_base}/v1/videos"
        body, content_type = _build_multipart_body(fields)
        merged_headers = {**headers, "Content-Type": content_type}
        return _get_http_client().post(
            url,
            content=body,
            headers=merged_headers,
            timeout=ZHANGYUAPI_timeout(timeout_seconds),
        )

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    @staticmethod
    def _poll_sora_task(api_base, headers, task_id, timeout_seconds,
                         retry_times, on_tick=None):
        """Poll ``GET /v1/videos/{task_id}`` until completion.

        Returns the completed task data dict.
        """
        url = f"{api_base}/v1/videos/{task_id}"
        start_ts = time.time()
        poll_start = time.time()
        consecutive_errors = 0

        # Adaptive polling stages: (threshold_s, interval_s)
        stages = ((15, 2.0), (60, 5.0), (float("inf"), 10.0))

        while True:
            elapsed = time.time() - start_ts
            poll_elapsed = time.time() - poll_start
            remaining = timeout_seconds - int(poll_elapsed + 0.999)

            if remaining <= 0:
                raise RuntimeError(
                    f"视频任务轮询超时 ({timeout_seconds}s)，"
                    f"已等待 {elapsed:.1f}s"
                )

            try:
                response = ZHANGYUAPI_get(
                    url, remaining,
                    headers={**headers, "Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    consecutive_errors = 0
                    data = response.json()
                    status = str(data.get("status", "")).lower()

                    if status in ("completed", "succeeded", "success", "done"):
                        return data
                    if status in ("failed", "error", "cancelled", "canceled"):
                        err = data.get("error") or data.get("message") or status
                        raise RuntimeError(
                            f"视频任务失败 (task_id={task_id}): {err}"
                        )

                elif is_retryable_http_status(response.status_code):
                    consecutive_errors += 1
                    if consecutive_errors > retry_times:
                        raise RuntimeError(
                            f"轮询连续 {consecutive_errors} 次 HTTP "
                            f"{response.status_code} 错误，中止"
                        )
                else:
                    raise RuntimeError(
                        f"轮询失败 HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )

            except _RETRYABLE_EXCEPTIONS as exc:
                consecutive_errors += 1
                if consecutive_errors > retry_times:
                    raise RuntimeError(
                        f"轮询连续 {consecutive_errors} 次网络错误，中止: {exc}"
                    )

            # Wait before next poll
            if consecutive_errors > 0:
                time.sleep(_jittered_backoff_seconds(consecutive_errors))
            else:
                interval = 2.0
                for threshold, iv in stages:
                    if poll_elapsed < threshold:
                        interval = iv
                        break
                if on_tick is not None:
                    on_tick(elapsed, poll_elapsed, interval)
                time.sleep(interval)

    # ------------------------------------------------------------------
    # Video download
    # ------------------------------------------------------------------

    @staticmethod
    def _download_video(api_base, headers, video_id, timeout_seconds,
                         retry_times):
        """Download video bytes from ``GET /v1/videos/{id}/content``."""
        url = f"{api_base}/v1/videos/{video_id}/content"
        return _download_bytes_with_retry(
            url, headers, timeout_seconds, retry_times, label="Sora视频",
        )

    # ------------------------------------------------------------------
    # Save video to output directory
    # ------------------------------------------------------------------

    @staticmethod
    def _save_video(video_bytes, video_id):
        """Save video bytes to ComfyUI output directory and return path."""
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except Exception:
            output_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "output"
            )
            output_dir = os.path.abspath(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        safe_id = "".join(
            c for c in str(video_id) if c.isalnum() or c in "_-"
        )
        filename = f"zhangyuapi_sora_{safe_id}_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(video_bytes)
        print(f"[Comfyui-ZhangyuAPI-Sora] saved: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self, **kwargs):
        """Execute a Sora-format video generation request.

        1. Validates and sanitizes all inputs.
        2. Submits ``POST /v1/videos`` (multipart/form-data).
        3. Polls ``GET /v1/videos/{task_id}`` with adaptive intervals.
        4. Downloads video via ``GET /v1/videos/{id}/content``.
        5. Saves to ComfyUI output directory.

        Returns:
            ``(VideoFromFile | str, str, str)`` — video object,
            JSON response summary, and JSON model list.
        """
        # -- sanitize inputs --------------------------------------------------
        api_key = kwargs.get("api_key (API密钥)", "").strip()
        api_base = normalize_api_base(
            kwargs.get("api_base (接口域名)", DEFAULT_SORA_BASE))
        prompt = kwargs.get("prompt (提示词)", "")
        model = kwargs.get("model (模型)", "").strip() or "sora"
        size = safe_choice(
            kwargs.get("size (分辨率)", "1280x720"),
            self.SIZES, "1280x720")
        duration = safe_int(kwargs.get("duration (时长秒数)", 8), 8, 4, 60)
        seed = safe_int(kwargs.get("seed (种子)", 0), 0, 0, 2147483647)
        timeout_seconds = safe_int(
            kwargs.get("timeout_seconds (超时秒数)", DEFAULT_VIDEO_TIMEOUT),
            DEFAULT_VIDEO_TIMEOUT,
            DEFAULT_MIN_VIDEO_TIMEOUT,
            DEFAULT_MAX_VIDEO_TIMEOUT,
        )
        retry_times = safe_int(
            kwargs.get("retry_times (重试次数)", DEFAULT_VIDEO_RETRY_TIMES),
            DEFAULT_VIDEO_RETRY_TIMES, 1, 5)
        unique_id = kwargs.get("unique_id")
        start_ts = time.time()

        # -- optional fields --------------------------------------------------
        negative_prompt = normalize_prompt_text(
            kwargs.get("negative_prompt (反向提示词)", ""))
        fps = safe_int(kwargs.get("fps (帧率)", 0), 0, 0, 120)
        n = safe_int(kwargs.get("n (生成数量)", 1), 1, 1, 4)

        # Reference image (img2video)
        image_data_url = None
        image_tensor = kwargs.get("image_01 (参考图)")
        if image_tensor is not None:
            image_data_url = tensor_to_data_url(image_tensor)

        # -- validate ---------------------------------------------------------
        if not api_key:
            emit_runtime_status(unique_id, "error", "API Key 为空",
                                0.0, 0, retry_times, timeout_seconds)
            raise ValueError("API Key 不能为空")

        clean_prompt = normalize_prompt_text(prompt)
        if not clean_prompt:
            raise ValueError("prompt 不能为空")

        width, height = _parse_size(size)

        # -- resolve model ----------------------------------------------------
        try:
            model, model_list = resolve_and_validate_model(
                model, api_base, api_key, unique_id,
                placeholder="sora",
                filter_func=_filter_sora_models,
            )
        except ValueError as exc:
            emit_runtime_status(unique_id, "error", str(exc),
                                0.0, 0, retry_times, timeout_seconds)
            raise

        # -- prepare request --------------------------------------------------
        headers = {"Authorization": f"Bearer {api_key}"}
        fields = self._build_payload(
            model, clean_prompt, width, height, duration,
            negative_prompt=negative_prompt or None,
            fps=fps if fps > 0 else None,
            n=n,
            seed=seed if seed > 0 else None,
            image_data_url=image_data_url,
        )

        print(
            f"[Comfyui-ZhangyuAPI-Sora] model={model}, size={size} "
            f"({width}x{height}), duration={duration}s, seed={seed}"
            + (f", img2video" if image_data_url else "")
        )
        emit_runtime_status(unique_id, "running", "开始生成视频",
                            0.0, 0, retry_times, timeout_seconds)

        # -- retry loop -------------------------------------------------------
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
                    api_base, headers, fields, timeout_seconds,
                )

                if response.status_code not in (200, 201, 202):
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

                # -- Identify video / task ID ---------------------------------
                video_id = (
                    data.get("id")
                    or data.get("video_id")
                    or data.get("task_id")
                )
                if not video_id:
                    raise RuntimeError(
                        f"API 未返回 video id: "
                        f"{json.dumps(_sanitize_api_response(data), ensure_ascii=False)[:500]}"
                    )

                # -- Poll until completion ------------------------------------
                remaining = timeout_seconds - int(time.time() - start_ts)
                emit_runtime_status(
                    unique_id, "running",
                    f"视频任务已提交 (id={video_id})，轮询中",
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )

                def _on_poll_tick(elapsed, poll_elapsed, interval):
                    emit_runtime_status(
                        unique_id, "running",
                        f"轮询视频中 · 间隔{interval:.0f}s · "
                        f"已等待{poll_elapsed:.0f}s",
                        elapsed,
                        attempt, retry_times, timeout_seconds,
                    )

                polled_data = self._poll_sora_task(
                    api_base, headers, video_id, remaining, retry_times,
                    on_tick=_on_poll_tick,
                )

                # -- Download video --------------------------------------------
                emit_runtime_status(
                    unique_id, "running", "下载视频中",
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )

                video_bytes = self._download_video(
                    api_base, headers, video_id, timeout_seconds, retry_times,
                )
                filepath = self._save_video(video_bytes, video_id)

                # -- Build return ----------------------------------------------
                elapsed = time.time() - start_ts
                response_info = {
                    "status": "success",
                    "format": "Sora",
                    "api_base": denormalize_api_base(api_base),
                    "model": model,
                    "size": size,
                    "width": width,
                    "height": height,
                    "duration": duration,
                    "fps": fps if fps > 0 else None,
                    "seed": seed,
                    "video_id": video_id,
                    "filepath": filepath,
                    "input_image": image_data_url is not None,
                    "elapsed_seconds": round(elapsed, 2),
                }

                emit_runtime_status(
                    unique_id, "success",
                    f"视频生成成功 (耗时 {elapsed:.1f}s)",
                    elapsed, attempt, retry_times, timeout_seconds,
                )

                video_obj = VideoFromFile(filepath) if VideoFromFile is not None else filepath
                return (
                    video_obj,
                    json.dumps(response_info, ensure_ascii=False, indent=2),
                    json.dumps(model_list, ensure_ascii=False),
                )

            except _RETRYABLE_EXCEPTIONS as exc:
                last_error = str(exc)
                if attempt < retry_times:
                    emit_runtime_status(
                        unique_id, "running",
                        f"网络/超时，重试中 ({attempt}/{retry_times})",
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

        # -- all retries exhausted --------------------------------------------
        elapsed = time.time() - start_ts
        emit_runtime_status(
            unique_id, "error",
            f"连续 {retry_times} 次失败",
            elapsed, retry_times, retry_times, timeout_seconds,
        )
        raise RuntimeError(
            f"Comfyui-ZhangyuAPI-Sora 连续 {retry_times} 次失败，"
            f"最后错误: {last_error}"
        )


# ===================================================================
# ComfyUI node registration
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ComfyuiZhangyuAPISoraNode": ComfyuiZhangyuAPISoraNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiZhangyuAPISoraNode": "ComfyUI-zhangyuapi-Sora格式",
}
