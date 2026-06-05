#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — 可灵格式视频生成节点.

适配 NewAPI 的 Kling 兼容端点：
- 文生视频: ``POST /kling/v1/videos/text2video``
- 图生视频: ``POST /kling/v1/videos/image2video``
- 状态查询: ``GET /kling/v1/videos/text2video/{task_id}`` 或
   ``GET /kling/v1/videos/image2video/{task_id}``
- 视频下载: 从完成状态响应中的 ``url`` 字段直接下载

Auth: ``Authorization: Bearer <api_key>`` (所有端点).
"""

import hashlib
import json
import os
import random
import time
import uuid

from .zhangyu_gpt_img2 import (
    _get_http_client,
    ZHANGYUAPI_timeout,
    ZHANGYUAPI_get,
    normalize_api_base,
    denormalize_api_base,
    safe_choice,
    safe_int,
    normalize_prompt_text,
    _RETRYABLE_EXCEPTIONS,
    is_retryable_http_status,
    _jittered_sleep,
    _jittered_backoff_seconds,
    _download_bytes_with_retry,
    emit_runtime_status,
    resolve_and_validate_model,
    _sanitize_api_response,
    _extract_api_error_message,
    _filter_models_by_patterns,
    fetch_available_models_cached,
    tensor_to_data_url,
    _log,
    DEFAULT_API_BASE_URL,
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

DEFAULT_KLING_BASE = DEFAULT_API_BASE_URL
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

# Kling model filter patterns
_KLING_MODEL_PATTERNS = [
    "kling", "kling-v",
]


def _filter_kling_models(all_models):
    """Filter model list to Kling-compatible models."""
    return _filter_models_by_patterns(
        all_models, _KLING_MODEL_PATTERNS,
        mode="include", fallback_empty=True,
    )


# ---------------------------------------------------------------------------
# Backend route for Kling model fetching
# ---------------------------------------------------------------------------
try:
    import asyncio as _asyncio_import_check
    import server as _comfy_server
    from aiohttp import web as _aiohttp_web

    if (_comfy_server is not None
            and _comfy_server.PromptServer.instance is not None):
        _routes = _comfy_server.PromptServer.instance.routes

        @_routes.post("/zhangyuapi_fetch_kling_models")
        async def _zhangyuapi_fetch_kling_models_route(request):
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
                kling_models = _filter_kling_models(all_models)
                return _aiohttp_web.json_response(
                    {"status": "success", "models": kling_models},
                )
            except RuntimeError as exc:
                msg = str(exc)
                print(f"[Kling] fetch models error: {msg}")
                return _aiohttp_web.json_response(
                    {"status": "error", "message": msg}, status=502,
                )
            except Exception as exc:
                print(f"[Kling] fetch models error: {exc}")
                return _aiohttp_web.json_response(
                    {"status": "error", "message": str(exc)}, status=500,
                )
except Exception as _exc:
    print(f"Warning: Could not register Kling model-fetch route: {_exc}")


# ===================================================================
# Helper: parse size
# ===================================================================

def _parse_size(size_str):
    """Parse ``"WxH"`` → ``(width, height)``; defaults to ``(1280, 720)``."""
    try:
        w, h = str(size_str).split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return 1280, 720


# ===================================================================
# Node class
# ===================================================================

class ComfyuiZhangyuAPIKlingNode:
    """ComfyUI 可灵格式视频生成节点 — Kling API 兼容.

    根据是否提供参考图自动选择 text2video 或 image2video 模式。

    Node display name: **ComfyUI-zhangyuapi-可灵格式**
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
                    "STRING", {"default": "kling-v1", "multiline": False,
                               "placeholder": "kling-v1"}),
                "api_base (接口域名)": (
                    "STRING", {"default": DEFAULT_KLING_BASE, "multiline": False}),
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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_image(kwargs):
        """Collect reference image from ``image_01 (参考图)`` input.

        Returns:
            ``str | None`` — base64 data-URL string, or ``None``.
        """
        tensor = kwargs.get("image_01 (参考图)")
        if tensor is None:
            return None
        return tensor_to_data_url(tensor)

    @staticmethod
    def _build_payload(model, prompt, width, height, duration,
                        negative_prompt=None, fps=None, n=1, seed=None,
                        image_data_url=None):
        """Build JSON request body for Kling video generation.

        Returns:
            ``dict`` — JSON-serializable payload.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "duration": duration,
            "n": n,
        }

        if seed is not None and seed > 0:
            payload["seed"] = seed

        if fps is not None and fps > 0:
            payload["fps"] = fps

        if image_data_url:
            payload["image"] = image_data_url

        if negative_prompt:
            payload["metadata"] = {"negative_prompt": negative_prompt}

        return payload

    # ------------------------------------------------------------------
    # API request
    # ------------------------------------------------------------------

    def _request_generation(self, api_base, headers, payload,
                             is_image2video, timeout_seconds):
        """POST to Kling video endpoint (text2video or image2video)."""
        sub_path = "image2video" if is_image2video else "text2video"
        url = f"{api_base}/kling/v1/videos/{sub_path}"
        return _get_http_client().post(
            url,
            json=payload,
            headers={**headers, "Content-Type": "application/json"},
            timeout=ZHANGYUAPI_timeout(timeout_seconds),
        )

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    @staticmethod
    def _poll_kling_task(api_base, headers, task_id, is_image2video,
                          timeout_seconds, retry_times, on_tick=None):
        """Poll Kling task status endpoint until completion.

        Returns the completed task data dict (contains ``url`` field).
        """
        sub_path = "image2video" if is_image2video else "text2video"
        url = f"{api_base}/kling/v1/videos/{sub_path}/{task_id}"
        start_ts = time.time()
        poll_start = time.time()
        consecutive_errors = 0
        stages = ((15, 2.0), (60, 5.0), (float("inf"), 10.0))

        while True:
            elapsed = time.time() - start_ts
            poll_elapsed = time.time() - poll_start
            remaining = timeout_seconds - int(poll_elapsed + 0.999)

            if remaining <= 0:
                raise RuntimeError(
                    f"Kling 任务轮询超时 ({timeout_seconds}s)，"
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
                        if not data.get("url"):
                            raise RuntimeError(
                                "Kling 任务已完成但未返回视频 URL"
                            )
                        return data
                    if status in ("failed", "error", "cancelled", "canceled"):
                        err = (data.get("error") or data.get("message") or
                               json.dumps(data, ensure_ascii=False)[:300])
                        raise RuntimeError(
                            f"Kling 任务失败 (task_id={task_id}): {err}"
                        )

                elif is_retryable_http_status(response.status_code):
                    consecutive_errors += 1
                    if consecutive_errors > retry_times:
                        raise RuntimeError(
                            f"Kling 轮询连续 {consecutive_errors} 次 HTTP "
                            f"{response.status_code} 错误，中止"
                        )
                else:
                    raise RuntimeError(
                        f"Kling 轮询失败 HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )

            except _RETRYABLE_EXCEPTIONS as exc:
                consecutive_errors += 1
                if consecutive_errors > retry_times:
                    raise RuntimeError(
                        f"Kling 轮询连续 {consecutive_errors} 次网络错误，"
                        f"中止: {exc}"
                    )

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
    def _download_video_from_url(video_url, headers, timeout_seconds,
                                  retry_times):
        """Download video bytes from a direct URL."""
        return _download_bytes_with_retry(
            video_url, headers, timeout_seconds, retry_times, label="Kling视频",
        )

    # ------------------------------------------------------------------
    # Save video
    # ------------------------------------------------------------------

    @staticmethod
    def _save_video(video_bytes, video_id):
        """Save video bytes to ComfyUI output directory."""
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
        filename = f"zhangyuapi_kling_{safe_id}_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(video_bytes)
        print(f"[Comfyui-ZhangyuAPI-Kling] saved: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self, **kwargs):
        """Execute a Kling-format video generation request.

        Auto-selects text2video or image2video mode based on whether
        reference images are provided.

        Returns:
            ``(VideoFromFile | str, str, str)``.
        """
        # -- sanitize inputs --------------------------------------------------
        api_key = kwargs.get("api_key (API密钥)", "").strip()
        api_base = normalize_api_base(
            kwargs.get("api_base (接口域名)", DEFAULT_KLING_BASE))
        prompt = kwargs.get("prompt (提示词)", "")
        model = kwargs.get("model (模型)", "").strip() or "kling-v1"
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

        image_data_url = self._collect_image(kwargs)
        is_image2video = image_data_url is not None

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
                placeholder="kling-v1",
                filter_func=_filter_kling_models,
            )
        except ValueError as exc:
            emit_runtime_status(unique_id, "error", str(exc),
                                0.0, 0, retry_times, timeout_seconds)
            raise

        # -- prepare request --------------------------------------------------
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = self._build_payload(
            model, clean_prompt, width, height, duration,
            negative_prompt=negative_prompt or None,
            fps=fps if fps > 0 else None,
            n=n,
            seed=seed if seed > 0 else None,
            image_data_url=image_data_url,
        )

        mode_label = "图生视频" if is_image2video else "文生视频"
        print(
            f"[Comfyui-ZhangyuAPI-Kling] mode={mode_label}, model={model}, "
            f"size={size} ({width}x{height}), duration={duration}s, seed={seed}"
        )
        emit_runtime_status(unique_id, "running", f"开始生成视频 · {mode_label}",
                            0.0, 0, retry_times, timeout_seconds)

        # -- retry loop -------------------------------------------------------
        last_error = None
        for attempt in range(1, retry_times + 1):
            try:
                emit_runtime_status(
                    unique_id, "running",
                    f"{mode_label}请求中 ({attempt}/{retry_times})",
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )

                response = self._request_generation(
                    api_base, headers, payload, is_image2video,
                    timeout_seconds,
                )

                if response.status_code not in (200, 201, 202):
                    try:
                        err_data = response.json()
                        err_msg = _extract_api_error_message(err_data)
                    except Exception:
                        err_msg = response.text[:500]
                    last_error = f"Kling API 错误 {response.status_code}: {err_msg}"
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

                # -- Identify task ID -----------------------------------------
                task_id = (
                    data.get("task_id")
                    or data.get("id")
                    or data.get("video_id")
                )
                if not task_id:
                    raise RuntimeError(
                        f"Kling API 未返回 task_id: "
                        f"{json.dumps(_sanitize_api_response(data), ensure_ascii=False)[:500]}"
                    )

                # -- Poll until completion ------------------------------------
                remaining = timeout_seconds - int(time.time() - start_ts)
                emit_runtime_status(
                    unique_id, "running",
                    f"Kling 任务已提交 (id={task_id})，轮询中",
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )

                def _on_poll_tick(elapsed, poll_elapsed, interval):
                    emit_runtime_status(
                        unique_id, "running",
                        f"轮询 Kling 视频中 · 间隔{interval:.0f}s · "
                        f"已等待{poll_elapsed:.0f}s",
                        elapsed,
                        attempt, retry_times, timeout_seconds,
                    )

                polled_data = self._poll_kling_task(
                    api_base, headers, task_id, is_image2video,
                    remaining, retry_times,
                    on_tick=_on_poll_tick,
                )

                # -- Download video --------------------------------------------
                video_url = polled_data.get("url")
                if not video_url:
                    raise RuntimeError(
                        "Kling 任务完成但响应中没有 url 字段"
                    )

                emit_runtime_status(
                    unique_id, "running", "下载视频中",
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )

                video_bytes = self._download_video_from_url(
                    video_url, headers, timeout_seconds, retry_times,
                )
                filepath = self._save_video(video_bytes, task_id)

                # -- Build return ----------------------------------------------
                elapsed = time.time() - start_ts
                response_info = {
                    "status": "success",
                    "format": "Kling",
                    "mode": mode_label,
                    "api_base": denormalize_api_base(api_base),
                    "model": model,
                    "size": size,
                    "width": width,
                    "height": height,
                    "duration": duration,
                    "fps": fps if fps > 0 else None,
                    "seed": seed,
                    "task_id": task_id,
                    "video_url": video_url,
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
            f"Comfyui-ZhangyuAPI-Kling 连续 {retry_times} 次失败，"
            f"最后错误: {last_error}"
        )


# ===================================================================
# ComfyUI node registration
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ComfyuiZhangyuAPIKlingNode": ComfyuiZhangyuAPIKlingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiZhangyuAPIKlingNode": "ComfyUI-zhangyuapi-可灵格式",
}
