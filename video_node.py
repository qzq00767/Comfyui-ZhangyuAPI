#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — 视频生成节点.

基于 NewAPI / OpenAI 兼容的视频生成 API，复用已有的域名 + Key 认证体系、
自适应轮询、模型自动获取等全部基础设施。

Supported API patterns (auto-detected):
- Pattern A (OpenAI / LLM Gateway): ``POST /v1/videos`` → ``GET /v1/videos/{id}`` → ``GET /v1/videos/{id}/content``
- Pattern B (xAI / generations): ``POST /v1/videos/generations`` → ``GET /v1/videos/{id}`` (video.url in response)

Specification:
- Endpoint: ``POST /v1/videos`` (or ``/v1/videos/generations`` as fallback)
- Auth: ``Authorization: Bearer <api_key>``
- Async response: ``{"id": "video_xxx", "status": "queued"}`` → polled with adaptive intervals
- Download: ``GET /v1/videos/{id}/content`` → mp4 bytes
- Error: ``{"error": {"message": "..."}}``
"""

import json
import os
import time
import uuid

import httpx

# ---------------------------------------------------------------------------
# Shared utilities — imported from the sibling gpt-image-2 node module
# ---------------------------------------------------------------------------
from .zhangyu_gpt_img2 import (  # noqa: E402
    # HTTP client + helpers
    ZHANGYUAPI_post,
    ZHANGYUAPI_get,
    ZHANGYUAPI_timeout,
    # URL / prompt sanitizers
    normalize_api_base,
    denormalize_api_base,
    normalize_prompt_text,
    # Input coercers
    safe_choice,
    safe_int,
    # Async detection
    is_async_task_response,
    # Polling (reuse unified _poll_async_task from shared module)
    _poll_async_task,
    # Retry / backoff
    is_retryable_http_status,
    _jittered_sleep,
    _RETRYABLE_EXCEPTIONS,
    _download_bytes_with_retry,
    # Frontend status emitter
    emit_runtime_status,
    # Constants
    DEFAULT_API_BASE_URL,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_READ_TIMEOUT,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_NODE_TIMEOUT,
    DEFAULT_RETRY_TIMES,
    # Model resolution & validation
    resolve_and_validate_model,
    # Response sanitizer (strips base64 from API responses)
    _sanitize_api_response,
    _extract_api_error_message,
    # Unified model filtering
    _filter_models_by_patterns,
    _log,
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

DEFAULT_VIDEO_TIMEOUT = 900   # video generation is slower than image
DEFAULT_MIN_VIDEO_TIMEOUT = 120
DEFAULT_MAX_VIDEO_TIMEOUT = 3600
DEFAULT_VIDEO_RETRY_TIMES = 2

# Common video resolutions (width x height)
VIDEO_SIZES = [
    "1280x720",     # 720p landscape
    "720x1280",     # 720p portrait
    "1024x1792",    # portrait HD
    "1792x1024",    # landscape HD
    "1920x1080",    # 1080p landscape
    "1080x1920",    # 1080p portrait
]

# ---------------------------------------------------------------------------
# Video model fetch (backend route)
# ---------------------------------------------------------------------------

# Known video model name patterns — used to filter /v1/models results
_VIDEO_MODEL_PATTERNS = [
    "sora", "veo", "grok-imagine-video", "wan", "wanx",
    "kling", "seedance", "cogvideo", "mochi", "stable-video",
    "video", "vidu", "pika", "runway", "hailuo", "minimax-video",
    "gen", "ltx", "hunyuan-video", "pyramid-flow",
]


def _filter_video_models(all_models):
    """Filter a list of model IDs to only video-capable models.

    Uses known video model name patterns (include mode).  Falls back to
    returning all models if no matches are found (the API may use an
    unknown prefix).
    """
    return _filter_models_by_patterns(
        all_models, _VIDEO_MODEL_PATTERNS,
        mode="include", fallback_empty=True,
    )


# Register backend route for video-model fetching
try:
    import asyncio as _asyncio_import_check
    import server as _comfy_server
    from aiohttp import web as _aiohttp_web

    if (_comfy_server is not None
            and _comfy_server.PromptServer.instance is not None):
        _routes = _comfy_server.PromptServer.instance.routes

        @_routes.post("/zhangyuapi_fetch_video_models")
        async def _zhangyuapi_fetch_video_models_route(request):
            """Handle ``POST /zhangyuapi_fetch_video_models``.

            Expects JSON: ``{"api_base": "...", "api_key": "..."}``.
            Returns ``{"status": "success", "models": [...]}``.
            """
            try:
                data = await request.json()
                api_base = data.get("api_base", "")
                api_key = data.get("api_key", "")

                if not api_key or not api_key.strip():
                    return _aiohttp_web.json_response(
                        {"status": "error", "message": "API Key 不能为空"},
                        status=400,
                    )

                # Import fetch_available_models from zhangyu_gpt_img2 at runtime
                from .zhangyu_gpt_img2 import fetch_available_models

                loop = _asyncio_import_check.get_running_loop()
                all_models = await loop.run_in_executor(
                    None,
                    lambda: fetch_available_models(
                        api_base,
                        api_key.strip(),
                        timeout_seconds=30,
                    ),
                )

                video_models = _filter_video_models(all_models)
                return _aiohttp_web.json_response(
                    {"status": "success", "models": video_models},
                )
            except RuntimeError as exc:
                msg = str(exc)
                print(f"Comfyui-ZhangyuAPI: fetch video models error: {msg}")
                return _aiohttp_web.json_response(
                    {"status": "error", "message": msg}, status=502,
                )
            except Exception as exc:
                print(f"Comfyui-ZhangyuAPI: fetch video models error: {exc}")
                return _aiohttp_web.json_response(
                    {"status": "error", "message": str(exc)}, status=500,
                )
except Exception as _exc:
    print(f"Warning: Could not register video-model-fetch route: {_exc}")




# ===================================================================
# Node class
# ===================================================================

class ComfyuiZhangyuAPIVideoNode:
    """ComfyUI 视频生成节点 — 域名+Key 即用.

    接口自动探测（OpenAI / xAI 两种风格），模型自动从
    ``/v1/models`` 筛选，异步任务自适应轮询，视频下载到
    ComfyUI output 目录。

    Node display name: **ComfyUI-zhangyuapi-视频生成**
    """

    SIZES = VIDEO_SIZES

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "response")
    FUNCTION = "generate"
    CATEGORY = "Comfyui-ZhangyuAPI/视频"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key (API密钥)": (
                    "STRING", {"default": "", "multiline": False}),
                "api_base (接口域名)": (
                    "STRING", {"default": DEFAULT_API_BASE_URL, "multiline": False}),
                "prompt (提示词)": (
                    "STRING", {"default": "", "multiline": True}),
                "model (模型)": (
                    ["auto (自动选择)"],),
                "seconds (时长秒数)": (
                    "INT", {"default": 8, "min": 4, "max": 60}),
                "size (分辨率)": (
                    cls.SIZES, {"default": "1280x720"}),
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
                "aspect_ratio (画面比例)": (
                    ["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "custom_model (自定义模型名)": (
                    "STRING", {"default": "", "multiline": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    # ------------------------------------------------------------------
    # Payload builder
    # ------------------------------------------------------------------

    def _build_payload(self, model, prompt, negative_prompt, seconds,
                       size, n_images=1):
        """Build JSON request body for video generation.

        Args:
            model: Model ID or ``"auto (自动选择)"``.
            prompt: Cleaned prompt.
            negative_prompt: Negative prompt (sent only if non-empty).
            seconds: Video duration in seconds.
            size: Resolution e.g. ``"1280x720"``.
            n_images: Number of videos (usually 1).

        Returns:
            ``dict`` — JSON-serializable request body.
        """
        payload = {
            "prompt": prompt,
            "seconds": seconds,
            "size": size,
            "n": n_images,
        }

        if model and model != "auto (自动选择)":
            payload["model"] = model

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        return payload

    # ------------------------------------------------------------------
    # API request (auto-detects endpoint)
    # ------------------------------------------------------------------

    def _request_video_generation(self, api_base, headers, payload,
                                   timeout_seconds):
        """POST to the video generation endpoint.

        Tries ``/v1/videos`` first (OpenAI / LLM Gateway style).
        If 404, falls back to ``/v1/videos/generations`` (xAI style).

        Returns:
            ``(httpx.Response, str)`` — response and the resolved URL path
            (``"/v1/videos"`` or ``"/v1/videos/generations"``).

        Raises:
            RuntimeError: On non-retryable HTTP errors.
        """
        # Pattern A: POST /v1/videos
        url_a = f"{api_base}/v1/videos"
        response = ZHANGYUAPI_post(
            url_a, timeout_seconds,
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
        )

        if response.status_code == 404:
            # Pattern B: POST /v1/videos/generations
            url_b = f"{api_base}/v1/videos/generations"
            print("[Comfyui-ZhangyuAPI-视频] /v1/videos returned 404, "
                  "trying /v1/videos/generations")
            response = ZHANGYUAPI_post(
                url_b, timeout_seconds,
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
            )
            if response.status_code != 404:
                return response, "/v1/videos/generations"

        return response, "/v1/videos"

    # ------------------------------------------------------------------
    # Video download
    # ------------------------------------------------------------------

    @staticmethod
    def _download_video_content(api_base, headers, video_id,
                                 timeout_seconds, retry_times=3):
        """Download video bytes from ``GET /v1/videos/{id}/content``.

        Delegates to shared :func:`_download_bytes_with_retry`.
        """
        url = f"{api_base}/v1/videos/{video_id}/content"
        return _download_bytes_with_retry(
            url, headers, timeout_seconds, retry_times, label="视频",
        )

    @staticmethod
    def _download_video_from_url(video_url, headers, timeout_seconds, retry_times=3):
        """Download video bytes from a direct URL (xAI pattern).

        Delegates to shared :func:`_download_bytes_with_retry`.
        """
        return _download_bytes_with_retry(
            video_url, headers, timeout_seconds, retry_times, label="视频",
        )

    @staticmethod
    def _save_video(video_bytes, video_id):
        """Save video bytes to ComfyUI output directory.

        Returns the absolute file path.
        """
        # Save to ComfyUI output directory if available
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except Exception:
            output_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "output"
            )
            output_dir = os.path.abspath(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        safe_id = "".join(c for c in str(video_id) if c.isalnum() or c in "_-")
        filename = f"zhangyuapi_{safe_id}_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(video_bytes)
        print(f"[Comfyui-ZhangyuAPI-视频] saved: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self, **kwargs):
        """Execute a video generation request.

        1. Validates and sanitizes all inputs.
        2. Sends ``POST /v1/videos`` (or ``/v1/videos/generations``).
        3. Polls with adaptive intervals until completion.
        4. Downloads video and saves to output directory.
        5. Returns ``(VideoFromFile, JSON response string)``.

        Args:
            **kwargs: ComfyUI widget values.

        Returns:
            ``tuple[VideoFromFile | str, str]`` — video object and
            JSON response summary.
        """
        # -- sanitize inputs --------------------------------------------------
        api_key = kwargs.get("api_key (API密钥)", "")
        api_base = normalize_api_base(
                kwargs.get("api_base (接口域名)", DEFAULT_API_BASE_URL)
            )
        prompt = kwargs.get("prompt (提示词)", "")

        model = kwargs.get("model (模型)", "auto (自动选择)")
        custom_model = (kwargs.get("custom_model (自定义模型名)") or "").strip()
        if custom_model:
            model = custom_model
            _log("info", f"[用户自定义模型] {model}")

        seconds = safe_int(kwargs.get("seconds (时长秒数)", 8), 8, 4, 60)
        size = safe_choice(
            kwargs.get("size (分辨率)", "1280x720"),
            self.SIZES, "1280x720")
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

        # -- validate ---------------------------------------------------------
        if not api_key.strip():
            emit_runtime_status(unique_id, "error", "API Key 为空",
                                0.0, 0, retry_times, timeout_seconds)
            raise ValueError("API Key 不能为空")

        # -- resolve model (auto-detect if placeholder) ----------------------
        try:
            model, model_list = resolve_and_validate_model(
                model, api_base, api_key.strip(), unique_id,
                placeholder="auto (自动选择)",
                filter_func=_filter_video_models,
            )
        except ValueError as exc:
            emit_runtime_status(unique_id, "error", str(exc),
                                0.0, 0, retry_times, timeout_seconds)
            raise

        clean_prompt = normalize_prompt_text(prompt)
        if not clean_prompt:
            raise ValueError("prompt 不能为空")

        negative_prompt = kwargs.get("negative_prompt (反向提示词)", "")
        clean_negative = normalize_prompt_text(negative_prompt)

        # -- prepare request --------------------------------------------------
        headers = {"Authorization": f"Bearer {api_key.strip()}"}
        payload = self._build_payload(
            model, clean_prompt, clean_negative, seconds, size,
        )

        print(
            f"[Comfyui-ZhangyuAPI-视频] model={model}, seconds={seconds}s, "
            f"size={size}, seed={seed}"
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

                response, _ = self._request_video_generation(
                    api_base, headers, payload, timeout_seconds,
                )

                # Parse error responses
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
                    or data.get("request_id")
                )
                if not video_id:
                    # May be a sync response — check for direct video URL
                    video_url = (
                        data.get("video", {}).get("url")
                        if isinstance(data.get("video"), dict)
                        else data.get("url")
                    )
                    if video_url:
                        video_bytes = self._download_video_from_url(
                            video_url, headers, timeout_seconds, retry_times,
                        )
                        filepath = self._save_video(video_bytes, "direct")
                        return self._build_return(
                            filepath, data, api_base, model, seconds, size,
                            seed, start_ts, unique_id, attempt, retry_times,
                            timeout_seconds,
                        )
                    raise RuntimeError(
                        f"API 未返回 video id: "
                        f"{json.dumps(_sanitize_api_response(data), ensure_ascii=False)[:500]}"
                    )

                # -- Poll async task -------------------------------------------
                remaining = timeout_seconds - int(time.time() - start_ts)
                emit_runtime_status(
                    unique_id, "running",
                    f"视频任务已提交 (id={video_id})，自适应轮询中",
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

                polled_data = _poll_async_task(
                    api_base, headers, video_id, remaining, retry_times,
                    on_tick=_on_poll_tick,
                    poll_url=f"{api_base}/v1/videos/{video_id}",
                )

                # -- Download video --------------------------------------------
                emit_runtime_status(
                    unique_id, "running", "下载视频中",
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )

                # Pattern B (xAI): video.url in response
                video_info = polled_data.get("video", {})
                if isinstance(video_info, dict) and video_info.get("url"):
                    video_bytes = self._download_video_from_url(
                        video_info["url"], headers, timeout_seconds, retry_times,
                    )
                else:
                    # Pattern A (OpenAI): GET /v1/videos/{id}/content
                    video_bytes = self._download_video_content(
                        api_base, headers, video_id, timeout_seconds,
                        retry_times,
                    )

                filepath = self._save_video(video_bytes, video_id)
                return self._build_return(
                    filepath, polled_data, api_base, model, seconds, size,
                    seed, start_ts, unique_id, attempt, retry_times,
                    timeout_seconds,
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
            f"Comfyui-ZhangyuAPI-视频 连续 {retry_times} 次失败，"
            f"最后错误: {last_error}"
        )

    # ------------------------------------------------------------------
    # Return builder
    # ------------------------------------------------------------------

    def _build_return(self, filepath, data, api_base, model, seconds, size,
                       seed, start_ts, unique_id, attempt, retry_times,
                       timeout_seconds):
        """Construct the node return value (video object + JSON response)."""
        elapsed = time.time() - start_ts

        response_info = {
            "status": "success",
            "api_base": denormalize_api_base(api_base),
            "model": model,
            "seconds": seconds,
            "size": size,
            "seed": seed,
            "filepath": filepath,
            "usage": data.get("usage"),
            "elapsed_seconds": round(elapsed, 2),
        }

        emit_runtime_status(
            unique_id, "success",
            f"视频生成成功 (耗时 {elapsed:.1f}s)",
            elapsed, attempt, retry_times, timeout_seconds,
        )

        # Return VideoFromFile if available, otherwise file path
        if VideoFromFile is not None:
            video_obj = VideoFromFile(filepath)
        else:
            video_obj = filepath
            print("[Comfyui-ZhangyuAPI-视频] VideoFromFile not available, "
                  "returning file path string")

        return (
            video_obj,
            json.dumps(response_info, ensure_ascii=False, indent=2),
        )


# ===================================================================
# ComfyUI node registration
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ComfyuiZhangyuAPIVideoNode": ComfyuiZhangyuAPIVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiZhangyuAPIVideoNode": "ComfyUI-zhangyuapi-视频生成 🧪测试中",
}
