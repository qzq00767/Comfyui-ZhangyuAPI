#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — 即梦格式视频生成节点.

适配 NewAPI 的即梦 (Jimeng) 兼容端点：
- 提交任务: ``POST /jimeng/?Action=CVSync2AsyncSubmitTask&Version=...``
- 查询结果: ``POST /jimeng/?Action=CVSync2AsyncGetResult&Version=...``

即梦使用单一端点，通过 ``Action`` query param 区分操作类型：
- ``CVSync2AsyncSubmitTask`` → 提交视频生成任务
- ``CVSync2AsyncGetResult`` → 查询任务结果

Auth: ``Authorization: Bearer <api_key>``.
"""

import base64
import hashlib
import json
import os
import random
import time
import uuid

from .zhangyu_gpt_img2 import (
    _get_http_client,
    ZHANGYUAPI_timeout,
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
    _sanitize_api_response,
    _skip_error_return,
    _extract_api_error_message,
    _filter_models_by_patterns,
    fetch_available_models_cached,
    tensor_to_data_url,
    tensor_to_png_bytes,
    _auto_downscale,
    _log,
    DEFAULT_API_BASE_URL,
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

DEFAULT_JIMENG_BASE = DEFAULT_API_BASE_URL
DEFAULT_VIDEO_TIMEOUT = 900
DEFAULT_MIN_VIDEO_TIMEOUT = 120
DEFAULT_MAX_VIDEO_TIMEOUT = 3600
DEFAULT_VIDEO_RETRY_TIMES = 2

# Jimeng model filter patterns (for model_list output port)
_JIMENG_MODEL_PATTERNS = [
    "jimeng", "即梦",
]


def _filter_jimeng_models(all_models):
    """Filter model list to Jimeng-compatible models."""
    return _filter_models_by_patterns(
        all_models, _JIMENG_MODEL_PATTERNS,
        mode="include", fallback_empty=True,
    )


# ===================================================================
# Node class
# ===================================================================

class ComfyuiZhangyuAPIJimengNode:
    """ComfyUI 即梦格式视频生成节点 — Jimeng API 兼容.

    通过 ``Action`` query param 区分提交 / 查询操作。
    支持文生视频和图生视频（通过 binary_data_base64 传入参考图）。

    Node display name: **ComfyUI-zhangyuapi-即梦格式**
    """

    # Common Jimeng request keys
    REQ_KEYS = [
        "jimeng_t2v",           # 文生视频
        "jimeng_i2v",           # 图生视频
        "jimeng_ti2v",          # 图文生视频
    ]

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "response", "model_list")
    FUNCTION = "generate"
    CATEGORY = "Comfyui-ZhangyuAPI/🎬视频 Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key (API密钥)": (
                    "STRING", {"default": "", "multiline": False}),
                "prompt (提示词)": (
                    "STRING", {"default": "", "multiline": True}),
                "model (模型)": (
                    "STRING", {"default": "jimeng", "multiline": False,
                               "placeholder": "jimeng"}),
                "api_base (接口域名)": (
                    "STRING", {"default": DEFAULT_JIMENG_BASE, "multiline": False}),
                "req_key (请求类型)": (
                    cls.REQ_KEYS, {"default": "jimeng_t2v"}),
                "version (API版本)": (
                    "STRING", {"default": "2024-02-28", "multiline": False}),
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
                "duration (时长秒数)": (
                    "INT", {"default": 5, "min": 2, "max": 60}),
                "fps (帧率)": (
                    "INT", {"default": 24, "min": 1, "max": 120}),
                **{f"image_{i:02d} (参考图{i})": ("IMAGE",) for i in range(1, 5)},
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
             if not k.startswith("image_")},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.md5(key.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_images(kwargs):
        """Collect reference images as base64 strings (no data URL prefix).

        Returns:
            ``list[str]`` — pure base64-encoded PNG data (for
            ``binary_data_base64`` array).
        """
        images = []
        for i in range(1, 5):
            tensor = kwargs.get(f"image_{i:02d} (参考图{i})")
            if tensor is None:
                continue
            tensor = _auto_downscale(tensor)
            png_bytes = tensor_to_png_bytes(tensor)
            images.append(base64.b64encode(png_bytes).decode("utf-8"))
        return images

    @staticmethod
    def _build_submit_payload(model, prompt, req_key, negative_prompt=None,
                               duration=None, fps=None, seed=None,
                               image_b64_list=None):
        """Build JSON body for ``Action=CVSync2AsyncSubmitTask``.

        Only includes documented Jimeng API params (req_key, prompt,
        binary_data_base64) plus *model* for NewAPI backend routing.
        Extra params (seed, duration, fps, negative_prompt) are sent
        as top-level fields — NewAPI may translate or ignore them.

        Returns:
            ``dict`` — JSON-serializable payload.
        """
        payload = {
            "req_key": req_key,
            "prompt": prompt,
        }

        # model is required for NewAPI backend routing
        if model:
            payload["model"] = model

        if image_b64_list:
            payload["binary_data_base64"] = image_b64_list

        # Extra params — accepted by some NewAPI backends
        if seed is not None and seed > 0:
            payload["seed"] = seed
        if duration is not None and duration > 0:
            payload["duration"] = duration
        if fps is not None and fps > 0:
            payload["fps"] = fps
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        return payload

    @staticmethod
    def _build_query_payload(req_key, task_id):
        """Build JSON body for ``Action=CVSync2AsyncGetResult``."""
        return {
            "req_key": req_key,
            "task_id": task_id,
        }

    # ------------------------------------------------------------------
    # API request
    # ------------------------------------------------------------------

    @staticmethod
    def _request_jimeng(api_base, headers, action, version, payload,
                         timeout_seconds):
        """POST to ``/jimeng/`` with *action* and *version* query params."""
        url = f"{api_base}/jimeng/?Action={action}&Version={version}"
        return _get_http_client().post(
            url,
            json=payload,
            headers={**headers, "Content-Type": "application/json"},
            timeout=ZHANGYUAPI_timeout(timeout_seconds),
        )

    # ------------------------------------------------------------------
    # Polling (Jimeng-specific: POST with Action=CVSync2AsyncGetResult)
    # ------------------------------------------------------------------

    @staticmethod
    def _poll_jimeng_task(api_base, headers, req_key, version, task_id,
                           timeout_seconds, retry_times, on_tick=None):
        """Poll Jimeng task via ``POST /jimeng/?Action=CVSync2AsyncGetResult``.

        Returns the completed task data dict.
        """
        url = f"{api_base}/jimeng/?Action=CVSync2AsyncGetResult&Version={version}"
        query_payload = {"req_key": req_key, "task_id": task_id}
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
                    f"即梦任务轮询超时 ({timeout_seconds}s)，"
                    f"已等待 {elapsed:.1f}s"
                )

            try:
                response = _get_http_client().post(
                    url,
                    json=query_payload,
                    headers={**headers, "Content-Type": "application/json"},
                    timeout=ZHANGYUAPI_timeout(remaining),
                )

                if response.status_code == 200:
                    consecutive_errors = 0
                    data = response.json()
                    code = data.get("code", -1)

                    if code != 0:
                        msg = data.get("message", "未知错误")
                        raise RuntimeError(
                            f"即梦查询失败 (code={code}): {msg}"
                        )

                    result_data = data.get("data", {})
                    status = str(result_data.get("status", "")).lower()

                    if status in ("completed", "succeeded", "success", "done"):
                        return data
                    if status in ("failed", "error", "cancelled", "canceled"):
                        err = result_data.get("message") or status
                        raise RuntimeError(
                            f"即梦任务失败 (task_id={task_id}): {err}"
                        )
                    # queued / processing / running — continue

                elif is_retryable_http_status(response.status_code):
                    consecutive_errors += 1
                    if consecutive_errors > retry_times:
                        raise RuntimeError(
                            f"即梦轮询连续 {consecutive_errors} 次 HTTP "
                            f"{response.status_code} 错误，中止"
                        )
                else:
                    raise RuntimeError(
                        f"即梦轮询失败 HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )

            except _RETRYABLE_EXCEPTIONS as exc:
                consecutive_errors += 1
                if consecutive_errors > retry_times:
                    raise RuntimeError(
                        f"即梦轮询连续 {consecutive_errors} 次网络错误，"
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
    def _download_video(url, headers, timeout_seconds, retry_times):
        """Download video bytes from a URL."""
        return _download_bytes_with_retry(
            url, headers, timeout_seconds, retry_times, label="即梦视频",
        )

    # ------------------------------------------------------------------
    # Save video
    # ------------------------------------------------------------------

    @staticmethod
    def _save_video(video_bytes, task_id):
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
            c for c in str(task_id) if c.isalnum() or c in "_-"
        )
        filename = f"zhangyuapi_jimeng_{safe_id}_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(video_bytes)
        print(f"[Comfyui-ZhangyuAPI-Jimeng] saved: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Extract video URL from Jimeng response
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_video_url(data):
        """Extract video URL from Jimeng response ``data`` field.

        The Jimeng API nests the result in ``data``.  Attempts common
        key names: ``video_url``, ``url``, ``result_url``, ``data``.
        """
        inner = data.get("data", {})
        if isinstance(inner, dict):
            for key in ("video_url", "url", "result_url", "data"):
                value = inner.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    return value
        # Fallback: search entire response for an http URL
        raw = json.dumps(data)
        import re
        match = re.search(r'https?://[^\s"\']+\.mp4', raw)
        if match:
            return match.group(0)
        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self, **kwargs):
        """Thin wrapper with ``skip_error`` handling."""
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
                timeout_seconds=kwargs.get("timeout_seconds (超时秒数)", 600),
            )

    def _generate_impl(self, **kwargs):
        """Execute a Jimeng-format video generation request.

        Returns:
            ``(VideoFromFile | str, str, str)``.
        """
        # -- sanitize inputs --------------------------------------------------
        api_key = kwargs.get("api_key (API密钥)", "").strip()
        api_base = normalize_api_base(
            kwargs.get("api_base (接口域名)", DEFAULT_JIMENG_BASE))
        prompt = kwargs.get("prompt (提示词)", "")
        model = kwargs.get("model (模型)", "").strip() or "jimeng"
        req_key = safe_choice(
            kwargs.get("req_key (请求类型)", "jimeng_t2v"),
            self.REQ_KEYS, "jimeng_t2v")
        version = kwargs.get("version (API版本)", "2024-02-28").strip()
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
        duration = safe_int(kwargs.get("duration (时长秒数)", 0), 0, 0, 60)
        fps = safe_int(kwargs.get("fps (帧率)", 0), 0, 0, 120)

        image_b64_list = self._collect_images(kwargs)

        # -- validate ---------------------------------------------------------
        if not api_key:
            emit_runtime_status(unique_id, "error", "API Key 为空",
                                0.0, 0, retry_times, timeout_seconds)
            raise ValueError("API Key 不能为空")

        clean_prompt = normalize_prompt_text(prompt)
        if not clean_prompt:
            raise ValueError("prompt 不能为空")

        if not version:
            raise ValueError("version (API版本) 不能为空")

        # -- fetch model list for output port (best-effort) --------------------
        model_list = []
        try:
            all_models = fetch_available_models_cached(
                api_base, api_key)
            model_list = _filter_jimeng_models(all_models)
        except Exception as exc:
            print(f"[Jimeng] 模型列表获取失败（不影响生成）: {exc}")

        # -- prepare request --------------------------------------------------
        headers = {"Authorization": f"Bearer {api_key}"}
        submit_payload = self._build_submit_payload(
            model, clean_prompt, req_key,
            negative_prompt=negative_prompt or None,
            duration=duration if duration > 0 else None,
            fps=fps if fps > 0 else None,
            seed=seed if seed > 0 else None,
            image_b64_list=image_b64_list or None,
        )

        has_images = bool(image_b64_list)
        mode_label = "图生视频" if has_images else "文生视频"
        print(
            f"[Comfyui-ZhangyuAPI-Jimeng] mode={mode_label}, model={model}, "
            f"req_key={req_key}, version={version}, seed={seed}"
        )
        emit_runtime_status(unique_id, "running", f"开始生成视频 · {mode_label}",
                            0.0, 0, retry_times, timeout_seconds)

        # -- retry loop -------------------------------------------------------
        last_error = None
        for attempt in range(1, retry_times + 1):
            try:
                emit_runtime_status(
                    unique_id, "running",
                    f"即梦{mode_label}请求中 ({attempt}/{retry_times})",
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )

                # Submit task
                response = self._request_jimeng(
                    api_base, headers, "CVSync2AsyncSubmitTask", version,
                    submit_payload, timeout_seconds,
                )

                if response.status_code not in (200, 201, 202):
                    try:
                        err_data = response.json()
                        err_msg = _extract_api_error_message(err_data)
                    except Exception:
                        err_msg = response.text[:500]
                    last_error = f"即梦 API 错误 {response.status_code}: {err_msg}"
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
                code = data.get("code", -1)
                if code != 0:
                    msg = data.get("message", "未知错误")
                    raise RuntimeError(f"即梦提交失败 (code={code}): {msg}")

                # -- Extract task ID from response ----------------------------
                result_data = data.get("data", {})
                task_id = (
                    result_data.get("task_id")
                    or result_data.get("id")
                    or data.get("task_id")
                )
                if not task_id:
                    # May be a sync response — check for direct video URL
                    video_url = self._extract_video_url(data)
                    if video_url:
                        video_bytes = self._download_video(
                            video_url, headers, timeout_seconds, retry_times,
                        )
                        filepath = self._save_video(video_bytes, "direct")
                        elapsed = time.time() - start_ts
                        response_info = {
                            "status": "success",
                            "format": "Jimeng",
                            "mode": mode_label,
                            "sync": True,
                            "api_base": denormalize_api_base(api_base),
                            "model": model,
                            "req_key": req_key,
                            "version": version,
                            "seed": seed,
                            "video_url": video_url,
                            "filepath": filepath,
                            "input_images": len(image_b64_list),
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
                    raise RuntimeError(
                        f"即梦 API 未返回 task_id: "
                        f"{json.dumps(_sanitize_api_response(data), ensure_ascii=False)[:500]}"
                    )

                # -- Poll until completion ------------------------------------
                remaining = timeout_seconds - int(time.time() - start_ts)
                emit_runtime_status(
                    unique_id, "running",
                    f"即梦任务已提交 (id={task_id})，轮询中",
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )

                def _on_poll_tick(elapsed, poll_elapsed, interval):
                    emit_runtime_status(
                        unique_id, "running",
                        f"轮询即梦视频中 · 间隔{interval:.0f}s · "
                        f"已等待{poll_elapsed:.0f}s",
                        elapsed,
                        attempt, retry_times, timeout_seconds,
                    )

                polled_data = self._poll_jimeng_task(
                    api_base, headers, req_key, version, task_id,
                    remaining, retry_times,
                    on_tick=_on_poll_tick,
                )

                # -- Extract video URL -----------------------------------------
                video_url = self._extract_video_url(polled_data)
                if not video_url:
                    raise RuntimeError(
                        "即梦任务完成但未能从响应中提取视频 URL"
                    )

                # -- Download video --------------------------------------------
                emit_runtime_status(
                    unique_id, "running", "下载视频中",
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )

                video_bytes = self._download_video(
                    video_url, headers, timeout_seconds, retry_times,
                )
                filepath = self._save_video(video_bytes, task_id)

                # -- Build return ----------------------------------------------
                elapsed = time.time() - start_ts
                response_info = {
                    "status": "success",
                    "format": "Jimeng",
                    "mode": mode_label,
                    "api_base": denormalize_api_base(api_base),
                    "model": model,
                    "req_key": req_key,
                    "version": version,
                    "seed": seed,
                    "task_id": task_id,
                    "video_url": video_url,
                    "filepath": filepath,
                    "duration": duration if duration > 0 else None,
                    "fps": fps if fps > 0 else None,
                    "input_images": len(image_b64_list),
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
            f"Comfyui-ZhangyuAPI-Jimeng 连续 {retry_times} 次失败，"
            f"最后错误: {last_error}"
        )


# ===================================================================
# ComfyUI node registration
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ComfyuiZhangyuAPIJimengNode": ComfyuiZhangyuAPIJimengNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiZhangyuAPIJimengNode": "ComfyUI-zhangyuapi-即梦格式",
}
