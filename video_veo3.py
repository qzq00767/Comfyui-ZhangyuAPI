#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — Google Veo3 视频生成节点.

适配 Google Veo3 API 端点：
- 提交任务: ``POST /v2/videos/generations``
- 查询状态: ``GET /v2/videos/generations/{task_id}``

Auth: ``Authorization: Bearer <api_key>``
"""

import base64
import json
import os
import time
import uuid

from io import BytesIO

import numpy as np
from PIL import Image

from .zhangyu_gpt_img2 import (
    _get_http_client,
    ZHANGYUAPI_timeout,
    ZHANGYUAPI_get,
    ZHANGYUAPI_post,
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
    _on_retryable_error,
    emit_runtime_status,
    _sanitize_api_response,
    _skip_error_return,
    _extract_api_error_message,
    _filter_models_by_patterns,
    fetch_available_models_cached,
    tensor_to_data_url,
    _auto_downscale,
    _log,
    DEFAULT_API_BASE_URL,
    _safe_extract_error_from_response,
)

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

DEFAULT_VEO3_BASE = DEFAULT_API_BASE_URL
DEFAULT_VIDEO_TIMEOUT = 900
DEFAULT_MIN_VIDEO_TIMEOUT = 120
DEFAULT_MAX_VIDEO_TIMEOUT = 3600
DEFAULT_VIDEO_RETRY_TIMES = 2

VEO3_MODELS = [
    "veo3", "veo3-fast", "veo3-pro",
    "veo3.1", "veo3.1-fast", "veo3.1-pro",
    "veo3.1-4k", "veo3.1-pro-4k",
]

VEO3_ASPECT_RATIOS = ["16:9", "9:16"]


def _save_video(video_bytes, prefix="zhangyuapi_veo3"):
    """Save video bytes to ComfyUI output directory."""
    try:
        import folder_paths
        output_dir = folder_paths.get_output_directory()
    except Exception:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
        output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{prefix}_{uuid.uuid4().hex[:8]}.mp4")
    with open(filepath, "wb") as f:
        f.write(video_bytes)
    _log("info", f"视频已保存: {filepath}")
    return filepath


def _image_tensor_to_base64_data_url(image_tensor):
    """Convert ComfyUI IMAGE tensor to base64 data URL."""
    if image_tensor is None:
        return None
    try:
        i = 255.0 * image_tensor.cpu().numpy()[0]
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Resize if too large
        max_dim = 1536
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.Resampling.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=75, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        _log("warn", f"图片转base64失败: {e}")
        return None


# ===================================================================
# Node: Google Veo3 视频生成
# ===================================================================

class ComfyuiZhangyuAPIVeo3Node:
    """ComfyUI Google Veo3 视频生成节点."""

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "generate"
    CATEGORY = "Comfyui-ZhangyuAPI/🎬视频 Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key (API密钥)": ("STRING", {"default": ""}),
                "api_base (接口地址)": ("STRING", {"default": DEFAULT_VEO3_BASE, "multiline": False}),
                "prompt (提示词)": ("STRING", {"default": "", "multiline": True}),
                "model (模型)": (VEO3_MODELS, {"default": "veo3"}),
                "aspect_ratio (宽高比)": (VEO3_ASPECT_RATIOS, {"default": "16:9"}),
                "enhance_prompt (增强提示词)": ("BOOLEAN", {"default": False}),
                "timeout_seconds (超时秒数)": ("INT", {"default": 600, "min": 30, "max": 900}),
                "retry_times (重试次数)": ("INT", {"default": 10, "min": 1, "max": 30}),
            },
            "optional": {
                "image_01 (参考图1)": ("IMAGE",),
                "image_02 (参考图2)": ("IMAGE",),
                "image_03 (参考图3)": ("IMAGE",),
                "seed (种子)": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "skip_error (跳过错误)": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def generate(self, api_key, api_base, prompt, model, aspect_ratio, enhance_prompt,
                 timeout_seconds, retry_times,
                 image_01=None, image_02=None, image_03=None, seed=0,
                 skip_error=False, unique_id=None):
        try:
            return self._generate_impl(
                api_key, api_base, prompt, model, aspect_ratio, enhance_prompt,
                timeout_seconds, retry_times,
                [image_01, image_02, image_03], seed, unique_id,
            )
        except Exception as exc:
            if not skip_error:
                raise
            return _skip_error_return(str(exc), self.RETURN_TYPES, unique_id, retry_times, timeout_seconds)

    def _generate_impl(self, api_key, api_base, prompt, model, aspect_ratio, enhance_prompt,
                       timeout_seconds, retry_times, images, seed, unique_id):
        base = normalize_api_base(api_base or DEFAULT_VEO3_BASE)
        headers = {"Authorization": f"Bearer {api_key.strip()}"}

        # Build payload
        payload = {
            "prompt": prompt,
            "model": model,
            "enhance_prompt": enhance_prompt,
            "aspect_ratio": aspect_ratio,
        }

        if seed > 0:
            payload["seed"] = seed

        # Collect images
        image_urls = []
        for img in images:
            if img is not None:
                data_url = _image_tensor_to_base64_data_url(img)
                if data_url:
                    image_urls.append(data_url)

        if image_urls:
            payload["images"] = image_urls

        _log("info", f"[Veo3] 提交任务: model={model}, images={len(image_urls)}")

        if unique_id:
            emit_runtime_status(unique_id, "running", "Veo3 任务提交中", 0, 0, retry_times, timeout_seconds)

        # Submit
        url = f"{base}/v2/videos/generations"
        response = ZHANGYUAPI_post(url, timeout_seconds, headers=headers, json=payload)

        if response.status_code not in (200, 201, 202):
            raise RuntimeError(f"API 错误 {response.status_code}: {_safe_extract_error_from_response(response)}")

        data = response.json()
        task_id = data.get("task_id")
        if not task_id:
            raise RuntimeError(f"未返回 task_id: {json.dumps(data, ensure_ascii=False)[:500]}")

        _log("info", f"[Veo3] 任务已提交: task_id={task_id}")

        # Poll
        poll_url = f"{base}/v2/videos/generations/{task_id}"
        start_ts = time.time()
        poll_start = time.time()
        consecutive_errors = 0
        stages = ((30, 3.0), (120, 5.0), (float("inf"), 10.0))

        while True:
            elapsed = time.time() - start_ts
            poll_elapsed = time.time() - poll_start
            remaining = timeout_seconds - int(poll_elapsed + 0.999)

            if remaining <= 0:
                raise RuntimeError(f"Veo3 任务轮询超时 ({timeout_seconds}s)")

            try:
                status_resp = ZHANGYUAPI_get(poll_url, remaining, headers=headers)

                if status_resp.status_code == 200:
                    consecutive_errors = 0
                    status_data = status_resp.json()
                    status = str(status_data.get("status", "")).upper()

                    if status == "SUCCESS":
                        output = status_data.get("data", {}).get("output")
                        if output:
                            # Download video
                            video_bytes = _download_bytes_with_retry(output, headers, timeout_seconds, retry_times, label="Veo3视频")
                            filepath = _save_video(video_bytes, "zhangyuapi_veo3")

                            response_info = json.dumps({
                                "status": "success", "model": model,
                                "task_id": task_id, "video_url": output,
                            }, ensure_ascii=False, indent=2)

                            video_obj = VideoFromFile(filepath) if VideoFromFile else filepath
                            return (video_obj, output, response_info)
                        else:
                            raise RuntimeError("Veo3 任务成功但未返回视频 URL")

                    elif status == "FAILURE":
                        fail_reason = status_data.get("fail_reason", "未知错误")
                        raise RuntimeError(f"Veo3 任务失败: {fail_reason}")

                elif is_retryable_http_status(status_resp.status_code):
                    consecutive_errors += 1
                    if consecutive_errors > retry_times:
                        raise RuntimeError(f"Veo3 轮询连续 {consecutive_errors} 次 HTTP 错误")
                else:
                    raise RuntimeError(f"Veo3 轮询失败 HTTP {status_resp.status_code}")

            except _RETRYABLE_EXCEPTIONS as exc:
                consecutive_errors += 1
                _on_retryable_error(exc)
                if consecutive_errors > retry_times:
                    raise RuntimeError(f"Veo3 轮询连续 {consecutive_errors} 次网络错误: {exc}")

            if consecutive_errors > 0:
                time.sleep(_jittered_backoff_seconds(consecutive_errors))
            else:
                interval = 3.0
                for threshold, iv in stages:
                    if poll_elapsed < threshold:
                        interval = iv
                        break
                if unique_id:
                    emit_runtime_status(unique_id, "running", f"Veo3 轮询中 · 已等待{poll_elapsed:.0f}s", elapsed, 1, retry_times, timeout_seconds)
                time.sleep(interval)


# ===================================================================
# ComfyUI node registration
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ComfyuiZhangyuAPIVeo3Node": ComfyuiZhangyuAPIVeo3Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiZhangyuAPIVeo3Node": "ComfyUI-zhangyuapi-Veo3视频",
}
