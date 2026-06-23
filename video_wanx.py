#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — 通义万相 (WanX) 视频生成节点.

适配阿里云 DashScope 视频生成端点：
- 提交任务: ``POST /api/v1/services/aigc/video-generation/video-synthesis``
- 查询状态: ``GET /api/v1/tasks/{task_id}``

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

DEFAULT_WANX_BASE = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_VIDEO_TIMEOUT = 900
DEFAULT_MIN_VIDEO_TIMEOUT = 120
DEFAULT_MAX_VIDEO_TIMEOUT = 3600
DEFAULT_VIDEO_RETRY_TIMES = 2

WANX_RESOLUTIONS = ["1080P", "720P"]
WANX_DURATIONS = [5, 10, 15]


def _save_video(video_bytes, prefix="zhangyuapi_wanx"):
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


def _image_tensor_to_data_url(image_tensor):
    """Convert ComfyUI IMAGE tensor to base64 data URL with compression."""
    if image_tensor is None:
        return None
    try:
        if hasattr(image_tensor, 'shape'):
            if len(image_tensor.shape) == 4:
                image_tensor = image_tensor[0]
            if image_tensor.shape[0] == 3:
                image_tensor = image_tensor.permute(1, 2, 0)

        image_np = image_tensor.cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype('uint8')

        img = Image.fromarray(image_np)

        # Resize if too large
        max_dim = 1536
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.Resampling.LANCZOS)

        # Try JPEG first (smaller)
        buf = BytesIO()
        if img.mode in ('RGBA', 'LA'):
            jpeg_img = Image.new('RGB', img.size, 'white')
            if img.mode == 'RGBA':
                jpeg_img.paste(img, mask=img.split()[-1])
            else:
                jpeg_img.paste(img)
            jpeg_img.save(buf, format="JPEG", quality=75, optimize=True)
        else:
            img.save(buf, format="JPEG", quality=75, optimize=True)

        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        _log("warn", f"图片转data URL失败: {e}")
        return None


# ===================================================================
# Node: 通义万相视频生成
# ===================================================================

class ComfyuiZhangyuAPIWanxNode:
    """ComfyUI 通义万相 (WanX) 视频生成节点."""

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "generate"
    CATEGORY = "Comfyui-ZhangyuAPI/🎬视频 Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key (API密钥)": ("STRING", {"default": ""}),
                "prompt (提示词)": ("STRING", {"default": "", "multiline": True}),
                "resolution (分辨率)": (WANX_RESOLUTIONS, {"default": "1080P"}),
                "duration (时长秒数)": (WANX_DURATIONS, {"default": 5}),
                "timeout_seconds (超时秒数)": ("INT", {"default": 600, "min": 30, "max": 900}),
                "retry_times (重试次数)": ("INT", {"default": 10, "min": 1, "max": 30}),
            },
            "optional": {
                "image (参考图)": ("IMAGE",),
                "audio_url (音频URL)": ("STRING", {"default": ""}),
                "prompt_extend (提示词扩展)": ("BOOLEAN", {"default": True}),
                "shot_type (镜头类型)": (["single", "multi"], {"default": "multi"}),
                "audio_enabled (启用音频)": ("BOOLEAN", {"default": True}),
                "skip_error (跳过错误)": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def generate(self, api_key, prompt, resolution, duration, timeout_seconds, retry_times,
                 image=None, audio_url="", prompt_extend=True, shot_type="multi",
                 audio_enabled=True, skip_error=False, unique_id=None):
        try:
            return self._generate_impl(
                api_key, prompt, resolution, duration, timeout_seconds, retry_times,
                image, audio_url, prompt_extend, shot_type, audio_enabled, unique_id,
            )
        except Exception as exc:
            if not skip_error:
                raise
            return _skip_error_return(str(exc), self.RETURN_TYPES, unique_id, retry_times, timeout_seconds)

    def _generate_impl(self, api_key, prompt, resolution, duration, timeout_seconds, retry_times,
                       image, audio_url, prompt_extend, shot_type, audio_enabled, unique_id):
        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空")
        if len(prompt) > 1500:
            raise ValueError(f"提示词过长 ({len(prompt)} 字符)，最多 1500 字符")

        base = DEFAULT_WANX_BASE
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

        # Build request body
        request_body = {
            "model": "wan2.6-i2v",
            "input": {"prompt": prompt},
            "parameters": {
                "prompt_extend": prompt_extend,
                "resolution": resolution,
                "duration": duration,
                "shot_type": shot_type,
                "audio": audio_enabled,
            }
        }

        # Add image
        if image is not None:
            img_url = _image_tensor_to_data_url(image)
            if img_url:
                request_body["input"]["img_url"] = img_url
            else:
                _log("warn", "[WanX] 图片转换失败，将使用文生视频模式")

        # Add audio
        if audio_url and audio_url.strip():
            request_body["input"]["audio_url"] = audio_url
            request_body["parameters"]["audio"] = True

        _log("info", f"[WanX] 提交任务: resolution={resolution}, duration={duration}s")

        if unique_id:
            emit_runtime_status(unique_id, "running", "通义万相任务提交中", 0, 0, retry_times, timeout_seconds)

        # Submit task
        url = f"{base}/services/aigc/video-generation/video-synthesis"
        response = ZHANGYUAPI_post(url, timeout_seconds, headers=headers, json=request_body)

        if response.status_code not in (200, 201, 202):
            raise RuntimeError(f"API 错误 {response.status_code}: {_safe_extract_error_from_response(response)}")

        data = response.json()
        task_id = data.get("output", {}).get("task_id") or data.get("task_id")
        if not task_id:
            raise RuntimeError(f"未返回 task_id: {json.dumps(data, ensure_ascii=False)[:500]}")

        _log("info", f"[WanX] 任务已提交: task_id={task_id}")

        # Poll
        poll_url = f"{base}/tasks/{task_id}"
        poll_headers = {"Authorization": f"Bearer {api_key.strip()}"}
        start_ts = time.time()
        poll_start = time.time()
        consecutive_errors = 0
        stages = ((30, 3.0), (120, 5.0), (float("inf"), 10.0))

        while True:
            elapsed = time.time() - start_ts
            poll_elapsed = time.time() - poll_start
            remaining = timeout_seconds - int(poll_elapsed + 0.999)

            if remaining <= 0:
                raise RuntimeError(f"WanX 任务轮询超时 ({timeout_seconds}s)")

            try:
                status_resp = ZHANGYUAPI_get(poll_url, remaining, headers=poll_headers)

                if status_resp.status_code == 200:
                    consecutive_errors = 0
                    status_data = status_resp.json()
                    output = status_data.get("output", {})
                    status = str(output.get("task_status", "")).lower()

                    if status in ("succeeded", "success", "done"):
                        video_url = output.get("video_url")
                        if not video_url:
                            # Try to get from results
                            results = output.get("results", [])
                            if results:
                                video_url = results[0].get("url")
                        if video_url:
                            # Download video
                            video_bytes = _download_bytes_with_retry(video_url, poll_headers, timeout_seconds, retry_times, label="WanX视频")
                            filepath = _save_video(video_bytes, "zhangyuapi_wanx")

                            response_info = json.dumps({
                                "status": "success", "task_id": task_id,
                                "video_url": video_url, "resolution": resolution,
                                "duration": duration,
                            }, ensure_ascii=False, indent=2)

                            video_obj = VideoFromFile(filepath) if VideoFromFile else filepath
                            return (video_obj, video_url, response_info)
                        else:
                            raise RuntimeError("WanX 任务成功但未返回视频 URL")

                    elif status in ("failed", "error"):
                        fail_msg = output.get("message", "未知错误")
                        raise RuntimeError(f"WanX 任务失败: {fail_msg}")

                elif is_retryable_http_status(status_resp.status_code):
                    consecutive_errors += 1
                    if consecutive_errors > retry_times:
                        raise RuntimeError(f"WanX 轮询连续 {consecutive_errors} 次 HTTP 错误")
                else:
                    raise RuntimeError(f"WanX 轮询失败 HTTP {status_resp.status_code}")

            except _RETRYABLE_EXCEPTIONS as exc:
                consecutive_errors += 1
                _on_retryable_error(exc)
                if consecutive_errors > retry_times:
                    raise RuntimeError(f"WanX 轮询连续 {consecutive_errors} 次网络错误: {exc}")

            if consecutive_errors > 0:
                time.sleep(_jittered_backoff_seconds(consecutive_errors))
            else:
                interval = 3.0
                for threshold, iv in stages:
                    if poll_elapsed < threshold:
                        interval = iv
                        break
                if unique_id:
                    emit_runtime_status(unique_id, "running", f"通义万相轮询中 · 已等待{poll_elapsed:.0f}s", elapsed, 1, retry_times, timeout_seconds)
                time.sleep(interval)


# ===================================================================
# ComfyUI node registration
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ComfyuiZhangyuAPIWanxNode": ComfyuiZhangyuAPIWanxNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiZhangyuAPIWanxNode": "ComfyUI-zhangyuapi-通义万相视频",
}
