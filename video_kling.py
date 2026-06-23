#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — 可灵格式视频生成节点（多功能合一）.

支持模式：
- 文生视频: POST /kling/v1/videos/text2video
- 图生视频: POST /kling/v1/videos/image2video
- 多图转视频: POST /kling/v1/videos/multi-image2video
- 视频延长: POST /kling/v1/videos/video-extend
- 唇形同步: POST /kling/v1/videos/lip-sync

Auth: Authorization: Bearer <api_key>
"""

import base64
import hashlib
import json
import os
import random
import time
import uuid

import numpy as np
from PIL import Image
from io import BytesIO

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

DEFAULT_KLING_BASE = DEFAULT_API_BASE_URL
DEFAULT_VIDEO_TIMEOUT = 900
DEFAULT_MIN_VIDEO_TIMEOUT = 120
DEFAULT_MAX_VIDEO_TIMEOUT = 3600
DEFAULT_VIDEO_RETRY_TIMES = 2

KLING_MODELS = ["kling-v2-1-master", "kling-v2-master", "kling-v1-6", "kling-v1-5", "kling-v1"]
KLING_MODES = ["std", "pro"]
ASPECT_RATIOS = ["1:1", "16:9", "9:16"]
DURATIONS = ["5", "10"]

CAMERA_TYPES = [
    "none", "horizontal", "vertical", "zoom",
    "vertical_shake", "horizontal_shake", "rotate",
    "master_down_zoom", "master_zoom_up",
    "master_right_rotate_zoom", "master_left_rotate_zoom",
]

KLING_ZH_VOICES = [
    ("阳光少年", "genshin_vindi2"), ("懂事小弟", "zhinen_xuesheng"),
    ("运动少年", "tiyuxi_xuedi"), ("青春少女", "ai_shatang"),
    ("温柔小妹", "genshin_klee2"), ("元气少女", "genshin_kirara"),
    ("阳光男生", "ai_kaiya"), ("幽默小哥", "tiexin_nanyou"),
    ("甜美邻家", "girlfriend_1_speech02"), ("温柔姐姐", "chat1_female_new-3"),
]

KLING_EN_VOICES = [
    ("Sunny", "genshin_vindi2"), ("Sage", "zhinen_xuesheng"),
    ("Blossom", "ai_shatang"), ("Peppy", "genshin_klee2"),
    ("Shine", "ai_kaiya"), ("Anchor", "oversea_male1"),
    ("Tender", "chat1_female_new-3"),
]


# ===================================================================
# Helpers
# ===================================================================

def _get_camera_json(camera, camera_value=0):
    camera_map = {
        "none": {"type": "empty", "horizontal": 0, "vertical": 0, "zoom": 0, "tilt": 0, "pan": 0, "roll": 0},
        "horizontal": {"type": "horizontal", "horizontal": camera_value, "vertical": 0, "zoom": 0, "tilt": 0, "pan": 0, "roll": 0},
        "vertical": {"type": "vertical", "horizontal": 0, "vertical": camera_value, "zoom": 0, "tilt": 0, "pan": 0, "roll": 0},
        "zoom": {"type": "zoom", "horizontal": 0, "vertical": 0, "zoom": camera_value, "tilt": 0, "pan": 0, "roll": 0},
        "vertical_shake": {"type": "vertical_shake", "horizontal": 0, "vertical": camera_value, "zoom": 0.5, "tilt": 0, "pan": 0, "roll": 0},
        "horizontal_shake": {"type": "horizontal_shake", "horizontal": camera_value, "vertical": 0, "zoom": 0.5, "tilt": 0, "pan": 0, "roll": 0},
        "rotate": {"type": "rotate", "horizontal": 0, "vertical": 0, "zoom": 0, "tilt": 0, "pan": camera_value, "roll": 0},
        "master_down_zoom": {"type": "zoom", "horizontal": 0, "vertical": 0, "zoom": camera_value, "tilt": camera_value, "pan": 0, "roll": 0},
        "master_zoom_up": {"type": "zoom", "horizontal": 0.2, "vertical": 0, "zoom": camera_value, "tilt": 0, "pan": 0, "roll": 0},
        "master_right_rotate_zoom": {"type": "rotate", "horizontal": 0, "vertical": 0, "zoom": camera_value, "tilt": 0, "pan": 0, "roll": camera_value},
        "master_left_rotate_zoom": {"type": "rotate", "horizontal": 0, "vertical": 0, "zoom": camera_value, "tilt": 0, "pan": camera_value, "roll": 0},
    }
    return json.dumps(camera_map.get(camera, camera_map["none"]))


def _image_to_base64(image_tensor):
    if image_tensor is None:
        return None
    try:
        i = 255.0 * image_tensor.cpu().numpy()[0]
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        _log("warn", f"图片转base64失败: {e}")
        return None


def _poll_task(base, headers, task_id, endpoint, timeout, retries, on_tick=None):
    url = f"{base}/kling/v1/videos/{endpoint}/{task_id}"
    start = time.time()
    poll_start = time.time()
    errors = 0

    while True:
        elapsed = time.time() - start
        poll_elapsed = time.time() - poll_start
        remaining = timeout - int(poll_elapsed + 0.999)
        if remaining <= 0:
            raise RuntimeError(f"轮询超时 ({timeout}s)")

        try:
            resp = ZHANGYUAPI_get(url, remaining, headers={**headers, "Content-Type": "application/json"})
            if resp.status_code == 200:
                errors = 0
                data = resp.json().get("data", resp.json())
                status = str(data.get("task_status", data.get("status", ""))).lower()
                if status in ("succeed", "success", "completed", "done"):
                    return data
                if status in ("failed", "error"):
                    raise RuntimeError(f"任务失败: {data.get('task_status_msg', data.get('error', ''))}")
            elif is_retryable_http_status(resp.status_code):
                errors += 1
                if errors > retries:
                    raise RuntimeError(f"连续 {errors} 次 HTTP 错误")
            else:
                raise RuntimeError(f"轮询失败 HTTP {resp.status_code}: {_safe_extract_error_from_response(resp)}")
        except _RETRYABLE_EXCEPTIONS as exc:
            errors += 1
            _on_retryable_error(exc)
            if errors > retries:
                raise RuntimeError(f"连续 {errors} 次网络错误: {exc}")

        if errors > 0:
            time.sleep(_jittered_backoff_seconds(errors))
        else:
            interval = 2.0 if poll_elapsed < 15 else 5.0 if poll_elapsed < 60 else 10.0
            if on_tick:
                on_tick(elapsed, poll_elapsed, interval)
            time.sleep(interval)


def _download_video(video_url, headers, timeout, retries):
    return _download_bytes_with_retry(video_url, headers, timeout, retries, label="Kling视频")


def _save_video(video_bytes, prefix="zhangyuapi_kling"):
    try:
        import folder_paths
        output_dir = folder_paths.get_output_directory()
    except:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{prefix}_{uuid.uuid4().hex[:8]}.mp4")
    with open(filepath, "wb") as f:
        f.write(video_bytes)
    return filepath


def _extract_video_url(task_data):
    if "task_result" in task_data and "videos" in task_data["task_result"]:
        return task_data["task_result"]["videos"][0].get("url")
    return task_data.get("url")


# ===================================================================
# Node: 可灵多功能视频节点
# ===================================================================

class ComfyuiZhangyuAPIKlingNode:
    """ComfyUI 可灵多功能视频生成节点."""

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "response", "model_list")
    FUNCTION = "generate"
    CATEGORY = "Comfyui-ZhangyuAPI/🎬视频 Video"

    @classmethod
    def INPUT_TYPES(cls):
        zh_voices = [n for n, _ in KLING_ZH_VOICES]
        en_voices = [n for n, _ in KLING_EN_VOICES]
        return {
            "required": {
                "api_key (API密钥)": ("STRING", {"default": ""}),
                "mode (功能模式)": (
                    ["文生视频", "图生视频", "多图转视频", "视频延长", "唇形同步"],
                    {"default": "文生视频"}),
                "prompt (提示词)": ("STRING", {"default": "", "multiline": True}),
                "model (模型)": (KLING_MODELS, {"default": "kling-v1-6"}),
                "aspect_ratio (宽高比)": (ASPECT_RATIOS, {"default": "16:9"}),
                "duration (时长)": (DURATIONS, {"default": "5"}),
                "kling_mode (std/pro)": (KLING_MODES, {"default": "std"}),
                "seed (种子)": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "timeout_seconds (超时秒数)": ("INT", {"default": 600, "min": 30, "max": 900}),
                "retry_times (重试次数)": ("INT", {"default": 10, "min": 1, "max": 30}),
            },
            "optional": {
                "negative_prompt (反向提示词)": ("STRING", {"default": "", "multiline": True}),
                "imagination (想象力)": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "camera (镜头运动)": (CAMERA_TYPES, {"default": "none"}),
                "camera_value (镜头强度)": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.1}),
                "image_01 (参考图)": ("IMAGE",),
                "image_02 (参考图2)": ("IMAGE",),
                "image_03 (参考图3)": ("IMAGE",),
                "image_04 (参考图4)": ("IMAGE",),
                "video_id (视频ID，延长/唇形同步用)": ("STRING", {"default": "", "forceInput": True}),
                "task_id (任务ID，唇形同步用)": ("STRING", {"default": "", "forceInput": True}),
                "lip_text (唇形同步文本)": ("STRING", {"default": "", "multiline": True}),
                "voice_language (语言)": (["zh", "en"], {"default": "zh"}),
                "zh_voice (中文音色)": (zh_voices, {"default": zh_voices[0]}),
                "en_voice (英文音色)": (en_voices, {"default": en_voices[0]}),
                "voice_speed (语速)": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 2.0, "step": 0.1}),
                "skip_error (跳过错误)": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def generate(self, api_key, mode, prompt, model, aspect_ratio, duration, kling_mode,
                 seed, timeout_seconds, retry_times,
                 negative_prompt="", imagination=0.5, camera="none", camera_value=0,
                 image_01=None, image_02=None, image_03=None, image_04=None,
                 video_id="", task_id="", lip_text="", voice_language="zh",
                 zh_voice="", en_voice="", voice_speed=1.0,
                 skip_error=False, unique_id=None):
        try:
            base = normalize_api_base(DEFAULT_KLING_BASE)
            headers = {"Authorization": f"Bearer {api_key.strip()}"}

            if mode == "文生视频":
                return self._text2video(base, headers, prompt, model, aspect_ratio, duration,
                                        kling_mode, seed, timeout_seconds, retry_times,
                                        negative_prompt, imagination, camera, camera_value, unique_id)
            elif mode == "图生视频":
                return self._image2video(base, headers, prompt, model, aspect_ratio, duration,
                                         kling_mode, seed, timeout_seconds, retry_times,
                                         negative_prompt, imagination, camera, camera_value,
                                         image_01, unique_id)
            elif mode == "多图转视频":
                return self._multi_image(base, headers, prompt, model, aspect_ratio, duration,
                                         kling_mode, seed, timeout_seconds, retry_times,
                                         negative_prompt, [image_01, image_02, image_03, image_04], unique_id)
            elif mode == "视频延长":
                return self._extend(base, headers, video_id, prompt, timeout_seconds, retry_times, unique_id)
            elif mode == "唇形同步":
                return self._lip_sync(base, headers, video_id, task_id, lip_text, voice_language,
                                      zh_voice, en_voice, voice_speed, timeout_seconds, retry_times, unique_id)
        except Exception as exc:
            if not skip_error:
                raise
            return _skip_error_return(str(exc), self.RETURN_TYPES, unique_id, retry_times, timeout_seconds)

    def _text2video(self, base, headers, prompt, model, aspect_ratio, duration, kling_mode,
                    seed, timeout, retries, negative_prompt, imagination, camera, camera_value, uid):
        camera_json = _get_camera_json(camera, camera_value) if model == "kling-v1" else _get_camera_json("none", 0)
        payload = {
            "prompt": prompt, "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio, "duration": duration,
            "model_name": model, "imagination": imagination,
            "num_videos": 1, "camera_json": camera_json, "seed": seed,
        }
        if model != "kling-v2-master":
            payload["mode"] = kling_mode

        if uid:
            emit_runtime_status(uid, "running", "可灵文生视频请求中", 0, 0, retries, timeout)

        resp = ZHANGYUAPI_post(f"{base}/kling/v1/videos/text2video", timeout, headers=headers, json=payload)
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"API 错误 {resp.status_code}: {_safe_extract_error_from_response(resp)}")

        task_id = resp.json().get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError("未返回 task_id")

        task_data = _poll_task(base, headers, task_id, "text2video", timeout - 10, retries)
        video_url = _extract_video_url(task_data)
        if not video_url:
            raise RuntimeError("未返回视频 URL")

        video_bytes = _download_video(video_url, headers, timeout, retries)
        filepath = _save_video(video_bytes)
        video_obj = VideoFromFile(filepath) if VideoFromFile else filepath
        return (video_obj, json.dumps({"status": "success", "task_id": task_id}, ensure_ascii=False), "[]")

    def _image2video(self, base, headers, prompt, model, aspect_ratio, duration, kling_mode,
                     seed, timeout, retries, negative_prompt, imagination, camera, camera_value,
                     image_01, uid):
        camera_json = _get_camera_json(camera, camera_value) if model in ("kling-v1", "kling-v1-5", "kling-v1-6") else _get_camera_json("none", 0)
        img_b64 = _image_to_base64(image_01)

        payload = {
            "prompt": prompt, "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio, "duration": duration,
            "model_name": model, "imagination": imagination,
            "num_videos": 1, "camera_json": camera_json, "seed": seed,
        }
        if model != "kling-v2-master":
            payload["mode"] = kling_mode
        if img_b64:
            payload["image"] = img_b64

        if uid:
            emit_runtime_status(uid, "running", "可灵图生视频请求中", 0, 0, retries, timeout)

        resp = ZHANGYUAPI_post(f"{base}/kling/v1/videos/image2video", timeout, headers=headers, json=payload)
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"API 错误 {resp.status_code}: {_safe_extract_error_from_response(resp)}")

        task_id = resp.json().get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError("未返回 task_id")

        task_data = _poll_task(base, headers, task_id, "image2video", timeout - 10, retries)
        video_url = _extract_video_url(task_data)
        if not video_url:
            raise RuntimeError("未返回视频 URL")

        video_bytes = _download_video(video_url, headers, timeout, retries)
        filepath = _save_video(video_bytes)
        video_obj = VideoFromFile(filepath) if VideoFromFile else filepath
        return (video_obj, json.dumps({"status": "success", "task_id": task_id}, ensure_ascii=False), "[]")

    def _multi_image(self, base, headers, prompt, model, aspect_ratio, duration, kling_mode,
                     seed, timeout, retries, negative_prompt, images, uid):
        image_list = []
        for img in images:
            if img is not None:
                b64 = _image_to_base64(img)
                if b64:
                    image_list.append({"image": b64})
        if not image_list:
            raise ValueError("至少需要一张参考图")

        payload = {
            "model_name": model, "image_list": image_list,
            "prompt": prompt, "negative_prompt": negative_prompt,
            "mode": kling_mode, "duration": duration, "aspect_ratio": aspect_ratio,
        }
        if seed > 0:
            payload["seed"] = seed

        if uid:
            emit_runtime_status(uid, "running", f"可灵多图转视频 ({len(image_list)}张)", 0, 0, retries, timeout)

        resp = ZHANGYUAPI_post(f"{base}/kling/v1/videos/multi-image2video", timeout, headers=headers, json=payload)
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"API 错误 {resp.status_code}: {_safe_extract_error_from_response(resp)}")

        task_id = resp.json().get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError("未返回 task_id")

        task_data = _poll_task(base, headers, task_id, "multi-image2video", timeout - 10, retries)
        video_url = _extract_video_url(task_data)
        if not video_url:
            raise RuntimeError("未返回视频 URL")

        video_bytes = _download_video(video_url, headers, timeout, retries)
        filepath = _save_video(video_bytes, "zhangyuapi_kling_multi")
        video_obj = VideoFromFile(filepath) if VideoFromFile else filepath
        return (video_obj, json.dumps({"status": "success", "task_id": task_id}, ensure_ascii=False), "[]")

    def _extend(self, base, headers, video_id, prompt, timeout, retries, uid):
        if not video_id:
            raise ValueError("视频延长模式需要提供 video_id")

        payload = {"video_id": video_id, "prompt": prompt}

        if uid:
            emit_runtime_status(uid, "running", "可灵视频延长中", 0, 0, retries, timeout)

        resp = ZHANGYUAPI_post(f"{base}/kling/v1/videos/video-extend", timeout, headers=headers, json=payload)
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"API 错误 {resp.status_code}: {_safe_extract_error_from_response(resp)}")

        task_id = resp.json().get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError("未返回 task_id")

        task_data = _poll_task(base, headers, task_id, "video-extend", timeout - 10, retries)
        video_url = _extract_video_url(task_data)
        if not video_url:
            raise RuntimeError("未返回视频 URL")

        video_bytes = _download_video(video_url, headers, timeout, retries)
        filepath = _save_video(video_bytes, "zhangyuapi_kling_extend")
        video_obj = VideoFromFile(filepath) if VideoFromFile else filepath
        return (video_obj, json.dumps({"status": "success", "task_id": task_id}, ensure_ascii=False), "[]")

    def _lip_sync(self, base, headers, video_id, task_id, text, voice_language,
                  zh_voice, en_voice, voice_speed, timeout, retries, uid):
        if not video_id or not task_id:
            raise ValueError("唇形同步模式需要提供 video_id 和 task_id")

        voice_map = dict(KLING_ZH_VOICES if voice_language == "zh" else KLING_EN_VOICES)
        voice_name = zh_voice if voice_language == "zh" else en_voice
        voice_id = voice_map.get(voice_name, "")

        payload = {
            "input": {
                "task_id": task_id, "video_id": video_id,
                "mode": "text2video", "text": text,
                "voice_id": voice_id, "voice_language": voice_language,
                "voice_speed": voice_speed,
            }
        }

        if uid:
            emit_runtime_status(uid, "running", "可灵唇形同步中", 0, 0, retries, timeout)

        resp = ZHANGYUAPI_post(f"{base}/kling/v1/videos/lip-sync", timeout, headers=headers, json=payload)
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"API 错误 {resp.status_code}: {_safe_extract_error_from_response(resp)}")

        new_task_id = resp.json().get("data", {}).get("task_id")
        if not new_task_id:
            raise RuntimeError("未返回 task_id")

        task_data = _poll_task(base, headers, new_task_id, "lip-sync", timeout - 10, retries)
        video_url = _extract_video_url(task_data)
        if not video_url:
            raise RuntimeError("未返回视频 URL")

        video_bytes = _download_video(video_url, headers, timeout, retries)
        filepath = _save_video(video_bytes, "zhangyuapi_kling_lipsync")
        video_obj = VideoFromFile(filepath) if VideoFromFile else filepath
        return (video_obj, json.dumps({"status": "success", "task_id": new_task_id}, ensure_ascii=False), "[]")


# ===================================================================
# Registration
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ComfyuiZhangyuAPIKlingNode": ComfyuiZhangyuAPIKlingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiZhangyuAPIKlingNode": "ComfyUI-zhangyuapi-可灵视频",
}
