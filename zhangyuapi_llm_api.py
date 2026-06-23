#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — 通用 LLM API 节点.

支持文本、图片、视频输入，调用任意 OpenAI 兼容的 chat/completions 接口。

Endpoint: POST /v1/chat/completions
Auth: Authorization: Bearer <api_key>
"""

import base64
import io
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

from .zhangyu_gpt_img2 import (
    ZHANGYUAPI_post,
    ZHANGYUAPI_timeout,
    normalize_api_base,
    denormalize_api_base,
    tensor_to_data_url,
    emit_runtime_status,
    DEFAULT_API_BASE_URL,
    _log,
    safe_int,
    _safe_extract_error_from_response,
    _RETRYABLE_EXCEPTIONS,
    _jittered_sleep,
    is_retryable_http_status,
    fetch_available_models_cached,
    _filter_chat_models,
)

# ===================================================================
# Constants
# ===================================================================

CATEGORY = "Comfyui-ZhangyuAPI/📝文本 Text"


# ===================================================================
# Image / Video encoding helpers
# ===================================================================

def _encode_image_b64(image_tensor, max_dimension=1536):
    """将 ComfyUI IMAGE tensor 编码为 base64 JPEG，带压缩优化。

    Args:
        image_tensor: ComfyUI IMAGE tensor
        max_dimension: 最大边长限制

    Returns:
        str: base64 编码的图片数据
    """
    try:
        # Convert tensor to PIL Image
        i = 255.0 * image_tensor.cpu().numpy()[0]
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        original_size = img.size
        _log("debug", f"[LLM API] 原始图片尺寸: {original_size[0]}x{original_size[1]}")

        # Apply size limit
        if max(original_size) > max_dimension:
            ratio = max_dimension / max(original_size)
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            _log("debug", f"[LLM API] 缩放图片至: {new_size[0]}x{new_size[1]}")

        # Try multiple compression levels
        formats_to_try = [
            ('JPEG', {'quality': 75, 'optimize': True}),
            ('JPEG', {'quality': 60, 'optimize': True}),
            ('JPEG', {'quality': 50, 'optimize': True}),
        ]

        best_result = None
        smallest_size = float('inf')

        for format_name, save_kwargs in formats_to_try:
            try:
                buf = io.BytesIO()
                img.save(buf, format=format_name, **save_kwargs)
                img_bytes = buf.getvalue()

                if len(img_bytes) < smallest_size:
                    smallest_size = len(img_bytes)
                    best_result = base64.b64encode(img_bytes).decode('utf-8')

                    base64_size_mb = len(best_result) / (1024 * 1024)
                    _log("debug", f"[LLM API] 质量 {save_kwargs['quality']}: {base64_size_mb:.2f}MB base64")

                    if base64_size_mb < 2.0:
                        break
            except Exception as e:
                _log("debug", f"[LLM API] 编码失败 quality={save_kwargs['quality']}: {e}")
                continue

        if best_result:
            final_size_mb = len(best_result) / (1024 * 1024)
            _log("debug", f"[LLM API] 最终图片 base64 大小: {final_size_mb:.2f}MB")
            return best_result
        else:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        _log("warn", f"[LLM API] 编码图片失败: {str(e)}")
        i = 255.0 * image_tensor.cpu().numpy()[0]
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_video_file_path(video):
    """从 ComfyUI VIDEO 对象中提取文件路径。"""
    if hasattr(video, "_VideoFromFile__file"):
        path = getattr(video, "_VideoFromFile__file", None)
        if isinstance(path, str) and os.path.exists(path):
            return path

    if hasattr(video, "get_stream_source"):
        try:
            stream_source = video.get_stream_source()
            if isinstance(stream_source, str) and os.path.exists(stream_source):
                return stream_source
        except Exception:
            pass

    for attr in ("path", "file"):
        if hasattr(video, attr):
            path = getattr(video, attr, None)
            if isinstance(path, str) and os.path.exists(path):
                return path

    return None


def _encode_video_b64(video):
    """将 ComfyUI VIDEO 对象编码为 base64 MP4，带 ffmpeg 压缩。"""
    video_path = _get_video_file_path(video)
    temp_original = None

    if not video_path:
        if hasattr(video, "save_to"):
            temp_original = f"temp_video_original_{time.time()}.mp4"
            try:
                video.save_to(temp_original)
                video_path = temp_original
            except Exception as e:
                raise ValueError(f"无法保存视频: {str(e)}")
        else:
            raise ValueError(f"无法从 {type(video)} 类型读取视频数据")

    # Get original video info
    try:
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration',
            '-of', 'json',
            video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe_result.returncode == 0:
            probe_data = json.loads(probe_result.stdout)
            if 'streams' in probe_data and len(probe_data['streams']) > 0:
                stream = probe_data['streams'][0]
                width = stream.get('width', 0)
                height = stream.get('height', 0)
                duration = float(stream.get('duration', 0))
                _log("debug", f"[LLM API] 原始视频: {width}x{height}, {duration:.1f}s")
    except Exception as e:
        _log("debug", f"[LLM API] 无法探测视频: {e}")

    # Get original file size
    try:
        original_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        _log("debug", f"[LLM API] 原始视频文件大小: {original_size_mb:.2f}MB")
    except:
        original_size_mb = 0

    # Compress video using ffmpeg
    compressed_path = f"temp_video_compressed_{time.time()}.mp4"

    try:
        compress_cmd = [
            'ffmpeg', '-i', video_path,
            '-t', '5',
            '-vf', 'scale=\'min(1280,iw)\':-2',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '30',
            '-b:v', '400k',
            '-maxrate', '400k',
            '-bufsize', '800k',
            '-r', '10',
            '-an',
            '-y',
            compressed_path
        ]

        _log("debug", "[LLM API] 使用 ffmpeg 压缩视频 (仅前5秒)...")
        result = subprocess.run(compress_cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            _log("debug", f"[LLM API] FFmpeg 压缩失败: {result.stderr}")
            final_path = video_path
        else:
            compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
            _log("debug", f"[LLM API] 压缩后视频大小: {compressed_size_mb:.2f}MB")
            final_path = compressed_path

    except FileNotFoundError:
        _log("debug", "[LLM API] 未找到 ffmpeg，使用原始视频")
        final_path = video_path
    except subprocess.TimeoutExpired:
        _log("debug", "[LLM API] ffmpeg 超时，使用原始视频")
        final_path = video_path
    except Exception as e:
        _log("debug", f"[LLM API] 压缩失败 ({str(e)})，使用原始视频")
        final_path = video_path

    # Read and encode to base64
    try:
        with open(final_path, "rb") as f:
            video_bytes = f.read()
            base64_data = base64.b64encode(video_bytes).decode("utf-8")

        base64_size_mb = len(base64_data) / (1024 * 1024)
        _log("debug", f"[LLM API] 最终视频 base64 大小: {base64_size_mb:.2f}MB")

        if base64_size_mb > 10.0:
            _log("warn", f"[LLM API] base64 大小过大 ({base64_size_mb:.2f}MB)，可能导致 API 错误")

        return base64_data

    finally:
        try:
            if temp_original and os.path.exists(temp_original):
                os.remove(temp_original)
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
        except:
            pass


# ===================================================================
# Node class
# ===================================================================

class ZhangyuAPILLMApiNode:
    """通用 LLM API 节点 — 支持文本、图片、视频多模态输入。

    调用 OpenAI 兼容的 chat/completions 接口。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_base (接口地址)": (
                    "STRING", {
                        "default": DEFAULT_API_BASE_URL,
                        "multiline": False,
                    }),
                "api_key (API密钥)": (
                    "STRING", {
                        "default": "",
                        "multiline": False,
                    }),
                "model (模型)": (
                    "STRING", {
                        "default": "gpt-4o",
                        "multiline": False,
                    }),
                "system_prompt (系统提示词)": (
                    "STRING", {
                        "default": "You are a helpful assistant.",
                        "multiline": True,
                    }),
                "prompt (用户提示词)": (
                    "STRING", {
                        "default": "",
                        "multiline": True,
                    }),
                "temperature (温度)": (
                    "FLOAT", {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                    }),
                "timeout_seconds (超时秒数)": (
                    "INT", {
                        "default": 120,
                        "min": 30,
                        "max": 600,
                    }),
                "retry_times (重试次数)": (
                    "INT", {
                        "default": 3,
                        "min": 1,
                        "max": 10,
                    }),
            },
            "optional": {
                "ref_image (参考图片)": ("IMAGE",),
                "video (参考视频)": ("VIDEO",),
                "skip_error (跳过错误)": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response (回复内容)",)
    FUNCTION = "call_llm"
    CATEGORY = CATEGORY

    def call_llm(self, api_base, api_key, model, system_prompt, prompt,
                 temperature, timeout_seconds, retry_times,
                 ref_image=None, video=None, skip_error=False, unique_id=None):
        """执行 LLM 调用。"""
        try:
            return self._call_llm_impl(
                api_base, api_key, model, system_prompt, prompt,
                temperature, timeout_seconds, retry_times,
                ref_image, video, unique_id,
            )
        except Exception as exc:
            if not skip_error:
                raise
            _log("warn", f"[LLM API] 调用失败 (skip_error): {exc}")
            return (f"Error: {exc}",)

    def _call_llm_impl(self, api_base, api_key, model, system_prompt, prompt,
                       temperature, timeout_seconds, retry_times,
                       ref_image, video, unique_id):
        """LLM 调用实现。"""
        # Normalize API base
        base = normalize_api_base(api_base or DEFAULT_API_BASE_URL)
        url = f"{base}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }

        # Build messages
        messages = []

        # System prompt
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        # User content
        user_content = []

        # Text prompt
        if prompt and prompt.strip():
            user_content.append({"type": "text", "text": prompt.strip()})

        # Video input (priority: video > image)
        if video is not None:
            try:
                _log("info", "[LLM API] 处理视频输入...")
                base64_video = _encode_video_b64(video)
                user_content.append({
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{base64_video}"
                    }
                })
                _log("debug", f"[LLM API] 视频 base64 大小: {len(base64_video) / (1024*1024):.2f}MB")
            except Exception as e:
                _log("warn", f"[LLM API] 编码视频失败: {e}")
                if ref_image is not None:
                    _log("info", "[LLM API] 回退到图片输入")

        # Image input
        if video is None and ref_image is not None:
            try:
                _log("info", "[LLM API] 处理图片输入...")
                base64_image = _encode_image_b64(ref_image)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
                _log("debug", f"[LLM API] 图片 base64 大小: {len(base64_image) / (1024*1024):.2f}MB")
            except Exception as e:
                _log("warn", f"[LLM API] 编码图片失败: {e}")

        # Add user message
        if user_content:
            if len(user_content) == 1 and user_content[0].get("type") == "text":
                messages.append({"role": "user", "content": user_content[0]["text"]})
            else:
                messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt or "Hello"})

        # Build payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        _log("info", f"[LLM API] 调用 {url} 模型={model}")

        if unique_id:
            emit_runtime_status(unique_id, "running", f"LLM 调用中: {model}", 0, 0, retry_times, timeout_seconds)

        # Retry loop
        last_error = None
        for attempt in range(1, retry_times + 1):
            try:
                response = ZHANGYUAPI_post(
                    url, timeout_seconds,
                    headers=headers,
                    json=payload,
                )

                if response.status_code != 200:
                    error_msg = _safe_extract_error_from_response(response)
                    last_error = f"API 错误 {response.status_code}: {error_msg}"

                    if is_retryable_http_status(response.status_code) and attempt < retry_times:
                        _log("warn", f"[LLM API] {last_error}，重试 ({attempt}/{retry_times})")
                        _jittered_sleep(attempt)
                        continue
                    raise RuntimeError(last_error)

                data = response.json()
                choices = data.get("choices") or []
                if choices:
                    result = choices[0].get("message", {}).get("content", "")
                    _log("info", f"[LLM API] 成功，回复 {len(result)} 字符")
                    if unique_id:
                        emit_runtime_status(unique_id, "success", f"LLM 调用成功", 0, attempt, retry_times, timeout_seconds)
                    return (result,)
                else:
                    raise RuntimeError("API 未返回有效回复")

            except _RETRYABLE_EXCEPTIONS as exc:
                last_error = str(exc)
                _log("warn", f"[LLM API] 网络错误 (attempt={attempt}/{retry_times}): {last_error}")
                from .zhangyu_gpt_img2 import _on_retryable_error as _on_retry
                _on_retry(exc)
                if attempt < retry_times:
                    _jittered_sleep(attempt)
                    continue
                break
            except RuntimeError:
                raise
            except Exception as exc:
                last_error = str(exc)
                _log("warn", f"[LLM API] 未知错误 (attempt={attempt}/{retry_times}): {last_error}")
                if attempt < retry_times:
                    _jittered_sleep(attempt)
                    continue
                break

        raise RuntimeError(f"LLM API 连续 {retry_times} 次失败: {last_error}")


# ===================================================================
# NODE_CLASS_MAPPINGS
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ZhangyuAPILLMApiNode": ZhangyuAPILLMApiNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZhangyuAPILLMApiNode": "ComfyUI-zhangyuapi-通用LLM接口",
}
