#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — Nano Banana 生图节点.

使用 Chat Completions API 格式调用 Nano Banana 系列模型。
- Endpoint: POST /v1/chat/completions
- Auth: Authorization: Bearer <api_key>
- 支持流式响应，从响应中提取图片（base64或URL）
"""

import base64
import hashlib
import json
import re
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
    VideoFromFile = None


# ===================================================================
# Constants
# ===================================================================

DEFAULT_NANOBANANA_BASE = DEFAULT_API_BASE_URL

NANOBANANA_MODELS = [
    "nano-banana-pro",
    "nano-banana-2",
    "nano-banana",
    "nano-banana-hd",
    "gemini-2.5-flash-image-preview",
    "gemini-3-pro-image-preview",
]


# ===================================================================
# Helpers
# ===================================================================

def _image_tensor_to_base64(image_tensor):
    """Convert ComfyUI IMAGE tensor to base64 string (no data URL prefix)."""
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


def _extract_image_from_response(response_text):
    """从响应文本中提取图片（base64或URL）。"""
    # 1. 尝试提取 base64 图片
    base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
    base64_matches = re.findall(base64_pattern, response_text)
    if base64_matches:
        try:
            image_data = base64.b64decode(base64_matches[0])
            img = Image.open(BytesIO(image_data))
            return img, f"data:image/png;base64,{base64_matches[0]}"
        except Exception as e:
            _log("warn", f"解码base64图片失败: {e}")

    # 2. 尝试提取 Markdown 图片 URL
    image_pattern = r'!\[.*?\]\((.*?)\)'
    matches = re.findall(image_pattern, response_text)
    if not matches:
        # 3. 尝试提取普通图片 URL
        url_pattern = r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)'
        matches = re.findall(url_pattern, response_text)
    if not matches:
        # 4. 尝试提取所有 URL
        all_urls_pattern = r'https?://\S+'
        matches = re.findall(all_urls_pattern, response_text)

    return None, matches[0] if matches else None


# ===================================================================
# Node: Nano Banana 生图
# ===================================================================

class ComfyuiZhangyuAPINanoBananaNode:
    """ComfyUI Nano Banana 生图节点 — 使用 Chat Completions API."""

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate"
    CATEGORY = "Comfyui-ZhangyuAPI/🖼️图片 Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key (API密钥)": ("STRING", {"default": ""}),
                "prompt (提示词)": ("STRING", {"default": "", "multiline": True}),
                "model (模型)": (NANOBANANA_MODELS, {"default": "nano-banana-pro"}),
                "api_base (接口地址)": ("STRING", {"default": DEFAULT_NANOBANANA_BASE, "multiline": False}),
                "temperature (创造性)": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p (采样)": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens (最大token)": ("INT", {"default": 32768, "min": 1, "max": 32768}),
                "timeout_seconds (超时秒数)": ("INT", {"default": 300, "min": 30, "max": 900}),
                "retry_times (重试次数)": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
            "optional": {
                "image_01 (参考图1)": ("IMAGE",),
                "image_02 (参考图2)": ("IMAGE",),
                "image_03 (参考图3)": ("IMAGE",),
                "image_04 (参考图4)": ("IMAGE",),
                "seed (种子)": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "skip_error (跳过错误)": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def generate(self, api_key, prompt, model, api_base, temperature, top_p, max_tokens,
                 timeout_seconds, retry_times,
                 image_01=None, image_02=None, image_03=None, image_04=None,
                 seed=0, skip_error=False, unique_id=None):
        try:
            return self._generate_impl(
                api_key, prompt, model, api_base, temperature, top_p, max_tokens,
                timeout_seconds, retry_times,
                [image_01, image_02, image_03, image_04], seed, unique_id,
            )
        except Exception as exc:
            if not skip_error:
                raise
            _log("warn", f"[Nano Banana] skip_error: {exc}")
            return _skip_error_return(str(exc), self.RETURN_TYPES, unique_id, retry_times, timeout_seconds)

    def _generate_impl(self, api_key, prompt, model, api_base, temperature, top_p, max_tokens,
                       timeout_seconds, retry_times, images, seed, unique_id):
        base = normalize_api_base(api_base or DEFAULT_NANOBANANA_BASE)
        url = f"{base}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }

        # Build content
        content = [{"type": "text", "text": prompt}]

        # Add reference images
        images_added = 0
        for img in images:
            if img is not None:
                batch_size = img.shape[0]
                for i in range(min(batch_size, 1)):  # Only first frame
                    single_image = img[i:i+1]
                    img_b64 = _image_tensor_to_base64(single_image)
                    if img_b64:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        })
                        images_added += 1

        _log("info", f"[Nano Banana] 模型={model}, 参考图={images_added}张")

        if unique_id:
            emit_runtime_status(unique_id, "running", f"Nano Banana 请求中: {model}", 0, 0, retry_times, timeout_seconds)

        # Build payload
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if seed > 0:
            payload["seed"] = seed

        # Send streaming request
        last_error = None
        for attempt in range(1, retry_times + 1):
            try:
                response_text = self._send_streaming(url, headers, payload, timeout_seconds)
                break
            except Exception as exc:
                last_error = str(exc)
                _log("warn", f"[Nano Banana] 请求失败 (attempt={attempt}/{retry_times}): {last_error}")
                if attempt < retry_times:
                    _jittered_sleep(attempt)
                    continue
                raise RuntimeError(f"连续 {retry_times} 次失败: {last_error}")

        # Extract image from response
        img, image_url = _extract_image_from_response(response_text)

        if img is not None:
            import torch
            img_tensor = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0
            if unique_id:
                emit_runtime_status(unique_id, "success", "Nano Banana 生成成功", 0, attempt, retry_times, timeout_seconds)
            return (img_tensor, response_text, image_url or "")

        if image_url:
            # Download image from URL
            try:
                img_bytes = _download_bytes_with_retry(image_url, headers, timeout_seconds, retry_times, label="Nano Banana图片")
                img = Image.open(BytesIO(img_bytes))
                import torch
                img_tensor = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0
                if unique_id:
                    emit_runtime_status(unique_id, "success", "Nano Banana 生成成功", 0, attempt, retry_times, timeout_seconds)
                return (img_tensor, response_text, image_url)
            except Exception as e:
                _log("warn", f"[Nano Banana] 下载图片失败: {e}")

        # No image found
        if unique_id:
            emit_runtime_status(unique_id, "error", "Nano Banana 未返回图片", 0, attempt, retry_times, timeout_seconds)
        raise RuntimeError(f"Nano Banana 未生成图片。响应: {response_text[:500]}")

    def _send_streaming(self, url, headers, payload, timeout_seconds):
        """Send streaming request and collect full response."""
        client = _get_http_client()
        full_response = ""

        with client.stream(
            "POST", url,
            json=payload,
            headers=headers,
            timeout=ZHANGYUAPI_timeout(timeout_seconds),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8').strip()
                    if line_text.startswith('data: '):
                        data = line_text[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    full_response += delta['content']
                        except json.JSONDecodeError:
                            continue

        return full_response


# ===================================================================
# Registration
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ComfyuiZhangyuAPINanoBananaNode": ComfyuiZhangyuAPINanoBananaNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiZhangyuAPINanoBananaNode": "ComfyUI-zhangyuapi-NanoBanana生图",
}
