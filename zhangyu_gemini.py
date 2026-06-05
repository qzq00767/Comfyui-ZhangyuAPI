#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — Gemini 格式通用生图节点.

适配 NewAPI 的 Gemini 中继（Relay）端点，调用原生 Gemini
``generateContent`` 协议生成图片。UI 布局对齐 Image-2 节点。

- Endpoint: ``POST {api_base}/v1beta/models/{model}:generateContent``
- Auth: ``Authorization: Bearer {key}``
- Image output: ``candidates[].content.parts[].inlineData`` (base64)
- Model list: ``GET {api_base}/v1beta/models``
"""

import hashlib
import json
import random
import time
import torch

from .zhangyu_gpt_img2 import (
    ZHANGYUAPI_timeout,
    _RETRYABLE_EXCEPTIONS,
    DEFAULT_NODE_TIMEOUT,
    DEFAULT_MIN_NODE_TIMEOUT,
    DEFAULT_MAX_NODE_TIMEOUT,
    DEFAULT_RETRY_TIMES,
    _get_http_client,
    _jittered_sleep,
    normalize_api_base,
    denormalize_api_base,
    b64_json_to_uint8,
    tensor_to_data_url,
    emit_runtime_status,
    safe_int,
    safe_float,
    safe_choice,
    normalize_prompt_text,
    _log,
)

DEFAULT_GEMINI_BASE = "https://zhangyuapi.com"

_GEMINI_IMAGE_EXCLUDE = [
    "embedding", "text-embedding", "aqa",
]


# ===================================================================
# Gemini 模型列表获取
# ===================================================================

def _fetch_gemini_models(api_base, api_key, timeout=10):
    """从 ``GET /v1beta/models`` 获取支持 generateContent 的模型列表。"""
    base = normalize_api_base(api_base or DEFAULT_GEMINI_BASE)
    url = f"{base}/v1beta/models"
    client = _get_http_client()
    response = client.get(
        url,
        headers={
            "Authorization": f"Bearer {(api_key or '').strip()}",
            "Content-Type": "application/json",
        },
        timeout=ZHANGYUAPI_timeout(timeout),
    )
    response.raise_for_status()
    data = response.json()
    models = []
    for m in data.get("models", []):
        name = m.get("name", "")
        model_id = name.replace("models/", "") if name.startswith("models/") else name
        if not model_id:
            continue
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" not in methods:
            continue
        lowered = model_id.lower()
        if any(pat in lowered for pat in _GEMINI_IMAGE_EXCLUDE):
            continue
        models.append(model_id)
    return sorted(models)


# ===================================================================
# Gemini generateContent 请求
# ===================================================================

def _gemini_generate(api_base, api_key, model, prompt, n=1,
                     image_size="1K", aspect_ratio="1:1",
                     seed=0, temperature=0.7, output_format="jpeg",
                     image_data_urls=None,
                     timeout_seconds=DEFAULT_NODE_TIMEOUT):
    """调用 ``POST /v1beta/models/{model}:generateContent`` 生成图片。

    Returns:
        ``list[dict]`` — ``{"mime_type": str, "data": base64_str}``。
    """
    base = normalize_api_base(api_base or DEFAULT_GEMINI_BASE)
    url = f"{base}/v1beta/models/{model}:generateContent"
    headers = {
        "Authorization": f"Bearer {(api_key or '').strip()}",
        "Content-Type": "application/json",
    }

    parts = []

    # Reference images: add as inlineData parts (before text)
    if image_data_urls:
        for data_url in image_data_urls:
            # data URL: "data:image/png;base64,..."
            header, b64 = data_url.split(",", 1) if "," in data_url else ("image/png", data_url)
            mime = "image/png"
            if "image/" in header:
                mime = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
            parts.append({
                "inlineData": {"mimeType": mime, "data": b64}
            })

    parts.append({"text": prompt})

    # Map output_format → MIME type
    mime_map = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }

    generation_config = {
        "responseModalities": ["IMAGE", "TEXT"],
        "temperature": temperature,
        "imageConfig": {},
    }

    if image_size and image_size != "auto (不传size)":
        generation_config["imageConfig"]["imageSize"] = image_size

    if aspect_ratio and aspect_ratio != "auto":
        generation_config["imageConfig"]["aspectRatio"] = aspect_ratio

    if seed:
        generation_config["seed"] = seed

    if output_format in mime_map:
        generation_config["imageConfig"]["outputMimeType"] = mime_map[output_format]

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": generation_config,
    }

    all_images = []
    for _ in range(n):
        response = _get_http_client().post(
            url,
            json=payload,
            headers=headers,
            timeout=ZHANGYUAPI_timeout(timeout_seconds),
        )
        try:
            response.raise_for_status()
        except Exception as exc:
            body = response.text[:1000] if response.text else "<empty>"
            raise RuntimeError(
                f"Gemini API 请求失败: HTTP {response.status_code}; body={body}"
            ) from exc

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini 未返回候选: {str(data)[:500]}")

        for candidate in candidates:
            for part in candidate.get("content", {}).get("parts", []):
                inline = part.get("inlineData")
                if inline and inline.get("data"):
                    all_images.append({
                        "mime_type": inline.get("mimeType", "image/png"),
                        "data": inline["data"],
                    })

    if not all_images:
        texts = []
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    texts.append(part["text"])
        raise RuntimeError(
            f"Gemini 未生成图片。文本输出: {''.join(texts)[:500]}"
        )

    return all_images


# ===================================================================
# Node class
# ===================================================================

class ComfyuiZhangyuAPIGeminiNode:
    """ComfyUI Gemini 格式通用生图节点 — 对齐 Image-2 布局."""

    IMAGE_SIZES = ["auto (不传size)", "1K", "2K", "4K"]

    ASPECT_RATIOS = [
        "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9",
    ]

    OUTPUT_FORMATS = ["png", "jpeg", "webp"]

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "chats", "model_list")
    FUNCTION = "generate"
    CATEGORY = "Comfyui-ZhangyuAPI/生图"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key (API密钥)": (
                    "STRING", {"default": "", "multiline": False}),
                "prompt (提示词)": (
                    "STRING", {"multiline": True, "default": ""}),
                "model (模型)": (
                    "STRING", {"default": "nano-banana",
                               "multiline": False,
                               "placeholder": "nano-banana"}),
                "api_base (接口域名)": (
                    "STRING", {"default": DEFAULT_GEMINI_BASE, "multiline": False}),
                "image_size (分辨率)": (
                    cls.IMAGE_SIZES, {"default": "1K"}),
                "aspect_ratio (宽高比)": (
                    cls.ASPECT_RATIOS, {"default": "1:1"}),
                "n (生成数量)": (
                    "INT", {"default": 1, "min": 1, "max": 4}),
                "seed (种子)": (
                    "INT", {"default": 0, "min": 0, "max": 2147483647,
                            "control_after_generate": True}),
                "temperature (创造性)": (
                    "FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0,
                              "step": 0.1}),
                "timeout_seconds (超时秒数)": (
                    "INT", {"default": DEFAULT_NODE_TIMEOUT,
                            "min": DEFAULT_MIN_NODE_TIMEOUT,
                            "max": DEFAULT_MAX_NODE_TIMEOUT}),
                "retry_times (重试次数)": (
                    "INT", {"default": DEFAULT_RETRY_TIMES, "min": 1, "max": 5}),
            },
            "optional": {
                "output_format (输出格式)": (
                    cls.OUTPUT_FORMATS, {"default": "jpeg"}),
                **{f"image_{i:02d}": ("IMAGE",) for i in range(1, 9)},
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
             if k not in tuple(f"image_{i:02d}" for i in range(1, 9))},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.md5(key.encode()).hexdigest()

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
            data_urls.append(tensor_to_data_url(tensor))
        return data_urls

    def generate(self, **kwargs):
        api_key = kwargs.get("api_key (API密钥)", "").strip()
        api_base = normalize_api_base(
            kwargs.get("api_base (接口域名)", DEFAULT_GEMINI_BASE))
        prompt = kwargs.get("prompt (提示词)", "")
        model = kwargs.get("model (模型)", "").strip()
        image_size = safe_choice(
            kwargs.get("image_size (分辨率)", "1K"),
            self.IMAGE_SIZES, "1K")
        aspect_ratio = safe_choice(
            kwargs.get("aspect_ratio (宽高比)", "1:1"),
            self.ASPECT_RATIOS, "1:1")
        n = safe_int(kwargs.get("n (生成数量)", 1), 1, 1, 4)
        seed = int(kwargs.get("seed (种子)", 0))
        if seed == 0:
            seed = random.randint(1, 2147483647)
        temperature = safe_float(kwargs.get("temperature (创造性)", 0.7), 0.7, 0.0, 2.0)
        output_format = safe_choice(
            kwargs.get("output_format (输出格式)", "jpeg"),
            self.OUTPUT_FORMATS, "jpeg")
        timeout_seconds = safe_int(
            kwargs.get("timeout_seconds (超时秒数)", DEFAULT_NODE_TIMEOUT),
            DEFAULT_NODE_TIMEOUT, DEFAULT_MIN_NODE_TIMEOUT, DEFAULT_MAX_NODE_TIMEOUT)
        retry_times = safe_int(
            kwargs.get("retry_times (重试次数)", DEFAULT_RETRY_TIMES),
            DEFAULT_RETRY_TIMES, 1, 5)
        unique_id = kwargs.get("unique_id")
        start_ts = time.time()

        if not api_key:
            emit_runtime_status(unique_id, "error", "API Key 为空",
                                0.0, 0, retry_times, timeout_seconds)
            raise ValueError("API Key 不能为空")

        clean_prompt = normalize_prompt_text(prompt)
        if not clean_prompt:
            raise ValueError("prompt 不能为空")

        # Collect reference images
        image_data_urls = self._collect_images(kwargs)

        # -- 获取模型列表 ---------------------------------------------------------
        model_list = []
        try:
            model_list = _fetch_gemini_models(api_base, api_key)
        except Exception as exc:
            _log("warn", f"[Gemini] 模型列表获取失败: {exc}")

        if not model and model_list:
            model = model_list[0]
            _log("info", f"[Gemini] 自动选择模型: {model}")
        if not model:
            model = "nano-banana"

        effective_size = image_size if image_size != "auto (不传size)" else "1K"
        print(
            f"[Comfyui-ZhangyuAPI-Gemini] model={model}, "
            f"n={n}, image_size={image_size}, aspect_ratio={aspect_ratio}, "
            f"seed={seed}, temperature={temperature}, output_format={output_format}"
            + (f", ref_images={len(image_data_urls)}" if image_data_urls else "")
        )
        emit_runtime_status(unique_id, "running", "开始生成",
                            0.0, 0, retry_times, timeout_seconds)

        # -- 重试循环 ------------------------------------------------------------
        last_error = None
        for attempt in range(1, retry_times + 1):
            try:
                emit_runtime_status(
                    unique_id, "running",
                    f"请求生成中 ({attempt}/{retry_times})",
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )

                images = _gemini_generate(
                    api_base, api_key, model, clean_prompt,
                    n=n, image_size=effective_size,
                    aspect_ratio=aspect_ratio,
                    seed=seed, temperature=temperature,
                    output_format=output_format,
                    image_data_urls=image_data_urls,
                    timeout_seconds=timeout_seconds,
                )

                tensors = []
                for img in images:
                    img_tensor = b64_json_to_uint8(img["data"])
                    if img_tensor.dim() == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                    tensors.append(img_tensor)

                image_tensor = torch.cat(tensors, dim=0) if tensors else torch.zeros(1, 1, 1, 3)

                elapsed = time.time() - start_ts
                response_info = {
                    "api_base": denormalize_api_base(api_base),
                    "model": model,
                    "n": n,
                    "image_size": image_size,
                    "aspect_ratio": aspect_ratio,
                    "seed": seed,
                    "temperature": temperature,
                    "output_format": output_format,
                    "input_images": len(image_data_urls),
                    "output_images": len(images),
                    "elapsed_seconds": round(elapsed, 2),
                }
                emit_runtime_status(
                    unique_id, "success",
                    f"生成成功 (耗时 {elapsed:.1f}s)",
                    elapsed, attempt, retry_times, timeout_seconds,
                )
                return (
                    image_tensor,
                    json.dumps(response_info, ensure_ascii=False, indent=2),
                    json.dumps({"images": len(images), "model": model},
                               ensure_ascii=False, indent=2),
                    json.dumps(model_list, ensure_ascii=False),
                )

            except _RETRYABLE_EXCEPTIONS as exc:
                last_error = str(exc)
                if attempt < retry_times:
                    emit_runtime_status(
                        unique_id, "running",
                        f"网络错误，重试中 ({attempt}/{retry_times})",
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
            f"Gemini 连续 {retry_times} 次失败，最后错误: {last_error}"
        )


# ===================================================================
# ComfyUI node registration
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ComfyuiZhangyuAPIGeminiNode": ComfyuiZhangyuAPIGeminiNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiZhangyuAPIGeminiNode": "ComfyUI-zhangyuapi-通用Gemini格式",
}
