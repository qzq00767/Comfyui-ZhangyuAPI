#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — 通用生图接口节点.

OpenAI 兼容的图片生成节点，适配所有支持 ``POST /v1/images/generations`` 的
API 提供商（zhangyuapi.com / newapi.pro / 自建等）。

用户只需提供 **域名 + API Key**，节点自动完成：

* 模型列表获取（前端自动调用 ``GET /v1/models`` → 填充下拉框）
* 接口路径拼接（``POST /v1/images/generations``）
* 同步 / 异步任务自动识别与自适应轮询
* 返回格式自动解析（``url`` / ``b64_json``）

完全兼容 NewAPI / OpenAI 生图 API 规范。

Specification:
- Model list: ``GET /v1/models`` → ``data[].id`` (auto-fetched by frontend)
- Endpoint: ``POST /v1/images/generations``
- Auth: ``Authorization: Bearer <api_key>``
- Sync response: ``{"data": [{"url": "..."}]}`` or ``{"data": [{"b64_json": "..."}]}``
- Async response: ``{"task_id": "...", "status": "processing"}`` → polled with
  adaptive intervals until completion
- Standard error: ``{"error": {"message": "...", "type": "...", "code": "..."}}``

Features:
- HTTP/2 via ``httpx`` with forced direct connection (bypasses all proxies).
- Adaptive task polling with four-stage interval escalation.
- Concurrent async image downloads via ``asyncio.create_task``.
- Frontend runtime status bar with live progress updates.
- Auto-fetch model list from ``/v1/models`` on API key / domain change.
"""

import json
import re
import time

import torch

# ---------------------------------------------------------------------------
# Shared utilities — imported from the sibling gpt-image-2 node module
# ---------------------------------------------------------------------------
from .zhangyu_gpt_img2 import (  # noqa: E402
    # HTTP — reuse shared thread-safe infrastructure
    ZHANGYUAPI_timeout,
    ZHANGYUAPI_post,
    _RETRYABLE_EXCEPTIONS,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_READ_TIMEOUT,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_NODE_TIMEOUT,
    DEFAULT_MIN_NODE_TIMEOUT,
    DEFAULT_MAX_NODE_TIMEOUT,
    DEFAULT_RETRY_TIMES,
    DEFAULT_API_BASE_URL,
    # async / retry / polling
    _run_async_coroutine,
    _jittered_sleep,
    is_retryable_http_status,
    _adaptive_poll_interval,
    is_async_task_response,
    _poll_async_task,
    # image conversion
    b64_json_to_uint8,
    tensor_to_data_url,
    # response parsing & download (shared module-level functions)
    _parse_response_images,
    _download_images_async,
    _sanitize_api_response,
    _strip_image_data,
    # input sanitization
    safe_choice,
    safe_int,
    safe_float,
    normalize_prompt_text,
    normalize_api_base,
    # model discovery & filtering
    fetch_available_models,
    _filter_image_models,
    # model resolution & validation
    resolve_and_validate_model,
    # frontend status
    emit_runtime_status,
)



# ===================================================================
# Node class
# ===================================================================

class ComfyuiZhangyuAPIUniversalImageNode:
    """ComfyUI 通用生图节点 — 适配所有 OpenAI 兼容的图片生成 API.

    支持任意 ``POST /v1/images/generations`` 端点，兼容同步响应和异步任务
    （task_id 轮询）两种模式。

    Node display name: **ComfyUI-zhangyuapi-通用生图接口**
    """

    # Model list — initial default; the frontend auto-populates from GET /v1/models
    # (filtered to image-capable models via /zhangyuapi_fetch_image_models).
    MODELS = ["从接口自动获取模型列表"]

    # Size presets — common aspect ratios mapped to standard resolutions.
    # The label shows both ratio and pixel dimensions; the helper
    # :meth:`_parse_size_preset` extracts the ``WxH`` part at runtime.
    SIZES = [
        "auto (自动)",
        "1:1 正方形 · 1024x1024",
        "1:1 正方形 · 2048x2048",
        "16:9 宽屏 · 1792x1024",
        "9:16 竖屏 · 1024x1792",
        "4:3 横向 · 1536x1152",
        "3:4 竖向 · 1152x1536",
        "3:2 横向 · 1536x1024",
        "2:3 竖向 · 1024x1536",
        "21:9 超宽 · 2016x864",
    ]

    QUALITIES = ["standard", "hd"]

    STYLES = ["vivid", "natural"]

    # Sampler choices for diffusion models (SD / FLUX / Hunyuan / etc.)
    SAMPLERS = [
        "auto (自动)",
        "DDIM",
        "DDPM",
        "DPM++ 2M",
        "DPM++ SDE",
        "DPM++ 2M SDE",
        "Euler",
        "Euler a",
        "Heun",
        "LMS",
        "PLMS",
        "UniPC",
        "DPM2",
        "DPM2 a",
        "LCM",
    ]

    # Midjourney aspect ratio options (--ar parameter)
    MJ_AR_OPTIONS = [
        "auto",
        "1:1",
        "2:3",
        "3:2",
        "3:4",
        "4:3",
        "4:5",
        "5:4",
        "9:16",
        "16:9",
        "21:9",
    ]

    # ------------------------------------------------------------------
    # Model-type detection — heuristic substring matching against model ID.
    # Order matters: place more specific patterns before broader ones
    # (e.g. "dall-e-3" must be checked before "dall-e" in stable-diffusion).
    # ------------------------------------------------------------------
    _MODEL_TYPE_PATTERNS = {
        "dall-e-3":    ["dall-e-3", "dalle3"],
        "dall-e-2":    ["dall-e-2", "dalle2"],
        "midjourney":  ["midjourney", "mj-", "niji"],
        "flux":        ["flux", "schnell"],
        "stable-diffusion": [
            "stable-diffusion", "sdxl", "sd3", "sd-",
            "sd1", "sd2", "realistic-vision", "dreamshaper",
            "anything-", "counterfeit", "meinamix",
            "dark-sushi", "aam-xl", "juggernaut",
        ],
        "kolors":      ["kolors"],
        "hunyuan":     ["hunyuan"],
        "ideogram":    ["ideogram"],
        "recraft":     ["recraft"],
        "playground":  ["playground"],
    }

    # Model types that accept steps / cfg_scale / sampler parameters.
    _DIFFUSION_MODEL_TYPES = frozenset({
        "flux", "stable-diffusion", "kolors", "hunyuan",
        "ideogram", "recraft", "playground",
    })

    # DALL-E model size constraints
    _DALLE3_VALID_SIZES = {"1024x1024", "1792x1024", "1024x1792"}
    _DALLE2_VALID_SIZES = {"256x256", "512x512", "1024x1024"}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_urls", "chats", "model_list")
    FUNCTION = "generate"
    CATEGORY = "Comfyui-ZhangyuAPI/生图"

    # ------------------------------------------------------------------
    # ComfyUI protocol
    # ------------------------------------------------------------------

    @classmethod
    def INPUT_TYPES(cls):
        """Return the ComfyUI widget definition."""
        return {
            "required": {
                "api_key (API密钥)": (
                    "STRING", {"default": "", "multiline": False}),
                "api_base (接口域名)": (
                    "STRING", {"default": DEFAULT_API_BASE_URL, "multiline": False}),
                "prompt (正向提示词)": (
                    "STRING", {"default": "", "multiline": True}),
                "model (模型)": (
                    "STRING", {"default": "", "multiline": False,
                               "placeholder": "手动填写模型名，留空则自动从接口获取"}),
                "custom_model (自定义模型名)": (
                    "STRING", {"default": "", "multiline": False}),
                "n (生成数量)": (
                    "INT", {"default": 1, "min": 1, "max": 10}),
                "size (尺寸)": (
                    cls.SIZES, {"default": "auto (自动)"}),
                "quality (画质)": (
                    cls.QUALITIES, {"default": "standard"}),
                "style (风格)": (
                    cls.STYLES, {"default": "vivid"}),
                "response_format (响应格式)": (
                    ["b64_json", "url"], {"default": "url"}),
                "seed (种子)": (
                    "INT", {
                        "default": 0, "min": 0, "max": 2147483647,
                        "control_after_generate": True,
                    }),
                "timeout_seconds (超时秒数)": (
                    "INT", {
                        "default": DEFAULT_NODE_TIMEOUT,
                        "min": DEFAULT_MIN_NODE_TIMEOUT,
                        "max": DEFAULT_MAX_NODE_TIMEOUT,
                    }),
                "retry_times (重试次数)": (
                    "INT", {
                        "default": DEFAULT_RETRY_TIMES,
                        "min": 1, "max": 10,
                    }),
            },
            "optional": {
                "negative_prompt (反向提示词)": (
                    "STRING", {"default": "", "multiline": True}),
                # --- diffusion model core parameters ---
                "steps (采样步数)": (
                    "INT", {"default": 20, "min": 1, "max": 150}),
                "cfg_scale (提示词引导强度)": (
                    "FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0,
                               "step": 0.5}),
                "sampler (采样器)": (
                    cls.SAMPLERS, {"default": "auto (自动)"}),
                "denoising_strength (重绘强度)": (
                    "FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                               "step": 0.01}),
                # --- Midjourney parameters ---
                "mj_ar (MJ宽高比 --ar)": (
                    cls.MJ_AR_OPTIONS, {"default": "auto"}),
                "mj_stylize (MJ风格化 --stylize)": (
                    "INT", {"default": 100, "min": 0, "max": 1000}),
                "mj_chaos (MJ混乱度 --chaos)": (
                    "INT", {"default": 0, "min": 0, "max": 100}),
                "mj_weird (MJ怪异度 --weird)": (
                    "INT", {"default": 0, "min": 0, "max": 3000}),
                "mj_no (MJ排除内容 --no)": (
                    "STRING", {"default": "", "multiline": False}),
                # Reference image inputs (for img2img / multi-image fusion)
                **{f"image_{i:02d}": ("IMAGE",) for i in range(1, 9)},
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        """Validation is done inside :meth:`generate`."""
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # -- model-type detection & validation ------------------------------

    @classmethod
    def _detect_model_type(cls, model):
        """Detect the model family from the model ID string via heuristic
        pattern matching.

        Args:
            model: Model ID string (e.g. ``"dall-e-3"``, ``"flux-pro"``).

        Returns:
            ``str`` — one of ``"dall-e-3"``, ``"dall-e-2"``, ``"midjourney"``,
            ``"flux"``, ``"stable-diffusion"``, ``"kolors"``, ``"hunyuan"``,
            ``"ideogram"``, ``"recraft"``, ``"playground"``, or ``"unknown"``.

        """
        if not model:
            return "unknown"
        lowered = str(model).lower()
        for model_type, patterns in cls._MODEL_TYPE_PATTERNS.items():
            for pat in patterns:
                if pat in lowered:
                    return model_type
        return "unknown"

    @classmethod
    def _is_diffusion_model(cls, model_type):
        """Return ``True`` if *model_type* supports steps/cfg_scale/sampler."""
        return model_type in cls._DIFFUSION_MODEL_TYPES

    @classmethod
    def _validate_for_model(cls, model_type, size, n_images):
        """Validate *size* and *n_images* against model-specific constraints.

        Args:
            model_type: From :meth:`_detect_model_type`.
            size: Resolved size string (``"WxH"`` or ``"auto"``).
            n_images: Number of images (1–10).

        Returns:
            ``tuple[str, int]`` — ``(validated_size, validated_n)``.

        Raises:
            ValueError: If a constraint is violated.

        """
        validated_n = n_images

        if model_type == "dall-e-3":
            if size not in cls._DALLE3_VALID_SIZES:
                raise ValueError(
                    f"DALL-E 3 只支持以下尺寸: "
                    f"{', '.join(sorted(cls._DALLE3_VALID_SIZES))}，"
                    f"当前: {size}"
                )
            if validated_n != 1:
                raise ValueError(
                    f"DALL-E 3 每次只能生成 1 张图片 (n=1)，"
                    f"当前: n={validated_n}"
                )
        elif model_type == "dall-e-2":
            if size not in cls._DALLE2_VALID_SIZES:
                raise ValueError(
                    f"DALL-E 2 只支持以下尺寸: "
                    f"{', '.join(sorted(cls._DALLE2_VALID_SIZES))}，"
                    f"当前: {size}"
                )
            validated_n = max(1, min(validated_n, 10))

        return size, validated_n

    # -- Midjourney prompt suffix builder --------------------------------

    @staticmethod
    def _build_mj_suffix(mj_ar, mj_stylize, mj_chaos, mj_weird, mj_no):
        """Build a Midjourney-style ``--param value`` suffix for the prompt.

        Only includes parameters that differ from their defaults.

        Args:
            mj_ar: Aspect ratio (``"16:9"`` or ``"auto"``).
            mj_stylize: Stylize value (0–1000, default 100).
            mj_chaos: Chaos value (0–100, default 0).
            mj_weird: Weird value (0–3000, default 0).
            mj_no: Comma-separated exclusions (e.g. ``"cats,dogs"``).

        Returns:
            ``str`` — space-prefixed suffix like
            ``" --ar 16:9 --stylize 100"``, or ``""``.

        """
        parts = []
        ar = str(mj_ar).strip() if mj_ar else "auto"
        if ar.lower() != "auto" and ar:
            parts.append(f"--ar {ar}")

        stylize = safe_int(mj_stylize, 100, 0, 1000)
        if stylize != 100:
            parts.append(f"--stylize {stylize}")

        chaos = safe_int(mj_chaos, 0, 0, 100)
        if chaos != 0:
            parts.append(f"--chaos {chaos}")

        weird = safe_int(mj_weird, 0, 0, 3000)
        if weird != 0:
            parts.append(f"--weird {weird}")

        no_val = str(mj_no).strip() if mj_no else ""
        if no_val:
            parts.append(f"--no {no_val}")

        return (" " + " ".join(parts)) if parts else ""

    # -- size preset parser ----------------------------------------------

    @staticmethod
    def _parse_size_preset(preset_label):
        """Extract a ``WxH`` size string from a size-preset label.

        Handles both descriptive labels (``"16:9 宽屏 · 1792x1024"``) and
        raw dimension strings (``"1024x1024"``).  Returns ``"auto"`` for the
        auto placeholder.

        Args:
            preset_label: The value from the ``size (尺寸)`` widget.

        Returns:
            ``str`` — ``"WxH"`` or ``"auto"``.

        """
        if not preset_label:
            return "auto"
        label = str(preset_label).strip()
        if label.lower().startswith("auto"):
            return "auto"
        # Extract WxH from the label (works for both "16:9 宽屏 · 1792x1024"
        # and bare "1792x1024")
        match = re.search(r"(\d{2,5}x\d{2,5})", label)
        return match.group(1) if match else "auto"

    def _collect_images(self, kwargs):
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

    def _build_payload(self, model, model_type, prompt, negative_prompt,
                       n_images, size, quality, style, response_format,
                       seed, steps, cfg_scale, sampler,
                       denoising_strength, mj_ar, mj_stylize,
                       mj_chaos, mj_weird, mj_no, image_data_urls=None):
        """Build the JSON request body for ``POST /v1/images/generations``.

        Only includes non-default / non-empty fields.  Model-type-aware:
        diffusion params are only included for SD/FLUX/etc., and Midjourney
        parameters are rendered as prompt suffixes.

        Args:
            model: Resolved (non-placeholder) model ID.
            model_type: From :meth:`_detect_model_type`.
            prompt: Cleaned positive prompt.
            negative_prompt: Negative prompt (sent only if non-empty).
            n_images: Number of images to generate (1–10).
            size: Resolved size (``"WxH"`` or ``"auto"``).
            quality: ``"standard"`` or ``"hd"``.
            style: ``"vivid"`` or ``"natural"``.
            response_format: ``"url"`` or ``"b64_json"``.
            seed: Random seed (0 = server picks random).
            steps: Sampling steps (1–150).
            cfg_scale: CFG guidance scale (1.0–30.0).
            sampler: Sampler name or ``"auto (自动)"``.
            denoising_strength: Denoising strength (0.0–1.0).
            mj_ar: MJ aspect ratio (``"16:9"`` or ``"auto"``).
            mj_stylize: MJ stylize value (0–1000).
            mj_chaos: MJ chaos value (0–100).
            mj_weird: MJ weird value (0–3000).
            mj_no: MJ --no exclusions string.

        Returns:
            ``dict`` — JSON-serializable request body.

        """
        # -- Midjourney: render params as prompt suffix --------------------
        if model_type == "midjourney":
            mj_suffix = self._build_mj_suffix(
                mj_ar, mj_stylize, mj_chaos, mj_weird, mj_no)
            if mj_suffix:
                prompt = prompt + mj_suffix

        payload = {
            "model": model,       # always sent (resolved before this call)
            "prompt": prompt,
            "n": n_images,
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        resolved_size = self._parse_size_preset(size)
        if resolved_size != "auto":
            payload["size"] = resolved_size

        if quality != "standard":
            payload["quality"] = quality

        if style != "vivid":
            payload["style"] = style

        if response_format != "b64_json":
            payload["response_format"] = response_format

        # -- seed: always send (0 = server picks random) -------------------
        payload["seed"] = seed

        # -- diffusion parameters: only for SD / FLUX / etc. ---------------
        if self._is_diffusion_model(model_type):
            steps_val = safe_int(steps, 30, 1, 150)
            if steps_val != 30:
                payload["steps"] = steps_val

            cfg_val = safe_float(cfg_scale, 7.0, 1.0, 30.0)
            if abs(cfg_val - 7.0) > 0.01:
                payload["cfg_scale"] = round(cfg_val, 1)

            sampler_val = str(sampler or "").strip()
            if (sampler_val.lower() != "auto (自动)"
                    and sampler_val.lower() != "auto"
                    and sampler_val):
                payload["sampler"] = sampler_val

        # -- denoising strength: only send if < 1.0 (img2img scenario) ----
        ds_val = safe_float(denoising_strength, 1.0, 0.0, 1.0)
        if ds_val < 0.999:
            payload["denoising_strength"] = round(ds_val, 2)

        # Reference images: send as base64 data URLs if present
        if image_data_urls:
            payload["image_data"] = image_data_urls

        return payload

    # ------------------------------------------------------------------
    # API request
    # ------------------------------------------------------------------

    def _request_generation(self, api_base, headers, payload, timeout_seconds):
        """POST to ``/v1/images/generations``.

        Args:
            api_base: Normalized API base URL.
            headers: Request headers dict (must include ``Authorization``).
            payload: Request body dict from :meth:`_build_payload`.
            timeout_seconds: Read timeout.

        Returns:
            ``httpx.Response``.

        """
        url = f"{api_base}/v1/images/generations"
        return ZHANGYUAPI_post(
            url,
            timeout_seconds,
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
        )

    # ------------------------------------------------------------------
    # Error parsing (OpenAI-compatible error format)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_error_message(data):
        """Extract a human-readable error string from an API error response.

        Standard error format::

            {"error": {"message": "...", "type": "...", "code": "..."}}

        Falls back to the raw response text on parse failure.

        Args:
            data: Parsed JSON dict from an error response.

        Returns:
            ``str`` — error message.

        """
        if not isinstance(data, dict):
            return str(data)[:500]
        error = data.get("error")
        if isinstance(error, dict):
            return error.get("message") or json.dumps(error, ensure_ascii=False)
        if isinstance(error, str):
            return error
        return json.dumps(_sanitize_api_response(data), ensure_ascii=False)[:500]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self, **kwargs):
        """Execute an image generation request via the OpenAI-compatible API.

        ComfyUI calls this method from a thread-pool executor.  The method:

        1. Validates and sanitizes all inputs.
        2. Sends ``POST /v1/images/generations``.
        3. If the API returns an async task, polls with adaptive intervals.
        4. Downloads images (URLs concurrently, b64_json synchronously).
        5. Returns ``(IMAGE tensor, JSON response string)``.

        Args:
            **kwargs: ComfyUI widget values keyed by display name.

        Returns:
            ``tuple[torch.Tensor, str]`` — ``(image, response_json)``.

        Raises:
            ValueError: On missing / invalid inputs.
            RuntimeError: On API errors, network failures, or exhaustion
                of retry attempts.

        """
        # -- sanitize inputs --------------------------------------------------
        api_key = kwargs.get("api_key (API密钥)", "")
        api_base = normalize_api_base(
                kwargs.get("api_base (接口域名)", DEFAULT_API_BASE_URL)
            )
        prompt = kwargs.get("prompt (正向提示词)", "")

        model = kwargs.get("model (模型)", "").strip()
        custom_model = (kwargs.get("custom_model (自定义模型名)") or "").strip()
        if custom_model:
            model = custom_model

        n_images = safe_int(kwargs.get("n (生成数量)", 1), 1, 1, 10)
        size = self._parse_size_preset(
            kwargs.get("size (尺寸)", "auto (自动)"))
        negative_prompt = kwargs.get("negative_prompt (反向提示词)", "")
        quality = safe_choice(
            kwargs.get("quality (画质)", "standard"),
            self.QUALITIES, "standard")
        style = safe_choice(
            kwargs.get("style (风格)", "vivid"),
            self.STYLES, "vivid")
        response_format = safe_choice(
            kwargs.get("response_format (响应格式)", "b64_json"),
            ["b64_json", "url"], "b64_json")
        seed = safe_int(kwargs.get("seed (种子)", 0), 0, 0, 2147483647)
        timeout_seconds = safe_int(
            kwargs.get("timeout_seconds (超时秒数)", DEFAULT_NODE_TIMEOUT),
            DEFAULT_NODE_TIMEOUT,
            DEFAULT_MIN_NODE_TIMEOUT,
            DEFAULT_MAX_NODE_TIMEOUT,
        )
        retry_times = safe_int(
            kwargs.get("retry_times (重试次数)", DEFAULT_RETRY_TIMES),
            DEFAULT_RETRY_TIMES, 1, 10)

        # -- sanitize new diffusion parameters --------------------------------
        steps = safe_int(
            kwargs.get("steps (采样步数)", 30), 30, 1, 150)
        cfg_scale = safe_float(
            kwargs.get("cfg_scale (提示词引导强度)", 7.0), 7.0, 1.0, 30.0)
        sampler = safe_choice(
            kwargs.get("sampler (采样器)", "auto (自动)"),
            self.SAMPLERS, "auto (自动)")
        denoising_strength = safe_float(
            kwargs.get("denoising_strength (重绘强度)", 1.0), 1.0, 0.0, 1.0)

        # -- sanitize Midjourney parameters ----------------------------------
        mj_ar = safe_choice(
            kwargs.get("mj_ar (MJ宽高比 --ar)", "auto"),
            self.MJ_AR_OPTIONS, "auto")
        mj_stylize = safe_int(
            kwargs.get("mj_stylize (MJ风格化 --stylize)", 100), 100, 0, 1000)
        mj_chaos = safe_int(
            kwargs.get("mj_chaos (MJ混乱度 --chaos)", 0), 0, 0, 100)
        mj_weird = safe_int(
            kwargs.get("mj_weird (MJ怪异度 --weird)", 0), 0, 0, 3000)
        mj_no = str(kwargs.get("mj_no (MJ排除内容 --no)", "")).strip()

        unique_id = kwargs.get("unique_id")
        start_ts = time.time()

        # -- validate ---------------------------------------------------------
        if not api_key.strip():
            emit_runtime_status(unique_id, "error", "API Key 为空",
                                0.0, 0, retry_times, timeout_seconds)
            raise ValueError("API Key 不能为空")

        clean_prompt = normalize_prompt_text(prompt)
        if not clean_prompt:
            raise ValueError("prompt 不能为空")

        # Normalize optional inputs
        clean_negative = normalize_prompt_text(negative_prompt)

        # Collect reference images (image_01 through image_08)
        image_data_urls = self._collect_images(kwargs)

        # -- resolve model (auto-detect if placeholder) ----------------------
        try:
            model, model_list = resolve_and_validate_model(
                model, api_base, api_key.strip(), unique_id,
                placeholder="从接口自动获取模型列表",
                filter_func=_filter_image_models,
            )
        except ValueError as exc:
            emit_runtime_status(unique_id, "error", str(exc),
                                0.0, 0, retry_times, timeout_seconds)
            raise

        # -- detect model family --------------------------------------------
        model_type = self._detect_model_type(model)

        # -- validate params against model capabilities ---------------------
        try:
            size, n_images = self._validate_for_model(
                model_type, size, n_images)
        except ValueError as exc:
            emit_runtime_status(unique_id, "error", str(exc),
                                0.0, 0, retry_times, timeout_seconds)
            raise

        # -- prepare request --------------------------------------------------
        headers = {"Authorization": f"Bearer {api_key.strip()}"}
        payload = self._build_payload(
            model, model_type, clean_prompt, clean_negative, n_images,
            size, quality, style, response_format,
            seed, steps, cfg_scale, sampler,
            denoising_strength, mj_ar, mj_stylize,
            mj_chaos, mj_weird, mj_no,
            image_data_urls=image_data_urls,
        )

        print(
            f"[Comfyui-ZhangyuAPI-通用] model={model} "
            f"(type={model_type}), n={n_images}, "
            f"size={size}, quality={quality}, style={style}, "
            f"response_format={response_format}, seed={seed}, "
            f"steps={steps}, cfg_scale={cfg_scale}, sampler={sampler}, "
            f"denoising_strength={denoising_strength}"
            + (f", ref_images={len(image_data_urls)}" if image_data_urls else "")
            + (f", mj_ar={mj_ar}, mj_stylize={mj_stylize}, "
               f"mj_chaos={mj_chaos}, mj_weird={mj_weird}, mj_no={mj_no!r}"
               if model_type == "midjourney" else "")
        )
        emit_runtime_status(unique_id, "running", "开始生成",
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
                    api_base, headers, payload, timeout_seconds,
                )

                # Parse error responses
                if response.status_code != 200:
                    try:
                        err_data = response.json()
                        err_msg = self._extract_error_message(err_data)
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

                # If API returned an async task, poll with adaptive intervals
                if is_async_task_response(data):
                    task_id = data.get("task_id") or data.get("id")
                    remaining = timeout_seconds - int(time.time() - start_ts)
                    emit_runtime_status(
                        unique_id, "running",
                        f"任务已提交 (id={task_id})，自适应轮询中",
                        time.time() - start_ts,
                        attempt, retry_times, timeout_seconds,
                    )

                    def _on_poll_tick(elapsed, poll_elapsed, interval):
                        emit_runtime_status(
                            unique_id, "running",
                            f"轮询任务中 · 间隔{interval:.0f}s · "
                            f"已等待{poll_elapsed:.0f}s",
                            elapsed,
                            attempt, retry_times, timeout_seconds,
                        )

                    data = _poll_async_task(
                        api_base, headers, task_id, remaining, retry_times,
                        on_tick=_on_poll_tick,
                    )

                # Parse images from the (possibly polled) response
                image_tensor, image_urls = _parse_response_images(
                    data, timeout_seconds, error_prefix="通用生图")
                elapsed = time.time() - start_ts

                response_info = {
                    "status": "success",
                    "api_base": api_base,
                    "model": model,
                    "model_type": model_type,
                    "n": n_images,
                    "size": size,
                    "quality": quality,
                    "style": style,
                    "response_format": response_format,
                    "negative_prompt": clean_negative or None,
                    "seed": seed,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "sampler": sampler,
                    "denoising_strength": (
                        denoising_strength
                        if denoising_strength < 0.999 else None
                    ),
                    "mj_params": (
                        {
                            "ar": mj_ar,
                            "stylize": mj_stylize,
                            "chaos": mj_chaos,
                            "weird": mj_weird,
                            "no": mj_no or None,
                        }
                        if model_type == "midjourney" else None
                    ),
                    "payload": _strip_image_data(payload),
                    "input_images": len(image_data_urls),
                    "output_images": int(image_tensor.shape[0]),
                    "image_urls": image_urls,
                    "usage": data.get("usage"),
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
                    json.dumps(image_urls, ensure_ascii=False),
                    json.dumps(_sanitize_api_response(data), ensure_ascii=False, indent=2),
                    json.dumps(model_list, ensure_ascii=False),
                )

            except _RETRYABLE_EXCEPTIONS as exc:
                last_error = str(exc)
                if attempt < retry_times:
                    emit_runtime_status(
                        unique_id, "running",
                        f"网络/代理/超时，重试中 ({attempt}/{retry_times})",
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
            f"Comfyui-ZhangyuAPI-通用 连续 {retry_times} 次失败，"
            f"最后错误: {last_error}"
        )


# ===================================================================
# ComfyUI node registration
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ComfyuiZhangyuAPIUniversalImageNode": ComfyuiZhangyuAPIUniversalImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiZhangyuAPIUniversalImageNode": "ComfyUI-zhangyuapi-通用生图接口",
}
