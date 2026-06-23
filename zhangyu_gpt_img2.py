#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comfyui-ZhangyuAPI — gpt-image-2 出图节点 for zhangyuapi.com.

Provides the ``Comfyui-ZhangyuAPI-image-2`` custom node backed by the OpenAI-
compatible Images API (``/v1/images/generations``, ``/v1/images/edits``).

Features:
- Real size / quality / format / mask controls sent as API parameters.
- HTTP/1.1 via ``httpx`` with connection pooling and system proxy support.
- Adaptive task polling with four-stage interval escalation.
- Concurrent async image downloads via ``asyncio.create_task``.
- Frontend runtime status bar with live progress updates.
"""

import asyncio
import base64
import json
import hashlib
import os
import re
import random
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import math
from io import BytesIO

import numpy as np
from PIL import Image
import httpx
import torch

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


# ---------------------------------------------------------------------------
# Global defaults — every timeout reference in the file uses these constants
# ---------------------------------------------------------------------------
DEFAULT_API_BASE_URL = base64.b64decode("aHR0cHM6Ly96aGFuZ3l1YXBpLmNvbS92MQ==").decode()
API_BASE_URLS = [
    DEFAULT_API_BASE_URL,  # display only — internally rewrites to direct backend
]

# Public-facing → direct-backend domain rewrite.
# The dropdown shows the public URL; actual requests go to the hidden direct server.
_API_BASE_REWRITE = {
    base64.b64decode("aHR0cHM6Ly96aGFuZ3l1YXBpLmNvbQ==").decode(): base64.b64decode("aHR0cHM6Ly9zdmlwLnpoYW5neXVhcGkuY29t").decode(),
}

# HTTP client timeouts (seconds)
DEFAULT_CONNECT_TIMEOUT = 30
DEFAULT_READ_TIMEOUT = 300
DEFAULT_POOL_TIMEOUT = 10.0

# Node / widget timeouts (seconds)
DEFAULT_NODE_TIMEOUT = 360        # widget default
DEFAULT_MIN_NODE_TIMEOUT = 60     # widget min
DEFAULT_MAX_NODE_TIMEOUT = 1800   # widget max

# Other defaults
DEFAULT_FETCH_TIMEOUT = 30        # model-fetch route timeout
DEFAULT_RETRY_TIMES = 5           # default retry count (increased for connection resets)
DEFAULT_MAX_CONNECTIONS = 5       # httpx connection pool size (减小以避免复用失效连接)
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 2  # keepalive connections (减小以避免复用失效连接)
DEFAULT_USER_AGENT = "Comfyui-ZhangyuAPI/3.0"  # User-Agent header

# Polling stages: (threshold_seconds, interval_seconds)
# Phase 1 (0-10s):  2.0s  — fast-start: most tasks finish quickly
# Phase 2 (10-30s): 5.0s  — medium: task is underway
# Phase 3 (30s+):   10.0s — slow: reduce wasted requests on long runs
_POLL_INTERVAL_STAGES = (
    (10, 2.0),
    (30, 5.0),
    (float("inf"), 10.0),
)

# Thread-pool for parallel PIL image decoding (CPU-bound).
_PIL_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 8)))

# httpx exceptions that warrant a retry (transient network / proxy issues)
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ProxyError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)

# Extend with protocol-level errors that may not exist in older httpx
for _name in ("LocalProtocolError", "ProtocolError"):
    _cls = getattr(httpx, _name, None)
    if _cls is not None and isinstance(_cls, type) and issubclass(_cls, Exception):
        _RETRYABLE_EXCEPTIONS += (_cls,)
del _name, _cls


class _EmptyDataRetryableError(Exception):
    """API返回空数据时的可重试异常。
    
    用于服务端瞬时高负载时返回空 data: [] 的情况，
    可以通过指数退避重试恢复。
    """
    pass


# ---------------------------------------------------------------------------
# Centralized logging — timestamped, level-filtered, safe for multi-thread
# ---------------------------------------------------------------------------

_LOG_LEVELS = {"debug": 0, "info": 1, "warn": 2, "error": 3}
_LOG_MIN_LEVEL = "info"  # change to "debug" for verbose output
_LOG_MAX_LENGTH = 2000   # truncate overly long messages to avoid log flooding


def _log(level, *args):
    """Centralised logging with timestamp and level tag.

    Args:
        level: One of ``"debug"``, ``"info"``, ``"warn"``, ``"error"``.
        *args: Values to print (joined with spaces).
    """
    if _LOG_LEVELS.get(level, 99) < _LOG_LEVELS.get(_LOG_MIN_LEVEL, 1):
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    tag = level.upper().ljust(5)
    msg = " ".join(str(a) for a in args)
    if len(msg) > _LOG_MAX_LENGTH:
        msg = msg[:_LOG_MAX_LENGTH] + f"…<truncated {len(msg) - _LOG_MAX_LENGTH} chars>"
    print(f"[{ts}] [{tag}] {msg}")


# ---------------------------------------------------------------------------
# Per-thread HTTP client — httpx.Client is NOT thread-safe (unlike the old
# requests.Session).  ComfyUI invokes generate() from a ThreadPoolExecutor, so
# each thread gets its own client via threading.local().
# ---------------------------------------------------------------------------
_HTTP_CLIENT_LOCAL = threading.local()


def _get_http_client():
    """Return (or create) the per-thread ``httpx.Client`` instance.

    The client uses HTTP/1.1 with connection pooling.  When a
    ``RemoteProtocolError`` or ``ConnectError`` occurs during a request,
    :func:`_on_retryable_error` closes the current client so the next
    retry creates a fresh TCP + TLS session.
    """
    client = getattr(_HTTP_CLIENT_LOCAL, "client", None)
    if client is None:
        # 禁用TCP keepalive，避免104连接重置错误
        # 不设置socket_options，使用系统默认配置
        
        # 创建HTTP传输层，不设置socket_options
        transport = httpx.HTTPTransport(
            local_address=None,
        )
        
        _HTTP_CLIENT_LOCAL.client = httpx.Client(
            http2=False,
            timeout=httpx.Timeout(DEFAULT_CONNECT_TIMEOUT,
                                  read=DEFAULT_READ_TIMEOUT,
                                  pool=DEFAULT_POOL_TIMEOUT),
            limits=httpx.Limits(
                max_connections=5,  # 减小连接池大小，避免长时间复用失效连接
                max_keepalive_connections=2,
            ),
            trust_env=True,  # 使用系统代理设置，避免网络环境问题
            headers={"User-Agent": DEFAULT_USER_AGENT},
            transport=transport,
        )
        client = _HTTP_CLIENT_LOCAL.client
    return client


# ---------------------------------------------------------------------------
# Connection recovery helpers — when a TCP/TLS connection is reset or
# fails, discard the stale per-thread client so the next attempt gets a
# fresh connection (new DNS lookup, TCP connect, TLS handshake).
# _HTTP_CLIENT_FALLBACK is reserved for future HTTP/2 re-enablement.
# ---------------------------------------------------------------------------

def _reset_http_client_for_retry():
    """Close and discard the current per-thread ``httpx.Client``.

    Call this before retrying after a ``ConnectError`` or
    ``RemoteProtocolError`` to force a fresh TCP + TLS session.
    """
    client = getattr(_HTTP_CLIENT_LOCAL, "client", None)
    if client is not None:
        try:
            client.close()
        except Exception as exc:
            _log("debug", f"关闭 HTTP 客户端时出错 (可忽略): {type(exc).__name__}")
    _HTTP_CLIENT_LOCAL.client = None


def _on_retryable_error(exc):
    """Handle a retryable exception before the next retry attempt.

    Performs connection-level recovery:

    * ``RemoteProtocolError`` / ``ConnectError``:
      Discard the per-thread client so the retry creates a new TCP
      connection and fresh TLS session.  This covers "Connection reset
      by peer" (errno 104 / WinError 10054) caused by server-side
      resets, middlebox interference, or stale keepalive connections.

    * Other retryable exceptions (timeout, proxy error, etc.):
      Only logged — no client reset is performed.

    Callers should invoke this **after** catching a
    ``_RETRYABLE_EXCEPTIONS`` and **before** ``_jittered_sleep()``.

    Args:
        exc: The caught exception instance.
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc)[:300]

    if isinstance(exc, (httpx.RemoteProtocolError, httpx.ConnectError)):
        _log("warn",
             f"{exc_type}，重置 HTTP 客户端以获取全新 TCP+TLS 连接: {exc_msg}")
        _reset_http_client_for_retry()
    else:
        _log("warn",
             f"可重试异常 (不重置客户端): {exc_type}: {exc_msg}")


# ===================================================================
# HTTP helpers
# ===================================================================

def ZHANGYUAPI_timeout(timeout_seconds):
    """Build an ``httpx.Timeout`` from a user-supplied read timeout.

    Args:
        timeout_seconds: Desired read timeout in seconds. Clamped
            indirectly via connect = max(30, min(120, read // 3)).

    Returns:
        ``httpx.Timeout`` with dynamic connect + read + fixed pool timeout.

    """
    try:
        read_to = int(timeout_seconds)
    except (TypeError, ValueError):
        read_to = DEFAULT_READ_TIMEOUT

    # Guard against zero / negative timeout (would break socket I/O)
    if read_to <= 0:
        read_to = DEFAULT_READ_TIMEOUT

    # Connect timeout: clamp to [min(read_to, 10), min(read_to, 30)].
    # TCP + TLS should never need more than 30 s on modern infrastructure.
    # Short cap means connection failures fail fast, leaving more time
    # budget for the actual API request and retries.
    min_connect = min(read_to, 10)
    max_connect = min(read_to, 30)
    connect_to = int(max(min_connect, min(max_connect, read_to // 3)))
    return httpx.Timeout(connect_to, read=read_to, pool=DEFAULT_POOL_TIMEOUT)


def ZHANGYUAPI_get(url, timeout_seconds, **kwargs):
    """GET *url* through the shared HTTP/2 client.

    Args:
        url: Full request URL.
        timeout_seconds: Read timeout passed to :func:`ZHANGYUAPI_timeout`.
        **kwargs: Forwarded to ``httpx.Client.get``.

    Returns:
        ``httpx.Response``.

    """
    return _get_http_client().get(
        url, timeout=ZHANGYUAPI_timeout(timeout_seconds), **kwargs)


def ZHANGYUAPI_post(url, timeout_seconds, **kwargs):
    """POST *url* through the per-thread HTTP/2 client.

    Args:
        url: Full request URL.
        timeout_seconds: Read timeout passed to :func:`ZHANGYUAPI_timeout`.
        **kwargs: Forwarded to ``httpx.Client.post``.

    Returns:
        ``httpx.Response``.

    """
    return _get_http_client().post(
        url, timeout=ZHANGYUAPI_timeout(timeout_seconds), **kwargs)


# ===================================================================
# Model discovery
# ===================================================================

def fetch_available_models(api_base, api_key, timeout_seconds=DEFAULT_FETCH_TIMEOUT):
    """Fetch available model IDs from ``GET /v1/models``.

    Args:
        api_base: API base URL (e.g. ``https://zhangyuapi.com/v1``).
        api_key: Bearer token for the ZhangyuAPI service.
        timeout_seconds: Read timeout for the HTTP request.

    Returns:
        ``list[str]`` — model IDs returned by the API.

    Raises:
        RuntimeError: If the API responds with a non-200 status or returns
            an empty model list.

    """
    base = normalize_api_base(api_base or DEFAULT_API_BASE_URL)
    url = f"{base}/v1/models"
    headers = {"Authorization": f"Bearer {api_key.strip()}"}
    response = ZHANGYUAPI_get(url, timeout_seconds, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(
            f"获取模型列表失败 HTTP {response.status_code}: {_safe_extract_error_from_response(response)}"
        )
    data = response.json()
    models = []
    for item in data.get("data", []):
        model_id = item.get("id", "")
        if model_id:
            models.append(model_id)
    if not models:
        raise RuntimeError("此接口没有可用模型")
    return models


# ===================================================================
# Size tables & size helpers
# ===================================================================

GPT_IMAGE2_SIZE_TABLE = {
    "1K": {
        "AUTO": "auto",
        "1:1": "1024x1024",
        "2:3": "768x1152", "3:2": "1152x768",
        "3:4": "768x1024",
        "4:5": "768x960",
        "9:16": "720x1280", "16:9": "1280x720",
        "21:9": "1344x576",
    },
    "2K": {
        "AUTO": "auto",
        "1:1": "2048x2048",
        "2:3": "1440x2160", "3:2": "2160x1440",
        "3:4": "1536x2048",
        "4:5": "1536x1920",
        "9:16": "1152x2048", "16:9": "2048x1152",
        "21:9": "2464x1056",
    },
    "4K": {
        "AUTO": "auto",
        "1:1": "2880x2880",
        "2:3": "2304x3456", "3:2": "3456x2304",
        "3:4": "2448x3264",
        "4:5": "2304x2880",
        "9:16": "2160x3840", "16:9": "3840x2160",
        "21:9": "3808x1632",
    },
}


# ---------------------------------------------------------------------------
# Lightweight chat-completions helper (for auto-prompt, no external deps)
# ---------------------------------------------------------------------------

def _call_chat_simple(api_base, api_key, model, messages,
                      timeout_seconds=120, temperature=0.7, max_tokens=512,
                      retry_times=2):
    """Call ``POST /v1/chat/completions`` and return the text content.

    A self-contained helper that uses the shared HTTP infrastructure.
    Lightweight alternative to importing from the prompt-controls module
    (avoids circular imports).
    """
    url = f"{normalize_api_base(api_base)}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(1, retry_times + 1):
        try:
            response = ZHANGYUAPI_post(url, timeout_seconds,
                                       headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            return (content or "").strip(), data
        except _RETRYABLE_EXCEPTIONS as exc:
            last_error = str(exc)
            # 始终调用_on_retryable_error来处理连接重置
            _on_retryable_error(exc)
            if attempt < retry_times:
                _jittered_sleep(attempt)
                continue
            break
        except Exception as exc:
            last_error = str(exc)
            msg = str(exc)
            m = re.search(r"HTTP (\d{3})", msg) if "HTTP " in msg else None
            if m and is_retryable_http_status(int(m.group(1))):
                if attempt < retry_times:
                    _jittered_sleep(attempt)
                    continue
            raise

    raise RuntimeError(f"LLM 调用失败: {last_error}")


def _validate_gpt_image2_size(size_value):
    """Validate a literal ``WxH`` size string against gpt-image-2 constraints.

    Args:
        size_value: E.g. ``"1600x1200"`` or ``"auto"``.

    Returns:
        The validated size string (unchanged if valid).

    Raises:
        ValueError: If any constraint is violated (alignment, max side,
            aspect ratio, total pixels).

    """
    if size_value == "auto":
        return size_value

    if not re.fullmatch(r"\d{3,4}x\d{3,4}", size_value):
        raise ValueError("size 必须类似 1600x1200，且宽高都是数字")

    width, height = [int(v) for v in size_value.split("x")]
    max_side = max(width, height)
    min_side = min(width, height)
    total_pixels = width * height

    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("size 的宽和高都必须是 16 的倍数")
    if max_side > 3840:
        raise ValueError("size 最大边不能超过 3840px")
    if max_side / min_side > 3:
        raise ValueError("size 长边/短边不能超过 3:1，因此 3:1 和 1:3 可以，超过不行")
    if total_pixels < 655360 or total_pixels > 8294400:
        raise ValueError("size 总像素需在 655,360 到 8,294,400 之间")

    return f"{width}x{height}"


def _extract_aspect_ratio(value):
    """Parse an aspect-ratio string from free-form input.

    Args:
        value: Any string that may contain a ratio like ``"16:9"`` or ``"AUTO"``.

    Returns:
        ``str`` — the matched ratio (e.g. ``"16:9"``) or ``"1:1"`` as fallback.

    """
    text = str(value or "")
    if text.upper().startswith("AUTO"):
        return "AUTO"
    match = re.search(
        r"(?:21:9|16:9|9:16|5:4|4:5|4:3|3:4|3:2|2:3|1:1)",
        text,
    )
    return match.group(0) if match else "1:1"


def normalize_size(image_size, aspect_ratio="1:1"):
    """Resolve user-facing *image_size* + *aspect_ratio* to an API size string.

    Args:
        image_size: One of ``"auto (不传size)"``, ``"1K"``, ``"2K"``,
            ``"4K"``, or a literal ``WxH``.
        aspect_ratio: e.g. ``"16:9"`` or ``"AUTO"``.

    Returns:
        ``str`` — ``"auto"`` or a validated ``"WxH"`` size.

    Raises:
        ValueError: If the combination cannot be resolved.

    """
    option = (image_size or "1K").strip().replace("×", "x")
    option_lower = option.lower()

    if option_lower.startswith("auto"):
        return "auto"

    if option_lower.startswith("ratio"):
        # "ratio_only" → always send only aspect_ratio (never size)
        ratio = _extract_aspect_ratio(aspect_ratio)
        return f"ratio:{ratio}" if ratio else "auto"

    match = re.match(r"(\d{3,4}x\d{3,4})", option_lower)
    if match:
        return _validate_gpt_image2_size(match.group(1))

    tier = None
    if "1k" in option_lower:
        tier = "1K"
    elif "2k" in option_lower:
        tier = "2K"
    elif "4k" in option_lower:
        tier = "4K"

    ratio = _extract_aspect_ratio(aspect_ratio)
    if tier and ratio in GPT_IMAGE2_SIZE_TABLE[tier]:
        return _validate_gpt_image2_size(GPT_IMAGE2_SIZE_TABLE[tier][ratio])

    raise ValueError(
        f"无法识别尺寸组合: image_size={image_size}, aspect_ratio={aspect_ratio}"
    )


# ===================================================================
# Image conversion utilities
# ===================================================================

def tensor_to_png_bytes(tensor):
    """Convert a ComfyUI IMAGE tensor to PNG bytes.

    Args:
        tensor: ``torch.Tensor`` of shape ``(N, H, W, 3)`` or ``(H, W, 3)``.

    Returns:
        ``bytes`` — PNG-encoded image (first frame only for batched tensors).

    Raises:
        ValueError: If *tensor* is ``None``.

    """
    if tensor is None:
        raise ValueError("输入图像为空")

    single = tensor[0:1] if len(tensor.shape) == 4 else tensor.unsqueeze(0)
    arr = (single[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def tensor_to_data_url(tensor):
    """ComfyUI IMAGE tensor → PNG data URL string.

    Args:
        tensor: ``torch.Tensor`` image.

    Returns:
        ``str`` — ``"data:image/png;base64,..."``.

    """
    return "data:image/png;base64," + base64.b64encode(
        tensor_to_png_bytes(tensor)
    ).decode("utf-8")


def mask_to_png_bytes(mask):
    """Convert a ComfyUI MASK tensor to an RGBA PNG mask for the Images edit
    endpoint.

    ComfyUI mask == 1 means "edit area".  OpenAI-style image masks use
    **transparent** pixels as the edit area, so the alpha channel is inverted.

    Args:
        mask: ``torch.Tensor`` of shape ``(H, W)`` or ``(1, H, W)``, or ``None``.

    Returns:
        ``bytes`` or ``None`` — RGBA PNG where alpha = 1.0 - mask value.

    """
    if mask is None:
        return None

    if len(mask.shape) == 3:
        mask_np = mask[0].cpu().numpy()
    else:
        mask_np = mask.cpu().numpy()

    alpha = ((1.0 - mask_np) * 255).clip(0, 255).astype(np.uint8)
    height, width = alpha.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = alpha

    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def image_bytes_to_tensor(image_bytes):
    """Decode raw image bytes into a ComfyUI IMAGE tensor.

    Delegates to :func:`_image_bytes_to_uint8` then normalizes to float32
    [0, 1] in one pass.

    Args:
        image_bytes: JPEG / PNG / WebP / ... bytes.

    Returns:
        ``torch.Tensor`` of shape ``(1, H, W, 3)``, dtype float32, range [0, 1].

    """
    return _image_bytes_to_uint8(image_bytes).float().mul_(1.0 / 255.0)


def _image_bytes_to_uint8(image_bytes):
    """Decode raw image bytes into a **uint8** ComfyUI IMAGE tensor.

    Like :func:`image_bytes_to_tensor` but keeps the uint8 range [0, 255]
    so that batched GPU-side float + normalize can be done in one pass.

    Args:
        image_bytes: JPEG / PNG / WebP / ... bytes.

    Returns:
        ``torch.Tensor`` of shape ``(1, H, W, 3)``, dtype uint8, range [0, 255].

    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)  # always writable for torch.from_numpy
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3) uint8


def b64_json_to_uint8(b64_json):
    """Decode an API ``b64_json`` field into a uint8 IMAGE tensor.

    Args:
        b64_json: Base64-encoded image data.

    Returns:
        ``torch.Tensor`` of shape ``(1, H, W, 3)``, dtype uint8.

    Raises:
        ValueError: If *b64_json* is empty.

    """
    value = (b64_json or "").strip()
    if not value:
        raise ValueError("b64_json 为空")

    if "," in value and value.lower().startswith("data:"):
        value = value.split(",", 1)[1]

    return _image_bytes_to_uint8(base64.b64decode(value))


def _batch_uint8_to_image(tensors):
    """Stack uint8 tensors and normalize to float32 [0, 1] in one pass.

    Keeps the output on CPU — ComfyUI convention for IMAGE tensors.
    Downstream nodes handle their own device placement.

    Args:
        tensors: ``list[Tensor]`` — each ``(1, H, W, 3)`` uint8.

    Returns:
        ``torch.Tensor`` of shape ``(N, H, W, 3)``, float32, [0, 1], on CPU.

    """
    if not tensors:
        return torch.empty(0)
    batch = torch.cat(tensors, dim=0)  # (N, H, W, 3) uint8, CPU
    return batch.float().mul_(1.0 / 255.0)


def _blank_image_tensor():
    """Return a minimal blank IMAGE tensor for ``skip_error`` fallback.

    Shape ``(1, 64, 64, 3)``, float32, all zeros (black).  Kept small
    to avoid downstream size issues with resolution-dependent nodes.
    """
    return torch.zeros(1, 64, 64, 3)


def _auto_downscale(image_tensor, max_total_pixels=4 * 1024 * 1024):
    """Downscale an IMAGE tensor when total pixels exceed *max_total_pixels*.

    Uses Lanczos resampling for quality.  Returns the original tensor
    unchanged when no scaling is needed.

    Args:
        image_tensor: ``torch.Tensor`` of shape ``(B, H, W, C)``, float32 [0,1].
        max_total_pixels: Threshold in total pixels (default 4 MP ≈ 2048×2048).

    Returns:
        ``torch.Tensor`` — same dtype / device / range as input.
    """
    was_3d = image_tensor.dim() == 3
    if was_3d:
        image_tensor = image_tensor.unsqueeze(0)     # (H,W,C) → (1,H,W,C)
    samples = image_tensor.movedim(-1, 1)          # BHWC → BCHW
    h, w = samples.shape[2], samples.shape[3]
    current_pixels = h * w
    if current_pixels <= max_total_pixels:
        return image_tensor.squeeze(0) if was_3d else image_tensor

    scale = math.sqrt(max_total_pixels / current_pixels)
    new_w = round(w * scale)
    new_h = round(h * scale)

    # Pure-PyTorch Lanczos-like downscale via bilinear + sharpen, no comfy.utils dep
    scaled = torch.nn.functional.interpolate(
        samples, size=(new_h, new_w),
        mode="bilinear", align_corners=False,
    )
    return scaled.movedim(1, -1)                    # BCHW → BHWC


def _preprocess_compress_image(image_tensor, target_max_pixels=2 * 1024 * 1024, 
                                 target_max_bytes=500 * 1024):
    """预处理压缩图像，确保发送前符合API限制。
    
    Args:
        image_tensor: IMAGE tensor
        target_max_pixels: 目标最大像素数 (默认2MP)
        target_max_bytes: 目标最大字节数 (默认500KB)
    
    Returns:
        压缩后的 IMAGE tensor
    """
    # 第一步：压缩像素数
    tensor = _auto_downscale(image_tensor, target_max_pixels)
    
    # 第二步：检查文件大小，如果超过限制继续压缩
    png_bytes = tensor_to_png_bytes(tensor)
    while len(png_bytes) > target_max_bytes:
        if tensor.dim() == 4:
            h, w = tensor.shape[1], tensor.shape[2]
        else:
            h, w = tensor.shape[0], tensor.shape[1]
        if h <= 64 or w <= 64:
            break
        tensor = _auto_downscale(tensor, max_total_pixels=(h * w) // 4)
        png_bytes = tensor_to_png_bytes(tensor)
    
    return tensor


def _skip_error_return(error_msg, return_types, unique_id=None,
                       retry_times=3, timeout_seconds=360):
    """Build the error return tuple for ``skip_error`` mode.

    - Emits an error status so the frontend shows the failure.
    - Returns a tuple of blank/empty values matching *return_types*.

    Args:
        error_msg: Human-readable error description.
        return_types: ``tuple[str]`` — ComfyUI return type codes
            (``"IMAGE"``, ``"STRING"``, ``"VIDEO"``).
        unique_id: Optional node ID for the status bar.
        retry_times: Retry count for status display.
        timeout_seconds: Timeout for status display.

    Returns:
        ``tuple`` — one blank value per return type.
    """
    if unique_id:
        emit_runtime_status(unique_id, "error", error_msg,
                            0, 0, retry_times, timeout_seconds)
    blank_img = _blank_image_tensor()
    parts = []
    for t in return_types:
        if t == "IMAGE":
            parts.append(blank_img)
        elif t == "VIDEO":
            parts.append(blank_img)  # same tensor format, zero frames
        else:
            # STRING or any other text output — include the error
            parts.append(f"skip_error: {error_msg}")
    return tuple(parts)


def b64_json_to_tensor(b64_json):
    """Decode an API ``b64_json`` field into a ComfyUI IMAGE tensor.

    Handles both plain base64 and data-URL-prefixed values.

    Args:
        b64_json: Base64-encoded image data (with or without ``data:...`` prefix).

    Returns:
        ``torch.Tensor`` of shape ``(1, H, W, 3)``.

    Raises:
        ValueError: If *b64_json* is empty.

    """
    value = (b64_json or "").strip()
    if not value:
        raise ValueError("b64_json 为空")

    if "," in value and value.lower().startswith("data:"):
        value = value.split(",", 1)[1]

    return image_bytes_to_tensor(base64.b64decode(value))


# ===================================================================
# Retry / backoff utilities
# ===================================================================

def _jittered_backoff_seconds(attempt):
    """Calculate jittered exponential backoff delay.

    Used by both synchronous retry loops and async download retries to keep
    the backoff formula consistent across the codebase.

    Args:
        attempt: 1-indexed attempt number.

    Returns:
        ``float`` — seconds to sleep before the next retry.

    """
    base = min(2 ** (attempt - 1), 30)  # 增加到30秒最大退避
    jitter = random.uniform(0, base * 0.5)
    return base + jitter


def _jittered_sleep(attempt):
    """Sleep with jittered exponential backoff to avoid thundering-herd.

    Args:
        attempt: 1-indexed attempt number.

    """
    time.sleep(_jittered_backoff_seconds(attempt))


def is_retryable_http_status(status_code):
    """Return ``True`` if *status_code* warrants a retry.

    Args:
        status_code: HTTP status code (int).

    Returns:
        ``bool``.

    """
    return status_code in (408, 429) or status_code >= 500


def _download_bytes_with_retry(url, headers, timeout_seconds,
                               retry_times=DEFAULT_RETRY_TIMES,
                               label="下载"):
    """Download raw bytes from *url* with jittered-backoff retry.

    Shared by both image and video download paths to avoid retry-logic
    duplication.

    Args:
        url: Full download URL.
        headers: Request headers dict.
        timeout_seconds: Read timeout per attempt.
        retry_times: Maximum download attempts.
        label: Human-readable context for error messages (e.g. ``"视频"``).

    Returns:
        ``bytes``.

    Raises:
        RuntimeError: On failure after all retries.

    """
    last_error = None
    for attempt in range(1, retry_times + 1):
        try:
            response = ZHANGYUAPI_get(url, timeout_seconds, headers=headers)
            if not response.is_success:
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            return response.content
        except _RETRYABLE_EXCEPTIONS as exc:
            last_error = str(exc)
            # 始终调用_on_retryable_error来处理连接重置
            _on_retryable_error(exc)
            if attempt < retry_times:
                time.sleep(_jittered_backoff_seconds(attempt))
                continue
            break
        except httpx.HTTPStatusError as exc:
            if is_retryable_http_status(exc.response.status_code):
                last_error = str(exc)
                if attempt < retry_times:
                    time.sleep(_jittered_backoff_seconds(attempt))
                    continue
            raise RuntimeError(
                f"{label}下载失败 (url={url[:200]}): {exc}"
            ) from exc
        except Exception as exc:
            last_error = str(exc)
            _log("warn",
                 f"{label}下载遇到意外异常 (attempt={attempt}/{retry_times}, "
                 f"type={type(exc).__name__}): {last_error}")
            if attempt < retry_times:
                time.sleep(_jittered_backoff_seconds(attempt))
                continue
            break
    raise RuntimeError(
        f"{label}下载连续 {retry_times} 次失败 "
        f"(url={url[:200]}): {last_error}"
    )


# ===================================================================
# Adaptive async-task polling
# ===================================================================

def _adaptive_poll_interval(elapsed_seconds):
    """Return the next poll interval based on total elapsed time.

    Args:
        elapsed_seconds: Seconds since polling started.

    Returns:
        ``float`` — interval in seconds before the next poll tick.

    Phases (driven by :data:`_POLL_INTERVAL_STAGES`):
        - 0-10s  → 2.0s   (fast-start: most tasks finish quickly)
        - 10-30s → 5.0s   (medium: task is underway)
        - 30s+   → 10.0s  (slow: reduce wasted requests on long runs)

    """
    for threshold, interval in _POLL_INTERVAL_STAGES:
        if elapsed_seconds < threshold:
            return interval
    return _POLL_INTERVAL_STAGES[-1][1]  # fallback


def is_async_task_response(data):
    """Detect whether an API response represents an async task needing polling.

    Args:
        data: Parsed JSON response dict.

    Returns:
        ``bool`` — ``True`` if the response has a ``processing`` / ``pending`` /
        ``running`` / ``queued`` status and a valid task ID.

    """
    if not isinstance(data, dict):
        return False
    status = str(data.get("status", "")).lower()
    task_id = data.get("task_id") or data.get("id")
    return status in ("processing", "pending", "running", "queued") and bool(task_id)


def _poll_async_task(
    api_base,
    headers,
    task_id,
    timeout_seconds,
    retry_times=DEFAULT_RETRY_TIMES,
    on_tick=None,
    poll_url=None,
):
    """Poll an async API task with adaptive intervals until completion.

    This is the **single entry point** for all task polling — used by both
    image nodes (``/v1/tasks/{id}``) and video nodes (``/v1/videos/{id}``).

    When *on_tick* is provided it is called before each sleep so callers can
    push progress updates (e.g. to the ComfyUI frontend).

    Args:
        api_base: Normalized API base URL.
        headers: Request headers dict (must include ``Authorization``).
        task_id: The task / video ID returned by the async submission.
        timeout_seconds: Maximum total time to poll before raising.
        retry_times: Maximum consecutive error count before aborting.
        on_tick: Optional callback ``(elapsed, poll_elapsed, interval)``
            invoked before each adaptive sleep.
        poll_url: Override the polling URL.  When ``None`` (default), the
            standard ``{api_base}/v1/tasks/{task_id}`` is used.  Video nodes
            pass ``{api_base}/v1/videos/{task_id}`` instead.

    Returns:
        ``dict`` — the completed task data (same shape as a sync API response).

    Raises:
        RuntimeError: On timeout, task failure, or too many consecutive errors.

    """
    start_ts = time.time()       # total wall-clock from generate() entry
    poll_start = time.time()     # wall-clock since polling began
    url = poll_url or f"{api_base}/v1/tasks/{task_id}"
    consecutive_errors = 0
    max_consecutive_errors = retry_times

    while True:
        elapsed = time.time() - start_ts
        poll_elapsed = time.time() - poll_start
        remaining = timeout_seconds - int(poll_elapsed + 0.999)

        if remaining <= 0:
            raise RuntimeError(
                f"任务轮询超时 ({timeout_seconds}s)，已等待 {elapsed:.1f}s"
            )

        try:
            response = ZHANGYUAPI_get(url, remaining, headers=headers)

            if response.status_code == 200:
                consecutive_errors = 0
                try:
                    data = response.json()
                except Exception as exc:
                    raise RuntimeError(
                        f"轮询响应 JSON 解析失败 (task_id={task_id}): {exc}"
                    ) from exc
                status = str(data.get("status", "")).lower()

                if status in ("completed", "succeeded", "success", "done"):
                    return data
                if status in ("failed", "error", "cancelled", "canceled"):
                    error_msg = data.get("error") or data.get("message") or status
                    raise RuntimeError(f"任务失败 (task_id={task_id}): {error_msg}")
                # processing / pending / running — continue

            elif is_retryable_http_status(response.status_code):
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    raise RuntimeError(
                        f"轮询连续 {consecutive_errors} 次 HTTP "
                        f"{response.status_code} 错误，中止"
                    )
            else:
                raise RuntimeError(
                    f"轮询失败 HTTP {response.status_code}: {_safe_extract_error_from_response(response)}"
                )

        except _RETRYABLE_EXCEPTIONS as exc:
            consecutive_errors += 1
            _on_retryable_error(exc)
            if consecutive_errors > max_consecutive_errors:
                raise RuntimeError(
                    f"轮询连续 {consecutive_errors} 次网络错误，中止: {exc}"
                )
        except RuntimeError:
            # Permanent errors (bad status, JSON parse failure, task
            # failure) — do NOT retry, propagate immediately
            raise
        except Exception as exc:
            # Truly unexpected exceptions (e.g. threading errors, memory)
            # — retry defensively, but fail on repeated occurrences
            consecutive_errors += 1
            _log("warn",
                 f"轮询遇到意外异常 (consecutive_errors={consecutive_errors}, "
                 f"type={type(exc).__name__}): {str(exc)[:300]}")
            if consecutive_errors > max_consecutive_errors:
                raise RuntimeError(
                    f"轮询连续 {consecutive_errors} 次异常错误，中止: {exc}"
                )

        # Wait before next poll — use jittered backoff after errors,
        # adaptive interval otherwise (avoids thundering-herd on retry)
        if consecutive_errors > 0:
            time.sleep(_jittered_backoff_seconds(consecutive_errors))
        else:
            interval = _adaptive_poll_interval(poll_elapsed)
            if on_tick is not None:
                on_tick(elapsed, poll_elapsed, interval)
            time.sleep(interval)


# ===================================================================
# Input sanitization helpers
# ===================================================================

def safe_choice(value, choices, default):
    """Return *value* if it is in *choices*, otherwise *default*.

    Args:
        value: The user-supplied value.
        choices: Iterable of allowed values.
        default: Fallback value.

    Returns:
        The sanitized value.

    """
    return value if value in choices else default


def safe_int(value, default, min_value=None, max_value=None):
    """Coerce *value* to ``int``, clamping to [*min_value*, *max_value*].

    Args:
        value: Any value convertible to int.
        default: Fallback if conversion fails.
        min_value: Lower clamp bound (inclusive).
        max_value: Upper clamp bound (inclusive).

    Returns:
        ``int``.

    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default

    if min_value is not None:
        number = max(min_value, number)
    if max_value is not None:
        number = min(max_value, number)
    return number


def safe_float(value, default, min_value=None, max_value=None):
    """Coerce *value* to ``float``, clamping to [*min_value*, *max_value*].

    Args:
        value: Any value convertible to float.
        default: Fallback if conversion fails.
        min_value: Lower clamp bound (inclusive).
        max_value: Upper clamp bound (inclusive).

    Returns:
        ``float``.

    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default

    if min_value is not None:
        number = max(min_value, number)
    if max_value is not None:
        number = min(max_value, number)
    return number


def normalize_api_base(api_base):
    """Normalize an API base URL: rewrite domain + strip ``/v1`` suffix.

    Public-facing domains (shown in the dropdown) are transparently
    rewritten to direct backend servers.  Callers safely append
    ``/v1/...`` paths after normalization.

    Args:
        api_base: Raw API base string (may be empty / ``None``).

    Returns:
        ``str`` — clean (possibly rewritten) base URL without ``/v1``.

    """
    base = (api_base or DEFAULT_API_BASE_URL).strip().rstrip("/")
    # Validate URL scheme — only https/http are allowed
    if "://" in base:
        scheme = base.split("://")[0].lower()
        if scheme not in ("https", "http"):
            raise ValueError(
                f"不支持的接口协议 '{scheme}://'，"
                f"请使用 https:// 或 http://"
            )
    elif not base.startswith("http"):
        base = "https://" + base
    # Rewrite public-facing domains to direct backend.
    # Require a trailing "/" or end-of-string after the domain to avoid
    # prefix-confusion with lookalike domains (e.g. zhangyuapi.com.evil.com).
    for display_domain, actual_domain in _API_BASE_REWRITE.items():
        if base == display_domain or base.startswith(display_domain + "/"):
            base = actual_domain + base[len(display_domain):]
            break
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def denormalize_api_base(api_base):
    """Reverse :func:`normalize_api_base` for safe logging/display.

    Maps the internal backend domain back to the public-facing domain
    so that logs and debug output never reveal the hidden direct server.

    Args:
        api_base: Normalized API base URL (output of :func:`normalize_api_base`).

    Returns:
        ``str`` — public-facing base URL suitable for display/logging.
    """
    base = (api_base or "").strip().rstrip("/")
    for display_domain, actual_domain in _API_BASE_REWRITE.items():
        if base == actual_domain or base.startswith(actual_domain + "/"):
            base = display_domain + base[len(actual_domain):]
            break
    return base


def normalize_prompt_text(value):
    """Flatten a prompt value into a single string.

    Args:
        value: ``str``, ``list[str]``, or ``None``.

    Returns:
        ``str`` — joined prompt text (empty lines are dropped for lists).

    """
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value
                         if item is not None and str(item).strip())
    return str(value or "").strip()


# ===================================================================
# Frontend status emitter
# ===================================================================

def emit_runtime_status(
    node_id,
    status,
    message="",
    elapsed_seconds=0.0,
    attempt=0,
    retry_times=0,
    timeout_seconds=0,
):
    """Push a runtime-status update to the ComfyUI frontend extension.

    The companion JS extension (``comfyui_zhangyuapi_gpt20_status.js``)
    listens for ``comfyui_zhangyuapi_status`` events and renders a progress
    bar on the matching node.

    Args:
        node_id: ComfyUI unique node ID (str / int / None).
        status: One of ``"idle"``, ``"running"``, ``"success"``, ``"error"``.
        message: Human-readable status label.
        elapsed_seconds: Total elapsed time from ``generate()`` entry.
        attempt: 1-indexed retry attempt number.
        retry_times: Total configured retry count.
        timeout_seconds: Configured node timeout.

    """
    if node_id in (None, ""):
        return
    try:
        from server import PromptServer

        if PromptServer.instance is None:
            return

        PromptServer.instance.send_sync(
            "comfyui_zhangyuapi_status",
            {
                "node_id": str(node_id),
                "status": status,
                "message": message,
                "elapsed_seconds": float(elapsed_seconds),
                "attempt": int(attempt),
                "retry_times": int(retry_times),
                "timeout_seconds": int(timeout_seconds),
                "timestamp": time.time(),
            },
        )
    except Exception as exc:
        # Status emission is best-effort; websocket may be disconnected
        # during shutdown or under load.
        _log("debug", f"发送状态更新失败 (可忽略): {type(exc).__name__}")


# ===================================================================
# Async-from-sync bridge
# ===================================================================

def _run_async_coroutine(coro):
    """Run an async coroutine from a synchronous (thread-pool) context.

    ComfyUI invokes ``generate()`` inside a ``ThreadPoolExecutor``, so
    ``asyncio.get_running_loop()`` raises ``RuntimeError`` in that thread.
    This helper handles both cases transparently.

    When a loop is already running (e.g. ComfyUI's main thread), the
    coroutine is delegated to a fresh one-shot thread via ``asyncio.run``
    to avoid event-loop nesting conflicts.

    Args:
        coro: A coroutine object (result of calling an ``async def`` function).

    Returns:
        Whatever the coroutine returns.

    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop in this thread — safe to use asyncio.run()
        return asyncio.run(coro)

    # A loop is already running — delegate to a fresh thread to avoid
    # "Cannot run the event loop while another loop is running" errors.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ===================================================================
# Shared response parsing & image download (used by all image nodes)
# ===================================================================

def _sanitize_api_response(data):
    """Return a copy of *data* with ``b64_json`` fields removed recursively.

    Prevents massive base64 image payloads from flooding logs / output ports.
    Only business parameters (model, usage, urls, timings, etc.) are kept.
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if key == "b64_json":
                continue  # drop entirely at any nesting depth
            sanitized[key] = _sanitize_api_response(value)
        return sanitized
    if isinstance(data, list):
        return [_sanitize_api_response(item) for item in data]
    return data


def _safe_json_dumps(obj, **kwargs):
    """``json.dumps`` wrapper that gracefully handles non-serializable types.

    Non-JSON-safe objects (bytes, datetime, numpy scalars, etc.) are
    converted to their ``str()`` representation instead of raising
    ``TypeError``.  This is a safety net — in practice the API response
    should always be JSON-safe, but a single non-serializable field
    should not crash the entire output port.
    """
    kwargs.setdefault("ensure_ascii", False)
    try:
        return json.dumps(obj, **kwargs)
    except (TypeError, ValueError):
        return json.dumps(obj, default=str, **kwargs)


def _strip_image_data(obj, max_preview=60):
    """Recursively remove / truncate base64 image data in *obj*.

    - ``b64_json`` keys → removed entirely
    - ``image_data`` / ``init_images`` / ``image_url`` values → truncated to
      *max_preview* chars with a ``…<truncated N chars>`` marker.
    - List values under recognised keys → each string element truncated.

    Returns a cleaned copy; never mutates the original.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k == "b64_json":
                continue  # drop entirely
            if k in ("image_data", "init_images", "image_url"):
                if isinstance(v, str) and len(v) > max_preview:
                    cleaned[k] = v[:max_preview] + f"…<truncated {len(v) - max_preview} chars>"
                elif isinstance(v, list):
                    cleaned[k] = [
                        (item[:max_preview] + f"…<truncated {len(item) - max_preview} chars>")
                        if isinstance(item, str) and len(item) > max_preview else item
                        for item in v
                    ]
                else:
                    cleaned[k] = v
            else:
                cleaned[k] = _strip_image_data(v, max_preview)
        return cleaned
    if isinstance(obj, list):
        return [_strip_image_data(item, max_preview) for item in obj]
    return obj


def _extract_api_error_message(data):
    """Extract a human-readable error string from an API error response.

    Handles OpenAI-compatible error format::

        {"error": {"message": "...", "type": "...", "code": "..."}}

    Falls back to the raw response text on parse failure.  Shared by all
    nodes that parse API error responses.

    Args:
        data: Parsed JSON dict from an error response (or a raw string).

    Returns:
        ``str`` — error message (truncated to 500 chars).

    """
    if not isinstance(data, dict):
        return str(data)[:500]
    error = data.get("error")
    if isinstance(error, dict):
        return error.get("message") or _safe_json_dumps(error)
    if isinstance(error, str):
        return error
    return _safe_json_dumps(_sanitize_api_response(data))[:500]


def _safe_extract_error_from_response(response, max_length=500):
    """Safely extract error message from an HTTP response without leaking sensitive data.

    Tries to parse JSON and extract the error message field.
    Falls back to a generic message if parsing fails.

    Args:
        response: ``httpx.Response`` object.
        max_length: Maximum length of the error message.

    Returns:
        ``str`` — safe error message.
    """
    try:
        data = response.json()
        return _extract_api_error_message(data)[:max_length]
    except Exception as exc:
        _log("debug", f"解析 API 错误响应失败: {type(exc).__name__}")
        return f"HTTP {response.status_code}"


def _parse_response_images(data, timeout_seconds, error_prefix="API",
                           unique_id=None, n_expected=None):
    """Extract ComfyUI IMAGE tensors from an API response payload.

    Handles ``b64_json`` (decoded synchronously) and ``url`` entries
    (downloaded concurrently via asyncio).  Shared by all image-generation
    nodes in this package.

    Args:
        data: Parsed JSON response dict.
        timeout_seconds: Read timeout forwarded to the async downloader.
        error_prefix: Context label for error messages (e.g. ``"gpt-image-2"``).
        unique_id: Optional ComfyUI node ID for status-bar updates.
        n_expected: Expected image count (for partial-failure warnings).

    Returns:
        ``(torch.Tensor, list[str], int)`` — batched IMAGE tensor
        ``(N, H, W, 3)``, list of raw image URLs, and count of
        images that were expected but could not be retrieved.

    Raises:
        RuntimeError: If no images can be extracted.

    """
    items = data.get("data")
    if not items:
        # 空数据可能是服务端瞬时高负载，抛出可重试异常
        raise _EmptyDataRetryableError(f"API 未返回图片数据 (可能服务端瞬时高负载): {data}")
    if not isinstance(items, list):
        items = [items]

    tensors = []
    b64_items = []
    url_items = []
    b64_failed = 0

    for item in items:
        if not isinstance(item, dict):
            _log("warn",
                 f"{error_prefix} 响应包含非 dict 条目 (type={type(item).__name__})，"
                 f"已跳过: {str(item)[:200]}")
            continue
        if item.get("b64_json"):
            b64_items.append(item["b64_json"])
        elif item.get("url"):
            url_items.append(item["url"])

    # b64_json: decode synchronously to uint8 (CPU-bound, fast)
    for idx, b64 in enumerate(b64_items):
        try:
            tensors.append(b64_json_to_uint8(b64))
        except Exception as exc:
            b64_failed += 1
            _log("warn",
                 f"第 {idx+1}/{len(b64_items)} 张图 base64 解码失败，已跳过: {exc}")

    # URLs: download all concurrently, returns (tensors, successful_urls)
    successful_urls = []
    if url_items:
        if unique_id:
            emit_runtime_status(unique_id, "running",
                                f"图片下载中 ({len(url_items)} 张)…",
                                0, 0, 0, 0)
        url_tensors, successful_urls = _download_images_async(
            url_items, timeout_seconds)
        tensors.extend(url_tensors)
        url_failed = len(url_items) - len(successful_urls)
        if unique_id and url_failed > 0:
            emit_runtime_status(unique_id, "running",
                                f"下载完成: {len(tensors)} 张成功, "
                                f"{b64_failed + url_failed} 张失败",
                                0, 0, 0, 0)
        elif unique_id:
            emit_runtime_status(unique_id, "running",
                                f"下载完成 ({len(tensors)} 张)",
                                0, 0, 0, 0)

    if not tensors:
        raise RuntimeError(
            f"未能解析 {error_prefix} 响应图片: "
            f"{_safe_json_dumps(_sanitize_api_response(data))[:500]}"
        )

    # Batch GPU transfer + normalize in one pass
    failed = (len(items) - len(tensors)) + (b64_failed if not url_items else 0)
    # Adjust failed count: items that were non-dict count as failed too
    if n_expected is not None and len(tensors) < n_expected:
        failed = n_expected - len(tensors)
    return _batch_uint8_to_image(tensors), successful_urls, failed


def _download_images_async(urls, timeout_seconds,
                           retry_times=DEFAULT_RETRY_TIMES):
    """Download multiple image URLs concurrently via ``asyncio.create_task()``.

    All downloads share a single ``httpx.AsyncClient`` for connection
    reuse (HTTP/1.1 keepalive).  PIL decoding is offloaded to
    :data:`_PIL_EXECUTOR` so CPU work overlaps with remaining network I/O.

    **Fault-tolerant**: individual image failures are logged and skipped;
    the batch continues as long as at least one image succeeds.

    Shared by all image-generation nodes in this package.

    Args:
        urls: ``list[str]`` — image URLs from the API response.
        timeout_seconds: Read timeout per request.
        retry_times: Max attempts per URL.

    Returns:
        ``(list[torch.Tensor], list[str])`` — uint8 tensors (one per
        successfully downloaded image) and the corresponding URL list
        (failed URLs are excluded).

    Raises:
        RuntimeError: If **all** URLs fail after *retry_times* attempts.

    """
    if not urls:
        return [], []

    _ACCEPT_HEADER = {
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }

    async def _download_bytes(client, url):
        """Download a single URL with jittered-backoff retry → bytes."""
        last_error = None
        for attempt in range(1, retry_times + 1):
            try:
                response = await client.get(url, headers=_ACCEPT_HEADER)
                if not response.is_success:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                return response.content
            except _RETRYABLE_EXCEPTIONS as exc:
                last_error = str(exc)
                _log("warn",
                     f"图片下载失败 (attempt={attempt}/{retry_times}, "
                     f"type={type(exc).__name__}): {last_error}")
                if attempt < retry_times:
                    await asyncio.sleep(
                        _jittered_backoff_seconds(attempt))
                    continue
                break
            except httpx.HTTPStatusError as exc:
                if is_retryable_http_status(exc.response.status_code):
                    last_error = str(exc)
                    _log("warn",
                         f"图片下载 HTTP {exc.response.status_code} "
                         f"(attempt={attempt}/{retry_times}): {last_error}")
                    if attempt < retry_times:
                        await asyncio.sleep(
                            _jittered_backoff_seconds(attempt))
                        continue
                raise RuntimeError(
                    f"下载图片失败 (url={url[:200]}): {exc}"
                ) from exc
        raise RuntimeError(
            f"下载图片连续 {retry_times} 次失败 "
            f"(url={url[:200]}): {last_error}"
        )

    async def _download_and_decode(client, url):
        """Download → immediately decode in thread-pool (pipelined)."""
        loop = asyncio.get_running_loop()
        img_bytes = await _download_bytes(client, url)
        return await loop.run_in_executor(
            _PIL_EXECUTOR, _image_bytes_to_uint8, img_bytes)

    async def _gather():
        req_timeout = ZHANGYUAPI_timeout(timeout_seconds)
        semaphore = asyncio.Semaphore(10)  # 限制最多10个并发下载
        
        async def _download_with_semaphore(client, url):
            async with semaphore:
                return await _download_and_decode(client, url)
        
        # 禁用TCP keepalive，避免104连接重置错误
        # 不设置socket_options，使用系统默认配置
        
        # 创建HTTP传输层，不设置socket_options
        transport = httpx.AsyncHTTPTransport(
            local_address=None,
        )
        
        async with httpx.AsyncClient(
            http2=False,
            timeout=req_timeout,
            trust_env=True,  # 使用系统代理设置，避免网络环境问题
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            transport=transport,
        ) as client:
            tasks = [
                asyncio.create_task(_download_with_semaphore(client, url))
                for url in urls
            ]
            results = await asyncio.gather(
                *tasks, return_exceptions=True)

        tensors = []
        successful_urls = []
        failed_count = 0
        for idx, r in enumerate(results):
            if isinstance(r, BaseException):
                failed_count += 1
                print(
                    f"[Comfyui-ZhangyuAPI] 警告: 第 {idx+1}/{len(urls)} 张图"
                    f" 下载失败，已跳过"
                    f" (url={urls[idx][:200]}): {r}"
                )
                continue
            tensors.append(r)
            successful_urls.append(urls[idx])

        if not tensors:
            raise RuntimeError(
                f"所有 {len(urls)} 张图片下载均失败，"
                f"无法继续（可能是网络或远端服务问题）"
            )
        if failed_count:
            print(
                f"[Comfyui-ZhangyuAPI] 图片下载: {len(tensors)}/{len(urls)} 成功, "
                f"{failed_count} 张被跳过"
            )
        return tensors, successful_urls

    return _run_async_coroutine(_gather())


# ===================================================================
# Node class
# ===================================================================

class ComfyuiZhangyuAPIImage2Node:
    """ComfyUI custom node for gpt-image-2 via the ZhangyuAPI Images API.

    Exposes real ``size``, ``quality``, ``output_format``, ``output_compression``,
    and mask controls.  Supports both synchronous image generation and async
    task submission with adaptive polling.

    Node display name: **Comfyui-ZhangyuAPI-image-2**
    """

    IMAGE_SIZES = ["auto (不传size)", "ratio_only (仅传比例)", "1K", "2K", "4K"]
    ASPECT_RATIOS = [
        "AUTO", "1:1",
        "2:3", "3:2", "3:4", "4:5",
        "9:16", "16:9", "21:9",
    ]
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_urls", "chats", "model_list")
    FUNCTION = "generate"
    CATEGORY = "Comfyui-ZhangyuAPI/🖼️图片 Image"

    # ------------------------------------------------------------------
    # ComfyUI protocol methods
    # ------------------------------------------------------------------

    @classmethod
    def INPUT_TYPES(cls):
        """Return the ComfyUI widget definition for this node.

        Returns:
            ``dict`` with ``"required"``, ``"optional"``, and ``"hidden"`` keys.

        """
        return {
            "required": {
                "api_key (API密钥)": (
                    "STRING", {
                        "default": "", 
                        "multiline": False,
                        "tooltip": "⚠️ 安全提示：如果密钥已泄露，请立即到后台重新生成！请勿将密钥提交到公开仓库。"
                    }),
                "prompt (提示词)": (
                    "STRING", {"default": "", "multiline": True}),
                "mode (模式)": (
                    ["AUTO", "text2img", "img2img"], {"default": "AUTO"}),
                "model (模型)": (
                    "STRING", {"default": "gpt-image-2", "multiline": False}),
                "n (生成数量-谨慎使用)": (
                    "INT", {"default": 1, "min": 1, "max": 5}),
                "api_base (接口域名)": (
                    "STRING", {"default": DEFAULT_API_BASE_URL, "multiline": False}),
                "image_size (分辨率)": (
                    cls.IMAGE_SIZES, {"default": "auto (不传size)"}),
                "aspect_ratio (宽高比)": (
                    cls.ASPECT_RATIOS, {"default": "1:1"}),
                "quality (画质)": (
                    ["auto", "low", "medium", "high"], {"default": "auto"}),
                "response_format (响应格式)": (
                    ["b64_json", "url"], {"default": "b64_json"}),
                "output_format (输出格式)": (
                    ["png", "jpeg", "webp"], {"default": "jpeg"}),
                "output_compression (压缩率)": (
                    "INT", {"default": 85, "min": 0, "max": 100}),
                "seed (本地种子-不发送API)": (
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
                    "INT", {"default": DEFAULT_RETRY_TIMES, "min": 1, "max": 10}),
            },
            "optional": {
                **{f"image_{i:02d}": ("IMAGE",) for i in range(1, 9)},
                "mask": ("MASK",),
                "background (背景)": (
                    ["auto", "transparent", "opaque"], {"default": "auto"}),
                "moderation (审核模式)": (
                    ["auto", "low"], {"default": "auto"}),
                "llm_model (反推用LLM模型)": (
                    "STRING", {"default": "gpt-5.5",
                               "multiline": False,
                               "placeholder": "prompt 留空时自动用此模型反推"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "skip_error": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        """Always return ``True`` — validation is done inside ``generate()``.

        Returns:
            ``bool`` — ``True``.

        """
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_images(self, kwargs):
        """Collect reference images from optional widget inputs.

        Args:
            kwargs: Full ``generate()`` kwargs dict.

        Returns:
            ``list[tuple[str, bytes]]`` — ``(filename, png_bytes)`` pairs
            for each non-None image_01 through image_08 input.

        """
        image_payloads = []
        base64_urls = []
        for i in range(1, 9):
            tensor = kwargs.get(f"image_{i:02d}")
            if tensor is None:
                continue
            # 预处理压缩：确保发送前符合API限制
            tensor = _preprocess_compress_image(tensor)
            png_bytes = tensor_to_png_bytes(tensor)
            image_payloads.append(
                (f"image_{i:02d}.png", png_bytes)
            )
            base64_urls.append(
                "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")
            )
        return image_payloads, base64_urls

    def _payload_fields(self, model, prompt, size, quality, response_format,
                        output_format, output_compression, n_images,
                        init_images=None, aspect_ratio=None,
                        background="auto", moderation="auto"):
        """Build the JSON payload fields for an Images API request.

        Only includes non-default values to keep the request minimal.

        Args:
            model: Model ID string.
            prompt: Cleaned prompt text.
            size: Resolved size string (``"WxH"`` or ``"auto"``) or
                ``"ratio:<W:H>"`` sentinel for ratio-only mode.
            quality: One of ``"auto"``, ``"low"``, ``"medium"``, ``"high"``.
            response_format: ``"url"`` or ``"b64_json"``.
            output_format: ``"png"``, ``"jpeg"``, or ``"webp"``.
            output_compression: 0-100 int.
            n_images: Number of images to generate (1-5).
            init_images: Optional ``list[str]`` of base64 data URLs for
                reference / init images.
            background: ``"auto"``, ``"transparent"``, or ``"opaque"``.
            moderation: ``"auto"`` or ``"low"``.

        Returns:
            ``dict`` — request body fields.

        """
        fields = {"model": model, "prompt": prompt, "n": n_images}
        if isinstance(size, str) and size.startswith("ratio:"):
            # Ratio-only mode: send aspect_ratio instead of concrete size.
            # "ratio:AUTO" means let the API decide — don't send any hint.
            ratio_val = size[6:]  # strip "ratio:" prefix
            if ratio_val and ratio_val.upper() != "AUTO":
                fields["aspect_ratio"] = ratio_val
        elif size != "auto":
            fields["size"] = size
        if quality != "auto":
            fields["quality"] = quality
        if response_format != "b64_json":
            fields["response_format"] = response_format
        # output format + compression
        fields["output_format"] = output_format
        if output_format == "png":
            # PNG is lossless — force compression to 100 regardless of widget
            fields["output_compression"] = 100
        else:
            fields["output_compression"] = output_compression
        if init_images:
            fields["init_images"] = init_images
        
        # background and moderation parameters (for gpt-image-2)
        if background and background != "auto":
            fields["background"] = background
        if moderation and moderation != "auto":
            fields["moderation"] = moderation
        
        # 强制过滤掉 OpenAI DALL-E 2 格式的 referenced_image_ids
        # 我们只使用 init_images 格式
        fields.pop("referenced_image_ids", None)
        
        return fields

    # ------------------------------------------------------------------
    # API request methods
    # ------------------------------------------------------------------

    def _request_text2img(self, api_base, headers, fields, timeout_seconds):
        """POST to ``/v1/images/generations`` (text-to-image).

        Args:
            api_base: Normalized API base URL.
            headers: Request headers (must include ``Authorization``).
            fields: Payload dict from :meth:`_payload_fields`.
            timeout_seconds: Read timeout.

        Returns:
            ``httpx.Response``.

        """
        return ZHANGYUAPI_post(
            f"{api_base}/v1/images/generations",
            timeout_seconds,
            headers={**headers, "Content-Type": "application/json"},
            json=fields,
        )

    def _request_img2img(self, api_base, headers, fields, image_payloads,
                         mask_bytes, timeout_seconds):
        """POST to ``/v1/images/edits`` (image-to-image / inpainting).

        Args:
            api_base: Normalized API base URL.
            headers: Request headers (must include ``Authorization``).
            fields: Payload dict from :meth:`_payload_fields`.
            image_payloads: ``list[tuple[str, bytes]]`` from :meth:`_collect_images`.
            mask_bytes: RGBA PNG bytes from :func:`mask_to_png_bytes`, or ``None``.
            timeout_seconds: Read timeout.

        Returns:
            ``httpx.Response``.

        """
        files = [
            ("image[]", (filename, BytesIO(image_bytes), "image/png"))
            for filename, image_bytes in image_payloads
        ]
        if mask_bytes is not None:
            files.append(
                ("mask", ("mask.png", BytesIO(mask_bytes), "image/png"))
            )

        data = {}
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, list):
                data[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (int, float, bool)):
                data[key] = value
            else:
                data[key] = str(value)
        return ZHANGYUAPI_post(
            f"{api_base}/v1/images/edits",
            timeout_seconds,
            headers=headers,
            data=data,
            files=files,
        )

    # ------------------------------------------------------------------
    # Response parsing & image download — delegated to module-level
    # shared functions (see _parse_api_response_images and
    # _download_images_async at module level).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self, **kwargs):
        """Thin wrapper: delegates to :meth:`_generate_impl` with ``skip_error``
        handling so the workflow can continue on failure.
        """
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
                timeout_seconds=kwargs.get("timeout_seconds (超时秒数)", 360),
            )

    def _generate_impl(self, **kwargs):
        """Execute a gpt-image-2 generation or edit request.

        ComfyUI calls this method from a thread-pool executor.  The method:

        1. Validates and sanitizes all inputs.
        2. Submits a synchronous or async API request.
        3. If the API returns an async task, polls with adaptive intervals.
        4. Downloads images (URLs concurrently, b64_json synchronously).
        5. Returns ``(IMAGE tensor, JSON response string)``.

        Args:
            **kwargs: ComfyUI widget values keyed by display name.

        Returns:
            ``tuple[torch.Tensor, str, str, str, str]`` —
            ``(image, response, image_urls, api_response, model_list)``.

        Raises:
            ValueError: On missing / invalid inputs.
            RuntimeError: On API errors, network failures, or exhaustion
                of retry attempts.

        """
        pbar = ProgressBar(100) if ProgressBar else None
        # -- sanitize inputs --------------------------------------------------
        api_key = kwargs.get("api_key (API密钥)", "").strip()
        prompt = kwargs.get("prompt (提示词)", "")
        mode = kwargs.get("mode (模式)", "AUTO")
        model = (kwargs.get("model (模型)") or "").strip() or "gpt-image-2"
        n_images = safe_int(kwargs.get("n (生成数量-谨慎使用)", 1), 1, 1, 5)
        api_base = normalize_api_base(
                kwargs.get("api_base (接口域名)", DEFAULT_API_BASE_URL)
            )
        image_size = kwargs.get(
            "image_size (分辨率)",
            kwargs.get("size_ratio (尺寸/比例)",
                       kwargs.get("size (尺寸)", "auto (不传size)")),
        )
        aspect_ratio = kwargs.get("aspect_ratio (宽高比)", "1:1")
        quality = safe_choice(
            kwargs.get("quality (画质)", "auto"),
            ["auto", "low", "medium", "high"], "auto")
        response_format = safe_choice(
            kwargs.get("response_format (响应格式)", "b64_json"),
            ["b64_json", "url"], "b64_json")
        output_format = safe_choice(
            kwargs.get("output_format (输出格式)", "jpeg"),
            ["png", "jpeg", "webp"], "jpeg")
        output_compression = safe_int(
            kwargs.get("output_compression (压缩率)", 85), 85, 0, 100)
        background = safe_choice(
            kwargs.get("background (背景)", "auto"),
            ["auto", "transparent", "opaque"], "auto")
        moderation = safe_choice(
            kwargs.get("moderation (审核模式)", "auto"),
            ["auto", "low"], "auto")
        seed = safe_int(
            kwargs.get("seed (本地种子-不发送API)",
                       kwargs.get("seed (种子)", 0)),
            0, 0, 2147483647)
        timeout_seconds = safe_int(
            kwargs.get("timeout_seconds (超时秒数)", DEFAULT_NODE_TIMEOUT),
            DEFAULT_NODE_TIMEOUT,
            DEFAULT_MIN_NODE_TIMEOUT,
            DEFAULT_MAX_NODE_TIMEOUT,
        )
        retry_times = safe_int(
            kwargs.get("retry_times (重试次数)", DEFAULT_RETRY_TIMES),
            DEFAULT_RETRY_TIMES, 1, 10)
        unique_id = kwargs.get("unique_id")
        start_ts = time.time()

        # -- validate ---------------------------------------------------------
        if not api_key.strip():
            emit_runtime_status(unique_id, "error", "API Key 为空",
                                0.0, 0, retry_times, timeout_seconds)
            raise ValueError("API Key 不能为空")

        # Collect images early — needed for auto-prompt fallback
        image_payloads, init_images = self._collect_images(kwargs)

        clean_prompt = normalize_prompt_text(prompt)
        if not clean_prompt and image_payloads:
            # Auto-generate prompt from reference images via LLM vision
            emit_runtime_status(unique_id, "running", "正在反推提示词…",
                                0.0, 0, retry_times, timeout_seconds)
            if pbar: pbar.update_absolute(5)
            try:
                # Use the first image as reference for prompt generation
                ref_url = init_images[0] if init_images else None
                if ref_url:
                    messages = [
                        {"role": "system",
                         "content": (
                             "你是专业图像提示词生成器。请基于用户提供的参考图片"
                             "反推提示词，生成标准 JSON 格式的绘画提示词：\n"
                             "核心还原：完整覆盖主体形象、构图视角、画面风格、"
                             "光影效果、画幅比例、场景环境、人物姿势、服饰发型、"
                             "配饰道具、材质纹理、摄影器材与焦段等全部关键视觉要素，"
                             "保障主体辨识度与画面结构高度还原\n"
                              "创意优化：在保留原图整体调性与核心特征的前提下，"
                              "对背景细节、光影层次、色彩氛围做适度创意升级，"
                              "不改变主体核心形态\n"
                              "输出规则：仅输出纯 JSON 文本，不添加任何解释、备注、"
                              "标签与多余话术，提示词总字数控制在 100 字以内"
                         )},
                        {"role": "user",
                         "content": [
                             {"type": "image_url",
                              "image_url": {"url": ref_url, "detail": "high"}},
                             {"type": "text",
                              "text": "请根据这张参考图反推一个图像生成用的提示词"},
                         ]},
                    ]
                    generated, _ = _call_chat_simple(
                        api_base, api_key.strip(),
                        kwargs.get("llm_model (反推用LLM模型)") or "gpt-5.5",
                        messages,
                        timeout_seconds=timeout_seconds,
                        temperature=0.7, max_tokens=512,
                        retry_times=retry_times,
                    )
                    # Strip markdown code fences before JSON parse (LLMs
                    # often wrap JSON in ```json ... ``` blocks)
                    clean_generated = generated.strip()
                    for fence in ("```json", "```"):
                        if clean_generated.startswith(fence):
                            clean_generated = clean_generated[len(fence):].lstrip("\n")
                        if clean_generated.endswith("```"):
                            clean_generated = clean_generated[:-3].rstrip("\n")
                    # Try JSON parse first; fall back to raw text
                    try:
                        parsed = json.loads(clean_generated)
                        if isinstance(parsed, dict):
                            clean_prompt = normalize_prompt_text(
                                parsed.get("prompt") or parsed.get("提示词") or "")
                        elif isinstance(parsed, str):
                            clean_prompt = normalize_prompt_text(parsed)
                        else:
                            clean_prompt = normalize_prompt_text(generated)
                    except (json.JSONDecodeError, ValueError):
                        clean_prompt = normalize_prompt_text(generated)
                    _log("info", f"自动反推提示词: {clean_prompt[:200]}")
            except Exception as exc:
                _log("warn", f"自动反推提示词失败: {exc}")
                # Fall through — will raise below if prompt is still empty

        if not clean_prompt:
            emit_runtime_status(unique_id, "error", "prompt 不能为空",
                                0.0, 0, retry_times, timeout_seconds)
            raise ValueError("prompt 不能为空")

        try:
            effective_size = normalize_size(image_size, aspect_ratio)
        except ValueError as exc:
            emit_runtime_status(unique_id, "error", str(exc),
                                0.0, 0, retry_times, timeout_seconds)
            raise
        mask_bytes = mask_to_png_bytes(kwargs.get("mask"))

        if mode == "AUTO":
            actual_mode = "img2img" if image_payloads else "text2img"
        else:
            actual_mode = mode

        if actual_mode == "img2img" and not image_payloads:
            emit_runtime_status(unique_id, "error",
                                "img2img 模式需要至少一张参考图",
                                0.0, 0, retry_times, timeout_seconds)
            raise ValueError("img2img 模式需要至少一张参考图")
        if mask_bytes is not None and not image_payloads:
            emit_runtime_status(unique_id, "error",
                                "mask 只能和 image_01 一起用于图片编辑",
                                0.0, 0, retry_times, timeout_seconds)
            raise ValueError("mask 只能和 image_01 一起用于图片编辑")

        # -- prepare request --------------------------------------------------
        headers = {"Authorization": f"Bearer {api_key.strip()}"}
        fields = self._payload_fields(
            model, clean_prompt, effective_size,
            quality, response_format, output_format, output_compression,
            n_images, init_images=init_images,
            aspect_ratio=aspect_ratio,
            background=background,
            moderation=moderation,
        )

        print(
            f"[Comfyui-ZhangyuAPI-image-2] mode={actual_mode}, "
            f"image_size={image_size}, aspect_ratio={aspect_ratio}, "
            f"fields={_strip_image_data(fields)}, seed={seed} (not sent to API)"
        )
        emit_runtime_status(unique_id, "running", "开始生成",
                            0.0, 0, retry_times, timeout_seconds)
        if pbar: pbar.update_absolute(10)

        # -- fetch model list for output port (best-effort) --------------------
        model_list = []
        try:
            model_list = fetch_available_models_cached(
                api_base, api_key.strip())
        except Exception as exc:
            _log("warn", f"获取模型列表失败（不影响生成）: {exc}")

        # -- retry loop -------------------------------------------------------
        last_error = None
        for attempt in range(1, retry_times + 1):
            try:
                emit_runtime_status(
                    unique_id, "running",
                    f"{'图片编辑' if actual_mode == 'img2img' else '文生图'}"
                    f"请求中 ({attempt}/{retry_times})",
                    time.time() - start_ts,
                    attempt, retry_times, timeout_seconds,
                )

                # Dispatch to the appropriate API endpoint
                if actual_mode == "img2img":
                    response = self._request_img2img(
                        api_base, headers, fields, image_payloads,
                        mask_bytes, timeout_seconds,
                    )
                else:
                    response = self._request_text2img(
                        api_base, headers, fields, timeout_seconds,
                    )

                if response.status_code != 200:
                    # Clear, immediately-failed messages for common client errors
                    if response.status_code == 401:
                        raise RuntimeError(
                            "API Key 无效 (401 Unauthorized)，"
                            "请检查 API 密钥是否正确"
                        )
                    if response.status_code == 403:
                        raise RuntimeError(
                            "API 访问被拒绝 (403 Forbidden)，"
                            "请检查账户权限或余额"
                        )

                    try:
                        err_data = response.json()
                        err_msg = _extract_api_error_message(err_data)
                    except Exception:
                        err_msg = _safe_extract_error_from_response(response)
                    last_error = (
                        f"API 错误 {response.status_code}: {err_msg}"
                    )
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
                if pbar: pbar.update_absolute(50)

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

                    # on_tick callback pushes live progress to the frontend
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
                image_tensor, image_urls, _failed = _parse_response_images(
                    data, timeout_seconds, error_prefix="gpt-image-2",
                    unique_id=unique_id, n_expected=n_images)
                if pbar: pbar.update_absolute(90)
                elapsed = time.time() - start_ts

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                response_info = (
                    f"## ZhangyuAPI 生成结果 ({timestamp})\n\n"
                    f"- **模型**：{model}\n"
                    f"- **模式**：{actual_mode}\n"
                    f"- **接口**：{denormalize_api_base(api_base)}\n"
                    f"- **分辨率**：{image_size}"
                    + (f" → {effective_size}" if effective_size != image_size else "") + "\n"
                    f"- **宽高比**：{aspect_ratio}\n"
                    f"- **画质**：{quality}\n"
                    f"- **输出格式**：{output_format}"
                    + (f" (压缩率 {output_compression})" if output_format != "png" else "") + "\n"
                    f"- **生成数量**：{n_images} 张\n"
                    f"- **成功**：{int(image_tensor.shape[0])} 张\n"
                    + (f"- **失败**：{_failed} 张\n"
                       f"- **警告**：请求 {n_images} 张，{_failed} 张下载/解码失败\n"
                       if _failed else "")
                    + (f"- **参考图**：{len(image_payloads)} 张\n" if image_payloads else "")
                    + (f"- **遮罩**：是\n" if mask_bytes is not None else "")
                    + (f"- **种子**：{seed} (仅本地，不发送 API)\n"
                       if seed else "")
                    + (f"- **耗时**：{elapsed:.1f}s (attempt {attempt}/{retry_times})\n"
                       f"- **Usage**：{data.get('usage')}\n"
                       if data.get("usage") else "")
                )

                emit_runtime_status(
                    unique_id, "success",
                    f"生成成功 (耗时 {elapsed:.1f}s)"
                    + (f"，{_failed} 张失败" if _failed else ""),
                    elapsed, attempt, retry_times, timeout_seconds,
                )
                if pbar: pbar.update_absolute(100)
                return (
                    image_tensor,
                    response_info,
                    _safe_json_dumps(image_urls),
                    _safe_json_dumps(_sanitize_api_response(data), indent=2),
                    _safe_json_dumps(model_list),
                )

            except _RETRYABLE_EXCEPTIONS as exc:
                last_error = str(exc)
                _log("warn",
                     f"Image-2 生成失败 (attempt={attempt}/{retry_times}, "
                     f"type={type(exc).__name__}): {last_error}")
                # 始终调用_on_retryable_error来处理连接重置
                _on_retryable_error(exc)
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
            except _EmptyDataRetryableError as exc:
                # 空数据可重试异常：服务端瞬时高负载，使用指数退避重试
                last_error = str(exc)
                _log("warn",
                     f"Image-2 空数据重试 (attempt={attempt}/{retry_times}): {last_error}")
                if attempt < retry_times:
                    emit_runtime_status(
                        unique_id, "running",
                        f"服务端返回空数据，重试中 ({attempt}/{retry_times})",
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
            f"Comfyui-ZhangyuAPI-image-2 连续 {retry_times} 次失败，"
            f"最后错误: {last_error}"
        )


# ===================================================================
# PromptServer routes — registered once at import time
# ===================================================================

# Filter patterns for chat models (prompt optimizer dropdown)
# Excludes known image-only / video-only models that don't support chat completions.
_CHAT_MODEL_EXCLUDE_PATTERNS = [
    "dall-e", "gpt-image", "flux", "sdxl", "stable-diffusion",
    "midjourney", "imagen", "sora", "veo", "wan", "wanx",
    "kling", "seedance", "cogvideo", "mochi", "vidu", "pika",
    "runway", "hailuo", "minimax-video", "ltx", "pyramid-flow",
]

# Filter patterns for image models (universal image gen node dropdown)
# Excludes known chat / video / embedding / TTS / STT / moderation models.
_IMAGE_MODEL_EXCLUDE_PATTERNS = [
    # Chat / LLM models (explicit versions to avoid matching "gpt-image-2")
    "gpt-4", "gpt-3", "gpt-5", "gpt-oss", "chatgpt", "o1-", "o3-", "o4-",
    "claude", "gemini", "llama", "qwen", "deepseek",
    "glm", "yi-", "mistral", "mixtral", "baichuan", "ernie", "command",
    # TTS / STT / embedding / moderation
    "tts-", "whisper", "embedding", "moderation", "babbage", "davinci",
    # Video models
    "sora", "veo", "wan", "wanx", "kling", "seedance", "cogvideo",
    "mochi", "vidu", "pika", "runway", "hailuo", "minimax-video",
    "ltx-video", "pyramid-flow",
]


def _filter_models_by_patterns(all_models, patterns, mode="exclude",
                               fallback_empty=True):
    """Generic model-list filter shared by all nodes.

    Args:
        all_models: ``list[str]`` — full model list from the API.
        patterns: ``list[str]`` — substrings to match (lowercased).
        mode: ``"exclude"`` to keep models NOT matching any pattern;
            ``"include"`` to keep only models that DO match.
        fallback_empty: If ``True`` and the result is empty, return the
            original list instead (lenient filtering).

    Returns:
        ``list[str]``.
    """
    if not all_models:
        return []
    if not patterns:
        return list(all_models)

    if mode == "include":
        result = [m for m in all_models
                  if any(p in m.lower() for p in patterns)]
    else:
        result = [m for m in all_models
                  if not any(p in m.lower() for p in patterns)]

    if fallback_empty and not result:
        return list(all_models)
    return result


def _filter_chat_models(all_models):
    """Thin wrapper — exclude known image/video-only models."""
    return _filter_models_by_patterns(all_models, _CHAT_MODEL_EXCLUDE_PATTERNS,
                                       mode="exclude", fallback_empty=False)


def _filter_image_models(all_models):
    """Thin wrapper — exclude known chat/video/tts/embedding models."""
    return _filter_models_by_patterns(all_models, _IMAGE_MODEL_EXCLUDE_PATTERNS,
                                       mode="exclude", fallback_empty=False)

# ===================================================================
# TTL-cached model-list fetching (with thundering-herd prevention)
# ===================================================================

_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_FETCH_LOCKS = {}        # per-key lock — prevents concurrent fetches
_MODEL_FETCH_LOCKS_LOCK = threading.Lock()  # guards _MODEL_FETCH_LOCKS dict
DEFAULT_MODEL_CACHE_TTL = 900  # seconds (15 min — model lists change rarely)
DEFAULT_MODEL_FETCH_TIMEOUT = 10  # seconds (quick fail, not worth waiting)
_MODEL_FETCH_LOCKS_MAX_SIZE = 100  # 最大锁数量，防止内存泄漏


def _make_cache_key(api_base, api_key):
    """Build a cache key from credentials (API key is hashed)."""
    key_hash = hashlib.sha256((api_key or "").strip().encode()).hexdigest()
    return (normalize_api_base(api_base or DEFAULT_API_BASE_URL), key_hash)


def _get_or_create_key_lock(cache_key):
    """Return the per-cache-key ``threading.Lock``, creating it if needed.

    Serialises fetches for the same credentials so that when N threads
    encounter a cold/expired cache simultaneously, only **one** thread
    hits the remote API — the others wait on the lock and then read
    the freshly populated cache.
    """
    with _MODEL_FETCH_LOCKS_LOCK:
        # 清理机制：如果锁字典过大，删除最旧的条目
        if len(_MODEL_FETCH_LOCKS) >= _MODEL_FETCH_LOCKS_MAX_SIZE:
            # 删除第一个条目（假设是最早的）
            oldest_key = next(iter(_MODEL_FETCH_LOCKS))
            del _MODEL_FETCH_LOCKS[oldest_key]
        
        key_lock = _MODEL_FETCH_LOCKS.get(cache_key)
        if key_lock is None:
            key_lock = threading.Lock()
            _MODEL_FETCH_LOCKS[cache_key] = key_lock
        return key_lock


def fetch_available_models_cached(api_base, api_key, ttl=None,
                                  timeout_seconds=None,
                                  force_refresh=False):
    """Fetch model IDs with in-memory TTL cache + thundering-herd protection.

    - Cache hit → return immediately (unless *force_refresh*).
    - Cache miss / expired / forced → live fetch; if another thread is
      already fetching for the same credentials, wait for it to finish
      instead of making a duplicate API call.
    - Live fetch fails + stale cache exists → return stale + log warning.
    - Live fetch fails + no stale → raise.

    Thread-safe.  The per-key lock ensures only one ``GET /v1/models``
    is in-flight per credential set at any time, so short bursts of
    concurrent executions won't trigger rate-limiting.
    """
    ttl = ttl if ttl is not None else DEFAULT_MODEL_CACHE_TTL
    timeout = (timeout_seconds if timeout_seconds is not None
               else DEFAULT_MODEL_FETCH_TIMEOUT)
    cache_key = _make_cache_key(api_base, api_key)

    # Fast path — cache hit (no lock contention for the common case)
    if not force_refresh:
        with _MODEL_CACHE_LOCK:
            entry = _MODEL_CACHE.get(cache_key)
            if entry is not None:
                age = time.time() - entry["fetched_at"]
                if age < ttl:
                    return entry["models"][:]

    # Slow path — serialise per key so only one thread fetches
    key_lock = _get_or_create_key_lock(cache_key)

    with key_lock:
        # Double-check: another thread may have populated the cache
        # while we were waiting for key_lock
        if not force_refresh:
            with _MODEL_CACHE_LOCK:
                entry = _MODEL_CACHE.get(cache_key)
                if entry is not None:
                    age = time.time() - entry["fetched_at"]
                    if age < ttl:
                        return entry["models"][:]

        try:
            models = fetch_available_models(api_base, api_key, timeout)
        except Exception as exc:
            # Graceful degradation: fall back to stale cache
            with _MODEL_CACHE_LOCK:
                stale = _MODEL_CACHE.get(cache_key)
                if stale is not None:
                    print(
                        f"[Comfyui-ZhangyuAPI] 模型列表刷新失败，使用过期缓存 "
                        f"(age={time.time() - stale['fetched_at']:.0f}s): {exc}"
                    )
                    return stale["models"][:]
            raise  # no stale cache — propagate

        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[cache_key] = {"models": models, "fetched_at": time.time()}
        return models[:]


def clear_model_cache(api_base=None, api_key=None):
    """Clear the model-fetch cache.

    If credentials are given, only that entry is cleared;
    otherwise the entire cache is purged (including stale per-key locks).
    """
    with _MODEL_CACHE_LOCK:
        if api_base is not None and api_key is not None:
            cache_key = _make_cache_key(api_base, api_key)
            _MODEL_CACHE.pop(cache_key, None)
            # Also clean up the corresponding per-key lock
            with _MODEL_FETCH_LOCKS_LOCK:
                _MODEL_FETCH_LOCKS.pop(cache_key, None)
        else:
            _MODEL_CACHE.clear()
            with _MODEL_FETCH_LOCKS_LOCK:
                _MODEL_FETCH_LOCKS.clear()


# ===================================================================
# Shared model resolution & validation
# ===================================================================

def resolve_and_validate_model(model, api_base, api_key, unique_id,
                                placeholder="auto (自动选择)",
                                filter_func=None, cache_ttl=None):
    """Resolve placeholder model name and validate against real API list.

    All node ``generate()``/``optimize()`` methods call this before
    making API requests.

    Args:
        model: Raw model widget value (may be placeholder).
        api_base: Normalized API base URL.
        api_key: Stripped API key string.
        unique_id: ComfyUI node ID for status emission.
        placeholder: The placeholder string to detect.
        filter_func: Optional ``list[str] → list[str]`` to filter models
            (e.g. ``_filter_image_models``).
        cache_ttl: Cache TTL override; ``None`` uses default 300 s.

    Returns:
        ``(resolved_model, model_list)`` — non-placeholder model ID
        and the full/filtered list for the output port.

    Raises:
        ValueError: Placeholder present but models cannot be fetched.

    """
    need_fetch = (not model or model == placeholder)

    if not need_fetch:
        # Explicit model — validate existence (soft: warn, don't block)
        try:
            all_models = fetch_available_models_cached(
                api_base, api_key, ttl=cache_ttl)
            filtered = filter_func(all_models) if filter_func else all_models
            if model in filtered or not filtered:
                return model, filtered

            # Model not in cached list — force-refresh then recheck
            # (JS may have fetched a newer list that Python's cache missed)
            print(
                f"[Comfyui-ZhangyuAPI] 模型 '{model}' 不在缓存列表中，"
                f"强制刷新后重试"
            )
            all_models = fetch_available_models_cached(
                api_base, api_key, ttl=cache_ttl, force_refresh=True)
            filtered = filter_func(all_models) if filter_func else all_models
            if model in filtered:
                return model, filtered

            # Still not found after force-refresh — warn but allow
            print(
                f"[Comfyui-ZhangyuAPI] 警告: 模型 '{model}' 不在当前可用列表中，"
                f"将继续执行（API 可能会拒绝）"
            )
            emit_runtime_status(
                unique_id, "running",
                f"模型 '{model}' 暂不在列表中，尝试执行",
                0.0, 0, 1, DEFAULT_FETCH_TIMEOUT,
            )
            return model, filtered
        except Exception as exc:
            # Cannot fetch — allow execution (API will reject if truly invalid)
            print(f"[Comfyui-ZhangyuAPI] 无法获取模型列表验证 '{model}': {exc}")
            emit_runtime_status(
                unique_id, "running",
                f"无法验证模型 '{model}'，将继续执行",
                0.0, 0, 1, DEFAULT_FETCH_TIMEOUT,
            )
            return model, []

    # Placeholder — must fetch and auto-select
    last_error = None
    for attempt in (1, 2):
        try:
            emit_runtime_status(
                unique_id, "running",
                f"自动获取模型列表中 ({attempt}/2)",
                0.0, attempt, 2, DEFAULT_FETCH_TIMEOUT,
            )
            all_models = fetch_available_models_cached(
                api_base, api_key, ttl=cache_ttl)
            filtered = filter_func(all_models) if filter_func else all_models
            if filtered:
                resolved = filtered[0]
                print(
                    f"[Comfyui-ZhangyuAPI] 自动选择模型: {resolved} "
                    f"(从 {len(filtered)} 个模型中)"
                )
                return resolved, filtered
            last_error = "API 没有返回符合条件的模型"
        except Exception as exc:
            last_error = str(exc)
            if attempt == 1:
                _jittered_sleep(1)
                clear_model_cache(api_base, api_key)
                continue

    raise ValueError(
        f"模型未选择且自动获取失败: {last_error}。"
        f"请点击 '🔄 获取模型' 按钮或手动输入模型名。"
    )


# ===================================================================
# ComfyUI node registration
# ===================================================================

NODE_CLASS_MAPPINGS = {
    "ComfyuiZhangyuAPIImage2Node": ComfyuiZhangyuAPIImage2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyuiZhangyuAPIImage2Node": "Comfyui-ZhangyuAPI-image-2",
}
