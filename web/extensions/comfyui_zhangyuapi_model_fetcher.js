/**
 * ComfyUI Model Fetcher Extension
 * Adds a "🔄 获取模型" button to image generation nodes, fetches available
 * models from the /v1/models endpoint using the configured api_key/api_base.
 *
 * For the auto-discovery node, model fetching also happens **automatically**
 * when the api_key or api_base widget value changes (debounced 800 ms).
 */
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const TARGET_NODE_TYPES = new Set([
    "ComfyuiZhangyuAPIImage2Node",
    "ComfyuiZhangyuAPIUniversalImageNode",
    "ZhangyuAPIPromptOptimizer",
    "ComfyuiZhangyuAPIVideoNode",
]);

const MODEL_FETCH_ROUTE = "/zhangyuapi_fetch_models";

/** Route override per node type (video nodes use a filtered model list). */
const MODEL_FETCH_ROUTE_MAP = {
    "ComfyuiZhangyuAPIVideoNode": "/zhangyuapi_fetch_video_models",
    "ZhangyuAPIPromptOptimizer": "/zhangyuapi_fetch_chat_models",
    "ComfyuiZhangyuAPIUniversalImageNode": "/zhangyuapi_fetch_image_models",
};

/** Debounce window for auto-fetch after widget value changes (ms). */
const AUTO_FETCH_DEBOUNCE_MS = 800;

/** Outstanding debounce timers keyed by node id. */
const nodeDebounceTimers = new Map();

// ---------------------------------------------------------------------------
// TTL model cache (in-memory + localStorage)
// ---------------------------------------------------------------------------

const MODEL_CACHE_TTL_MS = 300000; // 5 minutes

let _modelCache = new Map();

/**
 * Simple djb2 hash — fast, deterministic, avoids storing raw API keys
 * in localStorage.  NOT cryptographic; only used as a cache-key component.
 */
function _hashKey(str) {
    let h = 5381;
    for (let i = 0; i < str.length; i++) {
        h = ((h << 5) + h + str.charCodeAt(i)) | 0;
    }
    return (h >>> 0).toString(36);
}

function _makeCacheKey(apiBase, apiKey, nodeType) {
    const base = String(apiBase || "").trim().toLowerCase();
    const keyHash = _hashKey(String(apiKey || "").trim());
    return `${base}::${keyHash}::${nodeType}`;
}

function _loadCacheFromStorage() {
    try {
        const raw = localStorage.getItem("comfyui_zhangyuapi.model_cache");
        if (!raw) return;
        const entries = JSON.parse(raw);
        if (!Array.isArray(entries)) return;
        const now = Date.now();
        for (const entry of entries) {
            if (!entry || !entry.k || !Array.isArray(entry.m)) continue;
            if (now - (entry.t || 0) < MODEL_CACHE_TTL_MS) {
                _modelCache.set(entry.k, { models: entry.m, fetchedAt: entry.t });
            }
        }
        console.log(
            `[ZhangyuAPI] 从 localStorage 加载了 ${_modelCache.size} 个缓存模型列表`
        );
    } catch (e) {
        console.warn("[ZhangyuAPI] 加载模型缓存失败:", e);
    }
}

function _saveCacheToStorage() {
    try {
        const entries = [];
        for (const [k, v] of _modelCache.entries()) {
            entries.push({ k, m: v.models, t: v.fetchedAt });
        }
        localStorage.setItem(
            "comfyui_zhangyuapi.model_cache",
            JSON.stringify(entries)
        );
    } catch (e) {
        console.warn("[ZhangyuAPI] 持久化模型缓存失败:", e);
    }
}

function _cacheGet(apiBase, apiKey, nodeType) {
    const cacheKey = _makeCacheKey(apiBase, apiKey, nodeType);
    const entry = _modelCache.get(cacheKey);
    if (!entry) return null;
    if (Date.now() - entry.fetchedAt > MODEL_CACHE_TTL_MS) {
        _modelCache.delete(cacheKey);
        _saveCacheToStorage();
        return null;
    }
    return entry.models;
}

function _cacheSet(apiBase, apiKey, nodeType, models) {
    const cacheKey = _makeCacheKey(apiBase, apiKey, nodeType);
    _modelCache.set(cacheKey, { models: [...models], fetchedAt: Date.now() });
    _saveCacheToStorage();
}

// Initialize cache from localStorage at load time
_loadCacheFromStorage();


// ---------------------------------------------------------------------------
// Widget finders (look up widgets by canonical name)
// ---------------------------------------------------------------------------

function findModelWidget(node) {
    if (!node || !Array.isArray(node.widgets)) {
        return null;
    }
    return node.widgets.find(
        (w) => w && w.name === "model (模型)"
    );
}

function findApiKeyWidget(node) {
    if (!node || !Array.isArray(node.widgets)) {
        return null;
    }
    return node.widgets.find(
        (w) => w && w.name === "api_key (API密钥)"
    );
}

function findApiBaseWidget(node) {
    if (!node || !Array.isArray(node.widgets)) {
        return null;
    }
    return node.widgets.find(
        (w) => w && w.name === "api_base (接口域名)"
    );
}


// ---------------------------------------------------------------------------
// Model dropdown update
// ---------------------------------------------------------------------------

function updateModelDropdown(node, models) {
    const widget = findModelWidget(node);
    if (!widget) {
        console.warn("Comfyui-ZhangyuAPI: model widget not found on node", node?.type);
        return false;
    }

    if (!Array.isArray(models) || models.length === 0) {
        alert("此接口没有可用模型");
        return false;
    }

    // Always keep the placeholder as the first option; real models follow.
    const AUTO_OPTION = "从接口自动获取模型列表";
    const finalModels = [AUTO_OPTION, ...models.filter((m) => m !== AUTO_OPTION)];

    const currentValue = widget.value;
    widget.options.values = finalModels;

    // If the current value is the placeholder or no longer in the list,
    // auto-select the first *real* model (skip the placeholder at index 0).
    if (currentValue === AUTO_OPTION || !finalModels.includes(currentValue)) {
        widget.value = finalModels.length > 1 ? finalModels[1] : finalModels[0];
    }

    node.setDirtyCanvas(true, true);
    return true;
}


/**
 * Resolve the effective API base URL for model-fetch requests.
 *
 * @param {object} node - The ComfyUI node instance.
 * @returns {string} effective base URL, or ``""`` if not set.
 */
function getEffectiveApiBase(node) {
    const baseWidget = findApiBaseWidget(node);
    return baseWidget ? String(baseWidget.value || "").trim() : "";
}


// ---------------------------------------------------------------------------
// Fetch models from backend route
// ---------------------------------------------------------------------------

/**
 * Fetch available models from the API provider.
 *
 * Uses TTL cache (in-memory + localStorage): cache hit → instant dropdown
 * update; cache miss → live fetch → cache → update dropdown.
 *
 * @param {object}  node - The ComfyUI node instance.
 * @param {object}  [opts] - Options.
 * @param {boolean} [opts.silent=false] - If true, suppress alert() on errors.
 * @param {boolean} [opts.forceRefresh=false] - If true, skip cache.
 */
async function fetchModels(node, opts) {
    if (!node) return;

    const silent = !!(opts && opts.silent);
    const forceRefresh = !!(opts && opts.forceRefresh);

    const apiKeyWidget = findApiKeyWidget(node);
    const apiKey = apiKeyWidget ? String(apiKeyWidget.value || "").trim() : "";
    const effectiveBase = getEffectiveApiBase(node);

    if (!apiKey) {
        if (!silent) alert("请先填写 API Key");
        return;
    }

    const fetchBtn = node.__fetchModelsBtn;
    const fetchRoute = MODEL_FETCH_ROUTE_MAP[node.type] || MODEL_FETCH_ROUTE;

    // -- Cache hit (non-forced): serve immediately --------------------------
    if (!forceRefresh) {
        const cachedModels = _cacheGet(effectiveBase, apiKey, node.type);
        if (cachedModels && cachedModels.length > 0) {
            updateModelDropdown(node, cachedModels);
            console.log(
                `[ZhangyuAPI] 缓存命中: ${cachedModels.length} 个模型 (${node.type})`
            );
            // Background refresh if cache is > 80% of TTL old
            const cacheKey = _makeCacheKey(effectiveBase, apiKey, node.type);
            const entry = _modelCache.get(cacheKey);
            if (entry && (Date.now() - entry.fetchedAt) > MODEL_CACHE_TTL_MS * 0.8) {
                console.log("[ZhangyuAPI] 缓存即将过期，后台刷新中...");
                fetchModels(node, { silent: true, forceRefresh: true });
            }
            return;
        }
    }

    // -- Live fetch ---------------------------------------------------------
    try {
        if (fetchBtn && !silent) {
            fetchBtn.name = "⏳ 获取中...";
            node.setDirtyCanvas(true, true);
        }

        const response = await api.fetchApi(fetchRoute, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                api_base: effectiveBase,
                api_key: apiKey,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const msg = errorData.message || `请求失败 (${response.status})`;
            if (silent) console.warn(`[ZhangyuAPI] 模型接口获取失败: ${msg}`);
            else alert(`模型接口获取失败: ${msg}`);
            return;
        }

        const data = await response.json();
        if (data.status === "success" && Array.isArray(data.models)) {
            updateModelDropdown(node, data.models);
            _cacheSet(effectiveBase, apiKey, node.type, data.models);
            console.log(
                `[ZhangyuAPI] 拉取并缓存 ${data.models.length} 个模型 (${node.type})`
            );
        } else {
            const msg = data.message || "未知错误";
            if (silent) console.warn(`[ZhangyuAPI] 模型接口获取失败: ${msg}`);
            else alert(`模型接口获取失败: ${msg}`);
        }
    } catch (error) {
        if (silent) console.warn("[ZhangyuAPI] 模型接口获取失败:", error.message);
        else {
            console.error("[ZhangyuAPI] 模型接口获取失败:", error);
            alert(`模型接口获取失败: ${error.message}`);
        }
        // Fallback: try stale cache on network error
        const stale = _cacheGet(effectiveBase, apiKey, node.type);
        if (stale && stale.length > 0) {
            updateModelDropdown(node, stale);
            console.warn("[ZhangyuAPI] 网络错误，回退到过期缓存");
        }
    } finally {
        if (fetchBtn && !silent) {
            fetchBtn.name = "🔄 获取模型";
            node.setDirtyCanvas(true, true);
        }
    }
}


// ---------------------------------------------------------------------------
// Debounced auto-fetch (triggered on widget value change)
// ---------------------------------------------------------------------------

/**
 * Schedule a debounced auto-fetch for *node*.
 *
 * Called from widget callbacks on every keystroke; only fires after
 * {@link AUTO_FETCH_DEBOUNCE_MS} of inactivity AND only when both
 * ``api_key`` and ``api_base`` are non-empty.
 */
function debouncedAutoFetch(node) {
    if (!node || !node.id) {
        return;
    }
    const nodeId = node.id;

    if (nodeDebounceTimers.has(nodeId)) {
        clearTimeout(nodeDebounceTimers.get(nodeId));
    }

    nodeDebounceTimers.set(nodeId, setTimeout(() => {
        nodeDebounceTimers.delete(nodeId);

        // Guard: node may have been removed while the timer was pending
        if (!node.graph) {
            return;
        }

        const apiKey = findApiKeyWidget(node);
        const keyVal = apiKey ? String(apiKey.value || "").trim() : "";
        const baseVal = getEffectiveApiBase(node);

        if (keyVal && baseVal) {
            fetchModels(node, { silent: true });
        }
    }, AUTO_FETCH_DEBOUNCE_MS));
}


// ---------------------------------------------------------------------------
// Hook widget callbacks so value changes trigger auto-fetch
// ---------------------------------------------------------------------------

/**
 * Wrap *widget*'s ``callback`` so that every value change also schedules
 * a debounced auto-fetch on *node*.
 *
 * Preserves any existing callback (ComfyUI internal or user-set).
 */
function hookWidgetForAutoFetch(widget, node) {
    if (!widget || !node) {
        return;
    }
    const origCallback = widget.callback;
    widget.callback = function (value) {
        // Call the original callback first (ComfyUI internals, seed control, etc.)
        if (origCallback) {
            origCallback.call(this, value);
        }
        // Schedule a debounced model-list refresh
        debouncedAutoFetch(node);
    };
}


// ---------------------------------------------------------------------------
// Monkey-patch LGraphCanvas.prompt to catch keystrokes in text widgets
// ---------------------------------------------------------------------------

/**
 * Widget names whose text-input dialogs should trigger a debounced auto-fetch
 * on every keystroke (ComfyUI STRING widgets only fire ``callback`` on
 * Enter / blur, so the regular ``hookWidgetForAutoFetch`` wrapper never sees
 * intermediate typing).
 */
const TARGET_INPUT_WIDGET_NAMES = new Set([
    "api_key (API密钥)",
]);

/**
 * Monkey-patch ``LGraphCanvas.prototype.prompt`` so that when a user edits
 * the *api_key* STRING widget, every keystroke triggers
 * the debounced auto-fetch path.
 *
 * The patch follows the exact same pattern as
 * ``pysssss.UseNumberInputPrompt`` (useNumberInputPrompt.js).  It is
 * installed once in ``setup()`` via a ``_patched`` guard.
 */
function setupPromptInputPatching() {
    if (setupPromptInputPatching._patched) {
        return;
    }
    setupPromptInputPatching._patched = true;

    const origPrompt = LGraphCanvas.prototype.prompt;

    LGraphCanvas.prototype.prompt = function () {
        // Call the original prompt first — another extension may have
        // already wrapped it, so this keeps the chain intact.
        const dialog = origPrompt.apply(this, arguments);

        // LiteGraph sets app.canvas.node_widget to the [node, widget] pair
        // that is currently being edited BEFORE calling prompt().
        const nodeWidget = app.canvas?.node_widget;
        if (!nodeWidget) {
            return dialog;
        }

        const [node, widget] = nodeWidget;
        if (!node || !widget) {
            return dialog;
        }
        if (!TARGET_NODE_TYPES.has(node.type)) {
            return dialog;
        }
        if (!TARGET_INPUT_WIDGET_NAMES.has(widget.name)) {
            return dialog;
        }

        // Find the <input> element inside the popup dialog
        const inputEl = dialog.querySelector("input");
        if (!inputEl) {
            return dialog;
        }

        // Attach a native 'input' event listener so we hear every keystroke.
        // The existing debouncedAutoFetch() ensures we don't hammer the API.
        inputEl.addEventListener("input", function () {
            debouncedAutoFetch(node);
        });

        return dialog;
    };
}


// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "Comfyuizhangyuapi.ModelFetcher",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!TARGET_NODE_TYPES.has(nodeData?.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            // Model selection is now handled via fixed combo + custom_model field.
            // No fetch button or auto-fetch hooks are injected.
            return result;
        };
    },
});

console.log("[ZhangyuAPI] Model Fetcher extension loaded");
