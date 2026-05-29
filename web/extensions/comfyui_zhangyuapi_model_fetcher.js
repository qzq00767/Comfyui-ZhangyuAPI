/**
 * ComfyUI Model Fetcher Extension
 * Adds a "🔄 获取模型" button to image generation nodes, fetches available
 * models from the /v1/models endpoint using the configured api_key/api_base.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE_TYPES = new Set([
    "ComfyuiZhangyuAPINode",
    "ComfyuiZhangyuAPIImage2VipNode",
    "ComfyuiZhangyuAPIImage2Node",
]);

const MODEL_FETCH_ROUTE = "/zhangyuapi_fetch_models";


function findModelWidget(node) {
    if (!node || !Array.isArray(node.widgets)) {
        return null;
    }
    return node.widgets.find(
        (w) => w && w.name === "model (模型)"
    );
}


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

    const currentValue = widget.value;
    widget.options.values = [...models];

    // Keep current value if still valid, otherwise switch to first model
    if (!models.includes(currentValue)) {
        widget.value = models[0];
    }

    node.setDirtyCanvas(true, true);
    return true;
}


async function fetchModels(node) {
    if (!node) {
        return;
    }

    const apiKeyWidget = node.widgets.find(
        (w) => w && w.name === "api_key (API密钥)"
    );
    const apiBaseWidget = node.widgets.find(
        (w) => w && w.name === "api_base (接口域名)"
    );

    const apiKey = apiKeyWidget ? String(apiKeyWidget.value || "").trim() : "";
    const apiBase = apiBaseWidget ? String(apiBaseWidget.value || "").trim() : "";

    if (!apiKey) {
        alert("请先填写 API Key");
        return;
    }

    const fetchBtn = node.__fetchModelsBtn;

    try {
        if (fetchBtn) {
            fetchBtn.name = "⏳ 获取中...";
            node.setDirtyCanvas(true, true);
        }

        const response = await api.fetchApi(MODEL_FETCH_ROUTE, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                api_base: apiBase,
                api_key: apiKey,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const msg = errorData.message || `请求失败 (${response.status})`;
            alert(`获取模型列表失败: ${msg}`);
            return;
        }

        const data = await response.json();
        if (data.status === "success" && Array.isArray(data.models)) {
            updateModelDropdown(node, data.models);
            console.log(
                `Comfyui-ZhangyuAPI: fetched ${data.models.length} model(s) for ${node.type}`,
                data.models
            );
        } else {
            alert(`获取模型列表失败: ${data.message || "未知错误"}`);
        }
    } catch (error) {
        console.error("Comfyui-ZhangyuAPI: fetch models failed:", error);
        alert(`获取模型列表失败: ${error.message}`);
    } finally {
        if (fetchBtn) {
            fetchBtn.name = "🔄 获取模型";
            node.setDirtyCanvas(true, true);
        }
    }
}


app.registerExtension({
    name: "Comfyuizhangyuapi.ModelFetcher",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODE_TYPES.has(nodeData?.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // Add "Fetch Models" button — store reference on node for later lookup
            const fetchButton = this.addWidget(
                "button",
                "🔄 获取模型",
                "🔄 获取模型",
                () => fetchModels(this)
            );
            fetchButton.serialize = false;
            this.__fetchModelsBtn = fetchButton;

            return result;
        };
    },
});

console.log("Comfyui-ZhangyuAPI Model Fetcher extension loaded");
