/**
 * ComfyUI ZhangyuAPI — Preset template helper.
 *
 * Stripped-down extension: the model-fetch button and auto-fetch logic have
 * been removed per user request ("全部节点取消自动获取模型这个功能，改成手动输入模型").
 *
 * Only the preset → prompt_template auto-fill for ZhangyuAPIPromptOptimizer
 * is preserved.
 */
import { app } from "../../../scripts/app.js";

const TARGET_NODE_TYPES = new Set([
    "ZhangyuAPIPromptOptimizer",
]);

// Preset → prompt_template mapping for ZhangyuAPIPromptOptimizer
const PRESET_TEMPLATES = {
    "通用反推模板": "以json格式描述这幅图，描述准确复刻原始图像所需的所有方面，包括主体、视角构图、风格、光线、图片比例、有关物品、服装、发型、复杂细节、配饰、摄影器材、环境、身体姿势以及任何其他相关元素的具体信息,确保能够精确地重现原始图像的每一个细节。要求输出json格式提示词，字数750字以内。",
    "复刻提示词": "对这张目标图片进行完整深度解析，按照主题内容、场景设定、风格参考、色调色彩、构图视角、细节补充六个维度逐一项精细化描述，最终生成一段结构完整、逻辑清晰可直接用于AI文生图工具复刻同款画面的专业正向提示词，字数750字以内。",
};

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "Comfyuizhangyuapi.PresetHelper",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!TARGET_NODE_TYPES.has(nodeData?.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // ---- Preset → prompt auto-fill (prompt optimizer only) ----
            const presetWidget = this.widgets?.find(
                (w) => w && w.name === "preset (预设)"
            );
            const promptWidget = this.widgets?.find(
                (w) => w && w.name === "prompt (提示词)"
            );
            if (presetWidget && promptWidget) {
                const origCallback = presetWidget.callback;
                presetWidget.callback = function (value) {
                    if (origCallback) origCallback.call(this, value);
                    const template = PRESET_TEMPLATES[value];
                    if (template) {
                        // Only fill if prompt is currently empty or contains a previous template
                        const cur = (promptWidget.value || "").trim();
                        const isTemplate = Object.values(PRESET_TEMPLATES).includes(cur);
                        if (!cur || isTemplate) {
                            promptWidget.value = template;
                        }
                    }
                };
            }

            return result;
        };
    },
});

console.log("[ZhangyuAPI] Preset Helper extension loaded (model fetcher removed)");
