<script lang="ts">
    interface Props {
        modelType: "text" | "vision";
        onSelect: (type: "text" | "vision") => void;
    }

    let { modelType, onSelect }: Props = $props();
</script>

<div>
    <div class="block text-sm font-medium text-gray-700 mb-2">Model Type *</div>
    <div class="grid grid-cols-2 gap-3">
        <button
            type="button"
            onclick={() => onSelect("text")}
            class={`p-4 border-2 rounded-lg text-left transition-all ${
                modelType === "text"
                    ? "border-primary-500 bg-primary-50"
                    : "border-gray-300 hover:border-gray-400"
            }`}
        >
            <div class="flex items-center gap-2">
                <div
                    class={`w-4 h-4 rounded-full border-2 ${
                        modelType === "text"
                            ? "border-primary-500 bg-primary-500"
                            : "border-gray-400"
                    }`}
                >
                    {#if modelType === "text"}
                        <div
                            class="w-full h-full rounded-full bg-white scale-50"
                        ></div>
                    {/if}
                </div>
                <span class="font-medium">Text-Only (LLM)</span>
            </div>
            <p class="text-sm text-gray-600 mt-2 ml-6">
                Fine-tune language models for text generation tasks
            </p>
        </button>

        <button
            type="button"
            onclick={() => onSelect("vision")}
            class={`p-4 border-2 rounded-lg text-left transition-all ${
                modelType === "vision"
                    ? "border-primary-500 bg-primary-50"
                    : "border-gray-300 hover:border-gray-400"
            }`}
        >
            <div class="flex items-center gap-2">
                <div
                    class={`w-4 h-4 rounded-full border-2 ${
                        modelType === "vision"
                            ? "border-primary-500 bg-primary-500"
                            : "border-gray-400"
                    }`}
                >
                    {#if modelType === "vision"}
                        <div
                            class="w-full h-full rounded-full bg-white scale-50"
                        ></div>
                    {/if}
                </div>
                <span class="font-medium">Vision-Language (VLM)</span>
            </div>
            <p class="text-sm text-gray-600 mt-2 ml-6">
                Fine-tune multimodal models for image + text tasks
            </p>
        </button>
    </div>
</div>
