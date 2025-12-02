<script lang="ts">
    import type { RegistryModelInfo } from "$lib/api/client";

    interface Props {
        baseModel: string;
        modelType: "text" | "vision";
        useCustomModel: boolean;
        loadingModels: boolean;
        loadError: string;
        textModels: RegistryModelInfo[];
        visionModels: RegistryModelInfo[];
        selectedModelInfo: RegistryModelInfo | null;
        onBaseModelChange: (value: string) => void;
        onUseCustomModelChange: (value: boolean) => void;
    }

    let {
        baseModel,
        modelType,
        useCustomModel,
        loadingModels,
        loadError,
        textModels,
        visionModels,
        selectedModelInfo,
        onBaseModelChange,
        onUseCustomModelChange,
    }: Props = $props();

    let currentModels = $derived(
        modelType === "vision" ? visionModels : textModels,
    );
</script>

<div>
    <label
        for="base_model"
        class="block text-sm font-medium text-gray-700 mb-1"
    >
        Base Model *
    </label>

    <!-- Model Selection Type Toggle -->
    <div class="mb-3 flex items-center gap-4">
        <button
            type="button"
            onclick={() => onUseCustomModelChange(false)}
            class={`px-3 py-1.5 text-sm rounded-lg border ${
                !useCustomModel
                    ? "bg-primary-50 border-primary-500 text-primary-700 font-medium"
                    : "bg-white border-gray-300 text-gray-700 hover:bg-gray-50"
            }`}
        >
            📋 Registry Models
        </button>
        <button
            type="button"
            onclick={() => onUseCustomModelChange(true)}
            class={`px-3 py-1.5 text-sm rounded-lg border ${
                useCustomModel
                    ? "bg-primary-50 border-primary-500 text-primary-700 font-medium"
                    : "bg-white border-gray-300 text-gray-700 hover:bg-gray-50"
            }`}
        >
            🔧 Custom HuggingFace Model
        </button>
    </div>

    {#if !useCustomModel}
        <!-- Registry Model Dropdown -->
        <select
            id="base_model"
            value={baseModel}
            onchange={(e) => onBaseModelChange(e.currentTarget.value)}
            class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            required
            disabled={loadingModels}
        >
            {#if loadingModels}
                <option value="">Loading models...</option>
            {:else}
                {#each currentModels as model}
                    <option value={model.id}
                        >{model.name} ({model.parameters})</option
                    >
                {/each}
            {/if}
        </select>
        {#if loadError}
            <p class="text-xs text-yellow-600 mt-1">
                ⚠️ Using fallback models: {loadError}
            </p>
        {/if}
    {:else}
        <!-- Custom Model Input -->
        <input
            type="text"
            id="base_model_custom"
            value={baseModel}
            oninput={(e) => onBaseModelChange(e.currentTarget.value)}
            placeholder={modelType === "vision"
                ? "e.g., Qwen/Qwen2-VL-7B-Instruct"
                : "e.g., meta-llama/Llama-2-7b-hf"}
            class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 font-mono text-sm"
            required
        />
        <div class="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p class="text-xs text-blue-800 mb-2">
                <strong>💡 Enter any HuggingFace model ID:</strong>
            </p>
            <ul class="text-xs text-blue-700 space-y-1">
                {#if modelType === "vision"}
                    <li>
                        • Format: <code class="bg-blue-100 px-1 rounded"
                            >organization/model-name</code
                        >
                    </li>
                    <li>
                        • Examples: <code class="bg-blue-100 px-1 rounded"
                            >Qwen/Qwen2-VL-7B-Instruct</code
                        >,
                        <code class="bg-blue-100 px-1 rounded"
                            >llava-hf/llava-1.5-7b-hf</code
                        >
                    </li>
                    <li>• ⚠️ Model must support vision-language tasks</li>
                {:else}
                    <li>
                        • Format: <code class="bg-blue-100 px-1 rounded"
                            >organization/model-name</code
                        >
                    </li>
                    <li>
                        • Examples: <code class="bg-blue-100 px-1 rounded"
                            >meta-llama/Llama-2-7b-hf</code
                        >,
                        <code class="bg-blue-100 px-1 rounded"
                            >mistralai/Mistral-7B-v0.1</code
                        >
                    </li>
                    <li>
                        • ⚠️ Private models require HF_TOKEN environment
                        variable
                    </li>
                {/if}
            </ul>
        </div>
    {/if}

    {#if modelType === "vision" && !useCustomModel}
        <p class="text-xs text-gray-500 mt-1">
            🎨 Vision-language models can analyze images and text together
        </p>
    {/if}

    <!-- Show model info card if available -->
    {#if selectedModelInfo && !loadingModels && !useCustomModel}
        <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div class="flex items-start justify-between">
                <div class="flex-1">
                    <h4 class="text-sm font-semibold text-blue-900 mb-1">
                        📊 {selectedModelInfo.name}
                    </h4>
                    <p class="text-xs text-blue-800 mb-2">
                        {selectedModelInfo.description}
                    </p>

                    {#if selectedModelInfo.requirements}
                        <div class="space-y-1">
                            <p class="text-xs text-blue-700">
                                <strong>VRAM:</strong>
                                {selectedModelInfo.requirements.min_vram_gb}GB
                                minimum,
                                {selectedModelInfo.requirements
                                    .recommended_vram_gb}GB recommended
                            </p>
                            {#if selectedModelInfo.capabilities?.context_window}
                                <p class="text-xs text-blue-700">
                                    <strong>Context:</strong>
                                    {selectedModelInfo.capabilities.context_window.toLocaleString()}
                                    tokens
                                </p>
                            {/if}
                        </div>
                    {/if}

                    {#if selectedModelInfo.recommended_for && selectedModelInfo.recommended_for.length > 0}
                        <p class="text-xs text-blue-700 mt-2">
                            <strong>Best for:</strong>
                            {selectedModelInfo.recommended_for.join(", ")}
                        </p>
                    {/if}
                </div>
            </div>
        </div>
    {:else if useCustomModel}
        <div class="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p class="text-xs text-yellow-800">
                <strong>⚠️ Custom Model:</strong> Default hyperparameters may not
                be optimal for this model. You may need to adjust learning rate,
                batch size, and LoRA settings based on the model architecture.
            </p>
        </div>
    {/if}
</div>
