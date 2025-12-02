<script lang="ts">
    import type { RegistryModelInfo } from "$lib/api/client";

    type LoRAConfig = {
        r: number;
        lora_alpha: number;
        lora_dropout: number;
        lora_bias: string;
        use_rslora: boolean;
        use_gradient_checkpointing: string | boolean;
        random_state: number;
        target_modules: string[] | null;
        task_type: string;
        loftq_config: any;
        finetune_vision_layers: boolean;
        finetune_language_layers: boolean;
        finetune_attention_modules: boolean;
        finetune_mlp_modules: boolean;
    };

    interface Props {
        loraConfig: LoRAConfig;
        modelType: "text" | "vision";
        selectedModelInfo: RegistryModelInfo | null;
        showAdvanced: boolean;
        onToggleAdvanced: () => void;
    }

    let {
        loraConfig = $bindable(),
        modelType,
        selectedModelInfo,
        showAdvanced,
        onToggleAdvanced,
    }: Props = $props();

    // Handle target modules input
    function handleTargetModulesInput(e: Event) {
        const value = (e.target as HTMLInputElement).value.trim();
        if (value) {
            loraConfig.target_modules = value
                .split(",")
                .map((s) => s.trim())
                .filter((s) => s.length > 0);
        } else {
            loraConfig.target_modules = null;
        }
    }
</script>

<div>
    <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-gray-900">LoRA Configuration</h3>
        {#if selectedModelInfo?.training_defaults?.lora_config}
            <span class="text-xs text-green-600 bg-green-50 px-2 py-1 rounded">
                ✓ Using registry defaults
            </span>
        {/if}
    </div>

    <!-- Essential LoRA Parameters -->
    <div class="grid grid-cols-3 gap-4 mb-4">
        <div>
            <label
                for="lora_r"
                class="block text-sm font-medium text-gray-700 mb-1"
            >
                LoRA Rank (r)
            </label>
            <input
                type="number"
                id="lora_r"
                bind:value={loraConfig.r}
                min="1"
                max="256"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
            <p class="text-xs text-gray-500 mt-1">
                Higher = more parameters (16 typical, 64+ for complex tasks)
            </p>
        </div>

        <div>
            <label
                for="lora_alpha"
                class="block text-sm font-medium text-gray-700 mb-1"
            >
                LoRA Alpha
            </label>
            <input
                type="number"
                id="lora_alpha"
                bind:value={loraConfig.lora_alpha}
                min="1"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
            <p class="text-xs text-gray-500 mt-1">
                Scaling factor (typically equal to rank)
            </p>
        </div>

        <div>
            <label
                for="lora_dropout"
                class="block text-sm font-medium text-gray-700 mb-1"
            >
                LoRA Dropout
            </label>
            <input
                type="number"
                id="lora_dropout"
                bind:value={loraConfig.lora_dropout}
                min="0"
                max="0.5"
                step="0.05"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
            <p class="text-xs text-gray-500 mt-1">
                Regularization (0.0-0.3, 0 = no dropout)
            </p>
        </div>
    </div>

    <!-- Advanced LoRA Toggle -->
    <div class="mb-4">
        <button
            type="button"
            onclick={onToggleAdvanced}
            class="flex items-center gap-2 px-4 py-2 text-sm font-medium text-primary-700 bg-primary-50 border border-primary-200 rounded-lg hover:bg-primary-100 transition-colors"
        >
            <span>{showAdvanced ? "▼" : "▶"}</span>
            Advanced LoRA Settings
        </button>
    </div>

    {#if showAdvanced}
        <div class="p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <div class="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <label
                        for="lora_bias"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        LoRA Bias
                    </label>
                    <select
                        id="lora_bias"
                        bind:value={loraConfig.lora_bias}
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                        <option value="none">None (default)</option>
                        <option value="all">All bias terms</option>
                        <option value="lora_only">LoRA layers only</option>
                    </select>
                    <p class="text-xs text-gray-500 mt-1">
                        How to handle bias parameters in LoRA layers
                    </p>
                </div>

                <div>
                    <label
                        for="use_gradient_checkpointing"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Gradient Checkpointing
                    </label>
                    <select
                        id="use_gradient_checkpointing"
                        bind:value={loraConfig.use_gradient_checkpointing}
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                        <option value="unsloth"
                            >Unsloth (30% less VRAM, minor quality loss)</option
                        >
                        <option value="true">Standard (better quality)</option>
                        <option value="false"
                            >Disabled (best quality, most VRAM)</option
                        >
                    </select>
                    <p class="text-xs text-gray-500 mt-1">
                        Tradeoff between memory usage and training quality
                    </p>
                </div>
            </div>

            <div class="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <div class="flex items-center mt-2">
                        <input
                            type="checkbox"
                            id="use_rslora"
                            bind:checked={loraConfig.use_rslora}
                            class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                        />
                        <label
                            for="use_rslora"
                            class="ml-2 block text-sm text-gray-700"
                        >
                            Use RSLoRA (Rank-Stabilized LoRA)
                        </label>
                    </div>
                    <p class="text-xs text-gray-500 mt-1">
                        Better stability for high ranks (r > 16)
                    </p>
                </div>

                <div>
                    <label
                        for="random_state"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Random Seed
                    </label>
                    <input
                        type="number"
                        id="random_state"
                        bind:value={loraConfig.random_state}
                        min="0"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p class="text-xs text-gray-500 mt-1">
                        Seed for reproducible results (42 is popular)
                    </p>
                </div>
            </div>

            <div class="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <label
                        for="task_type"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Task Type
                    </label>
                    <select
                        id="task_type"
                        bind:value={loraConfig.task_type}
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                        <option value="CAUSAL_LM"
                            >Causal LM (Text Generation)</option
                        >
                        <option value="SEQ_2_SEQ_LM"
                            >Sequence-to-Sequence</option
                        >
                        <option value="TOKEN_CLS">Token Classification</option>
                        <option value="SEQ_CLS">Sequence Classification</option>
                        <option value="QUESTION_ANS">Question Answering</option>
                    </select>
                    <p class="text-xs text-gray-500 mt-1">
                        Type of task for PEFT optimization
                    </p>
                </div>

                <div>
                    <label
                        for="target_modules_input"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Target Modules (Advanced)
                    </label>
                    <input
                        type="text"
                        id="target_modules_input"
                        value={loraConfig.target_modules?.join(", ") || ""}
                        oninput={handleTargetModulesInput}
                        placeholder="q_proj, k_proj, v_proj (leave empty for auto)"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p class="text-xs text-gray-500 mt-1">
                        Comma-separated list of layers to apply LoRA
                        (auto-detected if empty)
                    </p>
                </div>
            </div>

            <!-- Vision-Specific Layer Fine-tuning (FastVisionModel) -->
            {#if modelType === "vision"}
                <div
                    class="mb-4 p-4 bg-purple-50 border border-purple-200 rounded-lg"
                >
                    <h4
                        class="text-md font-medium text-purple-900 mb-3 flex items-center gap-2"
                    >
                        🎨 Selective Layer Fine-tuning (Vision Models)
                    </h4>
                    <p class="text-sm text-purple-800 mb-4">
                        Control which parts of the vision-language model to
                        train. Disable layers you don't want to modify:
                    </p>

                    <div class="space-y-3">
                        <div class="flex items-start gap-3">
                            <input
                                type="checkbox"
                                id="finetune_vision_layers"
                                bind:checked={loraConfig.finetune_vision_layers}
                                class="h-4 w-4 mt-0.5 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                            />
                            <div class="flex-1">
                                <label
                                    for="finetune_vision_layers"
                                    class="text-sm font-medium text-gray-700"
                                >
                                    Fine-tune Vision Encoder Layers
                                </label>
                                <p class="text-xs text-gray-500 mt-0.5">
                                    Train the image processing layers. Disable
                                    to freeze vision encoder and only adapt
                                    language model.
                                </p>
                            </div>
                        </div>

                        <div class="flex items-start gap-3">
                            <input
                                type="checkbox"
                                id="finetune_language_layers"
                                bind:checked={
                                    loraConfig.finetune_language_layers
                                }
                                class="h-4 w-4 mt-0.5 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                            />
                            <div class="flex-1">
                                <label
                                    for="finetune_language_layers"
                                    class="text-sm font-medium text-gray-700"
                                >
                                    Fine-tune Language Model Layers
                                </label>
                                <p class="text-xs text-gray-500 mt-0.5">
                                    Train the text generation layers. Disable to
                                    freeze language model and only adapt vision
                                    encoder.
                                </p>
                            </div>
                        </div>

                        <div class="flex items-start gap-3">
                            <input
                                type="checkbox"
                                id="finetune_attention_modules"
                                bind:checked={
                                    loraConfig.finetune_attention_modules
                                }
                                class="h-4 w-4 mt-0.5 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                            />
                            <div class="flex-1">
                                <label
                                    for="finetune_attention_modules"
                                    class="text-sm font-medium text-gray-700"
                                >
                                    Fine-tune Attention Modules
                                </label>
                                <p class="text-xs text-gray-500 mt-0.5">
                                    Train attention layers (Q, K, V, O
                                    projections). Disable for faster training
                                    with slightly lower quality.
                                </p>
                            </div>
                        </div>

                        <div class="flex items-start gap-3">
                            <input
                                type="checkbox"
                                id="finetune_mlp_modules"
                                bind:checked={loraConfig.finetune_mlp_modules}
                                class="h-4 w-4 mt-0.5 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                            />
                            <div class="flex-1">
                                <label
                                    for="finetune_mlp_modules"
                                    class="text-sm font-medium text-gray-700"
                                >
                                    Fine-tune MLP Modules
                                </label>
                                <p class="text-xs text-gray-500 mt-0.5">
                                    Train feed-forward layers (gate, up, down
                                    projections). Disable for faster training
                                    with slightly lower quality.
                                </p>
                            </div>
                        </div>
                    </div>

                    <div
                        class="mt-4 p-3 bg-purple-100 border border-purple-300 rounded-lg"
                    >
                        <p class="text-xs text-purple-900 font-medium mb-2">
                            💡 Common Configurations:
                        </p>
                        <ul class="text-xs text-purple-800 space-y-1">
                            <li>
                                <strong>All enabled (default):</strong> Full model
                                fine-tuning - best quality, slowest
                            </li>
                            <li>
                                <strong>Language only:</strong> Disable vision layers
                                - adapt text generation while keeping vision frozen
                            </li>
                            <li>
                                <strong>Vision only:</strong> Disable language layers
                                - adapt image understanding while keeping language
                                frozen
                            </li>
                            <li>
                                <strong>Attention only:</strong> Disable MLPs - focus
                                on cross-modal attention mechanisms
                            </li>
                        </ul>
                    </div>
                </div>
            {/if}

            <div class="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <h4 class="text-sm font-semibold text-blue-900 mb-2">
                    💡 LoRA Tips
                </h4>
                <ul class="text-xs text-blue-800 space-y-1">
                    <li>
                        <strong>Rank (r):</strong> Start with 16, increase to 64+
                        for complex tasks or large datasets
                    </li>
                    <li>
                        <strong>Alpha:</strong> Usually equal to rank. Higher alpha
                        = stronger adaptation
                    </li>
                    <li>
                        <strong>Dropout:</strong> Add 0.1-0.3 if overfitting, keep
                        0 for small datasets
                    </li>
                    <li>
                        <strong>RSLoRA:</strong> Enable for ranks > 16 to improve
                        training stability
                    </li>
                    <li>
                        <strong>Target Modules:</strong> Leave empty for auto-detection.
                        Common: "q_proj,k_proj,v_proj,o_proj" for attention layers
                    </li>
                    <li>
                        <strong>Task Type:</strong> Use "CAUSAL_LM" for text generation,
                        "SEQ_2_SEQ_LM" for translation/summarization
                    </li>
                </ul>
            </div>
        </div>
    {/if}
</div>
