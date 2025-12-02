<script lang="ts">
    import type { RegistryModelInfo } from "$lib/api/client";

    type Hyperparameters = {
        learning_rate: number;
        num_epochs: number;
        batch_size: number;
        max_steps: number;
        gradient_accumulation_steps: number;
        warmup_steps: number;
        logging_steps: number;
        save_steps: number;
        eval_steps: number | null;
        optim: string;
        weight_decay: number;
        lr_scheduler_type: string;
        max_grad_norm: number;
        adam_beta1: number;
        adam_beta2: number;
        adam_epsilon: number;
        dataloader_num_workers: number;
        dataloader_pin_memory: boolean;
        eval_strategy: string;
        load_best_model_at_end: boolean;
        metric_for_best_model: string;
        save_total_limit: number;
    };

    interface Props {
        hyperparameters: Hyperparameters;
        modelType: "text" | "vision";
        hasValidationDataset: boolean;
        selectedModelInfo: RegistryModelInfo | null;
        showAdvanced: boolean;
        onToggleAdvanced: () => void;
    }

    let {
        hyperparameters = $bindable(),
        modelType,
        hasValidationDataset,
        selectedModelInfo,
        showAdvanced,
        onToggleAdvanced,
    }: Props = $props();
</script>

<div>
    <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-gray-900">
            Training Hyperparameters
        </h3>
        {#if selectedModelInfo?.training_defaults}
            <span class="text-xs text-green-600 bg-green-50 px-2 py-1 rounded">
                ✓ Using registry defaults
            </span>
        {/if}
    </div>

    {#if modelType === "vision"}
        <div class="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p class="text-sm text-yellow-800">
                ⚠️ <strong>Vision models require:</strong> Lower batch size (1-2),
                higher gradient accumulation (8+), and lower learning rate (2e-5)
            </p>
        </div>
    {/if}

    <!-- Essential Training Parameters -->
    <div class="mb-6">
        <h4
            class="text-md font-medium text-gray-800 mb-3 flex items-center gap-2"
        >
            🎯 Essential Parameters
        </h4>
        <div class="grid grid-cols-2 gap-4">
            <div>
                <label
                    for="learning_rate"
                    class="block text-sm font-medium text-gray-700 mb-1"
                >
                    Learning Rate
                </label>
                <input
                    type="number"
                    id="learning_rate"
                    bind:value={hyperparameters.learning_rate}
                    step="0.000001"
                    min="0"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <p class="text-xs text-gray-500 mt-1">
                    {#if modelType === "vision"}2e-5 recommended for vision
                        models{:else}2e-4 typical for text models{/if}
                </p>
            </div>

            <div>
                <label
                    for="num_epochs"
                    class="block text-sm font-medium text-gray-700 mb-1"
                >
                    Number of Epochs
                </label>
                <input
                    type="number"
                    id="num_epochs"
                    bind:value={hyperparameters.num_epochs}
                    min="1"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <p class="text-xs text-gray-500 mt-1">
                    Number of complete passes through dataset
                </p>
            </div>

            <div>
                <label
                    for="batch_size"
                    class="block text-sm font-medium text-gray-700 mb-1"
                >
                    Batch Size per GPU
                </label>
                <input
                    type="number"
                    id="batch_size"
                    bind:value={hyperparameters.batch_size}
                    min="1"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <p class="text-xs text-gray-500 mt-1">
                    {#if modelType === "vision"}Use 1 for vision models{:else}2-4
                        typical for text models{/if}
                </p>
            </div>

            <div>
                <label
                    for="gradient_accumulation"
                    class="block text-sm font-medium text-gray-700 mb-1"
                >
                    Gradient Accumulation Steps
                </label>
                <input
                    type="number"
                    id="gradient_accumulation"
                    bind:value={hyperparameters.gradient_accumulation_steps}
                    min="1"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <p class="text-xs text-gray-500 mt-1">
                    Effective batch size = batch_size ×
                    gradient_accumulation_steps
                </p>
            </div>

            <div>
                <label
                    for="max_steps"
                    class="block text-sm font-medium text-gray-700 mb-1"
                >
                    Max Steps (Optional)
                </label>
                <input
                    type="number"
                    id="max_steps"
                    bind:value={hyperparameters.max_steps}
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <p class="text-xs text-gray-500 mt-1">
                    Override epochs with exact step count (-1 for full epochs)
                </p>
            </div>

            <div>
                <label
                    for="optim"
                    class="block text-sm font-medium text-gray-700 mb-1"
                >
                    Optimizer
                </label>
                <select
                    id="optim"
                    bind:value={hyperparameters.optim}
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                    <option value="adamw_8bit"
                        >AdamW 8-bit (Memory Efficient)</option
                    >
                    <option value="adamw_torch">AdamW (Better Quality)</option>
                    <option value="adamw_torch_fused"
                        >AdamW Fused (Best Quality/Speed)</option
                    >
                    <option value="adafactor"
                        >Adafactor (Most Memory Efficient)</option
                    >
                    <option value="sgd">SGD</option>
                </select>
                <p class="text-xs text-gray-500 mt-1">
                    8-bit saves memory, standard/fused offers better quality
                </p>
            </div>
        </div>
    </div>

    <!-- Checkpoint & Logging -->
    <div class="mb-6">
        <h4
            class="text-md font-medium text-gray-800 mb-3 flex items-center gap-2"
        >
            💾 Checkpoints & Logging
        </h4>
        <div class="grid grid-cols-3 gap-4">
            <div>
                <label
                    for="logging_steps"
                    class="block text-sm font-medium text-gray-700 mb-1"
                >
                    Logging Steps
                </label>
                <input
                    type="number"
                    id="logging_steps"
                    bind:value={hyperparameters.logging_steps}
                    min="1"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <p class="text-xs text-gray-500 mt-1">
                    Log metrics every N steps
                </p>
            </div>

            <div>
                <label
                    for="save_steps"
                    class="block text-sm font-medium text-gray-700 mb-1"
                >
                    Save Steps
                </label>
                <input
                    type="number"
                    id="save_steps"
                    bind:value={hyperparameters.save_steps}
                    min="1"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <p class="text-xs text-gray-500 mt-1">
                    Save checkpoint every N steps
                </p>
            </div>

            <div>
                <label
                    for="save_total_limit"
                    class="block text-sm font-medium text-gray-700 mb-1"
                >
                    Max Checkpoints
                </label>
                <input
                    type="number"
                    id="save_total_limit"
                    bind:value={hyperparameters.save_total_limit}
                    min="1"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <p class="text-xs text-gray-500 mt-1">
                    Keep only N most recent checkpoints
                </p>
            </div>
        </div>
    </div>

    <!-- Evaluation Settings (only if validation dataset provided) -->
    {#if hasValidationDataset}
        <div class="mb-6">
            <h4
                class="text-md font-medium text-gray-800 mb-3 flex items-center gap-2"
            >
                📊 Evaluation Settings
            </h4>
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <label
                        for="eval_strategy"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Evaluation Strategy
                    </label>
                    <select
                        id="eval_strategy"
                        bind:value={hyperparameters.eval_strategy}
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                        <option value="steps">Every N steps</option>
                        <option value="epoch">Every epoch</option>
                        <option value="no">No evaluation</option>
                    </select>
                </div>

                <div>
                    <label
                        for="eval_steps"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Evaluation Steps
                    </label>
                    <input
                        type="number"
                        id="eval_steps"
                        bind:value={hyperparameters.eval_steps}
                        placeholder="Auto (same as save_steps)"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p class="text-xs text-gray-500 mt-1">
                        Evaluate every N steps (leave empty for auto)
                    </p>
                </div>

                <div>
                    <label
                        for="metric_for_best_model"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Best Model Metric
                    </label>
                    <select
                        id="metric_for_best_model"
                        bind:value={hyperparameters.metric_for_best_model}
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                        <option value="eval_loss"
                            >Validation Loss (lower is better)</option
                        >
                        <option value="eval_accuracy"
                            >Accuracy (higher is better)</option
                        >
                        <option value="eval_f1"
                            >F1 Score (higher is better)</option
                        >
                    </select>
                </div>

                <div>
                    <div class="flex items-center mt-6">
                        <input
                            type="checkbox"
                            id="load_best_model_at_end"
                            bind:checked={
                                hyperparameters.load_best_model_at_end
                            }
                            class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                        />
                        <label
                            for="load_best_model_at_end"
                            class="ml-2 block text-sm text-gray-700"
                        >
                            Load best model at end
                        </label>
                    </div>
                    <p class="text-xs text-gray-500 mt-1">
                        Automatically load checkpoint with best validation
                        metric
                    </p>
                </div>
            </div>
        </div>
    {/if}

    <!-- Advanced Hyperparameters Toggle -->
    <div class="mb-4">
        <button
            type="button"
            onclick={onToggleAdvanced}
            class="flex items-center gap-2 px-4 py-2 text-sm font-medium text-primary-700 bg-primary-50 border border-primary-200 rounded-lg hover:bg-primary-100 transition-colors"
        >
            <span>{showAdvanced ? "▼" : "▶"}</span>
            Advanced Hyperparameters
        </button>
    </div>

    {#if showAdvanced}
        <div class="mb-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <!-- Optimizer Settings -->
            <h4
                class="text-md font-medium text-gray-800 mb-3 flex items-center gap-2"
            >
                ⚙️ Optimizer Settings
            </h4>
            <div class="grid grid-cols-2 gap-4 mb-6">
                <div>
                    <label
                        for="weight_decay"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Weight Decay
                    </label>
                    <input
                        type="number"
                        id="weight_decay"
                        bind:value={hyperparameters.weight_decay}
                        step="0.001"
                        min="0"
                        max="1"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p class="text-xs text-gray-500 mt-1">
                        L2 regularization strength (0.01 typical)
                    </p>
                </div>

                <div>
                    <label
                        for="lr_scheduler_type"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        LR Scheduler
                    </label>
                    <select
                        id="lr_scheduler_type"
                        bind:value={hyperparameters.lr_scheduler_type}
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                        <option value="linear">Linear (default)</option>
                        <option value="cosine">Cosine (good for vision)</option>
                        <option value="constant">Constant</option>
                        <option value="constant_with_warmup"
                            >Constant with Warmup</option
                        >
                        <option value="polynomial">Polynomial</option>
                    </select>
                    <p class="text-xs text-gray-500 mt-1">
                        Learning rate schedule type
                    </p>
                </div>

                <div>
                    <label
                        for="warmup_steps"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Warmup Steps
                    </label>
                    <input
                        type="number"
                        id="warmup_steps"
                        bind:value={hyperparameters.warmup_steps}
                        min="0"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p class="text-xs text-gray-500 mt-1">
                        Steps to warmup learning rate from 0
                    </p>
                </div>

                <div>
                    <label
                        for="max_grad_norm"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Max Gradient Norm
                    </label>
                    <input
                        type="number"
                        id="max_grad_norm"
                        bind:value={hyperparameters.max_grad_norm}
                        step="0.1"
                        min="0"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p class="text-xs text-gray-500 mt-1">
                        Gradient clipping threshold (1.0 standard)
                    </p>
                </div>
            </div>

            <!-- Adam Parameters -->
            <h4
                class="text-md font-medium text-gray-800 mb-3 flex items-center gap-2"
            >
                🎛️ Adam Optimizer Parameters
            </h4>
            <div class="grid grid-cols-3 gap-4 mb-6">
                <div>
                    <label
                        for="adam_beta1"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Adam Beta1
                    </label>
                    <input
                        type="number"
                        id="adam_beta1"
                        bind:value={hyperparameters.adam_beta1}
                        step="0.01"
                        min="0"
                        max="1"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p class="text-xs text-gray-500 mt-1">
                        Exponential decay rate for 1st moment (0.9 default)
                    </p>
                </div>

                <div>
                    <label
                        for="adam_beta2"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Adam Beta2
                    </label>
                    <input
                        type="number"
                        id="adam_beta2"
                        bind:value={hyperparameters.adam_beta2}
                        step="0.001"
                        min="0"
                        max="1"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p class="text-xs text-gray-500 mt-1">
                        Exponential decay rate for 2nd moment (0.999 default)
                    </p>
                </div>

                <div>
                    <label
                        for="adam_epsilon"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Adam Epsilon
                    </label>
                    <input
                        type="number"
                        id="adam_epsilon"
                        bind:value={hyperparameters.adam_epsilon}
                        step="1e-9"
                        min="0"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p class="text-xs text-gray-500 mt-1">
                        Small constant for numerical stability (1e-8 default)
                    </p>
                </div>
            </div>

            <!-- Dataloader Settings -->
            <h4
                class="text-md font-medium text-gray-800 mb-3 flex items-center gap-2"
            >
                🔄 Data Loading Settings
            </h4>
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <label
                        for="dataloader_num_workers"
                        class="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Dataloader Workers
                    </label>
                    <input
                        type="number"
                        id="dataloader_num_workers"
                        bind:value={hyperparameters.dataloader_num_workers}
                        min="0"
                        max="16"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <p class="text-xs text-gray-500 mt-1">
                        Number of worker processes (0 = main process only)
                    </p>
                </div>

                <div>
                    <div class="flex items-center mt-6">
                        <input
                            type="checkbox"
                            id="dataloader_pin_memory"
                            bind:checked={hyperparameters.dataloader_pin_memory}
                            class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                        />
                        <label
                            for="dataloader_pin_memory"
                            class="ml-2 block text-sm text-gray-700"
                        >
                            Pin memory to GPU
                        </label>
                    </div>
                    <p class="text-xs text-gray-500 mt-1">
                        Faster data transfer to GPU (recommended)
                    </p>
                </div>
            </div>
        </div>
    {/if}
</div>
