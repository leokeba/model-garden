<script lang="ts">
    import { api, type Model } from "$lib/api/client";
    import Badge from "$lib/components/Badge.svelte";
    import Button from "$lib/components/Button.svelte";
    import Card from "$lib/components/Card.svelte";
    import { onMount } from "svelte";

    // Props
    interface Props {
        /** Compact mode hides advanced settings by default */
        compact?: boolean;
        /** Pre-selected local model path */
        selectedModelPath?: string;
        /** Pre-selected HuggingFace model ID */
        selectedHfModelId?: string;
        /** Callback when model is successfully loaded */
        onModelLoaded?: () => void;
        /** Callback when model is unloaded */
        onModelUnloaded?: () => void;
        /** Show local model selection */
        showLocalModels?: boolean;
        /** Show HuggingFace Hub option */
        showHuggingFaceHub?: boolean;
        /** Class to add to the root element */
        class?: string;
    }

    let {
        compact = false,
        selectedModelPath = "",
        selectedHfModelId = "",
        onModelLoaded,
        onModelUnloaded,
        showLocalModels = true,
        showHuggingFaceHub = true,
        class: className = "",
    }: Props = $props();

    // State
    let models: Model[] = $state([]);
    let loading = $state(true);
    let loadingModel = $state(false);
    let unloadingModel = $state(false);
    let error = $state("");
    let inferenceStatus = $state<any>(null);
    let showAdvanced = $state(!compact);

    // Form state
    let localModelPath = $state(selectedModelPath);
    let useHuggingFaceHub = $state(false);
    let huggingFaceModelId = $state("");

    // Basic settings
    let tensorParallelSize = $state(1);
    let gpuMemoryUtilization = $state(0.0);
    let maxModelLen = $state<number | null>(null);
    let dtype = $state("auto");
    let quantization = $state<string | null>(null);

    // Advanced settings
    let maxNumSeqs = $state<number | null>(null);
    let enforceEager = $state<boolean | null>(null);
    let limitImages = $state<number | null>(null);
    let limitVideos = $state<number | null>(null);

    // Poll interval for status updates during loading
    let pollInterval: ReturnType<typeof setInterval> | null = null;

    // Computed values
    let isLoaded = $derived(inferenceStatus?.loaded ?? false);
    let isLoading = $derived(!!inferenceStatus?.loading);
    let currentModelPath = $derived(
        inferenceStatus?.model_info?.model_path ?? null,
    );

    async function loadData() {
        try {
            loading = true;
            error = "";

            const [modelsResponse, statusResponse] = await Promise.all([
                showLocalModels
                    ? api.getModels()
                    : Promise.resolve({ items: [] }),
                api.getInferenceStatus(),
            ]);

            models = modelsResponse.items;
            inferenceStatus = statusResponse;

            // Pre-fill form if a model is loaded
            if (statusResponse.loaded && statusResponse.model_info) {
                const info = statusResponse.model_info;
                if (!localModelPath) {
                    localModelPath = info.model_path;
                }
                tensorParallelSize = info.tensor_parallel_size ?? 1;
                gpuMemoryUtilization = info.gpu_memory_utilization ?? 0.0;
                maxModelLen = info.max_model_len;
                dtype = info.dtype ?? "auto";
                quantization = info.quantization;
            }
        } catch (err) {
            error = err instanceof Error ? err.message : "Failed to load data";
        } finally {
            loading = false;
        }
    }

    function startPolling() {
        if (pollInterval) return;
        pollInterval = setInterval(async () => {
            try {
                inferenceStatus = await api.getInferenceStatus();
                // Stop polling when loading is complete
                if (!inferenceStatus?.loading) {
                    stopPolling();
                    if (inferenceStatus?.loaded) {
                        onModelLoaded?.();
                    }
                }
            } catch (e) {
                console.error("Poll error:", e);
            }
        }, 2000);
    }

    function stopPolling() {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    }

    async function handleLoadModel() {
        let modelPath = "";
        if (useHuggingFaceHub) {
            if (!huggingFaceModelId.trim()) {
                error = "Please enter a HuggingFace model ID";
                return;
            }
            modelPath = huggingFaceModelId.trim();
        } else {
            if (!localModelPath) {
                error = "Please select a model";
                return;
            }
            modelPath = localModelPath;
        }

        try {
            loadingModel = true;
            error = "";

            // If a model is already loaded, unload it first
            if (isLoaded) {
                await api.unloadModel();
            }

            // Build limit_mm_per_prompt if specified
            let limitMmPerPrompt: Record<string, number> | null = null;
            if (limitImages !== null || limitVideos !== null) {
                limitMmPerPrompt = {};
                if (limitImages !== null) limitMmPerPrompt.image = limitImages;
                if (limitVideos !== null) limitMmPerPrompt.video = limitVideos;
            }

            const result = await api.loadModel({
                model_path: modelPath,
                tensor_parallel_size: tensorParallelSize,
                gpu_memory_utilization: gpuMemoryUtilization,
                max_model_len: maxModelLen,
                max_num_seqs: maxNumSeqs,
                enforce_eager: enforceEager,
                limit_mm_per_prompt: limitMmPerPrompt,
                dtype: dtype,
                quantization: quantization,
            });

            if (result.success) {
                // Start polling for status updates
                startPolling();
            } else {
                error = result.message || "Failed to load model";
            }
        } catch (err) {
            error = err instanceof Error ? err.message : "Failed to load model";
        } finally {
            loadingModel = false;
        }
    }

    async function handleUnloadModel() {
        try {
            unloadingModel = true;
            error = "";

            const result = await api.unloadModel();

            if (result.success) {
                await loadData();
                onModelUnloaded?.();
            } else {
                error = result.message || "Failed to unload model";
            }
        } catch (err) {
            error =
                err instanceof Error ? err.message : "Failed to unload model";
        } finally {
            unloadingModel = false;
        }
    }

    async function handleReloadModel() {
        try {
            loadingModel = true;
            error = "";

            const result = await api.reloadModel();

            if (result.success) {
                startPolling();
            } else {
                error = result.message || "Failed to reload model";
            }
        } catch (err) {
            error =
                err instanceof Error ? err.message : "Failed to reload model";
        } finally {
            loadingModel = false;
        }
    }

    function formatBytes(bytes: number): string {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    }

    // Update localModelPath when prop changes
    $effect(() => {
        if (selectedModelPath && selectedModelPath !== localModelPath) {
            localModelPath = selectedModelPath;
            useHuggingFaceHub = false;
        }
    });

    // Update huggingFaceModelId when prop changes
    $effect(() => {
        if (selectedHfModelId && selectedHfModelId !== huggingFaceModelId) {
            huggingFaceModelId = selectedHfModelId;
            useHuggingFaceHub = true;
        }
    });

    onMount(() => {
        loadData();
        // Check for loading status on mount
        if (inferenceStatus?.loading) {
            startPolling();
        }

        return () => {
            stopPolling();
        };
    });
</script>

<div class={className}>
    {#if loading}
        <div class="flex items-center justify-center py-4">
            <div
                class="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"
            ></div>
        </div>
    {:else}
        <!-- Current Status -->
        {#if isLoaded && inferenceStatus?.model_info}
            <Card class="mb-4 bg-green-50 border-green-200">
                <div class="p-4 sm:p-5">
                    <div class="flex items-center gap-2 mb-3">
                        <Badge variant="success">Loaded</Badge>
                    </div>
                    <h4
                        class="text-sm font-semibold text-gray-900 mb-3 break-words"
                    >
                        {inferenceStatus.model_info.model_path
                            .split("/")
                            .pop() || inferenceStatus.model_info.model_path}
                    </h4>
                    <div class="space-y-2 text-sm text-gray-600">
                        <div class="flex justify-between">
                            <span class="text-gray-500">Memory:</span>
                            <span class="font-medium">
                                {#if inferenceStatus.model_info.gpu_memory_utilization === 0}
                                    Auto
                                {:else}
                                    {(
                                        inferenceStatus.model_info
                                            .gpu_memory_utilization * 100
                                    ).toFixed(0)}%
                                {/if}
                            </span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">Data Type:</span>
                            <span class="font-medium"
                                >{inferenceStatus.model_info.dtype}</span
                            >
                        </div>
                        {#if inferenceStatus.model_info.max_model_len}
                            <div class="flex justify-between">
                                <span class="text-gray-500">Context:</span>
                                <span class="font-medium"
                                    >{inferenceStatus.model_info.max_model_len.toLocaleString()}
                                    tokens</span
                                >
                            </div>
                        {/if}
                        {#if inferenceStatus.model_info.quantization}
                            <div class="flex justify-between">
                                <span class="text-gray-500">Quantization:</span>
                                <span class="font-medium"
                                    >{inferenceStatus.model_info
                                        .quantization}</span
                                >
                            </div>
                        {/if}
                        {#if inferenceStatus.model_info.tensor_parallel_size > 1}
                            <div class="flex justify-between">
                                <span class="text-gray-500">GPUs:</span>
                                <span class="font-medium"
                                    >{inferenceStatus.model_info
                                        .tensor_parallel_size}</span
                                >
                            </div>
                        {/if}
                        {#if inferenceStatus.model_info.is_lora_adapter}
                            <div class="flex flex-wrap gap-1 pt-1">
                                <Badge variant="info" size="sm"
                                    >LoRA Adapter</Badge
                                >
                                {#if inferenceStatus.model_info.is_vision_adapter}
                                    <Badge variant="warning" size="sm"
                                        >Vision (Merged)</Badge
                                    >
                                {/if}
                            </div>
                        {/if}
                    </div>
                    <div
                        class="flex flex-wrap gap-2 mt-4 pt-4 border-t border-green-200"
                    >
                        <Button
                            variant="secondary"
                            size="sm"
                            onclick={handleReloadModel}
                            disabled={loadingModel || unloadingModel}
                            loading={loadingModel}
                        >
                            🔄 Reload
                        </Button>
                        <Button
                            variant="danger"
                            size="sm"
                            onclick={handleUnloadModel}
                            disabled={loadingModel || unloadingModel}
                            loading={unloadingModel}
                        >
                            ⏏️ Unload
                        </Button>
                    </div>
                </div>
            </Card>
        {:else if isLoading && inferenceStatus?.loading}
            <Card class="mb-4 bg-blue-50 border-blue-200">
                <div class="p-4 sm:p-5">
                    <div class="flex items-center gap-3 mb-3">
                        <div
                            class="animate-spin rounded-full h-5 w-5 border-2 border-blue-600 border-t-transparent"
                        ></div>
                        <Badge variant="info">Loading Model</Badge>
                    </div>
                    <h4
                        class="text-sm font-semibold text-gray-900 mb-2 break-words"
                    >
                        {inferenceStatus.loading.model_path.split("/").pop() ||
                            inferenceStatus.loading.model_path}
                    </h4>
                    {#if inferenceStatus.loading.status_message}
                        <p class="text-sm text-gray-500">
                            {inferenceStatus.loading.status_message}
                        </p>
                    {/if}
                </div>
            </Card>
        {:else}
            <Card class="mb-4 bg-yellow-50 border-yellow-200">
                <div class="p-4 sm:p-5">
                    <div class="flex items-center gap-3">
                        <Badge variant="warning">No Model Loaded</Badge>
                    </div>
                    <p class="text-sm text-gray-600 mt-2">
                        Select and load a model below to start inference.
                    </p>
                </div>
            </Card>
        {/if}

        <!-- Error Message -->
        {#if error}
            <div
                class="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700"
            >
                <strong>Error:</strong>
                {error}
            </div>
        {/if}

        <!-- Load Model Form -->
        <Card>
            <div class="p-4 sm:p-5">
                <h3
                    class="text-base sm:text-lg font-semibold text-gray-900 mb-4"
                >
                    {isLoaded ? "Switch Model" : "Load Model"}
                </h3>

                <form
                    onsubmit={(e) => {
                        e.preventDefault();
                        handleLoadModel();
                    }}
                    class="space-y-4"
                >
                    <!-- Model Source Selection -->
                    {#if showHuggingFaceHub}
                        <div class="flex flex-wrap gap-4">
                            <label class="flex items-center cursor-pointer">
                                <input
                                    type="radio"
                                    bind:group={useHuggingFaceHub}
                                    value={false}
                                    class="w-4 h-4 text-primary-600"
                                />
                                <span class="ml-2 text-sm">Local Models</span>
                            </label>
                            <label class="flex items-center cursor-pointer">
                                <input
                                    type="radio"
                                    bind:group={useHuggingFaceHub}
                                    value={true}
                                    class="w-4 h-4 text-primary-600"
                                />
                                <span class="ml-2 text-sm">🤗 HuggingFace</span>
                            </label>
                        </div>
                    {/if}

                    {#if useHuggingFaceHub}
                        <!-- HuggingFace Model ID -->
                        <div>
                            <label
                                for="hf-model-id"
                                class="block text-sm font-medium text-gray-700 mb-1.5"
                            >
                                HuggingFace Model ID
                            </label>
                            <input
                                id="hf-model-id"
                                type="text"
                                bind:value={huggingFaceModelId}
                                placeholder="meta-llama/Llama-2-7b-chat-hf"
                                class="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm"
                            />
                        </div>
                    {:else if showLocalModels}
                        <!-- Local Model Selection -->
                        <div>
                            <label
                                for="local-model"
                                class="block text-sm font-medium text-gray-700 mb-1.5"
                            >
                                Select Model
                            </label>
                            {#if models.length === 0}
                                <p class="text-sm text-gray-500">
                                    No local models. <a
                                        href="/training/new"
                                        class="text-primary-600 hover:underline"
                                        >Train one</a
                                    >.
                                </p>
                            {:else}
                                <select
                                    id="local-model"
                                    bind:value={localModelPath}
                                    class="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm"
                                >
                                    <option value="">Select a model...</option>
                                    {#each models as model}
                                        <option value={model.path}>
                                            {model.name}
                                            {model.size_bytes
                                                ? `(${formatBytes(model.size_bytes)})`
                                                : ""}
                                        </option>
                                    {/each}
                                </select>
                            {/if}
                        </div>
                    {/if}

                    <!-- Basic Settings -->
                    <div class="space-y-4">
                        <!-- GPU Memory -->
                        <div>
                            <div class="flex items-center justify-between mb-2">
                                <label
                                    for="gpu-memory"
                                    class="text-sm font-medium text-gray-700"
                                >
                                    GPU Memory Utilization
                                </label>
                                <span
                                    class="text-sm font-semibold text-primary-600 tabular-nums"
                                >
                                    {gpuMemoryUtilization === 0
                                        ? "Auto"
                                        : `${(gpuMemoryUtilization * 100).toFixed(0)}%`}
                                </span>
                            </div>
                            <input
                                id="gpu-memory"
                                type="range"
                                min="0"
                                max="0.99"
                                step="0.05"
                                bind:value={gpuMemoryUtilization}
                                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
                            />
                            <div
                                class="flex justify-between text-xs text-gray-400 mt-1"
                            >
                                <span>Auto</span>
                                <span>50%</span>
                                <span>99%</span>
                            </div>
                        </div>

                        <!-- Data Type -->
                        <div>
                            <label
                                for="dtype"
                                class="block text-sm font-medium text-gray-700 mb-2"
                            >
                                Data Type
                            </label>
                            <select
                                id="dtype"
                                bind:value={dtype}
                                class="w-full px-3 py-2.5 border border-gray-300 rounded-lg text-sm bg-white"
                            >
                                <option value="auto">Auto (recommended)</option>
                                <option value="float16">Float16</option>
                                <option value="bfloat16">BFloat16</option>
                                <option value="float32">Float32</option>
                            </select>
                        </div>
                    </div>

                    <!-- Advanced Settings Toggle -->
                    <button
                        type="button"
                        onclick={() => (showAdvanced = !showAdvanced)}
                        class="text-sm text-primary-600 hover:text-primary-700 flex items-center gap-1.5 font-medium"
                    >
                        <span class="text-xs">{showAdvanced ? "▼" : "▶"}</span>
                        Advanced Settings
                    </button>

                    {#if showAdvanced}
                        <div class="space-y-4 pl-3 border-l-2 border-gray-200">
                            <!-- Tensor Parallel Size -->
                            <div>
                                <label
                                    for="tensor-parallel"
                                    class="block text-sm font-medium text-gray-700 mb-1.5"
                                >
                                    Tensor Parallel Size (GPUs)
                                </label>
                                <input
                                    id="tensor-parallel"
                                    type="number"
                                    min="1"
                                    max="8"
                                    bind:value={tensorParallelSize}
                                    class="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                                />
                            </div>

                            <!-- Max Model Length -->
                            <div>
                                <label
                                    for="max-len"
                                    class="block text-sm font-medium text-gray-700 mb-1.5"
                                >
                                    Max Context Length (tokens)
                                </label>
                                <input
                                    id="max-len"
                                    type="number"
                                    min="128"
                                    step="128"
                                    bind:value={maxModelLen}
                                    placeholder="Auto"
                                    class="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                                />
                            </div>

                            <!-- Max Concurrent Sequences -->
                            <div>
                                <label
                                    for="max-seqs"
                                    class="block text-sm font-medium text-gray-700 mb-1.5"
                                >
                                    Max Concurrent Sequences
                                </label>
                                <input
                                    id="max-seqs"
                                    type="number"
                                    min="1"
                                    max="256"
                                    bind:value={maxNumSeqs}
                                    placeholder="16 (default)"
                                    class="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                                />
                                <p class="text-xs text-gray-500 mt-1">
                                    Reduce for memory-constrained GPUs
                                </p>
                            </div>

                            <!-- Quantization -->
                            <div>
                                <label
                                    for="quant"
                                    class="block text-sm font-medium text-gray-700 mb-1.5"
                                >
                                    Quantization
                                </label>
                                <select
                                    id="quant"
                                    bind:value={quantization}
                                    class="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                                >
                                    <option value={null}>Auto-detect</option>
                                    <option value="awq">AWQ</option>
                                    <option value="gptq">GPTQ</option>
                                    <option value="squeezellm"
                                        >SqueezeLLM</option
                                    >
                                    <option value="fp8">FP8</option>
                                    <option value="bitsandbytes"
                                        >BitsAndBytes</option
                                    >
                                </select>
                            </div>

                            <!-- Enforce Eager -->
                            <div>
                                <label
                                    class="flex items-center gap-2 cursor-pointer"
                                >
                                    <input
                                        type="checkbox"
                                        bind:checked={enforceEager}
                                        class="w-4 h-4 text-primary-600 rounded"
                                    />
                                    <span class="text-sm text-gray-700">
                                        Disable CUDA Graphs (saves ~2GB)
                                    </span>
                                </label>
                            </div>

                            <!-- Multimodal Limits -->
                            <div>
                                <span
                                    class="block text-sm font-medium text-gray-700 mb-1.5"
                                >
                                    Vision Limits (per prompt)
                                </span>
                                <div class="grid grid-cols-2 gap-3">
                                    <div>
                                        <input
                                            id="limit-images"
                                            type="number"
                                            min="0"
                                            bind:value={limitImages}
                                            placeholder="Max images"
                                            class="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                                        />
                                    </div>
                                    <div>
                                        <input
                                            id="limit-videos"
                                            type="number"
                                            min="0"
                                            bind:value={limitVideos}
                                            placeholder="Max videos"
                                            class="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                                        />
                                    </div>
                                </div>
                                <p class="text-xs text-gray-500 mt-1">
                                    Limit multimodal inputs for vision models
                                </p>
                            </div>
                        </div>
                    {/if}

                    <!-- Submit -->
                    <Button
                        type="submit"
                        variant="primary"
                        fullWidth
                        disabled={loadingModel ||
                            isLoading ||
                            (useHuggingFaceHub
                                ? !huggingFaceModelId.trim()
                                : !localModelPath)}
                        loading={loadingModel || isLoading}
                    >
                        {#if loadingModel || isLoading}
                            Loading...
                        {:else if isLoaded}
                            Switch Model
                        {:else}
                            Load Model
                        {/if}
                    </Button>
                </form>
            </div>
        </Card>
    {/if}
</div>
