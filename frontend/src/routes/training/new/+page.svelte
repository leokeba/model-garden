<script lang="ts">
  import { goto } from "$app/navigation";
  import type { RegistryModelInfo } from "$lib/api/client";
  import { api } from "$lib/api/client";
  import Button from "$lib/components/Button.svelte";
  import Card from "$lib/components/Card.svelte";
  import { onMount } from "svelte";

  let formData = $state({
    name: "",
    model_type: "text", // 'text' or 'vision'
    base_model: "unsloth/tinyllama-bnb-4bit",
    dataset_path: "./data/sample.jsonl",
    validation_dataset_path: "",
    output_dir: "",
    hyperparameters: {
      learning_rate: 0.0002,
      num_epochs: 3,
      batch_size: 2,
      max_steps: -1,
      gradient_accumulation_steps: 4,
      warmup_steps: 10,
      logging_steps: 10,
      save_steps: 100,
      eval_steps: null as number | null,
      optim: "adamw_8bit",
      // Advanced Optimizer Parameters
      weight_decay: 0.01,
      lr_scheduler_type: "linear",
      max_grad_norm: 1.0,
      adam_beta1: 0.9,
      adam_beta2: 0.999,
      adam_epsilon: 1e-8,
      // Dataloader Settings
      dataloader_num_workers: 0,
      dataloader_pin_memory: true,
      // Evaluation Parameters
      eval_strategy: "steps",
      load_best_model_at_end: true,
      metric_for_best_model: "eval_loss",
      save_total_limit: 3,
    },
    lora_config: {
      r: 16,
      lora_alpha: 16,
      lora_dropout: 0.0,
      // Advanced LoRA Parameters
      lora_bias: "none",
      use_rslora: false,
      use_gradient_checkpointing: "unsloth",
      random_state: 42,
      target_modules: null as string[] | null,
      task_type: "CAUSAL_LM",
      loftq_config: null as any,
    },
    from_hub: false,
    validation_from_hub: false,
    save_method: "merged_16bit",
    selective_loss: false,
    selective_loss_level: "conservative",
    selective_loss_schema_keys: "",
    selective_loss_masking_start_step: 0,
    selective_loss_verbose: false,
    early_stopping_enabled: false,
    early_stopping_patience: 3,
    early_stopping_threshold: 0.0001,
  });

  let submitting = $state(false);
  let error = $state("");

  // Registry data - loaded from API
  let textModels = $state<RegistryModelInfo[]>([]);
  let visionModels = $state<RegistryModelInfo[]>([]);
  let selectedModelInfo = $state<RegistryModelInfo | null>(null);
  let loadingModels = $state(true);
  let loadError = $state("");

  // Load models from registry on mount
  onMount(async () => {
    try {
      loadingModels = true;
      const [textResponse, visionResponse] = await Promise.all([
        api.getRegistryModels("text-llm"),
        api.getRegistryModels("vision-vlm"),
      ]);

      textModels = textResponse.models;
      visionModels = visionResponse.models;

      // Set initial selected model info
      if (formData.model_type === "text" && textModels.length > 0) {
        selectedModelInfo = textModels[0];
        formData.base_model = textModels[0].id;
      } else if (formData.model_type === "vision" && visionModels.length > 0) {
        selectedModelInfo = visionModels[0];
        formData.base_model = visionModels[0].id;
      }
    } catch (err) {
      loadError =
        err instanceof Error
          ? err.message
          : "Failed to load models from registry";
      console.error("Failed to load registry models:", err);

      // Fallback to hardcoded models if registry fails
      textModels = [
        {
          id: "unsloth/tinyllama-bnb-4bit",
          name: "TinyLlama 1.1B (4-bit)",
          parameters: "1.1B",
        } as RegistryModelInfo,
        {
          id: "unsloth/phi-2-bnb-4bit",
          name: "Phi-2 2.7B (4-bit)",
          parameters: "2.7B",
        } as RegistryModelInfo,
        {
          id: "unsloth/mistral-7b-bnb-4bit",
          name: "Mistral 7B (4-bit)",
          parameters: "7B",
        } as RegistryModelInfo,
        {
          id: "unsloth/llama-2-7b-bnb-4bit",
          name: "Llama 2 7B (4-bit)",
          parameters: "7B",
        } as RegistryModelInfo,
        {
          id: "unsloth/llama-3-8b-bnb-4bit",
          name: "Llama 3 8B (4-bit)",
          parameters: "8B",
        } as RegistryModelInfo,
      ];
      visionModels = [
        {
          id: "Qwen/Qwen2.5-VL-3B-Instruct",
          name: "Qwen2.5-VL 3B",
          parameters: "3B",
        } as RegistryModelInfo,
        {
          id: "Qwen/Qwen2.5-VL-7B-Instruct",
          name: "Qwen2.5-VL 7B",
          parameters: "7B",
        } as RegistryModelInfo,
        {
          id: "Qwen/Qwen2.5-VL-72B-Instruct",
          name: "Qwen2.5-VL 72B",
          parameters: "72B",
        } as RegistryModelInfo,
        {
          id: "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
          name: "Qwen2.5-VL 3B (4-bit)",
          parameters: "3B",
        } as RegistryModelInfo,
        {
          id: "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
          name: "Qwen2.5-VL 7B (4-bit)",
          parameters: "7B",
        } as RegistryModelInfo,
      ];
    } finally {
      loadingModels = false;
    }
  });

  // Update selected model info and apply defaults when base_model changes
  $effect(() => {
    const currentModels =
      formData.model_type === "vision" ? visionModels : textModels;
    selectedModelInfo =
      currentModels.find((m) => m.id === formData.base_model) || null;

    // Apply registry defaults if model info is available
    if (selectedModelInfo?.training_defaults) {
      const defaults = selectedModelInfo.training_defaults;

      // Apply hyperparameter defaults
      if (defaults.hyperparameters) {
        formData.hyperparameters = {
          ...formData.hyperparameters,
          ...defaults.hyperparameters,
        };
      }

      // Apply LoRA defaults
      if (defaults.lora_config) {
        formData.lora_config = {
          ...formData.lora_config,
          ...defaults.lora_config,
        };
      }

      // Apply save method default
      if (defaults.save_method) {
        formData.save_method = defaults.save_method;
      }
    }
  });

  // Update available models when type changes
  $effect(() => {
    if (formData.model_type === "vision") {
      if (visionModels.length > 0) {
        formData.base_model = visionModels[0].id;
      }
      formData.dataset_path = "./data/vision_dataset.jsonl";
    } else {
      if (textModels.length > 0) {
        formData.base_model = textModels[0].id;
      }
      formData.dataset_path = "./data/sample.jsonl";
    }
  });

  // State for showing advanced settings
  let showAdvancedHyperparams = $state(false);
  let showAdvancedLora = $state(false);

  // Auto-update output directory when name changes
  $effect(() => {
    if (formData.name) {
      formData.output_dir = `./models/${formData.name.toLowerCase().replace(/[^a-z0-9]/g, "-")}`;
    }
  });

  async function handleSubmit(event: SubmitEvent) {
    event.preventDefault();

    if (!formData.name || !formData.base_model || !formData.dataset_path) {
      error = "Please fill in all required fields";
      return;
    }

    submitting = true;
    error = "";

    try {
      // Parse schema keys if provided
      let schema_keys_array = null;
      if (
        formData.selective_loss_schema_keys &&
        formData.selective_loss_schema_keys.trim()
      ) {
        schema_keys_array = formData.selective_loss_schema_keys
          .split(",")
          .map((k) => k.trim())
          .filter((k) => k.length > 0);
      }

      const response = await api.createTrainingJob({
        ...formData,
        // Add vision flag to the request
        is_vision: formData.model_type === "vision",
        // Add selective loss fields
        selective_loss: formData.selective_loss,
        selective_loss_level: formData.selective_loss_level,
        selective_loss_schema_keys: schema_keys_array,
        selective_loss_masking_start_step:
          formData.selective_loss_masking_start_step,
        selective_loss_verbose: formData.selective_loss_verbose,
        // Add early stopping fields
        early_stopping_enabled: formData.early_stopping_enabled,
        early_stopping_patience: formData.early_stopping_patience,
        early_stopping_threshold: formData.early_stopping_threshold,
      });
      if (response.success) {
        goto(`/training/${response.data.job_id}`);
      } else {
        error = "Failed to create training job";
      }
    } catch (err) {
      error =
        err instanceof Error ? err.message : "Failed to create training job";
    } finally {
      submitting = false;
    }
  }
</script>

<svelte:head>
  <title>New Training Job - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <!-- Header -->
  <div class="bg-white shadow">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center py-6">
        <div class="flex items-center">
          <Button href="/training" variant="ghost" size="sm"
            >← Training Jobs</Button
          >
          <h1 class="text-3xl font-bold text-gray-900 ml-4">
            New Training Job
          </h1>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <Card>
      <form onsubmit={handleSubmit} class="space-y-6">
        {#if error}
          <div class="p-4 bg-red-50 border border-red-200 rounded-lg">
            <p class="text-red-700">{error}</p>
          </div>
        {/if}

        <!-- Basic Configuration -->
        <div>
          <h3 class="text-lg font-semibold text-gray-900 mb-4">
            Basic Configuration
          </h3>

          <div class="grid grid-cols-1 gap-4">
            <!-- Model Type Selector -->
            <div>
              <div class="block text-sm font-medium text-gray-700 mb-2">
                Model Type *
              </div>
              <div class="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onclick={() => (formData.model_type = "text")}
                  class={`p-4 border-2 rounded-lg text-left transition-all ${
                    formData.model_type === "text"
                      ? "border-primary-500 bg-primary-50"
                      : "border-gray-300 hover:border-gray-400"
                  }`}
                >
                  <div class="flex items-center gap-2">
                    <div
                      class={`w-4 h-4 rounded-full border-2 ${
                        formData.model_type === "text"
                          ? "border-primary-500 bg-primary-500"
                          : "border-gray-400"
                      }`}
                    >
                      {#if formData.model_type === "text"}
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
                  onclick={() => (formData.model_type = "vision")}
                  class={`p-4 border-2 rounded-lg text-left transition-all ${
                    formData.model_type === "vision"
                      ? "border-primary-500 bg-primary-50"
                      : "border-gray-300 hover:border-gray-400"
                  }`}
                >
                  <div class="flex items-center gap-2">
                    <div
                      class={`w-4 h-4 rounded-full border-2 ${
                        formData.model_type === "vision"
                          ? "border-primary-500 bg-primary-500"
                          : "border-gray-400"
                      }`}
                    >
                      {#if formData.model_type === "vision"}
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

            <div>
              <label
                for="name"
                class="block text-sm font-medium text-gray-700 mb-1"
              >
                Model Name *
              </label>
              <input
                type="text"
                id="name"
                bind:value={formData.name}
                placeholder="my-finance-model"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                required
              />
            </div>

            <div>
              <label
                for="base_model"
                class="block text-sm font-medium text-gray-700 mb-1"
              >
                Base Model *
              </label>
              <select
                id="base_model"
                bind:value={formData.base_model}
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                required
                disabled={loadingModels}
              >
                {#if loadingModels}
                  <option value="">Loading models...</option>
                {:else if formData.model_type === "text"}
                  {#each textModels as model}
                    <option value={model.id}
                      >{model.name} ({model.parameters})</option
                    >
                  {/each}
                {:else}
                  {#each visionModels as model}
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
              {#if formData.model_type === "vision"}
                <p class="text-xs text-gray-500 mt-1">
                  🎨 Vision-language models can analyze images and text together
                </p>
              {/if}

              <!-- Show model info card if available -->
              {#if selectedModelInfo && !loadingModels}
                <div
                  class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg"
                >
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
              {/if}
            </div>

            <div>
              <label
                for="dataset_path"
                class="block text-sm font-medium text-gray-700 mb-1"
              >
                Dataset Path *
              </label>
              <input
                type="text"
                id="dataset_path"
                bind:value={formData.dataset_path}
                placeholder={formData.from_hub
                  ? "username/dataset-name"
                  : formData.model_type === "vision"
                    ? "./data/vision_dataset.jsonl"
                    : "./data/my-dataset.jsonl"}
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                required
              />
              <div class="mt-2 flex items-center">
                <input
                  type="checkbox"
                  id="from_hub"
                  bind:checked={formData.from_hub}
                  class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
                <label for="from_hub" class="ml-2 block text-sm text-gray-700">
                  Load from HuggingFace Hub
                </label>
              </div>
              <p class="text-xs text-gray-500 mt-1">
                {#if formData.from_hub}
                  Enter a HuggingFace dataset identifier (e.g.,
                  "username/dataset-name")<br />
                  For specific files, use: "username/repo::train.jsonl"
                {:else if formData.model_type === "vision"}
                  Path to your JSONL dataset with image paths/base64 or local
                  file
                {:else}
                  Path to your JSONL dataset file
                {/if}
              </p>
            </div>

            <!-- Validation Dataset (Optional) -->
            <div>
              <label
                for="validation_dataset_path"
                class="block text-sm font-medium text-gray-700 mb-1"
              >
                Validation Dataset Path (Optional)
              </label>
              <input
                type="text"
                id="validation_dataset_path"
                bind:value={formData.validation_dataset_path}
                placeholder={formData.validation_from_hub
                  ? "username/val-dataset-name"
                  : formData.model_type === "vision"
                    ? "./data/vision_val_dataset.jsonl"
                    : "./data/my-val-dataset.jsonl"}
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
              <div class="mt-2 flex items-center">
                <input
                  type="checkbox"
                  id="validation_from_hub"
                  bind:checked={formData.validation_from_hub}
                  class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
                <label
                  for="validation_from_hub"
                  class="ml-2 block text-sm text-gray-700"
                >
                  Load validation dataset from HuggingFace Hub
                </label>
              </div>
              <p class="text-xs text-gray-500 mt-1">
                📊 Optional: Provide a validation dataset to track validation
                loss during training<br />
                {#if formData.validation_from_hub}
                  Use HuggingFace format: "username/repo" or
                  "username/repo::validation.jsonl"
                {/if}
              </p>
            </div>

            {#if formData.model_type === "vision"}
              <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <h4 class="text-sm font-semibold text-blue-900 mb-2">
                  📋 Vision Dataset Format
                </h4>
                {#if formData.from_hub}
                  <p class="text-sm text-blue-800 mb-2">
                    HuggingFace datasets should use OpenAI messages format with
                    base64 images:
                  </p>
                  <pre
                    class="text-xs bg-blue-100 p-2 rounded overflow-x-auto"><code
                      >{`{"messages": [{"role": "user", "content": [{"type": "image", "image": "data:image/jpeg;base64,..."}, {"type": "text", "text": "What is shown?"}]}]}`}</code
                    ></pre>
                  <p class="text-xs text-blue-700 mt-2">
                    <strong>Example:</strong>
                    <code>Barth371/train_pop_valet_no_wrong_doc</code>
                  </p>
                {:else}
                  <p class="text-sm text-blue-800 mb-2">
                    Your dataset should be in JSONL format with:
                  </p>
                  <pre
                    class="text-xs bg-blue-100 p-2 rounded overflow-x-auto"><code
                      >{`{"text": "What is in this image?", "image": "/path/to/image.jpg", "response": "A cat sitting on a table"}`}</code
                    ></pre>
                  <p class="text-xs text-blue-700 mt-2">
                    <strong>Tip:</strong> Use
                    <code>model-garden create-vision-dataset</code> CLI to generate
                    sample data
                  </p>
                {/if}
              </div>
            {/if}

            <div>
              <label
                for="output_dir"
                class="block text-sm font-medium text-gray-700 mb-1"
              >
                Output Directory
              </label>
              <input
                type="text"
                id="output_dir"
                bind:value={formData.output_dir}
                placeholder="./models/my-model"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>
        </div>

        <!-- Training Hyperparameters -->
        <div>
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-gray-900">
              Training Hyperparameters
            </h3>
            {#if selectedModelInfo?.training_defaults}
              <span
                class="text-xs text-green-600 bg-green-50 px-2 py-1 rounded"
              >
                ✓ Using registry defaults
              </span>
            {/if}
          </div>

          {#if formData.model_type === "vision"}
            <div
              class="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg"
            >
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
                  bind:value={formData.hyperparameters.learning_rate}
                  step="0.00001"
                  min="0"
                  class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <p class="text-xs text-gray-500 mt-1">
                  {#if formData.model_type === "vision"}2e-5 recommended for
                    vision models{:else}2e-4 typical for text models{/if}
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
                  bind:value={formData.hyperparameters.num_epochs}
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
                  bind:value={formData.hyperparameters.batch_size}
                  min="1"
                  class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <p class="text-xs text-gray-500 mt-1">
                  {#if formData.model_type === "vision"}Use 1 for vision models{:else}2-4
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
                  bind:value={
                    formData.hyperparameters.gradient_accumulation_steps
                  }
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
                  bind:value={formData.hyperparameters.max_steps}
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
                  bind:value={formData.hyperparameters.optim}
                  class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="adamw_8bit"
                    >AdamW 8-bit (Recommended - Memory Efficient)</option
                  >
                  <option value="adamw_torch">AdamW (PyTorch)</option>
                  <option value="adamw_torch_fused">AdamW Fused (Faster)</option
                  >
                  <option value="adafactor"
                    >Adafactor (Very Memory Efficient)</option
                  >
                  <option value="sgd">SGD</option>
                </select>
                <p class="text-xs text-gray-500 mt-1">
                  8-bit AdamW reduces memory usage significantly
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
                  bind:value={formData.hyperparameters.logging_steps}
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
                  bind:value={formData.hyperparameters.save_steps}
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
                  bind:value={formData.hyperparameters.save_total_limit}
                  min="1"
                  class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <p class="text-xs text-gray-500 mt-1">
                  Keep only N most recent checkpoints
                </p>
              </div>
            </div>
          </div>

          <!-- Evaluation Settings -->
          {#if formData.validation_dataset_path}
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
                    bind:value={formData.hyperparameters.eval_strategy}
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
                    bind:value={formData.hyperparameters.eval_steps}
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
                    bind:value={formData.hyperparameters.metric_for_best_model}
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  >
                    <option value="eval_loss"
                      >Validation Loss (lower is better)</option
                    >
                    <option value="eval_accuracy"
                      >Accuracy (higher is better)</option
                    >
                    <option value="eval_f1">F1 Score (higher is better)</option>
                  </select>
                </div>

                <div>
                  <div class="flex items-center mt-6">
                    <input
                      type="checkbox"
                      id="load_best_model_at_end"
                      bind:checked={
                        formData.hyperparameters.load_best_model_at_end
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
                    Automatically load checkpoint with best validation metric
                  </p>
                </div>
              </div>
            </div>

            <!-- Early Stopping Configuration -->
            <div class="mb-6">
              <h4
                class="text-md font-medium text-gray-800 mb-3 flex items-center gap-2"
              >
                ⏸️ Automatic Early Stopping
              </h4>

              <div
                class="p-4 bg-blue-50 border border-blue-200 rounded-lg mb-4"
              >
                <p class="text-sm text-blue-800 mb-2">
                  <strong>Automatic Early Stopping:</strong> Stops training when
                  validation loss stops improving, preventing overfitting and saving
                  compute time.
                </p>
                <p class="text-xs text-blue-700">
                  This is different from the manual "Stop Early" button on the
                  training page. This monitors validation metrics and stops
                  automatically.
                </p>
              </div>

              <div class="space-y-4">
                <div>
                  <div class="flex items-center">
                    <input
                      type="checkbox"
                      id="early_stopping_enabled"
                      bind:checked={formData.early_stopping_enabled}
                      class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                    />
                    <label
                      for="early_stopping_enabled"
                      class="ml-2 block text-sm font-medium text-gray-700"
                    >
                      Enable Automatic Early Stopping
                    </label>
                  </div>
                  <p class="text-xs text-gray-500 mt-1 ml-6">
                    Monitor validation loss and stop when it stops improving
                  </p>
                </div>

                {#if formData.early_stopping_enabled}
                  <div
                    class="ml-6 space-y-4 p-4 bg-white border border-gray-200 rounded-lg"
                  >
                    <div>
                      <label
                        for="early_stopping_patience"
                        class="block text-sm font-medium text-gray-700 mb-1"
                      >
                        Patience (evaluations)
                      </label>
                      <input
                        type="number"
                        id="early_stopping_patience"
                        bind:value={formData.early_stopping_patience}
                        min="1"
                        max="20"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                      />
                      <p class="text-xs text-gray-500 mt-1">
                        Number of evaluations with no improvement before
                        stopping (3-5 typical)
                      </p>
                    </div>

                    <div>
                      <label
                        for="early_stopping_threshold"
                        class="block text-sm font-medium text-gray-700 mb-1"
                      >
                        Improvement Threshold
                      </label>
                      <input
                        type="number"
                        id="early_stopping_threshold"
                        bind:value={formData.early_stopping_threshold}
                        min="0"
                        step="0.0001"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                      />
                      <p class="text-xs text-gray-500 mt-1">
                        Minimum change to qualify as improvement (0.0001 =
                        0.01%, smaller = more sensitive)
                      </p>
                    </div>

                    <div
                      class="p-3 bg-green-50 border border-green-200 rounded-lg"
                    >
                      <p class="text-xs text-green-800">
                        <strong>💡 Example:</strong> With patience=3 and threshold=0.0001,
                        training stops if validation loss doesn't improve by at least
                        0.01% for 3 consecutive evaluations.
                      </p>
                    </div>
                  </div>
                {/if}
              </div>
            </div>
          {/if}

          <!-- Advanced Hyperparameters Toggle -->
          <div class="mb-4">
            <button
              type="button"
              onclick={() =>
                (showAdvancedHyperparams = !showAdvancedHyperparams)}
              class="flex items-center gap-2 px-4 py-2 text-sm font-medium text-primary-700 bg-primary-50 border border-primary-200 rounded-lg hover:bg-primary-100 transition-colors"
            >
              <span>{showAdvancedHyperparams ? "▼" : "▶"}</span>
              Advanced Hyperparameters
            </button>
          </div>

          {#if showAdvancedHyperparams}
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
                    bind:value={formData.hyperparameters.weight_decay}
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
                    bind:value={formData.hyperparameters.lr_scheduler_type}
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
                    bind:value={formData.hyperparameters.warmup_steps}
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
                    bind:value={formData.hyperparameters.max_grad_norm}
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
                    bind:value={formData.hyperparameters.adam_beta1}
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
                    bind:value={formData.hyperparameters.adam_beta2}
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
                    bind:value={formData.hyperparameters.adam_epsilon}
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
                    bind:value={formData.hyperparameters.dataloader_num_workers}
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
                      bind:checked={
                        formData.hyperparameters.dataloader_pin_memory
                      }
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

        <!-- LoRA Configuration -->
        <div>
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-gray-900">
              LoRA Configuration
            </h3>
            {#if selectedModelInfo?.training_defaults?.lora_config}
              <span
                class="text-xs text-green-600 bg-green-50 px-2 py-1 rounded"
              >
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
                bind:value={formData.lora_config.r}
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
                bind:value={formData.lora_config.lora_alpha}
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
                bind:value={formData.lora_config.lora_dropout}
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
              onclick={() => (showAdvancedLora = !showAdvancedLora)}
              class="flex items-center gap-2 px-4 py-2 text-sm font-medium text-primary-700 bg-primary-50 border border-primary-200 rounded-lg hover:bg-primary-100 transition-colors"
            >
              <span>{showAdvancedLora ? "▼" : "▶"}</span>
              Advanced LoRA Settings
            </button>
          </div>

          {#if showAdvancedLora}
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
                    bind:value={formData.lora_config.lora_bias}
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
                    bind:value={formData.lora_config.use_gradient_checkpointing}
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  >
                    <option value="unsloth">Unsloth (recommended)</option>
                    <option value="true">Standard PyTorch</option>
                    <option value="false">Disabled</option>
                  </select>
                  <p class="text-xs text-gray-500 mt-1">
                    Reduces memory at cost of compute time
                  </p>
                </div>
              </div>

              <div class="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <div class="flex items-center mt-2">
                    <input
                      type="checkbox"
                      id="use_rslora"
                      bind:checked={formData.lora_config.use_rslora}
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
                    bind:value={formData.lora_config.random_state}
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
                    bind:value={formData.lora_config.task_type}
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  >
                    <option value="CAUSAL_LM"
                      >Causal LM (Text Generation)</option
                    >
                    <option value="SEQ_2_SEQ_LM">Sequence-to-Sequence</option>
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
                    value={formData.lora_config.target_modules?.join(", ") ||
                      ""}
                    oninput={(e) => {
                      const value = e.currentTarget.value.trim();
                      if (value) {
                        formData.lora_config.target_modules = value
                          .split(",")
                          .map((s) => s.trim())
                          .filter((s) => s.length > 0);
                      } else {
                        formData.lora_config.target_modules = null;
                      }
                    }}
                    placeholder="q_proj, k_proj, v_proj (leave empty for auto)"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  />
                  <p class="text-xs text-gray-500 mt-1">
                    Comma-separated list of layers to apply LoRA (auto-detected
                    if empty)
                  </p>
                </div>
              </div>

              <div class="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <h4 class="text-sm font-semibold text-blue-900 mb-2">
                  💡 LoRA Tips
                </h4>
                <ul class="text-xs text-blue-800 space-y-1">
                  <li>
                    <strong>Rank (r):</strong> Start with 16, increase to 64+ for
                    complex tasks or large datasets
                  </li>
                  <li>
                    <strong>Alpha:</strong> Usually equal to rank. Higher alpha =
                    stronger adaptation
                  </li>
                  <li>
                    <strong>Dropout:</strong> Add 0.1-0.3 if overfitting, keep 0
                    for small datasets
                  </li>
                  <li>
                    <strong>RSLoRA:</strong> Enable for ranks > 16 to improve training
                    stability
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

        <!-- Model Save Options -->
        <div>
          <h3 class="text-lg font-semibold text-gray-900 mb-4">
            Model Save Options
          </h3>

          <div>
            <label
              for="save_method"
              class="block text-sm font-medium text-gray-700 mb-2"
            >
              Save Method
            </label>
            <select
              id="save_method"
              bind:value={formData.save_method}
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="merged_16bit"
                >Save Merged Model (16-bit) - Recommended</option
              >
              <option value="merged_4bit"
                >Save Merged Model (4-bit) - Smaller Size</option
              >
              <option value="lora">Save LoRA Adapters Only - Advanced</option>
            </select>
            <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p class="text-sm text-blue-800">
                {#if formData.save_method === "merged_16bit"}
                  <strong>✅ Merged 16-bit (Recommended):</strong> Full model with
                  LoRA weights merged using Unsloth. Creates split files for vLLM
                  compatibility.
                {:else if formData.save_method === "merged_4bit"}
                  <strong>📦 Merged 4-bit:</strong> Full model with LoRA weights
                  merged in 4-bit quantized format. Smaller file size.
                {:else}
                  <strong>🔧 LoRA Adapters Only (Advanced):</strong> Saves only the
                  adapter weights. Requires the base model to load.
                {/if}
              </p>
            </div>
          </div>
        </div>

        <!-- Selective Loss for Structured Outputs (Vision Models Only) -->
        {#if formData.model_type === "vision"}
          <div>
            <h3 class="text-lg font-semibold text-gray-900 mb-4">
              🎯 Selective Loss (Structured Outputs)
            </h3>

            <div
              class="p-4 bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg mb-4"
            >
              <p class="text-sm text-gray-800 mb-2">
                <strong>🔬 Experimental Feature:</strong> Optimize training for structured
                outputs (JSON, forms, etc.)
              </p>
              <p class="text-xs text-gray-700">
                Masks structural tokens (braces, colons, whitespace) so the
                model focuses on semantic content. Useful for form extraction,
                structured data generation, and similar tasks.
              </p>
            </div>

            <div class="space-y-4">
              <div>
                <div class="flex items-center">
                  <input
                    type="checkbox"
                    id="selective_loss"
                    bind:checked={formData.selective_loss}
                    class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                  />
                  <label
                    for="selective_loss"
                    class="ml-2 block text-sm font-medium text-gray-700"
                  >
                    Enable Selective Loss Masking
                  </label>
                </div>
                <p class="text-xs text-gray-500 mt-1 ml-6">
                  Automatically mask JSON structural tokens during training
                </p>
              </div>

              {#if formData.selective_loss}
                <div
                  class="ml-6 space-y-4 p-4 bg-white border border-gray-200 rounded-lg"
                >
                  <div>
                    <label
                      for="selective_loss_level"
                      class="block text-sm font-medium text-gray-700 mb-2"
                    >
                      Masking Level
                    </label>
                    <select
                      id="selective_loss_level"
                      bind:value={formData.selective_loss_level}
                      class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                      <option value="conservative"
                        >Conservative (Structure Only)</option
                      >
                      <option value="moderate"
                        >Moderate (Structure + null)</option
                      >
                      <option value="aggressive"
                        >Aggressive (Structure + null + Schema Keys)</option
                      >
                    </select>
                    <div class="mt-2 p-3 bg-gray-50 rounded-lg">
                      <p class="text-xs text-gray-700">
                        {#if formData.selective_loss_level === "conservative"}
                          <strong>Conservative:</strong> Masks JSON structural
                          characters: <code>{`{, }, [, ], :, ,, "`}</code> and
                          whitespace. Masks ~31% of tokens.
                          <em>Recommended for most cases.</em>
                        {:else if formData.selective_loss_level === "moderate"}
                          <strong>Moderate:</strong> Conservative + masks
                          <code>null</code> keyword. Good when null values are predictable.
                        {:else}
                          <strong>Aggressive:</strong> Moderate + masks schema field
                          names. Maximum focus on semantic content. Requires specifying
                          schema keys below.
                        {/if}
                      </p>
                    </div>
                  </div>

                  {#if formData.selective_loss_level === "aggressive"}
                    <div>
                      <label
                        for="selective_loss_schema_keys"
                        class="block text-sm font-medium text-gray-700 mb-1"
                      >
                        Schema Keys to Mask
                      </label>
                      <input
                        type="text"
                        id="selective_loss_schema_keys"
                        bind:value={formData.selective_loss_schema_keys}
                        placeholder="Marque,Modele,contents,confidence_score"
                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                      />
                      <p class="text-xs text-gray-500 mt-1">
                        Comma-separated list of JSON field names to mask (e.g.,
                        "name,address,phone")
                      </p>
                      <div
                        class="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded"
                      >
                        <p class="text-xs text-yellow-800">
                          ⚠️ Only mask keys that are predictable and don't carry
                          semantic meaning. The model should still learn what
                          values go with each key.
                        </p>
                      </div>
                    </div>
                  {/if}

                  <div>
                    <label
                      for="selective_loss_masking_start_step"
                      class="block text-sm font-medium text-gray-700 mb-2"
                    >
                      Masking Start Step: {formData.selective_loss_masking_start_step}
                    </label>
                    <input
                      type="range"
                      id="selective_loss_masking_start_step"
                      bind:value={formData.selective_loss_masking_start_step}
                      min="0"
                      max="500"
                      step="10"
                      class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-500"
                    />
                    <div
                      class="flex justify-between text-xs text-gray-500 mt-1"
                    >
                      <span>0 (Immediate)</span>
                      <span>100</span>
                      <span>200</span>
                      <span>500 steps</span>
                    </div>
                    <div class="mt-2 p-3 bg-blue-50 rounded-lg">
                      <p class="text-xs text-blue-700">
                        <strong>💡 Tip:</strong> Setting this to 50-200 lets the
                        model learn JSON structure first before applying
                        selective masking. This can prevent degeneration issues
                        with aggressive masking.
                        {#if formData.selective_loss_masking_start_step === 0}
                          <br /><em
                            >Currently: Masking starts immediately (traditional
                            approach)</em
                          >
                        {:else}
                          <br /><em
                            >Currently: Model learns structure for {formData.selective_loss_masking_start_step}
                            steps, then masking begins</em
                          >
                        {/if}
                      </p>
                    </div>
                  </div>

                  <div>
                    <div class="flex items-center">
                      <input
                        type="checkbox"
                        id="selective_loss_verbose"
                        bind:checked={formData.selective_loss_verbose}
                        class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                      />
                      <label
                        for="selective_loss_verbose"
                        class="ml-2 block text-sm text-gray-700"
                      >
                        Verbose mode (print masking statistics)
                      </label>
                    </div>
                    <p class="text-xs text-gray-500 mt-1">
                      Display detailed token masking stats during training
                    </p>
                  </div>

                  <div class="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <h4 class="text-sm font-semibold text-blue-900 mb-2">
                      📊 What Gets Masked?
                    </h4>
                    <ul class="text-xs text-blue-800 space-y-1">
                      <li>
                        ✓ Structural: <code>{`{ } [ ] : ,`}</code> and whitespace
                        (spaces, newlines, tabs)
                      </li>
                      <li>
                        ✓ Quotes: <code>"</code> (string delimiters - purely structural)
                      </li>
                      <li>
                        ✓ Null keyword: <code>null</code> (moderate/aggressive only)
                      </li>
                      <li>
                        ✗ NOT masked: <code>true</code>, <code>false</code> (can
                        be semantic)
                      </li>
                      <li>
                        ✓ Schema keys: Field names like <code>name</code> (aggressive
                        only)
                      </li>
                    </ul>
                    <p class="text-xs text-blue-700 mt-2">
                      <strong>Example:</strong> In
                      <code>{`{"name": "John", "age": 30}`}</code>, conservative
                      mode masks
                      <code>{`{ } : , "`}</code> and spaces (~31% of tokens),
                      trains on <code>name John age 30</code>
                    </p>
                  </div>
                </div>
              {/if}
            </div>
          </div>
        {/if}

        <!-- Submit Buttons -->
        <div class="flex gap-4 pt-4">
          <Button
            type="submit"
            variant="primary"
            loading={submitting}
            disabled={submitting}
          >
            {submitting ? "Creating..." : "Start Training"}
          </Button>
          <Button href="/training" variant="secondary">Cancel</Button>
        </div>
      </form>
    </Card>
  </div>
</div>
