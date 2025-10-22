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
      learning_rate: 0.00002,
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
      // Vision-specific LoRA parameters (FastVisionModel)
      finetune_vision_layers: true,
      finetune_language_layers: true,
      finetune_attention_modules: true,
      finetune_mlp_modules: true,
    },
    from_hub: false,
    validation_from_hub: false,
    save_method: "merged_16bit",
    selective_loss: false,
    selective_loss_level: "conservative",
    selective_loss_schema_keys: "",
    selective_loss_masking_start_step: 0,
    selective_loss_masking_start_epoch: 0.0,
    selective_loss_verbose: false,
    early_stopping_enabled: false,
    early_stopping_patience: 3,
    early_stopping_threshold: 0.0001,
    quality_mode: false,
    load_in_16bit: false,
    load_in_8bit: false,
  });

  let submitting = $state(false);
  let error = $state("");

  // Toggle between registry and custom model input
  let useCustomModel = $state(false);

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
    // Only apply registry defaults if using registry model
    if (!useCustomModel) {
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
    } else {
      // Clear selected model info for custom models
      selectedModelInfo = null;
    }
  });

  // Update available models when type changes
  $effect(() => {
    // Only auto-select from registry if not using custom model
    if (!useCustomModel) {
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
    } else {
      // Just update dataset path for custom models
      if (formData.model_type === "vision") {
        formData.dataset_path = "./data/vision_dataset.jsonl";
      } else {
        formData.dataset_path = "./data/sample.jsonl";
      }
    }
  });

  // State for showing advanced settings
  let showAdvancedHyperparams = $state(false);
  let showAdvancedLora = $state(false);

  // Auto-update output directory when name changes
  $effect(() => {
    if (formData.name) {
      formData.output_dir = formData.name
        .toLowerCase()
        .replace(/[^a-z0-9]/g, "-");
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
        selective_loss_masking_start_epoch:
          formData.selective_loss_masking_start_epoch,
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

              <!-- Model Selection Type Toggle -->
              <div class="mb-3 flex items-center gap-4">
                <button
                  type="button"
                  onclick={() => (useCustomModel = false)}
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
                  onclick={() => (useCustomModel = true)}
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
              {:else}
                <!-- Custom Model Input -->
                <input
                  type="text"
                  id="base_model_custom"
                  bind:value={formData.base_model}
                  placeholder={formData.model_type === "vision"
                    ? "e.g., Qwen/Qwen2-VL-7B-Instruct"
                    : "e.g., meta-llama/Llama-2-7b-hf"}
                  class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 font-mono text-sm"
                  required
                />
                <div
                  class="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg"
                >
                  <p class="text-xs text-blue-800 mb-2">
                    <strong>💡 Enter any HuggingFace model ID:</strong>
                  </p>
                  <ul class="text-xs text-blue-700 space-y-1">
                    {#if formData.model_type === "vision"}
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

              {#if formData.model_type === "vision" && !useCustomModel}
                <p class="text-xs text-gray-500 mt-1">
                  🎨 Vision-language models can analyze images and text together
                </p>
              {/if}

              <!-- Show model info card if available -->
              {#if selectedModelInfo && !loadingModels && !useCustomModel}
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
              {:else if useCustomModel}
                <div
                  class="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg"
                >
                  <p class="text-xs text-yellow-800">
                    <strong>⚠️ Custom Model:</strong> Default hyperparameters may
                    not be optimal for this model. You may need to adjust learning
                    rate, batch size, and LoRA settings based on the model architecture.
                  </p>
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
                placeholder="my-model"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
              <p class="mt-1 text-sm text-gray-500">
                Model will be saved to models/{formData.output_dir ||
                  "my-model"}
              </p>
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

        <!-- Quality Settings -->
        <div>
          <h3
            class="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2"
          >
            🎯 Quality Settings
          </h3>

          <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg mb-4">
            <div class="flex items-start gap-3">
              <div class="flex-shrink-0">
                <svg
                  class="w-5 h-5 text-blue-600 mt-0.5"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fill-rule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                    clip-rule="evenodd"
                  />
                </svg>
              </div>
              <div class="flex-1">
                <p class="text-sm text-blue-800 font-medium mb-1">
                  Quality vs Memory Tradeoff
                </p>
                <p class="text-xs text-blue-700">
                  Default settings prioritize memory efficiency. Enable quality
                  mode or adjust individual settings for better accuracy at the
                  cost of 2-4x more VRAM.
                </p>
              </div>
            </div>
          </div>

          <div class="space-y-4">
            <!-- Quality Mode Toggle -->
            <div
              class="p-4 bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg"
            >
              <div class="flex items-start justify-between">
                <div class="flex-1">
                  <div class="flex items-center gap-3 mb-2">
                    <input
                      type="checkbox"
                      id="quality_mode"
                      bind:checked={formData.quality_mode}
                      class="h-5 w-5 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                    />
                    <label
                      for="quality_mode"
                      class="text-base font-semibold text-gray-900"
                    >
                      🏆 Quality Mode (Recommended for Production)
                    </label>
                  </div>
                  <p class="text-sm text-gray-700 ml-8">
                    Automatically enables 16-bit precision, better optimizer,
                    and optimized settings for maximum accuracy.
                  </p>
                  <div
                    class="mt-3 ml-8 p-3 bg-white border border-purple-100 rounded-lg"
                  >
                    <p class="text-xs font-medium text-purple-900 mb-2">
                      Quality mode includes:
                    </p>
                    <ul class="text-xs text-gray-600 space-y-1">
                      <li>✓ 16-bit precision (better than 4-bit)</li>
                      <li>
                        ✓ Standard gradient checkpointing (better than
                        "unsloth")
                      </li>
                      <li>✓ AdamW optimizer (better than 8-bit version)</li>
                      <li>✓ RSLoRA for ranks ≥ 32</li>
                      <li>⚠️ Requires ~4x more VRAM</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            <!-- Manual Precision Controls -->
            <div class="p-4 bg-gray-50 border border-gray-200 rounded-lg">
              <h4 class="text-sm font-semibold text-gray-900 mb-3">
                Manual Precision Settings
              </h4>
              <p class="text-xs text-gray-600 mb-3">
                Override individual settings (quality mode will take precedence
                if enabled)
              </p>

              <div class="space-y-3">
                <div class="flex items-start gap-3">
                  <input
                    type="checkbox"
                    id="load_in_16bit"
                    bind:checked={formData.load_in_16bit}
                    disabled={formData.quality_mode}
                    class="h-4 w-4 mt-0.5 text-primary-600 focus:ring-primary-500 border-gray-300 rounded disabled:opacity-50"
                  />
                  <div class="flex-1">
                    <label
                      for="load_in_16bit"
                      class="text-sm font-medium text-gray-700"
                    >
                      Load in 16-bit precision
                    </label>
                    <p class="text-xs text-gray-500 mt-0.5">
                      Best quality, uses 4x more VRAM than 4-bit
                    </p>
                  </div>
                </div>

                <div class="flex items-start gap-3">
                  <input
                    type="checkbox"
                    id="load_in_8bit"
                    bind:checked={formData.load_in_8bit}
                    disabled={formData.quality_mode || formData.load_in_16bit}
                    class="h-4 w-4 mt-0.5 text-primary-600 focus:ring-primary-500 border-gray-300 rounded disabled:opacity-50"
                  />
                  <div class="flex-1">
                    <label
                      for="load_in_8bit"
                      class="text-sm font-medium text-gray-700"
                    >
                      Load in 8-bit precision
                    </label>
                    <p class="text-xs text-gray-500 mt-0.5">
                      Balanced quality/memory, uses 2x more VRAM than 4-bit
                    </p>
                  </div>
                </div>

                {#if !formData.quality_mode && !formData.load_in_16bit && !formData.load_in_8bit}
                  <div
                    class="text-xs text-gray-600 bg-blue-50 border border-blue-100 rounded px-3 py-2"
                  >
                    ℹ️ Using default 4-bit quantization (most memory efficient)
                  </div>
                {/if}
              </div>
            </div>
          </div>
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

              <!-- Vision-Specific Layer Fine-tuning (FastVisionModel) -->
              {#if formData.model_type === "vision"}
                <div
                  class="mb-4 p-4 bg-purple-50 border border-purple-200 rounded-lg"
                >
                  <h4
                    class="text-md font-medium text-purple-900 mb-3 flex items-center gap-2"
                  >
                    🎨 Selective Layer Fine-tuning (Vision Models)
                  </h4>
                  <p class="text-sm text-purple-800 mb-4">
                    Control which parts of the vision-language model to train.
                    Disable layers you don't want to modify:
                  </p>

                  <div class="space-y-3">
                    <div class="flex items-start gap-3">
                      <input
                        type="checkbox"
                        id="finetune_vision_layers"
                        bind:checked={
                          formData.lora_config.finetune_vision_layers
                        }
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
                          Train the image processing layers. Disable to freeze
                          vision encoder and only adapt language model.
                        </p>
                      </div>
                    </div>

                    <div class="flex items-start gap-3">
                      <input
                        type="checkbox"
                        id="finetune_language_layers"
                        bind:checked={
                          formData.lora_config.finetune_language_layers
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
                          Train the text generation layers. Disable to freeze
                          language model and only adapt vision encoder.
                        </p>
                      </div>
                    </div>

                    <div class="flex items-start gap-3">
                      <input
                        type="checkbox"
                        id="finetune_attention_modules"
                        bind:checked={
                          formData.lora_config.finetune_attention_modules
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
                          Train attention layers (Q, K, V, O projections).
                          Disable for faster training with slightly lower
                          quality.
                        </p>
                      </div>
                    </div>

                    <div class="flex items-start gap-3">
                      <input
                        type="checkbox"
                        id="finetune_mlp_modules"
                        bind:checked={formData.lora_config.finetune_mlp_modules}
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
                          projections). Disable for faster training with
                          slightly lower quality.
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
                        <strong>All enabled (default):</strong> Full model fine-tuning
                        - best quality, slowest
                      </li>
                      <li>
                        <strong>Language only:</strong> Disable vision layers - adapt
                        text generation while keeping vision frozen
                      </li>
                      <li>
                        <strong>Vision only:</strong> Disable language layers - adapt
                        image understanding while keeping language frozen
                      </li>
                      <li>
                        <strong>Attention only:</strong> Disable MLPs - focus on
                        cross-modal attention mechanisms
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
                      Masking Start Step (Legacy): {formData.selective_loss_masking_start_step}
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
                        <strong>💡 Legacy Method:</strong> Step-based delay
                        depends on batch configuration. For more predictable
                        results, use epoch-based delay above. Setting this to
                        50-200 lets the model learn JSON structure first before
                        applying selective masking. This can prevent
                        degeneration issues with aggressive masking.
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
                    <label
                      for="selective_loss_masking_start_epoch"
                      class="block text-sm font-medium text-gray-700 mb-2"
                    >
                      Masking Start Epoch: {formData.selective_loss_masking_start_epoch}
                    </label>
                    <input
                      type="range"
                      id="selective_loss_masking_start_epoch"
                      bind:value={formData.selective_loss_masking_start_epoch}
                      min="0"
                      max="2"
                      step="0.1"
                      class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-500"
                    />
                    <div
                      class="flex justify-between text-xs text-gray-500 mt-1"
                    >
                      <span>0.0 (Immediate)</span>
                      <span>0.5</span>
                      <span>1.0</span>
                      <span>2.0 epochs</span>
                    </div>
                    <div class="mt-2 p-3 bg-green-50 rounded-lg">
                      <p class="text-xs text-green-700">
                        <strong>🎯 Recommended:</strong> Epoch-based masking is
                        more robust than step-based as it's not affected by
                        batch size or gradient accumulation changes.
                        {#if formData.selective_loss_masking_start_epoch === 0.0}
                          <br /><em>Currently: Masking starts immediately</em>
                        {:else}
                          <br /><em
                            >Currently: Model learns structure for {formData.selective_loss_masking_start_epoch}
                            epochs, then masking begins</em
                          >
                        {/if}
                        {#if formData.selective_loss_masking_start_epoch > 0.0 && formData.selective_loss_masking_start_step > 0}
                          <br /><strong class="text-amber-600">Note:</strong> Both
                          epoch and step delays are set. Epoch-based will take precedence.
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
