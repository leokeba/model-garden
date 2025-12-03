<script lang="ts">
  import { goto } from "$app/navigation";
  import type { RegistryModelInfo, TrainingBackend } from "$lib/api/client";
  import { api } from "$lib/api/client";
  import Button from "$lib/components/Button.svelte";
  import Card from "$lib/components/Card.svelte";
  import {
    BackendSelector,
    BaseModelSelector,
    DatasetInput,
    EarlyStoppingSection,
    HyperparametersSection,
    LoRAConfigSection,
    ModelTypeSelector,
    QualitySettingsSection,
    SaveOptionsSection,
    SelectiveLossSection,
    ValidationDatasetInput,
    VisionDatasetInfo,
  } from "$lib/components/training";
  import { onMount, untrack } from "svelte";

  let formData = $state({
    name: "",
    model_type: "text" as "text" | "vision",
    base_model: "unsloth/tinyllama-bnb-4bit",
    dataset_path: "./data/sample.jsonl",
    validation_dataset_path: "",
    output_dir: "",
    backend: "unsloth",
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
      weight_decay: 0.01,
      lr_scheduler_type: "linear",
      max_grad_norm: 1.0,
      adam_beta1: 0.9,
      adam_beta2: 0.999,
      adam_epsilon: 1e-8,
      dataloader_num_workers: 0,
      dataloader_pin_memory: true,
      eval_strategy: "steps",
      load_best_model_at_end: true,
      metric_for_best_model: "eval_loss",
      save_total_limit: 3,
    },
    lora_config: {
      r: 16,
      lora_alpha: 16,
      lora_dropout: 0.0,
      lora_bias: "none",
      use_rslora: false,
      use_gradient_checkpointing: "unsloth" as string | boolean,
      random_state: 42,
      target_modules: null as string[] | null,
      task_type: "CAUSAL_LM",
      loftq_config: null as any,
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
    selective_loss_masking_strategy: "epoch_based",
    selective_loss_masking_start_epoch: 0.0,
    selective_loss_mask_every_n_steps: 100,
    selective_loss_mask_for_n_steps: 50,
    selective_loss_structural_weight: 0.1,
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

  // Training backends - loaded from API
  let backends = $state<TrainingBackend[]>([]);
  let loadingBackends = $state(true);

  // State for showing advanced settings
  let showAdvancedHyperparams = $state(false);
  let showAdvancedLora = $state(false);

  // Selective loss config object for the component
  let selectiveLossConfig = $derived({
    selective_loss: formData.selective_loss,
    selective_loss_level: formData.selective_loss_level,
    selective_loss_schema_keys: formData.selective_loss_schema_keys,
    selective_loss_masking_strategy: formData.selective_loss_masking_strategy,
    selective_loss_masking_start_epoch:
      formData.selective_loss_masking_start_epoch,
    selective_loss_mask_every_n_steps:
      formData.selective_loss_mask_every_n_steps,
    selective_loss_mask_for_n_steps: formData.selective_loss_mask_for_n_steps,
    selective_loss_structural_weight: formData.selective_loss_structural_weight,
    selective_loss_verbose: formData.selective_loss_verbose,
  });

  // Load models from registry on mount
  onMount(async () => {
    // Load backends
    try {
      loadingBackends = true;
      const backendsResponse = await api.getBackends();
      backends = backendsResponse.backends;
    } catch (err) {
      console.error("Failed to load backends:", err);
      // Fallback to default backends
      backends = [
        {
          name: "unsloth",
          description:
            "Unsloth-optimized training with 2x speedup and 60% memory savings",
          supports_text: true,
          supports_vision: true,
        },
        {
          name: "transformers",
          description:
            "Standard HuggingFace Transformers + PEFT (maximum compatibility, slower than Unsloth)",
          supports_text: true,
          supports_vision: true,
        },
      ];
    } finally {
      loadingBackends = false;
    }

    // Load models
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
    const baseModel = formData.base_model;
    const modelType = formData.model_type;
    const isCustom = useCustomModel;

    untrack(() => {
      if (!isCustom) {
        const currentModels =
          modelType === "vision" ? visionModels : textModels;
        selectedModelInfo =
          currentModels.find((m) => m.id === baseModel) || null;

        if (selectedModelInfo?.training_defaults) {
          const defaults = selectedModelInfo.training_defaults;

          if (defaults.hyperparameters) {
            formData.hyperparameters = {
              ...formData.hyperparameters,
              ...defaults.hyperparameters,
            };
          }

          if (defaults.lora_config) {
            formData.lora_config = {
              ...formData.lora_config,
              ...defaults.lora_config,
            };
          }

          if (defaults.save_method) {
            formData.save_method = defaults.save_method;
          }
        }
      } else {
        selectedModelInfo = null;
      }
    });
  });

  // Update available models when type changes
  let previousModelType = $state<string | null>(null);

  $effect(() => {
    const modelType = formData.model_type;

    // Only run if model type actually changed
    if (previousModelType === modelType) return;
    previousModelType = modelType;

    untrack(() => {
      if (!useCustomModel) {
        if (modelType === "vision") {
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
        if (modelType === "vision") {
          formData.dataset_path = "./data/vision_dataset.jsonl";
        } else {
          formData.dataset_path = "./data/sample.jsonl";
        }
      }
    });
  });

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
        is_vision: formData.model_type === "vision",
        selective_loss: formData.selective_loss,
        selective_loss_level: formData.selective_loss_level,
        selective_loss_schema_keys: schema_keys_array,
        selective_loss_masking_strategy:
          formData.selective_loss_masking_strategy,
        selective_loss_masking_start_epoch:
          formData.selective_loss_masking_start_epoch,
        selective_loss_mask_every_n_steps:
          formData.selective_loss_mask_every_n_steps,
        selective_loss_mask_for_n_steps:
          formData.selective_loss_mask_for_n_steps,
        selective_loss_structural_weight:
          formData.selective_loss_structural_weight,
        selective_loss_verbose: formData.selective_loss_verbose,
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
            <ModelTypeSelector
              modelType={formData.model_type}
              onSelect={(type) => (formData.model_type = type)}
            />

            <!-- Model Name -->
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

            <!-- Base Model Selector -->
            <BaseModelSelector
              baseModel={formData.base_model}
              modelType={formData.model_type}
              {useCustomModel}
              {loadingModels}
              {loadError}
              {textModels}
              {visionModels}
              {selectedModelInfo}
              onBaseModelChange={(value) => (formData.base_model = value)}
              onUseCustomModelChange={(value) => (useCustomModel = value)}
            />

            <!-- Dataset Input -->
            <DatasetInput
              datasetPath={formData.dataset_path}
              fromHub={formData.from_hub}
              modelType={formData.model_type}
              onDatasetPathChange={(value) => (formData.dataset_path = value)}
              onFromHubChange={(value) => (formData.from_hub = value)}
            />

            <!-- Validation Dataset Input -->
            <ValidationDatasetInput
              validationDatasetPath={formData.validation_dataset_path}
              validationFromHub={formData.validation_from_hub}
              modelType={formData.model_type}
              onValidationDatasetPathChange={(value) =>
                (formData.validation_dataset_path = value)}
              onValidationFromHubChange={(value) =>
                (formData.validation_from_hub = value)}
            />

            {#if formData.model_type === "vision"}
              <VisionDatasetInfo fromHub={formData.from_hub} />
            {/if}

            <!-- Output Directory -->
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
        <HyperparametersSection
          bind:hyperparameters={formData.hyperparameters}
          modelType={formData.model_type}
          hasValidationDataset={!!formData.validation_dataset_path}
          {selectedModelInfo}
          showAdvanced={showAdvancedHyperparams}
          onToggleAdvanced={() =>
            (showAdvancedHyperparams = !showAdvancedHyperparams)}
        />

        <!-- Early Stopping (only if validation dataset provided) -->
        {#if formData.validation_dataset_path}
          <EarlyStoppingSection
            earlyStoppingEnabled={formData.early_stopping_enabled}
            earlyStoppingPatience={formData.early_stopping_patience}
            earlyStoppingThreshold={formData.early_stopping_threshold}
            onEarlyStoppingEnabledChange={(value) =>
              (formData.early_stopping_enabled = value)}
            onEarlyStoppingPatienceChange={(value) =>
              (formData.early_stopping_patience = value)}
            onEarlyStoppingThresholdChange={(value) =>
              (formData.early_stopping_threshold = value)}
          />
        {/if}

        <!-- Quality Settings -->
        <QualitySettingsSection
          qualityMode={formData.quality_mode}
          loadIn16bit={formData.load_in_16bit}
          loadIn8bit={formData.load_in_8bit}
          onQualityModeChange={(value) => (formData.quality_mode = value)}
          onLoadIn16bitChange={(value) => (formData.load_in_16bit = value)}
          onLoadIn8bitChange={(value) => (formData.load_in_8bit = value)}
        />

        <!-- Training Backend -->
        <BackendSelector
          backend={formData.backend}
          modelType={formData.model_type}
          {backends}
          loading={loadingBackends}
          onBackendChange={(value) => (formData.backend = value)}
        />

        <!-- LoRA Configuration -->
        <LoRAConfigSection
          bind:loraConfig={formData.lora_config}
          modelType={formData.model_type}
          {selectedModelInfo}
          showAdvanced={showAdvancedLora}
          onToggleAdvanced={() => (showAdvancedLora = !showAdvancedLora)}
        />

        <!-- Model Save Options -->
        <SaveOptionsSection
          saveMethod={formData.save_method}
          onSaveMethodChange={(value) => (formData.save_method = value)}
        />

        <!-- Selective Loss for Structured Outputs (Vision Models Only) -->
        {#if formData.model_type === "vision"}
          <SelectiveLossSection
            bind:config={selectiveLossConfig}
            numEpochs={formData.hyperparameters.num_epochs}
          />
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
