<script lang="ts">
  import { goto } from '$app/navigation';
  import Button from '$lib/components/Button.svelte';
  import Card from '$lib/components/Card.svelte';
  import { api } from '$lib/api/client';

  let formData = $state({
    name: '',
    model_type: 'text', // 'text' or 'vision'
    base_model: 'unsloth/tinyllama-bnb-4bit',
    dataset_path: './data/sample.jsonl',
    validation_dataset_path: '',
    output_dir: '',
    hyperparameters: {
      learning_rate: 0.0002,
      num_epochs: 3,
      batch_size: 2,
      max_steps: -1,
      gradient_accumulation_steps: 4,
      eval_steps: null as number | null,
    },
    lora_config: {
      r: 16,
      lora_alpha: 16,
    },
    from_hub: false,
    validation_from_hub: false,
    save_method: 'merged_16bit'
  });

  let submitting = $state(false);
  let error = $state('');

  const textModels = [
    'unsloth/tinyllama-bnb-4bit',
    'unsloth/phi-2-bnb-4bit',
    'unsloth/mistral-7b-bnb-4bit',
    'unsloth/llama-2-7b-bnb-4bit',
    'unsloth/llama-3-8b-bnb-4bit',
  ];

  const visionModels = [
    'Qwen/Qwen2.5-VL-3B-Instruct',
    'Qwen/Qwen2.5-VL-7B-Instruct',
    'Qwen/Qwen2.5-VL-72B-Instruct',
    'unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit',
    'unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit',
  ];

  // Update available models when type changes
  $effect(() => {
    if (formData.model_type === 'vision') {
      formData.base_model = visionModels[0];
      formData.dataset_path = './data/vision_dataset.jsonl';
      // Vision models need smaller batch size and larger gradient accumulation
      formData.hyperparameters.batch_size = 1;
      formData.hyperparameters.gradient_accumulation_steps = 8;
      formData.hyperparameters.learning_rate = 0.00002; // Lower LR for vision
    } else {
      formData.base_model = textModels[0];
      formData.dataset_path = './data/sample.jsonl';
      formData.hyperparameters.batch_size = 2;
      formData.hyperparameters.gradient_accumulation_steps = 4;
      formData.hyperparameters.learning_rate = 0.0002;
    }
  });

  function updateOutputDir() {
    if (formData.name && !formData.output_dir) {
      formData.output_dir = `./models/${formData.name.toLowerCase().replace(/[^a-z0-9]/g, '-')}`;
    }
  }

  async function handleSubmit(event: SubmitEvent) {
    event.preventDefault();
    
    if (!formData.name || !formData.base_model || !formData.dataset_path) {
      error = 'Please fill in all required fields';
      return;
    }

    submitting = true;
    error = '';

    try {
      const response = await api.createTrainingJob({
        ...formData,
        // Add vision flag to the request
        is_vision: formData.model_type === 'vision'
      });
      if (response.success) {
        goto(`/training/${response.data.job_id}`);
      } else {
        error = 'Failed to create training job';
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to create training job';
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
          <Button href="/training" variant="ghost" size="sm">← Training Jobs</Button>
          <h1 class="text-3xl font-bold text-gray-900 ml-4">New Training Job</h1>
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
          <h3 class="text-lg font-semibold text-gray-900 mb-4">Basic Configuration</h3>
          
          <div class="grid grid-cols-1 gap-4">
            <!-- Model Type Selector -->
            <div>
              <div class="block text-sm font-medium text-gray-700 mb-2">
                Model Type *
              </div>
              <div class="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onclick={() => formData.model_type = 'text'}
                  class={`p-4 border-2 rounded-lg text-left transition-all ${
                    formData.model_type === 'text'
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-300 hover:border-gray-400'
                  }`}
                >
                  <div class="flex items-center gap-2">
                    <div class={`w-4 h-4 rounded-full border-2 ${
                      formData.model_type === 'text'
                        ? 'border-primary-500 bg-primary-500'
                        : 'border-gray-400'
                    }`}>
                      {#if formData.model_type === 'text'}
                        <div class="w-full h-full rounded-full bg-white scale-50"></div>
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
                  onclick={() => formData.model_type = 'vision'}
                  class={`p-4 border-2 rounded-lg text-left transition-all ${
                    formData.model_type === 'vision'
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-300 hover:border-gray-400'
                  }`}
                >
                  <div class="flex items-center gap-2">
                    <div class={`w-4 h-4 rounded-full border-2 ${
                      formData.model_type === 'vision'
                        ? 'border-primary-500 bg-primary-500'
                        : 'border-gray-400'
                    }`}>
                      {#if formData.model_type === 'vision'}
                        <div class="w-full h-full rounded-full bg-white scale-50"></div>
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
              <label for="name" class="block text-sm font-medium text-gray-700 mb-1">
                Model Name *
              </label>
              <input
                type="text"
                id="name"
                bind:value={formData.name}
                oninput={updateOutputDir}
                placeholder="my-finance-model"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                required
              />
            </div>

            <div>
              <label for="base_model" class="block text-sm font-medium text-gray-700 mb-1">
                Base Model *
              </label>
              <select
                id="base_model"
                bind:value={formData.base_model}
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                required
              >
                {#if formData.model_type === 'text'}
                  {#each textModels as model}
                    <option value={model}>{model}</option>
                  {/each}
                {:else}
                  {#each visionModels as model}
                    <option value={model}>{model}</option>
                  {/each}
                {/if}
              </select>
              {#if formData.model_type === 'vision'}
                <p class="text-xs text-gray-500 mt-1">
                  🎨 Vision-language models can analyze images and text together
                </p>
              {/if}
            </div>

            <div>
              <label for="dataset_path" class="block text-sm font-medium text-gray-700 mb-1">
                Dataset Path *
              </label>
              <input
                type="text"
                id="dataset_path"
                bind:value={formData.dataset_path}
                placeholder={formData.from_hub ? 'username/dataset-name' : (formData.model_type === 'vision' ? './data/vision_dataset.jsonl' : './data/my-dataset.jsonl')}
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
                  Enter a HuggingFace dataset identifier (e.g., "username/dataset-name")<br/>
                  For specific files, use: "username/repo::train.jsonl"
                {:else if formData.model_type === 'vision'}
                  Path to your JSONL dataset with image paths/base64 or local file
                {:else}
                  Path to your JSONL dataset file
                {/if}
              </p>
            </div>

            <!-- Validation Dataset (Optional) -->
            <div>
              <label for="validation_dataset_path" class="block text-sm font-medium text-gray-700 mb-1">
                Validation Dataset Path (Optional)
              </label>
              <input
                type="text"
                id="validation_dataset_path"
                bind:value={formData.validation_dataset_path}
                placeholder={formData.validation_from_hub ? 'username/val-dataset-name' : (formData.model_type === 'vision' ? './data/vision_val_dataset.jsonl' : './data/my-val-dataset.jsonl')}
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
              <div class="mt-2 flex items-center">
                <input
                  type="checkbox"
                  id="validation_from_hub"
                  bind:checked={formData.validation_from_hub}
                  class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
                <label for="validation_from_hub" class="ml-2 block text-sm text-gray-700">
                  Load validation dataset from HuggingFace Hub
                </label>
              </div>
              <p class="text-xs text-gray-500 mt-1">
                📊 Optional: Provide a validation dataset to track validation loss during training<br/>
                {#if formData.validation_from_hub}
                  Use HuggingFace format: "username/repo" or "username/repo::validation.jsonl"
                {/if}
              </p>
            </div>

            {#if formData.model_type === 'vision'}
              <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <h4 class="text-sm font-semibold text-blue-900 mb-2">📋 Vision Dataset Format</h4>
                {#if formData.from_hub}
                  <p class="text-sm text-blue-800 mb-2">HuggingFace datasets should use OpenAI messages format with base64 images:</p>
                  <pre class="text-xs bg-blue-100 p-2 rounded overflow-x-auto"><code>{`{"messages": [{"role": "user", "content": [{"type": "image", "image": "data:image/jpeg;base64,..."}, {"type": "text", "text": "What is shown?"}]}]}`}</code></pre>
                  <p class="text-xs text-blue-700 mt-2">
                    <strong>Example:</strong> <code>Barth371/train_pop_valet_no_wrong_doc</code>
                  </p>
                {:else}
                  <p class="text-sm text-blue-800 mb-2">Your dataset should be in JSONL format with:</p>
                  <pre class="text-xs bg-blue-100 p-2 rounded overflow-x-auto"><code>{`{"text": "What is in this image?", "image": "/path/to/image.jpg", "response": "A cat sitting on a table"}`}</code></pre>
                  <p class="text-xs text-blue-700 mt-2">
                    <strong>Tip:</strong> Use <code>model-garden create-vision-dataset</code> CLI to generate sample data
                  </p>
                {/if}
              </div>
            {/if}

            <div>
              <label for="output_dir" class="block text-sm font-medium text-gray-700 mb-1">
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
          <h3 class="text-lg font-semibold text-gray-900 mb-4">Training Hyperparameters</h3>
          
          {#if formData.model_type === 'vision'}
            <div class="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p class="text-sm text-yellow-800">
                ⚠️ <strong>Vision models require:</strong> Lower batch size (1-2), higher gradient accumulation (8+), and lower learning rate (2e-5)
              </p>
            </div>
          {/if}
          
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label for="learning_rate" class="block text-sm font-medium text-gray-700 mb-1">
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
            </div>

            <div>
              <label for="num_epochs" class="block text-sm font-medium text-gray-700 mb-1">
                Epochs
              </label>
              <input
                type="number"
                id="num_epochs"
                bind:value={formData.hyperparameters.num_epochs}
                min="1"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>

            <div>
              <label for="batch_size" class="block text-sm font-medium text-gray-700 mb-1">
                Batch Size
              </label>
              <input
                type="number"
                id="batch_size"
                bind:value={formData.hyperparameters.batch_size}
                min="1"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>

            <div>
              <label for="gradient_accumulation" class="block text-sm font-medium text-gray-700 mb-1">
                Gradient Accumulation
              </label>
              <input
                type="number"
                id="gradient_accumulation"
                bind:value={formData.hyperparameters.gradient_accumulation_steps}
                min="1"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
              <p class="text-xs text-gray-500 mt-1">
                Higher for vision models (8+)
              </p>
            </div>

            <div>
              <label for="max_steps" class="block text-sm font-medium text-gray-700 mb-1">
                Max Steps
              </label>
              <input
                type="number"
                id="max_steps"
                bind:value={formData.hyperparameters.max_steps}
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
              <p class="text-xs text-gray-500 mt-1">
                -1 for full epochs
              </p>
            </div>

            <div>
              <label for="eval_steps" class="block text-sm font-medium text-gray-700 mb-1">
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
                Evaluate every N steps (only used if validation dataset provided)
              </p>
            </div>
          </div>
        </div>

        <!-- LoRA Configuration -->
        <div>
          <h3 class="text-lg font-semibold text-gray-900 mb-4">LoRA Configuration</h3>
          
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label for="lora_r" class="block text-sm font-medium text-gray-700 mb-1">
                LoRA Rank (r)
              </label>
              <input
                type="number"
                id="lora_r"
                bind:value={formData.lora_config.r}
                min="1"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>

            <div>
              <label for="lora_alpha" class="block text-sm font-medium text-gray-700 mb-1">
                LoRA Alpha
              </label>
              <input
                type="number"
                id="lora_alpha"
                bind:value={formData.lora_config.lora_alpha}
                min="1"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>
        </div>

        <!-- Model Save Options -->
        <div>
          <h3 class="text-lg font-semibold text-gray-900 mb-4">Model Save Options</h3>
          
          <div>
            <label for="save_method" class="block text-sm font-medium text-gray-700 mb-2">
              Save Method
            </label>
            <select
              id="save_method"
              bind:value={formData.save_method}
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="merged_16bit">Save Merged Model (16-bit) - Recommended</option>
              <option value="merged_4bit">Save Merged Model (4-bit) - Smaller Size</option>
              <option value="lora">Save LoRA Adapters Only - Advanced</option>
            </select>
            <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p class="text-sm text-blue-800">
                {#if formData.save_method === 'merged_16bit'}
                  <strong>✅ Merged 16-bit (Recommended):</strong> Full model with LoRA weights merged using Unsloth. Creates split files for vLLM compatibility.
                {:else if formData.save_method === 'merged_4bit'}
                  <strong>📦 Merged 4-bit:</strong> Full model with LoRA weights merged in 4-bit quantized format. Smaller file size.
                {:else}
                  <strong>🔧 LoRA Adapters Only (Advanced):</strong> Saves only the adapter weights. Requires the base model to load.
                {/if}
              </p>
            </div>
          </div>
        </div>

        <!-- Submit Buttons -->
        <div class="flex gap-4 pt-4">
          <Button 
            type="submit" 
            variant="primary" 
            loading={submitting}
            disabled={submitting}
          >
            {submitting ? 'Creating...' : 'Start Training'}
          </Button>
          <Button href="/training" variant="secondary">
            Cancel
          </Button>
        </div>
      </form>
    </Card>
  </div>
</div>