<script lang="ts">
  import { goto } from '$app/navigation';
  import Button from '$lib/components/Button.svelte';
  import Card from '$lib/components/Card.svelte';
  import { api } from '$lib/api/client';

  let formData = $state({
    name: '',
    base_model: 'unsloth/tinyllama-bnb-4bit',
    dataset_path: './data/sample.jsonl',
    output_dir: '',
    hyperparameters: {
      learning_rate: 0.0002,
      num_epochs: 3,
      batch_size: 2,
      max_steps: -1,
      gradient_accumulation_steps: 4,
    },
    lora_config: {
      r: 16,
      lora_alpha: 16,
    },
    from_hub: false
  });

  let submitting = $state(false);
  let error = $state('');

  const baseModels = [
    'unsloth/tinyllama-bnb-4bit',
    'unsloth/phi-2-bnb-4bit',
    'unsloth/mistral-7b-bnb-4bit',
    'unsloth/llama-2-7b-bnb-4bit',
    'unsloth/llama-3-8b-bnb-4bit',
  ];

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
      const response = await api.createTrainingJob(formData);
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
                {#each baseModels as model}
                  <option value={model}>{model}</option>
                {/each}
              </select>
            </div>

            <div>
              <label for="dataset_path" class="block text-sm font-medium text-gray-700 mb-1">
                Dataset Path *
              </label>
              <input
                type="text"
                id="dataset_path"
                bind:value={formData.dataset_path}
                placeholder="./data/my-dataset.jsonl"
                class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                required
              />
              <p class="text-xs text-gray-500 mt-1">
                Path to your JSONL dataset file
              </p>
            </div>

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