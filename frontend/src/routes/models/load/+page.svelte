<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import Button from '$lib/components/Button.svelte';
  import Card from '$lib/components/Card.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import { api, type Model } from '$lib/api/client';

  let models: Model[] = $state([]);
  let loading = $state(true);
  let error = $state('');
  let loadingModel = $state(false);
  let currentStatus = $state<any>(null);
  
  // Form state
  let selectedModelPath = $state('');
  let tensorParallelSize = $state(1);
  let gpuMemoryUtilization = $state(0.9);
  let maxModelLen = $state<number | null>(null);
  let dtype = $state('auto');
  let quantization = $state<string | null>(null);

  onMount(async () => {
    await loadData();
    
    // Pre-select model from URL parameter
    const modelParam = $page.url.searchParams.get('model');
    if (modelParam && !currentStatus?.loaded) {
      selectedModelPath = decodeURIComponent(modelParam);
    }
  });

  async function loadData() {
    try {
      loading = true;
      // Load available models
      const response = await api.getModels();
      models = response.items;
      
      // Load current inference status
      currentStatus = await api.getInferenceStatus();
      
      // If a model is already loaded, pre-fill the form with its settings
      if (currentStatus.loaded && currentStatus.model_info) {
        selectedModelPath = currentStatus.model_info.model_path;
        tensorParallelSize = currentStatus.model_info.tensor_parallel_size;
        gpuMemoryUtilization = currentStatus.model_info.gpu_memory_utilization;
        maxModelLen = currentStatus.model_info.max_model_len;
        dtype = currentStatus.model_info.dtype;
        quantization = currentStatus.model_info.quantization;
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load data';
    } finally {
      loading = false;
    }
  }

  async function handleLoadModel() {
    if (!selectedModelPath) {
      error = 'Please select a model';
      return;
    }

    try {
      loadingModel = true;
      error = '';
      
      // If a model is already loaded, unload it first
      if (currentStatus?.loaded) {
        await api.unloadModel();
      }
      
      // Load the new model
      const result = await api.loadModel({
        model_path: selectedModelPath,
        tensor_parallel_size: tensorParallelSize,
        gpu_memory_utilization: gpuMemoryUtilization,
        max_model_len: maxModelLen,
        dtype: dtype,
        quantization: quantization,
      });
      
      if (result.success) {
        // Refresh status
        await loadData();
        // Redirect to inference page
        goto('/inference');
      } else {
        error = result.message || 'Failed to load model';
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load model';
    } finally {
      loadingModel = false;
    }
  }

  async function handleUnloadModel() {
    try {
      loadingModel = true;
      error = '';
      
      const result = await api.unloadModel();
      
      if (result.success) {
        await loadData();
      } else {
        error = result.message || 'Failed to unload model';
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to unload model';
    } finally {
      loadingModel = false;
    }
  }

  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
</script>

<svelte:head>
  <title>Load Model - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <!-- Header -->
  <div class="bg-white shadow">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center py-6">
        <div class="flex items-center">
          <Button href="/models" variant="ghost" size="sm">← Models</Button>
          <h1 class="text-3xl font-bold text-gray-900 ml-4">Load Model for Inference</h1>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    {#if loading}
      <div class="text-center py-12">
        <div class="inline-block w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin"></div>
        <p class="mt-2 text-gray-600">Loading...</p>
      </div>
    {:else}
      <!-- Current Status -->
      {#if currentStatus?.loaded}
        <Card class="mb-6 bg-green-50 border-green-200">
          <div class="flex items-start justify-between">
            <div class="flex-1">
              <div class="flex items-center gap-2 mb-2">
                <Badge variant="success">Loaded</Badge>
                <h3 class="text-lg font-semibold text-gray-900">Model Currently Loaded</h3>
              </div>
              <div class="space-y-1 text-sm text-gray-600">
                <p><strong>Path:</strong> {currentStatus.model_info.model_path}</p>
                <p><strong>Tensor Parallel Size:</strong> {currentStatus.model_info.tensor_parallel_size}</p>
                <p><strong>GPU Memory:</strong> {(currentStatus.model_info.gpu_memory_utilization * 100).toFixed(0)}%</p>
                {#if currentStatus.model_info.max_model_len}
                  <p><strong>Max Length:</strong> {currentStatus.model_info.max_model_len} tokens</p>
                {/if}
                <p><strong>Data Type:</strong> {currentStatus.model_info.dtype}</p>
                {#if currentStatus.model_info.quantization}
                  <p><strong>Quantization:</strong> {currentStatus.model_info.quantization}</p>
                {/if}
              </div>
            </div>
            <Button 
              variant="danger" 
              onclick={handleUnloadModel}
              disabled={loadingModel}
            >
              {loadingModel ? 'Unloading...' : 'Unload Model'}
            </Button>
          </div>
        </Card>
      {:else}
        <Card class="mb-6 bg-yellow-50 border-yellow-200">
          <div class="flex items-center gap-2">
            <Badge variant="warning">No Model Loaded</Badge>
            <p class="text-gray-700">Load a model below to start using inference.</p>
          </div>
        </Card>
      {/if}

      <!-- Error Message -->
      {#if error}
        <div class="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      {/if}

      <!-- Load Model Form -->
      <Card>
        <h2 class="text-xl font-semibold text-gray-900 mb-6">
          {currentStatus?.loaded ? 'Load Different Model' : 'Load Model'}
        </h2>

        <form onsubmit={(e) => { e.preventDefault(); handleLoadModel(); }} class="space-y-6">
          <!-- Model Selection -->
          <div>
            <div class="block text-sm font-medium text-gray-700 mb-2">
              Select Model
            </div>
            {#if models.length === 0}
              <p class="text-sm text-gray-500">No models available. <a href="/training/new" class="text-primary-600 hover:text-primary-700">Train a model first</a>.</p>
            {:else}
              <div class="grid grid-cols-1 gap-3">
                {#each models as model}
                  <label class="relative flex items-center p-4 border rounded-lg cursor-pointer hover:bg-gray-50 transition-colors {selectedModelPath === model.path ? 'border-primary-600 bg-primary-50' : 'border-gray-300'}">
                    <input
                      type="radio"
                      name="model"
                      value={model.path}
                      bind:group={selectedModelPath}
                      class="w-4 h-4 text-primary-600 focus:ring-primary-500"
                    />
                    <div class="ml-3 flex-1">
                      <div class="flex items-center justify-between">
                        <div>
                          <div class="font-medium text-gray-900">{model.name}</div>
                          <div class="text-sm text-gray-500">{model.base_model}</div>
                          <div class="text-xs text-gray-400 mt-1">{model.path}</div>
                        </div>
                        <div class="text-right">
                          {#if model.size_bytes}
                            <div class="text-sm font-medium text-gray-700">{formatBytes(model.size_bytes)}</div>
                          {/if}
                          <Badge variant={model.status === 'available' ? 'success' : 'warning'} class="mt-1">
                            {model.status}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </label>
                {/each}
              </div>
            {/if}
          </div>

          <!-- Advanced Settings -->
          <details class="border border-gray-200 rounded-lg p-4">
            <summary class="font-medium text-gray-900 cursor-pointer">Advanced Settings</summary>
            
            <div class="mt-4 space-y-4">
              <!-- Tensor Parallel Size -->
              <div>
                <label for="tensor-parallel" class="block text-sm font-medium text-gray-700 mb-1">
                  Tensor Parallel Size
                </label>
                <input
                  id="tensor-parallel"
                  type="number"
                  min="1"
                  bind:value={tensorParallelSize}
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <p class="mt-1 text-xs text-gray-500">Number of GPUs to use for tensor parallelism (default: 1)</p>
              </div>

              <!-- GPU Memory Utilization -->
              <div>
                <label for="gpu-memory" class="block text-sm font-medium text-gray-700 mb-1">
                  GPU Memory Utilization
                </label>
                <div class="flex items-center gap-3">
                  <input
                    id="gpu-memory"
                    type="range"
                    min="0.1"
                    max="0.99"
                    step="0.05"
                    bind:value={gpuMemoryUtilization}
                    class="flex-1"
                  />
                  <span class="text-sm font-medium text-gray-700 w-16 text-right">
                    {(gpuMemoryUtilization * 100).toFixed(0)}%
                  </span>
                </div>
                <p class="mt-1 text-xs text-gray-500">Fraction of GPU memory to use (0.1-0.99, default: 0.9)</p>
              </div>

              <!-- Max Model Length -->
              <div>
                <label for="max-length" class="block text-sm font-medium text-gray-700 mb-1">
                  Max Model Length (tokens)
                </label>
                <input
                  id="max-length"
                  type="number"
                  min="128"
                  step="128"
                  bind:value={maxModelLen}
                  placeholder="Auto (from model config)"
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <p class="mt-1 text-xs text-gray-500">Maximum sequence length. Leave empty for auto-detection.</p>
              </div>

              <!-- Data Type -->
              <div>
                <label for="dtype" class="block text-sm font-medium text-gray-700 mb-1">
                  Data Type
                </label>
                <select
                  id="dtype"
                  bind:value={dtype}
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="auto">Auto</option>
                  <option value="float16">Float16</option>
                  <option value="bfloat16">BFloat16</option>
                  <option value="float32">Float32</option>
                </select>
                <p class="mt-1 text-xs text-gray-500">Precision for model weights (default: auto)</p>
              </div>

              <!-- Quantization -->
              <div>
                <label for="quantization" class="block text-sm font-medium text-gray-700 mb-1">
                  Quantization
                </label>
                <select
                  id="quantization"
                  bind:value={quantization}
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value={null}>None (Auto-detect)</option>
                  <option value="awq">AWQ</option>
                  <option value="gptq">GPTQ</option>
                  <option value="squeezellm">SqueezeLLM</option>
                  <option value="fp8">FP8</option>
                  <option value="bitsandbytes">BitsAndBytes</option>
                </select>
                <p class="mt-1 text-xs text-gray-500">Quantization method (default: auto-detect from model)</p>
              </div>
            </div>
          </details>

          <!-- Submit Button -->
          <div class="flex gap-3 pt-4">
            <Button
              type="submit"
              variant="primary"
              fullWidth
              disabled={loadingModel || !selectedModelPath}
            >
              {#if loadingModel}
                <div class="flex items-center gap-2">
                  <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Loading Model...</span>
                </div>
              {:else if currentStatus?.loaded}
                Switch to This Model
              {:else}
                Load Model
              {/if}
            </Button>
            <Button href="/models" variant="secondary">
              Cancel
            </Button>
          </div>
        </form>
      </Card>

      <!-- Help Text -->
      <div class="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h3 class="font-medium text-blue-900 mb-2">💡 Tips</h3>
        <ul class="text-sm text-blue-800 space-y-1">
          <li>• Only one model can be loaded at a time for inference</li>
          <li>• Loading a new model will automatically unload the current one</li>
          <li>• Larger models require more GPU memory - adjust GPU Memory Utilization if needed</li>
          <li>• Auto-detection works well for most models - advanced settings are optional</li>
          <li>• After loading, visit the <a href="/inference" class="underline font-medium">Inference page</a> to generate text</li>
        </ul>
      </div>
    {/if}
  </div>
</div>
