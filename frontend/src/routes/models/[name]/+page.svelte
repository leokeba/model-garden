<script lang="ts">
  import { page } from '$app/stores';
  import Button from '$lib/components/Button.svelte';
  import Card from '$lib/components/Card.svelte';
  import { api, type Model } from '$lib/api/client';
  import { onMount } from 'svelte';

  const modelName = $derived($page.params.name);
  
  let model: Model | null = $state(null);
  let loading = $state(true);
  let error = $state('');
  
  // Text generation
  let prompt = $state('');
  let generating = $state(false);
  let generatedText = $state('');
  let generationSettings = $state({
    max_length: 100,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 50,
  });

  async function loadModel() {
    if (!modelName) return;
    
    try {
      loading = true;
      const response = await api.getModel(modelName);
      model = response;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load model';
    } finally {
      loading = false;
    }
  }

  async function generateText() {
    if (!prompt.trim() || !modelName) return;
    
    try {
      generating = true;
      generatedText = '';
      
      const response = await api.generateText(modelName, {
        prompt: prompt.trim(),
        ...generationSettings
      });
      
      if (response.success) {
        generatedText = response.data.generated_text;
      } else {
        error = 'Failed to generate text';
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to generate text';
    } finally {
      generating = false;
    }
  }

  function formatFileSize(bytes: number) {
    const sizes = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }

  function formatDate(dateString: string) {
    return new Date(dateString).toLocaleString();
  }

  onMount(() => {
    loadModel();
  });
</script>

<svelte:head>
  <title>Model: {modelName} - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <!-- Header -->
  <div class="bg-white shadow">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center py-6">
        <div class="flex items-center">
          <Button href="/models" variant="ghost" size="sm">← Models</Button>
          <h1 class="text-3xl font-bold text-gray-900 ml-4">{modelName}</h1>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    {#if loading}
      <div class="flex justify-center items-center h-64">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    {:else if error}
      <Card>
        <div class="text-center py-8">
          <p class="text-red-600 text-lg">{error}</p>
          <Button onclick={loadModel} variant="primary" class="mt-4">Try Again</Button>
        </div>
      </Card>
    {:else if model}
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Main Content -->
        <div class="lg:col-span-2 space-y-6">
          <!-- Text Generation -->
          <Card>
            <div class="p-6">
              <h2 class="text-xl font-semibold text-gray-900 mb-4">Text Generation</h2>
              
              <div class="space-y-4">
                <div>
                  <label for="prompt" class="block text-sm font-medium text-gray-700 mb-2">
                    Prompt
                  </label>
                  <textarea
                    id="prompt"
                    bind:value={prompt}
                    placeholder="Enter your prompt here..."
                    rows="4"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 resize-none"
                  ></textarea>
                </div>

                <div class="flex gap-4">
                  <Button 
                    onclick={generateText} 
                    variant="primary"
                    loading={generating}
                    disabled={generating || !prompt.trim()}
                  >
                    {generating ? 'Generating...' : 'Generate'}
                  </Button>
                  
                  <Button 
                    onclick={() => { prompt = ''; generatedText = ''; }}
                    variant="secondary"
                    disabled={generating}
                  >
                    Clear
                  </Button>
                </div>

                {#if generatedText}
                  <div>
                    <h4 class="text-sm font-medium text-gray-700 mb-2">
                      Generated Text
                    </h4>
                    <div class="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                      <p class="whitespace-pre-wrap text-gray-900">{generatedText}</p>
                    </div>
                  </div>
                {/if}
              </div>
            </div>
          </Card>

          <!-- Model Information -->
          <Card>
            <div class="p-6">
              <h2 class="text-xl font-semibold text-gray-900 mb-4">Model Information</h2>
              
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <dt class="text-sm font-medium text-gray-700">Name</dt>
                  <dd class="mt-1 text-sm text-gray-900">{model.name}</dd>
                </div>
                
                <div>
                  <dt class="text-sm font-medium text-gray-700">Type</dt>
                  <dd class="mt-1 text-sm text-gray-900">{model.type || 'Fine-tuned Model'}</dd>
                </div>
                
                <div>
                  <dt class="text-sm font-medium text-gray-700">Size</dt>
                  <dd class="mt-1 text-sm text-gray-900">{formatFileSize(model.size_bytes || model.size || 0)}</dd>
                </div>
                
                <div>
                  <dt class="text-sm font-medium text-gray-700">Created</dt>
                  <dd class="mt-1 text-sm text-gray-900">{formatDate(model.created_at)}</dd>
                </div>
                
                <div class="col-span-2">
                  <dt class="text-sm font-medium text-gray-700">Path</dt>
                  <dd class="mt-1 text-sm text-gray-900 font-mono bg-gray-50 px-2 py-1 rounded">
                    {model.path}
                  </dd>
                </div>
              </div>
            </div>
          </Card>

          <!-- Files -->
          {#if model.files && model.files.length > 0}
            <Card>
              <div class="p-6">
                <h2 class="text-xl font-semibold text-gray-900 mb-4">Model Files</h2>
                
                <div class="overflow-hidden">
                  <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                      <tr>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          File
                        </th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Size
                        </th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Modified
                        </th>
                      </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                      {#each model.files as file}
                        <tr>
                          <td class="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                            {file.name}
                          </td>
                          <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                            {formatFileSize(file.size)}
                          </td>
                          <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                            {formatDate(file.modified)}
                          </td>
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
              </div>
            </Card>
          {/if}
        </div>

        <!-- Sidebar -->
        <div class="space-y-6">
          <!-- Generation Settings -->
          <Card>
            <div class="p-6">
              <h3 class="text-lg font-semibold text-gray-900 mb-4">Generation Settings</h3>
              
              <div class="space-y-4">
                <div>
                  <label for="max_length" class="block text-sm font-medium text-gray-700 mb-1">
                    Max Length
                  </label>
                  <input
                    type="number"
                    id="max_length"
                    bind:value={generationSettings.max_length}
                    min="1"
                    max="1000"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  />
                </div>

                <div>
                  <label for="temperature" class="block text-sm font-medium text-gray-700 mb-1">
                    Temperature: {generationSettings.temperature}
                  </label>
                  <input
                    type="range"
                    id="temperature"
                    bind:value={generationSettings.temperature}
                    min="0.1"
                    max="2.0"
                    step="0.1"
                    class="w-full"
                  />
                </div>

                <div>
                  <label for="top_p" class="block text-sm font-medium text-gray-700 mb-1">
                    Top P: {generationSettings.top_p}
                  </label>
                  <input
                    type="range"
                    id="top_p"
                    bind:value={generationSettings.top_p}
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    class="w-full"
                  />
                </div>

                <div>
                  <label for="top_k" class="block text-sm font-medium text-gray-700 mb-1">
                    Top K
                  </label>
                  <input
                    type="number"
                    id="top_k"
                    bind:value={generationSettings.top_k}
                    min="1"
                    max="100"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  />
                </div>
              </div>
            </div>
          </Card>

          <!-- Actions -->
          <Card>
            <div class="p-6">
              <h3 class="text-lg font-semibold text-gray-900 mb-4">Actions</h3>
              
              <div class="space-y-2">
                <Button href="/training/new" variant="primary" fullWidth>
                  Train New Model
                </Button>
                
                <Button onclick={loadModel} variant="secondary" fullWidth>
                  Refresh
                </Button>
                
                <Button href="/models" variant="secondary" fullWidth>
                  Back to Models
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    {/if}
  </div>
</div>