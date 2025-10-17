<script lang="ts">
  import { onMount } from 'svelte';
  import Button from '$lib/components/Button.svelte';
  import Card from '$lib/components/Card.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import { api, type Model } from '$lib/api/client';

  let models: Model[] = $state([]);
  let loading = $state(true);
  let error = $state('');

  onMount(async () => {
    try {
      const response = await api.getModels();
      models = response.items;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load models';
    } finally {
      loading = false;
    }
  });

  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function formatDate(dateString: string): string {
    if (dateString === 'unknown') return 'Unknown';
    return new Date(dateString).toLocaleDateString();
  }

  async function handleDelete(modelId: string) {
    if (!confirm('Are you sure you want to delete this model?')) return;
    
    try {
      await api.deleteModel(modelId);
      models = models.filter(m => m.id !== modelId);
    } catch (err) {
      alert('Failed to delete model: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  }
</script>

<svelte:head>
  <title>Models - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <!-- Header -->
  <div class="bg-white shadow">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center py-6">
        <div class="flex items-center">
          <Button href="/" variant="ghost" size="sm">← Dashboard</Button>
          <h1 class="text-3xl font-bold text-gray-900 ml-4">Models</h1>
        </div>
        <div class="flex gap-3">
          <Button href="/models/browse" variant="secondary">
            🤗 Browse HuggingFace
          </Button>
          <Button href="/models/load" variant="secondary">
            🔌 Load Model
          </Button>
          <Button href="/training/new" variant="primary">
            + Train New Model
          </Button>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    {#if loading}
      <div class="text-center py-12">
        <div class="inline-block w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin"></div>
        <p class="mt-2 text-gray-600">Loading models...</p>
      </div>
    {:else if error}
      <div class="text-center py-12">
        <div class="text-red-600 text-lg">{error}</div>
        <Button onclick={() => window.location.reload()} variant="primary" class="mt-4">
          Retry
        </Button>
      </div>
    {:else if models.length === 0}
      <div class="text-center py-12">
        <div class="text-gray-400 text-6xl mb-4">📦</div>
        <h3 class="text-xl font-semibold text-gray-900 mb-2">No models yet</h3>
        <p class="text-gray-500 mb-6">Start by training your first model to see it here.</p>
        <Button href="/training/new" variant="primary">
          Train Your First Model
        </Button>
      </div>
    {:else}
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {#each models as model}
          <Card hoverable>
            <div class="space-y-4">
              <!-- Header -->
              <div class="flex items-start justify-between">
                <div>
                  <h3 class="text-lg font-semibold text-gray-900">
                    {model.name}
                  </h3>
                  <p class="text-sm text-gray-500">{model.base_model}</p>
                </div>
                <Badge variant={model.status === 'available' ? 'success' : 'warning'}>
                  {model.status}
                </Badge>
              </div>
              
              <!-- Metrics -->
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <div class="text-xs text-gray-500">Size</div>
                  <div class="font-medium">
                    {model.size_bytes ? formatBytes(model.size_bytes) : 'Unknown'}
                  </div>
                </div>
                <div>
                  <div class="text-xs text-gray-500">Created</div>
                  <div class="font-medium">
                    {formatDate(model.created_at)}
                  </div>
                </div>
              </div>
              
              <!-- Actions -->
              <div class="flex gap-2 pt-2 border-t">
                <Button href={`/models/load?model=${encodeURIComponent(model.path)}`} variant="primary" size="sm" fullWidth>
                  🔌 Load
                </Button>
                <Button variant="secondary" size="sm" href={`/inference?model=${model.id}`}>
                  Generate
                </Button>
                <Button 
                  variant="danger" 
                  size="sm" 
                  onclick={() => handleDelete(model.id)}
                >
                  🗑️
                </Button>
              </div>
            </div>
          </Card>
        {/each}
      </div>
    {/if}
  </div>
</div>