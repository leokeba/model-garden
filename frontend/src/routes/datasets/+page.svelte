<script lang="ts">
  import { onMount } from 'svelte';
  import Button from '$lib/components/Button.svelte';
  import Card from '$lib/components/Card.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import { api } from '$lib/api/client';

  type Dataset = {
    name: string;
    path: string;
    size: number;
    examples: number;
    format: string;
    created_at: string;
    modified_at?: string;
    metadata?: Record<string, any>;
  };

  let datasets: Dataset[] = $state([]);
  let loading = $state(true);
  let error = $state('');
  let uploading = $state(false);
  let uploadProgress = $state(0);

  // Upload
  let fileInput: HTMLInputElement | undefined = $state(undefined);
  let selectedFile: File | null = $state(null);
  let datasetName = $state('');
  let datasetType = $state('text');
  let showUploadModal = $state(false);

  // Preview
  let selectedDataset: Dataset | null = $state(null);
  let previewData: any[] = $state([]);
  let loadingPreview = $state(false);

  async function loadDatasets() {
    try {
      loading = true;
      error = '';
      const response = await api.get('/datasets');
      datasets = response.datasets || [];
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load datasets';
      datasets = [];
    } finally {
      loading = false;
    }
  }

  function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target.files && target.files[0]) {
      selectedFile = target.files[0];
      if (!datasetName) {
        // Auto-fill name from filename
        datasetName = target.files[0].name.replace(/\.[^/.]+$/, '');
      }
    }
  }

  async function uploadDataset() {
    if (!selectedFile || !datasetName.trim()) return;

    try {
      uploading = true;
      uploadProgress = 0;

      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('name', datasetName.trim());
      formData.append('type', datasetType);

      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          uploadProgress = (e.loaded / e.total) * 100;
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          showUploadModal = false;
          selectedFile = null;
          datasetName = '';
          uploadProgress = 0;
          loadDatasets();
        } else {
          error = 'Upload failed: ' + xhr.statusText;
        }
      });

      xhr.addEventListener('error', () => {
        error = 'Upload failed: Network error';
      });

      xhr.open('POST', 'http://localhost:8000/api/v1/datasets/upload');
      xhr.send(formData);
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to upload dataset';
    } finally {
      uploading = false;
    }
  }

  async function deleteDataset(name: string) {
    if (!confirm(`Are you sure you want to delete dataset "${name}"?`)) {
      return;
    }
    
    try {
      await api.delete(`/datasets/${name}`);
      await loadDatasets();
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to delete dataset';
    }
  }
  
  async function previewDataset(dataset: Dataset) {
    try {
      loadingPreview = true;
      selectedDataset = dataset;
      const response = await api.get(`/datasets/${dataset.name}/preview?limit=10`);
      previewData = response.samples || [];
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load preview';
      previewData = [];
    } finally {
      loadingPreview = false;
    }
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function formatDate(dateString: string): string {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  }

  function getFormatBadgeColor(format: string): 'success' | 'info' | 'warning' {
    const formats: Record<string, 'success' | 'info' | 'warning'> = {
      jsonl: 'success',
      json: 'success',
      csv: 'info',
      parquet: 'info',
      txt: 'warning',
    };
    return formats[format] || 'info';
  }

  onMount(() => {
    loadDatasets();
  });
</script>

<svelte:head>
  <title>Datasets - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50 pt-6">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <!-- Header -->
    <div class="flex justify-between items-center mb-8">
      <div>
        <h1 class="text-3xl font-bold text-gray-900">Datasets</h1>
        <p class="mt-2 text-sm text-gray-600">
          Manage training datasets for fine-tuning models
        </p>
      </div>
      <Button onclick={() => (showUploadModal = true)} variant="primary">
        + Upload Dataset
      </Button>
    </div>

    {#if error}
      <div class="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
        <div class="flex items-start">
          <span class="text-red-600 mr-2">⚠️</span>
          <div class="flex-1">
            <p class="text-sm text-red-800">{error}</p>
          </div>
          <button onclick={() => (error = '')} class="text-red-600 hover:text-red-800">
            ✕
          </button>
        </div>
      </div>
    {/if}

    {#if loading}
      <div class="flex justify-center items-center h-64">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    {:else if datasets.length === 0}
      <Card>
        <div class="text-center py-12">
          <div class="text-6xl mb-4">📊</div>
          <h3 class="text-xl font-semibold text-gray-700 mb-2">No datasets yet</h3>
          <p class="text-gray-500 mb-6">Upload a dataset to get started with training</p>
          <Button onclick={() => (showUploadModal = true)} variant="primary">
            Upload Your First Dataset
          </Button>
        </div>
      </Card>
    {:else}
      <!-- Datasets Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {#each datasets as dataset}
          <Card class="hover:shadow-lg transition-shadow">
            <div class="p-6">
              <!-- Header -->
              <div class="flex justify-between items-start mb-4">
                <div class="flex-1 min-w-0">
                  <h3 class="text-lg font-semibold text-gray-900 truncate mb-1">
                    {dataset.name}
                  </h3>
                  <Badge variant={getFormatBadgeColor(dataset.format)} size="sm">
                    {dataset.format.toUpperCase()}
                  </Badge>
                </div>
                <button
                  onclick={() => deleteDataset(dataset.name)}
                  class="text-gray-400 hover:text-red-600 transition-colors ml-2"
                  title="Delete dataset"
                >
                  🗑️
                </button>
              </div>

              <!-- Stats -->
              <div class="space-y-2 mb-4">
                <div class="flex justify-between text-sm">
                  <span class="text-gray-600">Examples:</span>
                  <span class="font-medium text-gray-900">
                    {dataset.examples.toLocaleString()}
                  </span>
                </div>
                <div class="flex justify-between text-sm">
                  <span class="text-gray-600">Size:</span>
                  <span class="font-medium text-gray-900">
                    {formatFileSize(dataset.size)}
                  </span>
                </div>
                <div class="flex justify-between text-sm">
                  <span class="text-gray-600">Created:</span>
                  <span class="font-medium text-gray-900 truncate ml-2" title={formatDate(dataset.created_at)}>
                    {new Date(dataset.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>

              <!-- Actions -->
              <div class="flex gap-2">
                <Button
                  onclick={() => previewDataset(dataset)}
                  variant="secondary"
                  size="sm"
                  fullWidth
                >
                  👁️ Preview
                </Button>
                <Button
                  href={`/training/new?dataset=${dataset.name}`}
                  variant="primary"
                  size="sm"
                  fullWidth
                >
                  🎓 Train
                </Button>
              </div>
            </div>
          </Card>
        {/each}
      </div>
    {/if}
  </div>
</div>

<!-- Upload Modal -->
{#if showUploadModal}
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
    <Card class="max-w-2xl w-full">
      <div class="p-6">
        <div class="flex justify-between items-center mb-6">
          <h2 class="text-2xl font-bold text-gray-900">Upload Dataset</h2>
          <button
            onclick={() => {
              showUploadModal = false;
              selectedFile = null;
              datasetName = '';
              uploadProgress = 0;
            }}
            class="text-gray-400 hover:text-gray-600"
            disabled={uploading}
          >
            ✕
          </button>
        </div>

        <div class="space-y-6">
          <!-- File Input -->
          <div>
            <label for="dataset-file" class="block text-sm font-medium text-gray-700 mb-2">
              Dataset File
            </label>
            <input
              type="file"
              id="dataset-file"
              bind:this={fileInput}
              onchange={handleFileSelect}
              accept=".json,.jsonl,.csv,.txt,.parquet"
              disabled={uploading}
              class="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-lg file:border-0
                file:text-sm file:font-semibold
                file:bg-primary-50 file:text-primary-700
                hover:file:bg-primary-100
                disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <p class="mt-2 text-xs text-gray-500">
              Supported formats: JSON, JSONL, CSV, TXT, Parquet
            </p>
          </div>

          <!-- Dataset Name -->
          <div>
            <label for="dataset-name" class="block text-sm font-medium text-gray-700 mb-2">
              Dataset Name
            </label>
            <input
              type="text"
              id="dataset-name"
              bind:value={datasetName}
              placeholder="my-dataset"
              disabled={uploading}
              class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:bg-gray-100"
            />
          </div>

          <!-- Dataset Type -->
          <div>
            <label for="dataset-type" class="block text-sm font-medium text-gray-700 mb-2">
              Dataset Type
            </label>
            <select
              id="dataset-type"
              bind:value={datasetType}
              disabled={uploading}
              class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:bg-gray-100"
            >
              <option value="text">Text</option>
              <option value="vision">Vision</option>
              <option value="multimodal">Multimodal</option>
            </select>
          </div>

          <!-- Upload Progress -->
          {#if uploading}
            <div>
              <div class="flex justify-between text-sm text-gray-600 mb-2">
                <span>Uploading...</span>
                <span>{Math.round(uploadProgress)}%</span>
              </div>
              <div class="w-full bg-gray-200 rounded-full h-2">
                <div
                  class="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style="width: {uploadProgress}%"
                ></div>
              </div>
            </div>
          {/if}

          <!-- Actions -->
          <div class="flex gap-3 pt-4">
            <Button
              onclick={() => {
                showUploadModal = false;
                selectedFile = null;
                datasetName = '';
                uploadProgress = 0;
              }}
              variant="secondary"
              fullWidth
              disabled={uploading}
            >
              Cancel
            </Button>
            <Button
              onclick={uploadDataset}
              variant="primary"
              fullWidth
              disabled={!selectedFile || !datasetName.trim() || uploading}
              loading={uploading}
            >
              {uploading ? 'Uploading...' : 'Upload'}
            </Button>
          </div>
        </div>
      </div>
    </Card>
  </div>
{/if}

<!-- Preview Modal -->
{#if selectedDataset}
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
    <Card class="max-w-4xl w-full max-h-[80vh] flex flex-col">
      <div class="p-6 border-b border-gray-200">
        <div class="flex justify-between items-center">
          <div>
            <h2 class="text-2xl font-bold text-gray-900">{selectedDataset.name}</h2>
            <p class="text-sm text-gray-600 mt-1">
              Showing first 10 samples
            </p>
          </div>
          <button
            onclick={() => {
              selectedDataset = null;
              previewData = [];
            }}
            class="text-gray-400 hover:text-gray-600"
          >
            ✕
          </button>
        </div>
      </div>

      <div class="flex-1 overflow-y-auto p-6">
        {#if loadingPreview}
          <div class="flex justify-center items-center h-32">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
          </div>
        {:else if previewData.length === 0}
          <p class="text-center text-gray-500 py-8">No data to preview</p>
        {:else}
          <div class="space-y-4">
            {#each previewData as sample, index}
              <div class="border border-gray-200 rounded-lg p-4 bg-gray-50">
                <div class="text-xs font-medium text-gray-500 mb-2">
                  Sample {index + 1}
                </div>
                <pre class="text-sm text-gray-900 whitespace-pre-wrap overflow-x-auto">{JSON.stringify(
                    sample,
                    null,
                    2
                  )}</pre>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <div class="p-6 border-t border-gray-200">
        <Button
          onclick={() => {
            selectedDataset = null;
            previewData = [];
          }}
          variant="secondary"
          fullWidth
        >
          Close
        </Button>
      </div>
    </Card>
  </div>
{/if}
