<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import Button from '$lib/components/Button.svelte';
  import Card from '$lib/components/Card.svelte';
  import Badge from '$lib/components/Badge.svelte';

  interface HuggingFaceModel {
    id: string;
    author: string;
    modelName: string;
    downloads: number;
    likes: number;
    tags: string[];
    pipeline_tag?: string;
    library_name?: string;
    lastModified: string;
    description?: string;
    size?: string;
  }

  // Popular models organized by category
  const popularModels: HuggingFaceModel[] = [
    // Conversational AI
    {
      id: 'microsoft/DialoGPT-large',
      author: 'microsoft',
      modelName: 'DialoGPT-large',
      downloads: 2500000,
      likes: 850,
      tags: ['conversational', 'dialogue', 'chatbot'],
      pipeline_tag: 'conversational',
      library_name: 'transformers',
      lastModified: '2024-01-15',
      description: 'Large-scale pretrained dialogue response generation model',
      size: '774M'
    },
    {
      id: 'microsoft/DialoGPT-medium',
      author: 'microsoft',
      modelName: 'DialoGPT-medium',
      downloads: 1800000,
      likes: 650,
      tags: ['conversational', 'dialogue', 'chatbot'],
      pipeline_tag: 'conversational',
      library_name: 'transformers',
      lastModified: '2024-01-15',
      description: 'Medium-scale pretrained dialogue response generation model',
      size: '345M'
    },
    
    // Text Generation
    {
      id: 'gpt2',
      author: 'openai-community',
      modelName: 'gpt2',
      downloads: 15000000,
      likes: 2100,
      tags: ['text-generation', 'gpt2', 'causal-lm'],
      pipeline_tag: 'text-generation',
      library_name: 'transformers',
      lastModified: '2024-02-01',
      description: 'GPT-2 Base model (117M parameters)',
      size: '117M'
    },
    {
      id: 'gpt2-medium',
      author: 'openai-community',
      modelName: 'gpt2-medium',
      downloads: 8500000,
      likes: 1200,
      tags: ['text-generation', 'gpt2', 'causal-lm'],
      pipeline_tag: 'text-generation',
      library_name: 'transformers',
      lastModified: '2024-02-01',
      description: 'GPT-2 Medium model (345M parameters)',
      size: '345M'
    },
    {
      id: 'gpt2-large',
      author: 'openai-community',
      modelName: 'gpt2-large',
      downloads: 4200000,
      likes: 890,
      tags: ['text-generation', 'gpt2', 'causal-lm'],
      pipeline_tag: 'text-generation',
      library_name: 'transformers',
      lastModified: '2024-02-01',
      description: 'GPT-2 Large model (774M parameters)',
      size: '774M'
    },

    // Code Generation
    {
      id: 'microsoft/CodeGPT-small-py',
      author: 'microsoft',
      modelName: 'CodeGPT-small-py',
      downloads: 950000,
      likes: 420,
      tags: ['code-generation', 'python', 'programming'],
      pipeline_tag: 'text-generation',
      library_name: 'transformers',
      lastModified: '2023-12-10',
      description: 'Small model for Python code generation',
      size: '124M'
    },

    // Instruction Following
    {
      id: 'distilgpt2',
      author: 'distilbert-base-uncased',
      modelName: 'distilgpt2',
      downloads: 6800000,
      likes: 890,
      tags: ['text-generation', 'distilgpt2', 'causal-lm'],
      pipeline_tag: 'text-generation',
      library_name: 'transformers',
      lastModified: '2024-01-20',
      description: 'Distilled version of GPT-2 (82M parameters)',
      size: '82M'
    }
  ];

  let searchQuery = $state('');
  let selectedCategory = $state('all');
  let selectedSize = $state('all');
  let filteredModels = $state<HuggingFaceModel[]>([]);

  const categories = [
    { value: 'all', label: 'All Categories' },
    { value: 'conversational', label: '💬 Conversational' },
    { value: 'text-generation', label: '✍️ Text Generation' },
    { value: 'code-generation', label: '💻 Code Generation' },
    { value: 'summarization', label: '📝 Summarization' },
    { value: 'question-answering', label: '❓ Q&A' }
  ];

  const sizes = [
    { value: 'all', label: 'All Sizes' },
    { value: 'small', label: '🟢 Small (<500M)' },
    { value: 'medium', label: '🟡 Medium (500M-2B)' },
    { value: 'large', label: '🔴 Large (>2B)' }
  ];

  function filterModels() {
    let filtered = popularModels;

    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(model => 
        model.id.toLowerCase().includes(query) ||
        model.modelName.toLowerCase().includes(query) ||
        model.description?.toLowerCase().includes(query) ||
        model.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }

    // Filter by category
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(model => 
        model.pipeline_tag === selectedCategory ||
        model.tags.includes(selectedCategory)
      );
    }

    // Filter by size
    if (selectedSize !== 'all') {
      filtered = filtered.filter(model => {
        if (!model.size) return false;
        const sizeMatch = model.size.match(/(\d+\.?\d*)([MGB])/);
        if (!sizeMatch) return false;
        
        const [, sizeNum, unit] = sizeMatch;
        const sizeInM = unit === 'G' ? parseFloat(sizeNum) * 1000 : 
                       unit === 'B' ? parseFloat(sizeNum) / 1000 : 
                       parseFloat(sizeNum);
        
        switch (selectedSize) {
          case 'small': return sizeInM < 500;
          case 'medium': return sizeInM >= 500 && sizeInM <= 2000;
          case 'large': return sizeInM > 2000;
          default: return true;
        }
      });
    }

    filteredModels = filtered;
  }

  function formatNumber(num: number): string {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  }

  function getPipelineTagIcon(tag?: string): string {
    switch (tag) {
      case 'conversational': return '💬';
      case 'text-generation': return '✍️';
      case 'code-generation': return '💻';
      case 'summarization': return '📝';
      case 'question-answering': return '❓';
      default: return '🤖';
    }
  }

  function handleLoadModel(modelId: string) {
    // Navigate to load page with the model ID pre-filled
    goto(`/models/load?hf_model=${encodeURIComponent(modelId)}`);
  }

  onMount(() => {
    filterModels();
  });

  // Update filters when inputs change
  $effect(() => {
    filterModels();
  });
</script>

<svelte:head>
  <title>Browse HuggingFace Models - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <!-- Header -->
  <div class="bg-white shadow">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center py-6">
        <div class="flex items-center">
          <Button href="/models" variant="ghost" size="sm">← Models</Button>
          <h1 class="text-3xl font-bold text-gray-900 ml-4">🤗 Browse HuggingFace Models</h1>
        </div>
        <Button href="/models/load" variant="primary">
          Load Custom Model
        </Button>
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <!-- Search and Filters -->
    <Card class="mb-8">
      <div class="p-6">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
          <!-- Search -->
          <div class="md:col-span-2">
            <label for="search" class="block text-sm font-medium text-gray-700 mb-2">
              Search Models
            </label>
            <input
              id="search"
              type="text"
              bind:value={searchQuery}
              placeholder="Search by name, description, or tags..."
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
          </div>

          <!-- Category Filter -->
          <div>
            <label for="category" class="block text-sm font-medium text-gray-700 mb-2">
              Category
            </label>
            <select
              id="category"
              bind:value={selectedCategory}
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              {#each categories as category}
                <option value={category.value}>{category.label}</option>
              {/each}
            </select>
          </div>

          <!-- Size Filter -->
          <div>
            <label for="size" class="block text-sm font-medium text-gray-700 mb-2">
              Model Size
            </label>
            <select
              id="size"
              bind:value={selectedSize}
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              {#each sizes as size}
                <option value={size.value}>{size.label}</option>
              {/each}
            </select>
          </div>
        </div>
      </div>
    </Card>

    <!-- Results Info -->
    <div class="mb-6">
      <p class="text-gray-600">
        Showing <span class="font-semibold">{filteredModels.length}</span> 
        {filteredModels.length === 1 ? 'model' : 'models'}
      </p>
    </div>

    <!-- Models Grid -->
    {#if filteredModels.length === 0}
      <Card>
        <div class="text-center py-12">
          <div class="text-gray-400 text-6xl mb-4">🔍</div>
          <h3 class="text-lg font-medium text-gray-900 mb-2">No models found</h3>
          <p class="text-gray-500 mb-4">
            Try adjusting your search criteria or browse all models.
          </p>
          <Button onclick={() => { searchQuery = ''; selectedCategory = 'all'; selectedSize = 'all'; }}>
            Clear Filters
          </Button>
        </div>
      </Card>
    {:else}
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {#each filteredModels as model}
          <Card class="hover:shadow-lg transition-shadow">
            <div class="p-6">
              <!-- Header -->
              <div class="flex items-start justify-between mb-3">
                <div class="flex-1 min-w-0">
                  <h3 class="text-lg font-semibold text-gray-900 truncate">
                    {getPipelineTagIcon(model.pipeline_tag)} {model.modelName}
                  </h3>
                  <p class="text-sm text-gray-500">by {model.author}</p>
                </div>
                {#if model.size}
                  <Badge variant="info" class="ml-2">{model.size}</Badge>
                {/if}
              </div>

              <!-- Description -->
              {#if model.description}
                <p class="text-sm text-gray-600 mb-4 line-clamp-2">
                  {model.description}
                </p>
              {/if}

              <!-- Tags -->
              <div class="flex flex-wrap gap-1 mb-4">
                {#each model.tags.slice(0, 3) as tag}
                  <Badge variant="info" class="text-xs">{tag}</Badge>
                {/each}
                {#if model.tags.length > 3}
                  <Badge variant="info" class="text-xs">+{model.tags.length - 3}</Badge>
                {/if}
              </div>

              <!-- Stats -->
              <div class="flex items-center justify-between text-sm text-gray-500 mb-4">
                <div class="flex items-center gap-4">
                  <span>📥 {formatNumber(model.downloads)}</span>
                  <span>❤️ {formatNumber(model.likes)}</span>
                </div>
                <span>{model.library_name || 'transformers'}</span>
              </div>

              <!-- Actions -->
              <div class="flex gap-2">
                <Button 
                  onclick={() => handleLoadModel(model.id)} 
                  variant="primary" 
                  fullWidth
                  size="sm"
                >
                  Load Model
                </Button>
                <a
                  href={`https://huggingface.co/${model.id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  class="btn btn-secondary btn-sm text-center"
                >
                  View on HF
                </a>
              </div>
            </div>
          </Card>
        {/each}
      </div>
    {/if}

    <!-- Help Section -->
    <div class="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <div class="p-6">
          <h3 class="text-lg font-semibold text-gray-900 mb-3">💡 Getting Started</h3>
          <ul class="text-sm text-gray-600 space-y-2">
            <li>• Browse popular models above or search by name/description</li>
            <li>• Click "Load Model" to instantly load any model for inference</li>
            <li>• Use filters to find models by category and size</li>
            <li>• All models support text generation and chat completion</li>
          </ul>
        </div>
      </Card>

      <Card>
        <div class="p-6">
          <h3 class="text-lg font-semibold text-gray-900 mb-3">🏷️ Model Categories</h3>
          <ul class="text-sm text-gray-600 space-y-2">
            <li>• <strong>Conversational:</strong> Optimized for dialogue and chat</li>
            <li>• <strong>Text Generation:</strong> General-purpose language models</li>
            <li>• <strong>Code Generation:</strong> Specialized for programming tasks</li>
            <li>• <strong>Instruction Following:</strong> Fine-tuned to follow instructions</li>
          </ul>
        </div>
      </Card>
    </div>
  </div>
</div>

<style>
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>