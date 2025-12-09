<script lang="ts">
  import { api } from "$lib/api/client";
  import Button from "$lib/components/Button.svelte";
  import Card from "$lib/components/Card.svelte";
  import {
    DatasetCard,
    HubDatasetCard,
    PreviewModal,
    UploadModal,
  } from "$lib/components/datasets";
  import { onMount } from "svelte";

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

  type HubDataset = {
    id: string;
    author: string;
    datasetName: string;
    downloads: number;
    likes: number;
    tags: string[];
    description?: string;
    size?: string;
  };

  let datasets: Dataset[] = $state([]);
  let loading = $state(true);
  let error = $state("");
  let uploading = $state(false);
  let uploadProgress = $state(0);

  // Filtering & search
  let localSearch = $state("");
  let formatFilter = $state<"all" | "text" | "vision" | "multimodal">("all");

  // Tab state
  let activeTab = $state<"local" | "hub">("local");

  // Upload
  let selectedFile: File | null = $state(null);
  let datasetName = $state("");
  let datasetType = $state("text");
  let showUploadModal = $state(false);

  // Preview
  let selectedDataset: Dataset | null = $state(null);
  let previewData: any[] = $state([]);
  let loadingPreview = $state(false);

  // Hub browsing
  let hubSearchQuery = $state("");
  let hubFilter = $state<"all" | "text" | "vision" | "audio">("all");
  let loadingHubDataset = $state(false);

  // Derived metrics
  const totalSize = $derived(
    datasets.reduce((sum, item) => sum + (item.size || 0), 0),
  );
  const totalExamples = $derived(
    datasets.reduce((sum, item) => sum + (item.examples || 0), 0),
  );
  const lastUpdated = $derived(
    datasets.reduce((latest: number | null, item) => {
      const candidate = item.modified_at || item.created_at;
      if (!candidate) return latest;
      const timestamp = new Date(candidate).getTime();
      if (Number.isNaN(timestamp)) return latest;
      if (latest === null || timestamp > latest) return timestamp;
      return latest;
    }, null),
  );
  const visionCount = $derived(
    datasets.filter((dataset) => inferDatasetType(dataset).includes("vision"))
      .length,
  );

  function formatBytes(bytes: number): string {
    if (!bytes && bytes !== 0) return "";
    if (bytes === 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.min(
      units.length - 1,
      Math.floor(Math.log(bytes) / Math.log(1024)),
    );
    const value = bytes / Math.pow(1024, i);
    return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[i]}`;
  }

  function inferDatasetType(dataset: Dataset): string {
    const metaType = (
      dataset.metadata?.type ||
      dataset.metadata?.modality ||
      ""
    )
      .toString()
      .toLowerCase();

    if (metaType.includes("vision") || metaType.includes("image"))
      return "vision";
    if (metaType.includes("multi")) return "multimodal";
    if (metaType.includes("audio")) return "audio";
    if (metaType) return metaType;
    const format = dataset.format?.toLowerCase?.() || "";
    if (
      ["json", "jsonl", "csv", "txt", "parquet"].some((f) => format.includes(f))
    ) {
      return "text";
    }
    if (format.includes("vision") || format.includes("image")) return "vision";
    return "text";
  }

  const filteredDatasets = $derived(
    datasets.filter((dataset) => {
      const type = inferDatasetType(dataset);
      const matchesType =
        formatFilter === "all" || type.includes(formatFilter.toLowerCase());
      const matchesSearch =
        localSearch.trim() === "" ||
        dataset.name.toLowerCase().includes(localSearch.toLowerCase()) ||
        dataset.path.toLowerCase().includes(localSearch.toLowerCase());

      return matchesType && matchesSearch;
    }),
  );

  // Popular datasets from HuggingFace
  const popularDatasets: HubDataset[] = [
    {
      id: "databricks/databricks-dolly-15k",
      author: "databricks",
      datasetName: "databricks-dolly-15k",
      downloads: 850000,
      likes: 1200,
      tags: ["instruction", "text-generation", "english"],
      description: "15k instruction-following examples for fine-tuning LLMs",
      size: "13.4 MB",
    },
    {
      id: "timdettmers/openassistant-guanaco",
      author: "timdettmers",
      datasetName: "openassistant-guanaco",
      downloads: 620000,
      likes: 890,
      tags: ["instruction", "chat", "conversational"],
      description: "Open Assistant conversations optimized for fine-tuning",
      size: "22.1 MB",
    },
    {
      id: "yahma/alpaca-cleaned",
      author: "yahma",
      datasetName: "alpaca-cleaned",
      downloads: 750000,
      likes: 980,
      tags: ["instruction", "text-generation"],
      description: "Cleaned version of the Alpaca instruction dataset",
      size: "44.7 MB",
    },
    {
      id: "HuggingFaceH4/ultrachat_200k",
      author: "HuggingFaceH4",
      datasetName: "ultrachat_200k",
      downloads: 420000,
      likes: 650,
      tags: ["chat", "conversational", "instruction"],
      description: "200k high-quality multi-turn conversations",
      size: "468 MB",
    },
    {
      id: "tatsu-lab/alpaca",
      author: "tatsu-lab",
      datasetName: "alpaca",
      downloads: 950000,
      likes: 1450,
      tags: ["instruction", "text-generation"],
      description: "Stanford Alpaca instruction-following dataset",
      size: "44.2 MB",
    },
    {
      id: "Anthropic/hh-rlhf",
      author: "Anthropic",
      datasetName: "hh-rlhf",
      downloads: 380000,
      likes: 720,
      tags: ["rlhf", "preference", "chat"],
      description: "Human preference data for RLHF",
      size: "286 MB",
    },
    {
      id: "OpenAssistant/oasst1",
      author: "OpenAssistant",
      datasetName: "oasst1",
      downloads: 510000,
      likes: 850,
      tags: ["chat", "instruction", "multilingual"],
      description: "Open Assistant conversation dataset",
      size: "89.4 MB",
    },
    {
      id: "teknium/GPT4-LLM-Cleaned",
      author: "teknium",
      datasetName: "GPT4-LLM-Cleaned",
      downloads: 290000,
      likes: 540,
      tags: ["instruction", "gpt4", "cleaned"],
      description: "GPT-4 generated instruction dataset (cleaned)",
      size: "31.5 MB",
    },
    {
      id: "garage-bAInd/Open-Platypus",
      author: "garage-bAInd",
      datasetName: "Open-Platypus",
      downloads: 180000,
      likes: 420,
      tags: ["instruction", "reasoning", "stem"],
      description: "STEM and logic-focused instruction dataset",
      size: "14.7 MB",
    },
    {
      id: "llamafactory/alpaca_gpt4_en",
      author: "llamafactory",
      datasetName: "alpaca_gpt4_en",
      downloads: 340000,
      likes: 610,
      tags: ["instruction", "gpt4", "english"],
      description: "English instruction dataset generated by GPT-4",
      size: "24.9 MB",
    },
  ];

  let filteredHubDatasets = $derived(
    popularDatasets.filter((dataset) => {
      const matchesSearch =
        hubSearchQuery === "" ||
        dataset.id.toLowerCase().includes(hubSearchQuery.toLowerCase()) ||
        dataset.description
          ?.toLowerCase()
          .includes(hubSearchQuery.toLowerCase());

      const matchesFilter =
        hubFilter === "all" || dataset.tags.includes(hubFilter);

      return matchesSearch && matchesFilter;
    }),
  );

  async function loadDatasets() {
    try {
      loading = true;
      error = "";
      const response = await api.get("/datasets");
      datasets = response.datasets || [];
    } catch (err) {
      error = err instanceof Error ? err.message : "Failed to load datasets";
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
        datasetName = target.files[0].name.replace(/\.[^/.]+$/, "");
      }
    }
  }

  async function uploadDataset() {
    if (!selectedFile || !datasetName.trim()) return;

    try {
      uploading = true;
      uploadProgress = 0;

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("name", datasetName.trim());
      formData.append("type", datasetType);

      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          uploadProgress = (e.loaded / e.total) * 100;
        }
      });

      xhr.addEventListener("load", () => {
        if (xhr.status === 200) {
          closeUploadModal();
          loadDatasets();
        } else {
          error = "Upload failed: " + xhr.statusText;
        }
      });

      xhr.addEventListener("error", () => {
        error = "Upload failed: Network error";
      });

      xhr.open("POST", `${window.location.origin}/api/v1/datasets/upload`);
      xhr.send(formData);
    } catch (err) {
      error = err instanceof Error ? err.message : "Failed to upload dataset";
    } finally {
      uploading = false;
    }
  }

  function closeUploadModal() {
    showUploadModal = false;
    selectedFile = null;
    datasetName = "";
    uploadProgress = 0;
  }

  async function deleteDataset(name: string) {
    if (!confirm(`Are you sure you want to delete dataset "${name}"?`)) {
      return;
    }

    try {
      await api.delete(`/datasets/${name}`);
      await loadDatasets();
    } catch (err) {
      error = err instanceof Error ? err.message : "Failed to delete dataset";
    }
  }

  async function previewDataset(dataset: Dataset) {
    try {
      loadingPreview = true;
      selectedDataset = dataset;
      const response = await api.get(
        `/datasets/${dataset.name}/preview?limit=10`,
      );
      previewData = response.samples || [];
    } catch (err) {
      error = err instanceof Error ? err.message : "Failed to load preview";
      previewData = [];
    } finally {
      loadingPreview = false;
    }
  }

  function closePreviewModal() {
    selectedDataset = null;
    previewData = [];
  }

  async function loadDatasetFromHub(datasetId: string) {
    if (!confirm(`Load dataset "${datasetId}" from HuggingFace Hub?`)) {
      return;
    }

    try {
      loadingHubDataset = true;
      error = "";

      const response = await api.post("/datasets/from-hub", {
        dataset_id: datasetId,
        split: "train",
      });

      if (response.success) {
        activeTab = "local";
        await loadDatasets();
      } else {
        error = response.message || "Failed to load dataset from Hub";
      }
    } catch (err) {
      error =
        err instanceof Error ? err.message : "Failed to load dataset from Hub";
    } finally {
      loadingHubDataset = false;
    }
  }

  onMount(() => {
    loadDatasets();
  });
</script>

<svelte:head>
  <title>Datasets - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gradient-to-b from-gray-50 via-white to-gray-100">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
    <!-- Hero -->
    <section
      class="relative overflow-hidden rounded-3xl bg-gradient-to-r from-slate-900 via-primary-800 to-primary-600 text-white shadow-2xl border border-white/10"
    >
      <div
        class="absolute inset-0 opacity-30 bg-[radial-gradient(circle_at_top_right,#ffffff55,transparent_40%)]"
      ></div>
      <div class="relative p-8 md:p-10">
        <div
          class="flex flex-col md:flex-row md:items-center md:justify-between gap-8"
        >
          <div class="space-y-4 max-w-3xl">
            <p class="text-[11px] uppercase tracking-[0.25em] text-white/70">
              Dataset Studio
            </p>
            <div class="space-y-2">
              <h1 class="text-3xl md:text-4xl font-bold leading-tight">
                Curate, inspect, and launch training datasets
              </h1>
              <p class="text-sm md:text-base text-white/80 max-w-2xl">
                Browse local files, pull from HuggingFace Hub, and spot-check
                samples (including vision data) before training.
              </p>
            </div>
            <div class="flex flex-wrap gap-3">
              <Button
                onclick={() => (showUploadModal = true)}
                variant="primary"
              >
                🚀 Upload dataset
              </Button>
              <Button onclick={() => (activeTab = "hub")} variant="secondary">
                🤗 Browse HuggingFace Hub
              </Button>
            </div>
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full md:w-auto">
            <div
              class="rounded-2xl bg-white/10 border border-white/20 p-4 backdrop-blur"
            >
              <p class="text-xs uppercase tracking-wide text-white/70 mb-1">
                Datasets
              </p>
              <div class="text-3xl font-semibold">{datasets.length}</div>
              <p class="text-xs text-white/70 mt-1">
                {visionCount} vision-enabled
              </p>
            </div>
            <div
              class="rounded-2xl bg-white/10 border border-white/20 p-4 backdrop-blur"
            >
              <p class="text-xs uppercase tracking-wide text-white/70 mb-1">
                Total size
              </p>
              <div class="text-3xl font-semibold">
                {formatBytes(totalSize) || "–"}
              </div>
              <p class="text-xs text-white/70 mt-1">
                {totalExamples.toLocaleString()} samples
              </p>
            </div>
            <div
              class="rounded-2xl bg-white/10 border border-white/20 p-4 backdrop-blur"
            >
              <p class="text-xs uppercase tracking-wide text-white/70 mb-1">
                Last updated
              </p>
              <div class="text-3xl font-semibold">
                {lastUpdated ? new Date(lastUpdated).toLocaleDateString() : "–"}
              </div>
              <p class="text-xs text-white/70 mt-1">Auto-refresh on upload</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    {#if error}
      <div class="p-4 bg-red-50 border border-red-200 rounded-xl shadow-sm">
        <div class="flex items-start gap-2">
          <span class="text-red-600">⚠️</span>
          <div class="flex-1">
            <p class="text-sm text-red-800">{error}</p>
          </div>
          <button
            onclick={() => (error = "")}
            class="text-red-600 hover:text-red-800"
          >
            ✕
          </button>
        </div>
      </div>
    {/if}

    <!-- Tabs / actions -->
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div
        class="flex items-center gap-2 bg-white border border-gray-200 rounded-2xl p-1 shadow-sm"
      >
        <button
          onclick={() => (activeTab = "local")}
          class="px-4 py-2 text-sm font-medium rounded-xl transition-colors {activeTab ===
          'local'
            ? 'bg-primary-600 text-white shadow'
            : 'text-gray-600 hover:text-gray-900'}"
        >
          📁 My datasets
        </button>
        <button
          onclick={() => (activeTab = "hub")}
          class="px-4 py-2 text-sm font-medium rounded-xl transition-colors {activeTab ===
          'hub'
            ? 'bg-primary-600 text-white shadow'
            : 'text-gray-600 hover:text-gray-900'}"
        >
          🤗 HuggingFace Hub
        </button>
      </div>

      <div class="flex gap-2">
        <Button onclick={() => (showUploadModal = true)} variant="secondary">
          + Upload
        </Button>
        <Button
          onclick={() => (activeTab = activeTab === "local" ? "hub" : "local")}
          variant="primary"
        >
          {activeTab === "local" ? "Browse Hub" : "Back to local"}
        </Button>
      </div>
    </div>

    {#if error}
      <div class="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
        <div class="flex items-start">
          <span class="text-red-600 mr-2">⚠️</span>
          <div class="flex-1">
            <p class="text-sm text-red-800">{error}</p>
          </div>
          <button
            onclick={() => (error = "")}
            class="text-red-600 hover:text-red-800"
          >
            ✕
          </button>
        </div>
      </div>
    {/if}

    <!-- Local Datasets Tab -->
    {#if activeTab === "local"}
      <Card class="border-primary-50 shadow-sm">
        <div class="p-4 flex flex-wrap items-center gap-3">
          <div class="relative flex-1 min-w-[240px]">
            <input
              type="text"
              bind:value={localSearch}
              placeholder="Search by name or path..."
              class="w-full px-4 py-2 rounded-xl border border-gray-200 bg-gray-50 focus:bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
            <span
              class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400"
              >⌕</span
            >
          </div>

          <div class="flex flex-wrap items-center gap-2">
            {#each ["all", "text", "vision", "multimodal"] as option}
              <button
                onclick={() => (formatFilter = option)}
                class="px-3 py-2 text-sm rounded-full border transition-colors {formatFilter ===
                option
                  ? 'bg-primary-600 text-white border-primary-600 shadow'
                  : 'bg-white text-gray-700 border-gray-200 hover:border-primary-300'}"
              >
                {option === "all"
                  ? "All types"
                  : option.charAt(0).toUpperCase() + option.slice(1)}
              </button>
            {/each}
          </div>

          <div class="flex items-center gap-2 ml-auto text-xs text-gray-500">
            <span
              class="px-3 py-2 rounded-full bg-gray-100 border border-gray-200"
            >
              {filteredDatasets.length} shown
            </span>
            <Button
              onclick={() => (showUploadModal = true)}
              variant="primary"
              size="sm"
            >
              + Upload
            </Button>
          </div>
        </div>
      </Card>

      {#if loading}
        <div class="flex justify-center items-center h-64">
          <div
            class="animate-spin rounded-full h-10 w-10 border-[3px] border-primary-200 border-t-primary-600"
          ></div>
        </div>
      {:else if datasets.length === 0}
        <Card>
          <div class="text-center py-14 space-y-4">
            <div class="text-6xl">📊</div>
            <div class="space-y-2">
              <h3 class="text-xl font-semibold text-gray-800">
                No datasets yet
              </h3>
              <p class="text-gray-500">
                Upload a dataset or browse the Hub to get started.
              </p>
            </div>
            <div class="flex gap-3 justify-center">
              <Button
                onclick={() => (showUploadModal = true)}
                variant="primary"
              >
                Upload dataset
              </Button>
              <Button onclick={() => (activeTab = "hub")} variant="secondary">
                Browse Hub
              </Button>
            </div>
          </div>
        </Card>
      {:else if filteredDatasets.length === 0}
        <Card>
          <div class="flex items-center justify-between gap-4 p-6">
            <div>
              <h3 class="text-lg font-semibold text-gray-800">
                No datasets match your filters
              </h3>
              <p class="text-sm text-gray-500">
                Try another keyword or reset filters.
              </p>
            </div>
            <Button
              variant="secondary"
              onclick={() => {
                formatFilter = "all";
                localSearch = "";
              }}
            >
              Reset
            </Button>
          </div>
        </Card>
      {:else}
        <!-- Datasets Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {#each filteredDatasets as dataset}
            <DatasetCard
              {dataset}
              onDelete={deleteDataset}
              onPreview={previewDataset}
            />
          {/each}
        </div>
      {/if}
    {/if}

    <!-- HuggingFace Hub Tab -->
    {#if activeTab === "hub"}
      <!-- Search and Filter -->
      <Card class="mb-6 border-primary-50 shadow-sm">
        <div class="p-4 space-y-3">
          <div class="flex flex-wrap gap-4">
            <!-- Search -->
            <div class="flex-1 min-w-[260px]">
              <input
                type="text"
                bind:value={hubSearchQuery}
                placeholder="Search HuggingFace datasets..."
                class="w-full px-4 py-2 rounded-xl border border-gray-200 bg-gray-50 focus:bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>

            <!-- Filter -->
            <div class="w-48">
              <select
                bind:value={hubFilter}
                class="w-full px-4 py-2 rounded-xl border border-gray-200 bg-gray-50 focus:bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="all">All Types</option>
                <option value="text">Text</option>
                <option value="vision">Vision</option>
                <option value="audio">Audio</option>
              </select>
            </div>
          </div>
          <p class="text-xs text-gray-500">
            We keep a curated set of high-signal instruction and vision datasets
            handy.
          </p>
        </div>
      </Card>

      <!-- Popular Datasets -->
      <div class="mb-4">
        <h2 class="text-lg font-semibold text-gray-900 mb-2">
          Popular instruction + vision datasets
        </h2>
        <p class="text-sm text-gray-600">One-click pull into your workspace.</p>
      </div>

      {#if filteredHubDatasets.length === 0}
        <Card>
          <div class="text-center py-12">
            <div class="text-6xl mb-4">🔍</div>
            <h3 class="text-xl font-semibold text-gray-700 mb-2">
              No datasets found
            </h3>
            <p class="text-gray-500">Try adjusting your search or filters</p>
          </div>
        </Card>
      {:else}
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {#each filteredHubDatasets as dataset}
            <HubDatasetCard
              {dataset}
              loading={loadingHubDataset}
              onLoad={loadDatasetFromHub}
            />
          {/each}
        </div>
      {/if}
    {/if}
  </div>
</div>

<!-- Upload Modal -->
<UploadModal
  show={showUploadModal}
  {uploading}
  {uploadProgress}
  onClose={closeUploadModal}
  onUpload={uploadDataset}
  {selectedFile}
  {datasetName}
  {datasetType}
  onFileSelect={handleFileSelect}
  onNameChange={(name) => (datasetName = name)}
  onTypeChange={(type) => (datasetType = type)}
/>

<!-- Preview Modal -->
<PreviewModal
  dataset={selectedDataset}
  {previewData}
  loading={loadingPreview}
  onClose={closePreviewModal}
/>
